"""
indexing_layer/indexing_pipeline.py

Loads processed JSON output and upserts into ChromaDB.

File loading order:
  Prefers encoded_*_chunks.json (output of encoding + embedding layers) which
  contain richer retrieval views, linkage metadata, and promoted_fields.
  Falls back to *_chunks.json if encoded files are not present.

Format routing:
  video → upsert_document + upsert_video_segments + upsert_video_frames
  spreadsheet → upsert_document + upsert_table_chunks
  document → upsert_document + upsert_text_chunks + upsert_image_chunks + upsert_table_chunks

Pydantic validation at disk→memory boundary before any upsert.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from indexing_layer.utils.chroma_store import ChromaStore
from shared.models.metadata_models import (
    validate_doc_metadata,
    validate_text_chunks,
    validate_image_chunks,
    validate_table_chunks,
    validate_video_segments,
    validate_video_frames,
)

VIDEO_FORMATS = {"mp4", "avi", "mkv", "mov", "webm", "m4v"}
SHEET_FORMATS = {"csv", "xlsx", "xls", "ods", "tsv"}

# Prefer encoded files (encoding + embedding layer output) over raw files
ENCODED_FILENAMES = {
    "text_chunks.json":    "encoded_text_chunks.json",
    "image_chunks.json":   "encoded_image_chunks.json",
    "table_chunks.json":   "encoded_table_chunks.json",
    "video_segments.json": "encoded_video_segments.json",
    "video_frames.json":   "encoded_video_frames.json",
}


def _load(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_encoded_or_raw(metadata_dir: Path, raw_name: str) -> list:
    """Load encoded file if available, otherwise fall back to raw filename."""
    encoded_name = ENCODED_FILENAMES.get(raw_name, raw_name)
    data = _load(metadata_dir / encoded_name)
    if data is not None:
        return data
    return _load(metadata_dir / raw_name) or []




def _merge_encoding_fields(
    raw_list:  list,
    dict_list: list,
) -> list:
    """
    Merge encoding_views and linkage fields from raw encoded records
    back onto the Pydantic-validated dicts.

    Pydantic strips any field not declared in the model schema.
    encoding_views (bm25_text, title_text, retrieval_text etc) and
    linkage (sibling_chunk_ids, related_section_ids etc) are encoding
    layer additions not in TextChunkMetadata — they get stripped.

    We re-merge them here so ChromaStore receives complete dicts with
    all retrieval signals intact.
    """
    if not raw_list or not dict_list:
        return dict_list

    # Build lookup: chunk_id → raw record
    raw_by_id = {}
    for r in raw_list:
        cid = r.get("chunk_id") if isinstance(r, dict) else None
        if cid:
            raw_by_id[cid] = r

    # Fields to carry forward from raw encoded records
    ENCODING_FIELDS = {
        "bm25_text", "title_text", "retrieval_text", "fused_retrieval_text",
        "summary_text", "entity_text", "rerank_text", "section_context_text",
        "local_context_text", "sibling_chunk_ids", "neighbor_chunk_ids",
        "related_section_ids", "semantic_anchor", "structure_unit_title",
        "evidence_role_confidence", "contextual_summary_confidence",
        "detected_codes_confidence", "salience_score", "evidence_role",
        "contextual_summary", "detected_codes", "entities",
        "image_caption_confidence", "depicted_component_confidence",
        "visible_annotations_confidence", "frame_role_confidence",
        "table_summary_confidence", "table_purpose_confidence",
    }

    merged = []
    for d in dict_list:
        cid = d.get("chunk_id", "")
        raw = raw_by_id.get(cid)
        if raw and isinstance(raw, dict):
            # Case 1: encoded file has nested encoding_views/linkage dicts
            for k, v in (raw.get("encoding_views") or {}).items():
                if k in ENCODING_FIELDS and v and not d.get(k):
                    d[k] = v
            for k, v in (raw.get("linkage") or {}).items():
                if k in ENCODING_FIELDS and v and not d.get(k):
                    d[k] = v
            # Case 2: encoded file is FLAT (embedding layer saves flat dicts)
            # bm25_text, title_text etc. sit directly on the raw dict
            for field in ENCODING_FIELDS:
                if field in raw and raw[field] is not None and (d.get(field) is None or d.get(field) == 0.0):
                    d[field] = raw[field]
        merged.append(d)
    return merged


def _enrich_section_map_with_anchors(doc_meta: dict, struct_units: list) -> dict:
    """
    Merge semantic_anchor, section_summary, AND structure_unit_id from
    structure_units into section_map entries.

    structure_unit_id is critical for sections_collection ID assignment:
    PDFs use _sec_NNN, PPTs use _slide_NNN — carrying the ID through here
    ensures chroma_store generates the correct section ID that matches
    the section_id / structure_unit_id stored in each chunk.
    """
    if not struct_units or not isinstance(struct_units, list):
        return doc_meta
    section_map = doc_meta.get('section_map', {})
    if not section_map:
        return doc_meta
    unit_lookup = {}
    for unit in struct_units:
        if not isinstance(unit, dict):
            continue
        title = (unit.get('title') or unit.get('structure_unit_title') or '').strip()
        if title:
            unit_lookup[title] = {
                'semantic_anchor':   (unit.get('semantic_anchor') or '').strip(),
                'section_summary':   (unit.get('section_summary') or unit.get('summary') or '').strip(),
                'structure_unit_id': (unit.get('structure_unit_id') or '').strip(),
            }
    enriched = 0
    for heading, entry in section_map.items():
        if not isinstance(entry, dict):
            continue
        unit = unit_lookup.get(heading, {})
        if unit.get('semantic_anchor') and not entry.get('semantic_anchor'):
            entry['semantic_anchor'] = unit['semantic_anchor']
            enriched += 1
        if unit.get('section_summary') and not entry.get('section_summary'):
            entry['section_summary'] = unit['section_summary']
        if unit.get('structure_unit_id') and not entry.get('structure_unit_id'):
            entry['structure_unit_id'] = unit['structure_unit_id']
    if enriched:
        from loguru import logger
        logger.info(f'  section_map enriched: {enriched} sections got semantic_anchor + structure_unit_id')
    doc_meta['section_map'] = section_map
    return doc_meta


class IndexingPipeline:

    def __init__(
        self,
        chroma_store:  ChromaStore,
        output_dir:    str  = "./output",
        text_embedder: Optional[Any] = None,
    ):
        """
        Args:
            chroma_store:  Initialised ChromaStore.
            output_dir:    Root output directory from the pipeline.
            text_embedder: Optional pre-built TextEmbedder to share with ChromaStore
                           for inline section / doc embedding. If omitted, ChromaStore
                           will lazy-init its own instance on first need.
        """
        self.store      = chroma_store
        self.output_dir = Path(output_dir)
        # Wire shared embedder into store so it is never re-instantiated per call
        if text_embedder is not None and chroma_store._embedder is None:
            chroma_store._embedder = text_embedder

    def index_document(self, doc_output_dir: Path) -> bool:
        """
        Load, validate, and upsert one document into ChromaDB.
        Returns True on success.
        """
        metadata_dir = doc_output_dir / "metadata"
        if not metadata_dir.exists():
            logger.warning(f"  No metadata/ in {doc_output_dir} — skipping")
            return False

        doc_name = doc_output_dir.name
        logger.info(f"\n{'='*55}")
        logger.info(f"  Indexing: {doc_name}")
        logger.info(f"{'='*55}")

        # ── Load JSON: prefer encoded files, fall back to raw ─────────
        raw_doc    = _load(metadata_dir / "doc_metadata.json") or {}
        raw_struct  = _load(metadata_dir / "structure_units.json") or []
        raw_text   = _load_encoded_or_raw(metadata_dir, "text_chunks.json")
        raw_img    = _load_encoded_or_raw(metadata_dir, "image_chunks.json")
        raw_tbl    = _load_encoded_or_raw(metadata_dir, "table_chunks.json")
        raw_segs   = _load_encoded_or_raw(metadata_dir, "video_segments.json")
        raw_frms   = _load_encoded_or_raw(metadata_dir, "video_frames.json")

        # ── Pydantic boundary validation ───────────────────────────────
        try:
            doc_model = validate_doc_metadata(raw_doc)
        except Exception as e:
            logger.error(f"  doc_metadata invalid: {e} — skipping {doc_name}")
            return False

        text_models = validate_text_chunks(raw_text)
        img_models  = validate_image_chunks(raw_img)
        tbl_models  = validate_table_chunks(raw_tbl)
        seg_models  = validate_video_segments(raw_segs)
        frm_models  = validate_video_frames(raw_frms)

        logger.info(
            f"  Validated: text={len(text_models)} img={len(img_models)} "
            f"tbl={len(tbl_models)} seg={len(seg_models)} frames={len(frm_models)}"
        )

        # Plain dicts for ChromaStore
        doc_meta     = doc_model.to_dict()
        text_chunks  = [m.to_dict() for m in text_models]
        image_chunks = [m.to_dict() for m in img_models]
        table_chunks = [m.to_dict() for m in tbl_models]
        vid_segments = [m.to_dict() for m in seg_models]
        vid_frames   = [m.to_dict() for m in frm_models]

        # ── Enrich section_map with semantic_anchor from structure_units ──
        # section_map (from doc_metadata.json) has summary + keywords but
        # NO semantic_anchor. structure_units.json has semantic_anchor from
        # DocMetaEnricher. Merge them so sections_collection gets rich
        # section embeddings (heading + anchor + summary) instead of
        # just heading alone.
        doc_meta = _enrich_section_map_with_anchors(doc_meta, raw_struct)

        # ── Merge encoding_views + linkage from raw encoded records ───
        # Pydantic validation (above) strips fields not in the model —
        # encoding_views (bm25_text, title_text, retrieval_text) and
        # linkage (sibling_chunk_ids, related_section_ids, neighbor_chunk_ids)
        # are stripped because TextChunkMetadata has no such fields.
        # We merge them back from the raw dicts AFTER validation so
        # ChromaStore can store them as metadata.
        text_chunks  = _merge_encoding_fields(raw_text,  text_chunks)
        image_chunks = _merge_encoding_fields(raw_img,   image_chunks)
        table_chunks = _merge_encoding_fields(raw_tbl,   table_chunks)
        vid_segments = _merge_encoding_fields(raw_segs,  vid_segments)
        vid_frames   = _merge_encoding_fields(raw_frms,  vid_frames)

        fmt = (doc_meta.get("file_format") or "").lower().strip(".")

        # ── Upsert by format ───────────────────────────────────────────
        self.store.upsert_document(doc_meta)

        if fmt in VIDEO_FORMATS:
            self.store.upsert_video_segments(vid_segments, doc_meta)
            self.store.upsert_video_frames(vid_frames, doc_meta)

        elif fmt in SHEET_FORMATS:
            self.store.upsert_table_chunks(table_chunks, doc_meta)

        else:
            self.store.upsert_text_chunks(text_chunks, doc_meta)
            self.store.upsert_image_chunks(image_chunks, doc_meta)
            self.store.upsert_table_chunks(table_chunks, doc_meta)

        logger.success(f"  Indexed: {doc_name}")
        return True

    def index_all(
        self,
        output_path: Optional[Path] = None,
        doc_name:    Optional[str]  = None,
        dry_run:     bool           = False,
    ) -> Dict[str, int]:
        output_path = output_path or self.output_dir
        if doc_name:
            doc_dirs = [output_path / doc_name]
        else:
            doc_dirs = sorted([
                d for d in output_path.iterdir()
                if d.is_dir() and (d / "metadata").exists()
            ])

        logger.info(f"Indexing {len(doc_dirs)} document(s) into ChromaDB")
        stats = {"total": len(doc_dirs), "succeeded": 0, "failed": 0, "skipped": 0}

        for doc_dir in doc_dirs:
            if dry_run:
                logger.info(f"[DRY RUN] Would index: {doc_dir.name}")
                stats["skipped"] += 1
                continue
            try:
                ok = self.index_document(doc_dir)
                if ok:
                    stats["succeeded"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                logger.error(f"Failed to index {doc_dir.name}: {e}")
                stats["failed"] += 1

        return stats

    def get_stats(self) -> Dict[str, int]:
        return self.store.get_collection_stats()