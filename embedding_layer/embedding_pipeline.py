"""
embedding_layer/embedding_pipeline.py

Orchestrator for the embedding layer.

Pipeline per document:
  1. Load JSON from disk — prefers encoded_*_chunks.json (encoding layer output),
     falls back to *_chunks.json if encoded files are not present.
  2. Flatten EncodedRecord dicts: merges metadata + encoding_views + linkage into
     a single flat dict so embedders can access all fields directly.
  3. Pydantic validation at disk→memory boundary
  4. Detect format from doc_metadata["file_format"]
  5. Route to correct embedders
  6. Dimension validation before saving
  7. Save enriched flat dicts back to encoded_*_chunks.json (preserving all
     encoding fields alongside the new embedding vectors)

Format routing:
  video (mp4/avi/mkv/mov/webm/m4v):
      embed video_segments (text) + video_frames (CLIP + text) + doc_embedding

  spreadsheet (csv/xlsx/xls/ods/tsv):
      embed table_chunks (semantic + HTML content) + doc_embedding

  document (pdf/docx/pptx):
      embed text_chunks + image_chunks + table_chunks + doc_embedding

Dimension validation:
  After embedding, every vector is checked against expected dimensions.
  Wrong-dim vectors are dropped with an error log (not crash).
  Pydantic validators on the models would also catch these at save time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from embedding_layer.embedders.text_embedder  import TextEmbedder
from embedding_layer.embedders.image_embedder import ImageEmbedder
from embedding_layer.embedders.video_embedder import VideoEmbedder

from shared.models.metadata_models import (
    TEXT_EMBED_DIM,
    CLIP_EMBED_DIM,
    validate_doc_metadata,
    validate_text_chunks,
    validate_image_chunks,
    validate_table_chunks,
    validate_video_segments,
    validate_video_frames,
)

VIDEO_FORMATS = {"mp4", "avi", "mkv", "mov", "webm", "m4v", "video"}
SHEET_FORMATS = {"csv", "xlsx", "xls", "ods", "tsv"}

# Encoded filename map: raw name → encoded name
ENCODED_FILENAMES = {
    "text_chunks.json":   "encoded_text_chunks.json",
    "image_chunks.json":  "encoded_image_chunks.json",
    "table_chunks.json":  "encoded_table_chunks.json",
    "video_segments.json": "encoded_video_segments.json",
    "video_frames.json":  "encoded_video_frames.json",
}


def _load(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_encoded_or_raw(metadata_dir: Path, raw_name: str) -> tuple:
    """Load encoded file if available, otherwise fall back to raw.

    Returns:
        (data, is_encoded): data is list of dicts, is_encoded indicates source.
    """
    encoded_name = ENCODED_FILENAMES.get(raw_name, raw_name)
    encoded_path = metadata_dir / encoded_name
    raw_path     = metadata_dir / raw_name

    if encoded_path.exists():
        data = _load(encoded_path) or []
        return data, True, encoded_path
    data = _load(raw_path) or []
    return data, False, raw_path


def _flatten_encoded_record(rec: dict) -> dict:
    """Flatten an EncodedRecord dict into a single dict for embedding.

    EncodedRecord structure:
        chunk_id, doc_id, source_id, source_modality, file_format,
        structure_unit_id, encoding_views{...}, linkage{...},
        promoted_fields{...}, metadata{...original chunk fields...}

    Returns a flat dict with:
        - all original chunk fields from metadata (base)
        - all encoding_views fields overlaid (fused_retrieval_text, retrieval_text, etc.)
        - all linkage fields overlaid (section_id, related_figures, etc.)
        - promoted_fields preserved as a top-level dict
        - top-level identity fields (chunk_id, doc_id, source_modality, file_format)
    """
    if "encoding_views" not in rec:
        return rec  # Already a flat raw chunk

    flat = dict(rec.get("metadata", {}))

    # Overlay encoding views (fused_retrieval_text, retrieval_text, etc.)
    for k, v in (rec.get("encoding_views") or {}).items():
        flat[k] = v

    # Overlay linkage fields (section_id, related_figures, related_tables, etc.)
    for k, v in (rec.get("linkage") or {}).items():
        flat[k] = v

    # Preserve promoted_fields for downstream indexing
    flat["promoted_fields"] = rec.get("promoted_fields", {})

    # Ensure top-level identity fields take precedence
    for key in ("chunk_id", "doc_id", "source_id", "source_modality",
                "file_format", "structure_unit_id"):
        if rec.get(key):
            flat[key] = rec[key]

    return flat


def _save(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _validate_text_dim(chunks: List[Dict]) -> List[Dict]:
    """Drop text_embedding vectors with wrong dimension."""
    for c in chunks:
        emb = c.get("text_embedding")
        if emb is not None and len(emb) != TEXT_EMBED_DIM:
            logger.error(
                f"  [DimCheck] {c.get('chunk_id')} text_embedding "
                f"{len(emb)}-dim ≠ {TEXT_EMBED_DIM} — dropping"
            )
            c["text_embedding"] = None
    return chunks


def _validate_clip_dim(chunks: List[Dict]) -> List[Dict]:
    """Drop clip_embedding vectors with wrong dimension."""
    for c in chunks:
        emb = c.get("clip_embedding")
        if emb is not None and len(emb) != CLIP_EMBED_DIM:
            logger.error(
                f"  [DimCheck] {c.get('chunk_id')} clip_embedding "
                f"{len(emb)}-dim ≠ {CLIP_EMBED_DIM} — dropping"
            )
            c["clip_embedding"] = None
    return chunks


def _validate_html_dim(chunks: List[Dict]) -> List[Dict]:
    """Drop html_text_embedding vectors with wrong dimension."""
    for c in chunks:
        emb = c.get("html_text_embedding")
        if emb is not None and len(emb) != TEXT_EMBED_DIM:
            logger.error(
                f"  [DimCheck] {c.get('chunk_id')} html_text_embedding "
                f"{len(emb)}-dim ≠ {TEXT_EMBED_DIM} — dropping"
            )
            c["html_text_embedding"] = None
    return chunks



def _merge_enrich_fields(enrich_list: list, dict_list: list) -> list:
    """Merge enrichment fields from original text_chunks.json into encoded dicts."""
    _ENRICH = {
        "evidence_role_confidence", "contextual_summary_confidence",
        "detected_codes_confidence", "salience_score", "evidence_role",
        "contextual_summary", "detected_codes", "entities",
        "image_caption_confidence", "depicted_component_confidence",
        "visible_annotations_confidence", "frame_role_confidence",
        "table_summary_confidence", "table_purpose_confidence",
    }
    if not enrich_list or not dict_list:
        return dict_list
    enrich_by_id = {
        r.get("chunk_id"): r
        for r in enrich_list
        if isinstance(r, dict) and r.get("chunk_id")
    }
    for d in dict_list:
        src = enrich_by_id.get(d.get("chunk_id", ""))
        if src:
            for field in _ENRICH:
                if field in src and src[field] is not None and (d.get(field) is None or d.get(field) == 0.0):
                    d[field] = src[field]
    return dict_list


def _merge_encoding_fields(raw_list: list, dict_list: list) -> list:
    """
    Re-merge encoding fields (bm25_text, title_text, sibling_chunk_ids etc.)
    from flattened raw records back onto Pydantic-validated dicts.
    Pydantic strips any field not in the model — this restores them.
    """
    if not raw_list or not dict_list:
        return dict_list
    raw_by_id = {
        r.get("chunk_id"): r
        for r in raw_list
        if isinstance(r, dict) and r.get("chunk_id")
    }
    encoding_fields = {
        "bm25_text", "title_text", "retrieval_text", "fused_retrieval_text",
        "summary_text", "entity_text", "rerank_text", "section_context_text",
        "local_context_text", "sibling_chunk_ids", "neighbor_chunk_ids",
        "related_section_ids", "semantic_anchor", "structure_unit_title",
    }
    ENCODING_FIELDS = {
        "bm25_text", "title_text", "retrieval_text", "fused_retrieval_text",
        "summary_text", "entity_text", "rerank_text", "section_context_text",
        "local_context_text", "sibling_chunk_ids", "neighbor_chunk_ids",
        "related_section_ids", "semantic_anchor", "structure_unit_title",
    }
    merged = []
    for d in dict_list:
        raw = raw_by_id.get(d.get("chunk_id", ""))
        if raw:
            # Case 1: nested encoding_views/linkage dicts
            for k, v in (raw.get("encoding_views") or {}).items():
                if k in ENCODING_FIELDS and v and not d.get(k):
                    d[k] = v
            for k, v in (raw.get("linkage") or {}).items():
                if k in ENCODING_FIELDS and v and not d.get(k):
                    d[k] = v
            # Case 2: flat dict — fields sit directly on raw
            for field in ENCODING_FIELDS:
                if field in raw and raw[field] and not d.get(field):
                    d[field] = raw[field]
        merged.append(d)
    return merged


class EmbeddingPipeline:

    def __init__(
        self,
        text_model:       str  = "text-embedding-3-small",
        clip_model:       str  = "ViT-B-32",
        clip_pretrained:  str  = "openai",
        skip_images:      bool = False,
        skip_tables:      bool = False,
        skip_video_frames: bool = False,
        dry_run:          bool = False,
    ):
        self.skip_images       = skip_images
        self.skip_tables       = skip_tables
        self.skip_video_frames = skip_video_frames
        self.dry_run           = dry_run

        if not dry_run:
            self.text_embedder  = TextEmbedder(model=text_model)
            if not skip_images:
                self.image_embedder = ImageEmbedder(
                    clip_model=clip_model,
                    clip_pretrained=clip_pretrained,
                    text_model=text_model,
                )
                self.video_embedder = VideoEmbedder(
                    text_embedder=self.text_embedder,
                    image_embedder=self.image_embedder,
                )
            else:
                self.image_embedder = None
                self.video_embedder = None

    def process_document(self, doc_output_dir: Path) -> Dict[str, Any]:
        doc_name     = doc_output_dir.name
        metadata_dir = doc_output_dir / "metadata"

        if not metadata_dir.exists():
            logger.warning(f"  No metadata/ in {doc_output_dir}")
            return {}

        logger.info(f"\n{'='*55}")
        logger.info(f"  Embedding: {doc_name}")
        logger.info(f"{'='*55}")

        # ── Load JSON: prefer encoded files, fall back to raw ─────────
        raw_doc  = _load(metadata_dir / "doc_metadata.json") or {}

        raw_text, text_encoded, text_save_path = _load_encoded_or_raw(metadata_dir, "text_chunks.json")
        raw_img,  img_encoded,  img_save_path  = _load_encoded_or_raw(metadata_dir, "image_chunks.json")
        raw_tbl,  tbl_encoded,  tbl_save_path  = _load_encoded_or_raw(metadata_dir, "table_chunks.json")
        raw_segs, segs_encoded, segs_save_path = _load_encoded_or_raw(metadata_dir, "video_segments.json")
        raw_frms, frms_encoded, frms_save_path = _load_encoded_or_raw(metadata_dir, "video_frames.json")

        # Load original enrichment files for merging enrichment fields
        # encoded_text_chunks.json has None for enrichment fields because
        # Pydantic stripped them during encoding. The original text_chunks.json
        # has the real values (erc=0.96 etc.) written by the enrichment layer.
        enrich_text = _load(metadata_dir / "text_chunks.json") or []
        enrich_img  = _load(metadata_dir / "image_chunks.json") or []
        enrich_tbl  = _load(metadata_dir / "table_chunks.json") or []

        if text_encoded or img_encoded or tbl_encoded:
            logger.info(f"  Using encoded files (encoding layer output)")

        # Flatten EncodedRecord dicts → plain dicts with all fields accessible
        raw_text = [_flatten_encoded_record(r) for r in raw_text]
        raw_img  = [_flatten_encoded_record(r) for r in raw_img]
        raw_tbl  = [_flatten_encoded_record(r) for r in raw_tbl]
        raw_segs = [_flatten_encoded_record(r) for r in raw_segs]
        raw_frms = [_flatten_encoded_record(r) for r in raw_frms]

        # ── Pydantic boundary validation ───────────────────────────────
        try:
            doc_model = validate_doc_metadata(raw_doc)
        except Exception as e:
            logger.error(f"  doc_metadata invalid: {e} — aborting {doc_name}")
            return {}

        text_models = validate_text_chunks(raw_text)
        img_models  = validate_image_chunks(raw_img)
        tbl_models  = validate_table_chunks(raw_tbl)
        seg_models  = validate_video_segments(raw_segs)
        frm_models  = validate_video_frames(raw_frms)

        logger.info(
            f"  Validated: text={len(text_models)} img={len(img_models)} "
            f"tbl={len(tbl_models)} seg={len(seg_models)} frames={len(frm_models)}"
        )

        if self.dry_run:
            logger.info(f"[DRY RUN] Would embed {doc_name}")
            return {"doc_name": doc_name, "dry_run": True}

        # Work on plain dicts
        doc_meta     = doc_model.to_dict()
        text_chunks  = [m.to_dict() for m in text_models]
        image_chunks = [m.to_dict() for m in img_models]
        table_chunks = [m.to_dict() for m in tbl_models]
        vid_segments = [m.to_dict() for m in seg_models]
        vid_frames   = [m.to_dict() for m in frm_models]

        # ── Merge encoding_views + linkage back after Pydantic strips them ──
        # Pydantic validation above strips fields not in the model schema.
        # bm25_text, title_text, sibling_chunk_ids, related_section_ids etc.
        # are encoding layer additions — they get stripped and lost.
        # We re-merge them from the raw (already-flattened) dicts so these
        # signals survive into the saved encoded files and ChromaDB metadata.
        text_chunks  = _merge_encoding_fields(raw_text,   text_chunks)
        image_chunks = _merge_encoding_fields(raw_img,    image_chunks)
        table_chunks = _merge_encoding_fields(raw_tbl,    table_chunks)
        vid_segments = _merge_encoding_fields(raw_segs,   vid_segments)
        vid_frames   = _merge_encoding_fields(raw_frms,   vid_frames)
        # Merge enrichment fields from original files (has real erc values)
        text_chunks  = _merge_enrich_fields(enrich_text, text_chunks)
        image_chunks = _merge_enrich_fields(enrich_img,  image_chunks)
        table_chunks = _merge_enrich_fields(enrich_tbl,  table_chunks)

        fmt = (doc_meta.get("file_format") or "").lower().strip(".")

        # ── Format-specific embedding ──────────────────────────────────
        if fmt in VIDEO_FORMATS:
            vid_segments = self.video_embedder.embed_segments(vid_segments)
            if not self.skip_video_frames and self.video_embedder:
                vid_frames = self.video_embedder.embed_frames(
                    vid_frames, doc_output_dir
                )
            doc_meta = self.text_embedder.embed_doc(doc_meta)

        elif fmt in SHEET_FORMATS:
            if not self.skip_tables:
                table_chunks = self.text_embedder.embed_table_chunks(table_chunks)
            doc_meta = self.text_embedder.embed_doc(doc_meta)

        else:
            # PDF / DOCX / PPTX
            text_chunks = self.text_embedder.embed_text_chunks(text_chunks)
            if not self.skip_images and self.image_embedder:
                image_chunks = self.image_embedder.embed_chunks(
                    image_chunks, doc_output_dir
                )
            if not self.skip_tables:
                table_chunks = self.text_embedder.embed_table_chunks(table_chunks)
            doc_meta = self.text_embedder.embed_doc(doc_meta)

        # ── Dimension validation before saving ────────────────────────
        text_chunks  = _validate_text_dim(text_chunks)
        image_chunks = _validate_text_dim(_validate_clip_dim(image_chunks))
        table_chunks = _validate_text_dim(_validate_html_dim(table_chunks))
        vid_segments = _validate_text_dim(vid_segments)
        vid_frames   = _validate_text_dim(_validate_clip_dim(vid_frames))

        # ── Save: write to encoded paths (or raw if encoding never ran) ─
        _save(doc_meta,     metadata_dir / "doc_metadata.json")
        _save(text_chunks,  text_save_path)
        _save(image_chunks, img_save_path)
        _save(table_chunks, tbl_save_path)
        _save(vid_segments, segs_save_path)
        _save(vid_frames,   frms_save_path)

        result = {
            "doc_name":       doc_name,
            "format":         fmt,
            "text_embedded":  sum(1 for c in text_chunks  if c.get("text_embedding")),
            "img_embedded":   sum(1 for c in image_chunks if c.get("clip_embedding")),
            "tbl_embedded":   sum(1 for c in table_chunks if c.get("text_embedding")),
            "seg_embedded":   sum(1 for s in vid_segments if s.get("text_embedding")),
            "frame_embedded": sum(1 for f in vid_frames   if f.get("clip_embedding")),
            "doc_embedded":   bool(doc_meta.get("doc_embedding")),
        }
        logger.success(
            f"  Embedding saved: {doc_name} | "
            f"text={result['text_embedded']} img={result['img_embedded']} "
            f"tbl={result['tbl_embedded']} seg={result['seg_embedded']} "
            f"frames={result['frame_embedded']}"
        )
        return result