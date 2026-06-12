"""
enrichment_layer/enrichment_pipeline.py

Orchestrator for the enrichment layer.

Pipeline per document:
  1. Load JSON from disk
  2. Pydantic validation at disk→memory boundary (catch corrupt files early)
  3. Detect format from doc_metadata["file_format"]
  4. Route to correct enrichers in the correct order
  5. Save enriched plain dicts back to disk

Format routing:
  video (mp4/avi/mkv/mov/webm/m4v):
      DocMetaEnricher → VideoSegmentEnricher → VideoFrameEnricher

  spreadsheet (csv/xlsx/xls/ods/tsv):
      DocMetaEnricher → TableEnricher (schema-aware)

  document (pdf/docx/pptx):
      DocMetaEnricher → TextEnricher → ImageEnricher → TableEnricher

Order is enforced:
  - DocMetaEnricher always first (section summaries + semantic_anchor upgrades
    are needed by TextEnricher / VideoSegmentEnricher)
  - VideoSegmentEnricher before VideoFrameEnricher (frames use enriched segments)

Idempotency:
  All enrichers skip chunks that already have their target field set.
  Safe to re-run if process crashes mid-way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from enrichment_layer.enrichers.doc_meta_enricher    import DocMetaEnricher
from enrichment_layer.enrichers.text_enricher        import TextEnricher
from enrichment_layer.enrichers.image_enricher       import ImageEnricher
from enrichment_layer.enrichers.table_enricher       import TableEnricher
from enrichment_layer.enrichers.video_segment_enricher import VideoSegmentEnricher
from enrichment_layer.enrichers.video_frame_enricher   import VideoFrameEnricher
from enrichment_layer.utils.llm_client import LLMClient, build_llm_client

from shared.models.metadata_models import (
    validate_doc_metadata,
    validate_structure_units,
    validate_text_chunks,
    validate_image_chunks,
    validate_table_chunks,
    validate_video_segments,
    validate_video_frames,
)

VIDEO_FORMATS = {"mp4", "avi", "mkv", "mov", "webm", "m4v", "video"}
SHEET_FORMATS = {"csv", "xlsx", "xls", "ods", "tsv"}


def _load(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class EnrichmentPipeline:

    def __init__(
        self,
        llm:           LLMClient,
        skip_images:   bool = False,
        skip_tables:   bool = False,
        skip_video_frames: bool = False,
        dry_run:       bool = False,
    ):
        self.llm              = llm
        self.skip_images      = skip_images
        self.skip_tables      = skip_tables
        self.skip_video_frames = skip_video_frames
        self.dry_run          = dry_run

        # Instantiate enrichers once — they hold no state between documents
        self.doc_meta_enricher      = DocMetaEnricher(llm)
        self.text_enricher          = TextEnricher(llm)
        self.image_enricher         = ImageEnricher(llm)
        self.table_enricher         = TableEnricher(llm)
        self.video_segment_enricher = VideoSegmentEnricher(llm)
        self.video_frame_enricher   = VideoFrameEnricher(llm)

    def process_document(self, doc_output_dir: Path) -> Dict[str, Any]:
        doc_name     = doc_output_dir.name
        metadata_dir = doc_output_dir / "metadata"

        if not metadata_dir.exists():
            logger.warning(f"  No metadata/ in {doc_output_dir}")
            return {}

        logger.info(f"\n{'='*55}")
        logger.info(f"  Enriching: {doc_name}")
        logger.info(f"{'='*55}")

        # ── Load raw JSON ──────────────────────────────────────────────
        raw_doc_meta     = _load(metadata_dir / "doc_metadata.json")    or {}
        raw_struct_units = _load(metadata_dir / "structure_units.json") or []
        raw_text         = _load(metadata_dir / "text_chunks.json")     or []
        raw_images       = _load(metadata_dir / "image_chunks.json")    or []
        raw_tables       = _load(metadata_dir / "table_chunks.json")    or []
        raw_vid_segs     = _load(metadata_dir / "video_segments.json")  or []
        raw_vid_frames   = _load(metadata_dir / "video_frames.json")    or []

        # ── Pydantic boundary validation ───────────────────────────────
        try:
            doc_meta_model = validate_doc_metadata(raw_doc_meta)
        except Exception as e:
            logger.error(f"  doc_metadata invalid: {e} — aborting {doc_name}")
            return {}

        su_models   = validate_structure_units(raw_struct_units)
        text_models = validate_text_chunks(raw_text)
        img_models  = validate_image_chunks(raw_images)
        tbl_models  = validate_table_chunks(raw_tables)
        seg_models  = validate_video_segments(raw_vid_segs)
        frm_models  = validate_video_frames(raw_vid_frames)

        logger.info(
            f"  Validated: text={len(text_models)} | img={len(img_models)} | "
            f"tbl={len(tbl_models)} | seg={len(seg_models)} | frames={len(frm_models)}"
        )

        # Work on plain dicts (enrichers never use Pydantic internally)
        doc_meta      = doc_meta_model.to_dict()
        struct_units  = [m.to_dict() for m in su_models]
        text_chunks   = [m.to_dict() for m in text_models]
        image_chunks  = [m.to_dict() for m in img_models]
        table_chunks  = [m.to_dict() for m in tbl_models]
        vid_segments  = [m.to_dict() for m in seg_models]
        vid_frames    = [m.to_dict() for m in frm_models]

        if self.dry_run:
            logger.info(f"[DRY RUN] Would enrich {doc_name}")
            return {"doc_name": doc_name, "dry_run": True}

        # ── Detect format ──────────────────────────────────────────────
        fmt = (doc_meta.get("file_format") or "").lower().strip(".")

        # Build full document text for enrichers that need it
        doc_text = " ".join(
            (c.get("text_original_content") or "")
            for c in sorted(text_chunks + vid_segments,
                            key=lambda c: c.get("chunk_index", 0))
        )

        # ── Step 1: Doc metadata (always first) ────────────────────────
        doc_meta, struct_units = self.doc_meta_enricher.enrich(
            doc_metadata    = doc_meta,
            structure_units = struct_units,
            text_chunks     = text_chunks if fmt not in VIDEO_FORMATS else vid_segments,
            doc_output_dir  = doc_output_dir,
        )

        # ── Step 2–4: Format-specific enrichment ──────────────────────
        if fmt in VIDEO_FORMATS:
            vid_segments = self.video_segment_enricher.enrich(
                video_segments  = vid_segments,
                structure_units = struct_units,
                doc_metadata    = doc_meta,
                doc_output_dir  = doc_output_dir,
            )
            if not self.skip_video_frames:
                vid_frames = self.video_frame_enricher.enrich(
                    video_frames    = vid_frames,
                    video_segments  = vid_segments,
                    structure_units = struct_units,
                    doc_metadata    = doc_meta,
                    doc_output_dir  = doc_output_dir,
                )

        elif fmt in SHEET_FORMATS:
            # Only table_chunks exist for spreadsheets
            table_chunks = self.table_enricher.enrich(
                table_chunks  = table_chunks,
                document_text = doc_text,
            )

        else:
            # PDF / DOCX / PPTX
            text_chunks = self.text_enricher.enrich(
                text_chunks     = text_chunks,
                structure_units = struct_units,
                document_text   = doc_text,
            )
            if not self.skip_images:
                image_chunks = self.image_enricher.enrich(
                    image_chunks   = image_chunks,
                    text_chunks    = text_chunks,
                    doc_output_dir = doc_output_dir,
                    document_text  = doc_text,
                    doc_metadata   = doc_meta,
                )
            if not self.skip_tables and table_chunks:
                table_chunks = self.table_enricher.enrich(
                    table_chunks  = table_chunks,
                    document_text = doc_text,
                )

        # ── Save enriched plain dicts back to disk ─────────────────────
        _save(doc_meta,     metadata_dir / "doc_metadata.json")
        _save(struct_units, metadata_dir / "structure_units.json")
        _save(text_chunks,  metadata_dir / "text_chunks.json")

        # image_chunks: skip for CSV/XLSX — spreadsheets never have images
        if fmt not in SHEET_FORMATS:
            _save(image_chunks, metadata_dir / "image_chunks.json")

        # table_chunks: skip for video — videos never have tables
        if fmt not in VIDEO_FORMATS:
            _save(table_chunks, metadata_dir / "table_chunks.json")

        # video_segments and video_frames: video only
        if fmt in VIDEO_FORMATS:
            _save(vid_segments, metadata_dir / "video_segments.json")
            _save(vid_frames,   metadata_dir / "video_frames.json")

        # ── Build summary ──────────────────────────────────────────────
        result = {
            "doc_name":          doc_name,
            "format":            fmt,
            "doc_summary_set":   bool(doc_meta.get("doc_summary")),
            "text_enriched":     sum(1 for c in text_chunks    if c.get("contextual_summary")),
            "image_enriched":    sum(1 for c in image_chunks   if c.get("contextual_summary")),
            "table_enriched":    sum(1 for c in table_chunks   if c.get("table_summary")),
            "seg_enriched":      sum(1 for s in vid_segments   if s.get("contextual_summary")),
            "frame_enriched":    sum(1 for f in vid_frames     if f.get("contextual_summary")),
        }
        logger.success(
            f"  Enrichment saved: {doc_name} | "
            f"text={result['text_enriched']} img={result['image_enriched']} "
            f"tbl={result['table_enriched']} seg={result['seg_enriched']} "
            f"frames={result['frame_enriched']}"
        )
        return result