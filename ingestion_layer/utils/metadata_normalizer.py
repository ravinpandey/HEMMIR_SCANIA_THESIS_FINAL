"""
ingestion_layer/utils/metadata_normalizer.py

Metadata Normalizer — post-processing pass after extraction.

Purpose:
    After any extractor writes its JSON files, the normalizer runs a final
    pass to guarantee:
        1. Every chunk has correct promoted_fields for ChromaDB insertion.
        2. format_specific dicts are present and populated.
        3. doc_metadata has all mandatory fields with safe default values.
        4. section_map entries link chunk_ids correctly.
        5. related_figures_str / related_tables_str string fields are set.
        6. No None values leak into ChromaDB metadata fields.
        7. structure_units.json is always written with fully linked
           chunk_ids, figure_ids, table_ids — derived from section_map
           AFTER all linking is complete.
        8. video_segments.json is normalized with promoted_fields.
        9. video_frames.json is written as empty [] at ingestion time
           so the enrichment pipeline never crashes on a missing file.

Changes from previous version:
    - Added full video support:
        * normalize() now reads and saves video_segments.json
        * _normalize_video_segment() adds promoted_fields to each segment
        * video_frames.json written as [] if not present
        * _write_structure_units() handles scene units via merge path
    - Added _write_structure_units() — builds structure_units.json AFTER
      _link_section_map() fills chunk_ids, figure_ids, table_ids.
    - If structure_units.json already exists (DOCX/PPTX/Video), merges
      the linked IDs into existing units — preserves format-specific fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models.multiformat_models import (
    ExtendedPromotedFields,
    FileFormat,
    SourceType,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _safe_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    if isinstance(v, list):
        return json.dumps(v)
    return str(v)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return default


VIDEO_FORMATS = {"mp4", "avi", "mkv", "mov", "webm", "m4v", "video"}


# ─────────────────────────────────────────────────────────────────────────────
# MetadataNormalizer
# ─────────────────────────────────────────────────────────────────────────────

class MetadataNormalizer:
    """
    Runs a normalization pass on all JSON files for one document.
    Handles all formats: PDF, DOCX, PPTX, CSV, XLSX, Image, Video.
    Idempotent — safe to run multiple times.
    """

    def normalize(self, doc_output_dir: Path) -> Dict[str, Any]:
        """
        Normalize all output JSON files in doc_output_dir/metadata/.
        Returns a summary dict with counts and any issues found.
        """
        metadata_dir = doc_output_dir / "metadata"
        if not metadata_dir.exists():
            logger.warning(f"Normalizer: no metadata/ in {doc_output_dir}")
            return {}

        doc_meta     = _load(metadata_dir / "doc_metadata.json") or {}
        text_chunks  = _load(metadata_dir / "text_chunks.json")  or []
        img_chunks   = _load(metadata_dir / "image_chunks.json") or []
        tbl_chunks   = _load(metadata_dir / "table_chunks.json") or []
        vid_segments = _load(metadata_dir / "video_segments.json") or []

        doc_id      = doc_meta.get("doc_id", "")
        file_format = doc_meta.get("file_format", FileFormat.PDF.value)

        issues = []

        # ── Normalize doc_metadata ────────────────────────────────────
        doc_meta = self._normalize_doc_meta(
            doc_meta, text_chunks, img_chunks, tbl_chunks, vid_segments
        )

        # ── Normalize chunk files ─────────────────────────────────────
        text_chunks = [
            self._normalize_chunk(c, doc_meta, "text", i)
            for i, c in enumerate(text_chunks, start=1)
        ]
        img_chunks = [
            self._normalize_chunk(c, doc_meta, "image", i)
            for i, c in enumerate(img_chunks, start=1)
        ]
        tbl_chunks = [
            self._normalize_chunk(c, doc_meta, "table", i)
            for i, c in enumerate(tbl_chunks, start=1)
        ]

        # ── Normalize video segments ──────────────────────────────────
        if vid_segments:
            vid_segments = [
                self._normalize_video_segment(seg, doc_meta, i)
                for i, seg in enumerate(vid_segments, start=1)
            ]

        # ── Link chunks into section_map ──────────────────────────────
        doc_meta = self._link_section_map(doc_meta, text_chunks, img_chunks, tbl_chunks)

        # ── Re-link figure_index_map and table_index_map ──────────────
        doc_meta = self._rebuild_index_maps(doc_meta, img_chunks, tbl_chunks)

        # ── Save standard JSON files — only write files meaningful for format ──
        _save(doc_meta,    metadata_dir / "doc_metadata.json")
        _save(text_chunks, metadata_dir / "text_chunks.json")

        # image_chunks: skip for CSV/XLSX (they never have images)
        fmt_lower = file_format.lower().strip(".")
        sheet_formats = {"csv", "xlsx", "xls", "ods", "tsv"}
        if fmt_lower not in sheet_formats:
            _save(img_chunks, metadata_dir / "image_chunks.json")

        # table_chunks: skip for video (videos never have tables)
        if fmt_lower not in VIDEO_FORMATS:
            _save(tbl_chunks, metadata_dir / "table_chunks.json")

        if vid_segments:
            _save(vid_segments, metadata_dir / "video_segments.json")

        # ── video_frames.json — do NOT write for non-video formats ──────
        # VideoExtractor now writes video_frames.json directly at ingestion
        # with proper VideoFrameChunkMetadata format including parent_id,
        # timestamp_s, frame_position etc. The normalizer must NOT overwrite
        # it with [] — that would erase the real keyframe data.
        # For non-video formats this file simply does not exist (correct).

        # ── Write structure_units.json — ALWAYS as final step ─────────
        # Must run AFTER _link_section_map so chunk_ids, figure_ids,
        # table_ids are fully populated before structure_units are built.
        n_units = self._write_structure_units(metadata_dir, doc_meta)

        logger.info(
            f"Normalized: {doc_output_dir.name} | "
            f"format={file_format} | "
            f"text={len(text_chunks)} img={len(img_chunks)} "
            f"tbl={len(tbl_chunks)} seg={len(vid_segments)} units={n_units}"
        )

        return {
            "doc_id":          doc_id,
            "file_format":     file_format,
            "text_count":      len(text_chunks),
            "image_count":     len(img_chunks),
            "table_count":     len(tbl_chunks),
            "segment_count":   len(vid_segments),
            "structure_units": n_units,
            "issues":          issues,
        }

    # ── structure_units.json writer ────────────────────────────────────────

    def _write_structure_units(
        self,
        metadata_dir: Path,
        doc_meta:     Dict,
    ) -> int:
        """
        Build and write structure_units.json from doc_metadata section_map.

        Case 1 — exists (DOCX/PPTX/Video wrote it): merge chunk_ids etc.
        Case 2 — does not exist (PDF): build fresh from section_map.
        Returns number of units written.
        """
        doc_id      = doc_meta.get("doc_id", "")
        section_map = doc_meta.get("section_map", {})

        if not section_map:
            _save([], metadata_dir / "structure_units.json")
            logger.debug(f"  structure_units: empty (no section_map) for {doc_id}")
            return 0

        su_path = metadata_dir / "structure_units.json"

        # ── Case 1: already exists — merge ────────────────────────────
        if su_path.exists():
            existing = _load(su_path) or []
            if existing:
                title_to_unit = {
                    u.get("title", ""): u
                    for u in existing
                    if isinstance(u, dict) and u.get("title")
                }
                for heading, entry in section_map.items():
                    if not isinstance(entry, dict):
                        continue
                    unit = title_to_unit.get(heading)
                    if unit is None:
                        continue
                    unit["chunk_ids"]  = entry.get("chunk_ids",  unit.get("chunk_ids",  []))
                    unit["figure_ids"] = entry.get("figure_ids", unit.get("figure_ids", []))
                    unit["table_ids"]  = entry.get("table_ids",  unit.get("table_ids",  []))
                    unit.setdefault("section_summary", None)
                    unit.setdefault("keywords",        [])
                    unit.setdefault("entities",        [])
                    unit.setdefault("subsections",     [])
                    unit.setdefault("semantic_anchor", heading)
                    unit.setdefault("source_id",       "")
                    unit.setdefault("synthetic",       entry.get("synthetic", False))

                _save(existing, su_path)
                logger.info(
                    f"  structure_units: merged {len(existing)} units for {doc_id}"
                )
                return len(existing)

        # ── Case 2: build fresh from section_map ──────────────────────
        units = []
        for idx, (heading, entry) in enumerate(section_map.items(), start=1):
            if not isinstance(entry, dict):
                continue
            units.append({
                "structure_unit_id":   f"{doc_id}_sec_{idx:03d}",
                "doc_id":              doc_id,
                "source_id":           "",
                "structure_unit_type": "section",
                "unit_index":          idx,
                "title":               heading,
                "semantic_anchor":     heading,
                "start_page":          entry.get("start_page"),
                "end_page":            entry.get("end_page"),
                "start_element_idx":   entry.get("start_element_idx"),
                "end_element_idx":     entry.get("end_element_idx"),
                "start_time_s":        None,
                "end_time_s":          None,
                "sheet_name":          None,
                "chunk_ids":           entry.get("chunk_ids",  []),
                "figure_ids":          entry.get("figure_ids", []),
                "table_ids":           entry.get("table_ids",  []),
                "frame_ids":           [],
                "synthetic":           entry.get("synthetic", False),
                "section_summary":     None,
                "keywords":            [],
                "entities":            [],
                "subsections":         [],
            })

        _save(units, su_path)
        logger.info(f"  structure_units: wrote {len(units)} units for {doc_id}")
        return len(units)

    # ── Doc metadata normalization ─────────────────────────────────────────

    def _normalize_doc_meta(
        self,
        meta:         Dict,
        text_chunks:  List[Dict],
        img_chunks:   List[Dict],
        tbl_chunks:   List[Dict],
        vid_segments: List[Dict] = [],
    ) -> Dict:
        total_chunks = len(text_chunks) + len(img_chunks) + len(tbl_chunks)

        meta.setdefault("doc_id",          "")
        meta.setdefault("doc_title",       meta.get("doc_id", ""))
        meta.setdefault("project_id",      meta.get("doc_title", ""))
        meta.setdefault("author",          None)
        meta.setdefault("language",        "English")
        meta.setdefault("document_type",   None)
        meta.setdefault("chunk_strategy",  "paragraph")
        meta.setdefault("total_pages",     0)
        meta.setdefault("chunk_size",      None)
        meta.setdefault("chunk_overlap",   None)
        meta.setdefault("tier1",           None)
        meta.setdefault("tier2",           None)
        meta.setdefault("last_modified",   None)
        meta.setdefault("file_format",     FileFormat.PDF.value)
        meta.setdefault("source_id",       "")
        meta.setdefault("source_type",     SourceType.LOCAL.value)
        meta.setdefault("folder_path",     "")

        meta["chunk_count"]     = total_chunks
        meta["related_figures"] = [c["chunk_id"] for c in img_chunks if "chunk_id" in c]
        meta["related_tables"]  = [c["chunk_id"] for c in tbl_chunks if "chunk_id" in c]

        meta["figure_count"] = len(img_chunks)
        meta["table_count"]  = len(tbl_chunks)
        meta["has_figures"]  = len(img_chunks) > 0
        meta["has_tables"]   = len(tbl_chunks) > 0

        # Video-specific
        if vid_segments:
            meta["total_segments"] = len(vid_segments)
            meta["has_video"]      = True
        else:
            meta.setdefault("has_video", False)

        if not meta.get("section_titles"):
            sm = meta.get("section_map", {})
            meta["section_titles"] = [
                k for k, v in sm.items()
                if isinstance(v, dict) and not v.get("synthetic", False)
            ]

        meta.setdefault("figure_index_map", {})
        meta.setdefault("table_index_map",  {})
        meta.setdefault("section_map",      {})

        for heading, entry in meta["section_map"].items():
            if not isinstance(entry, dict):
                continue
            entry.setdefault("keywords",    [])
            entry.setdefault("entities",    [])
            entry.setdefault("summary",     None)
            entry.setdefault("subsections", [])
            entry.setdefault("chunk_ids",   [])
            entry.setdefault("figure_ids",  [])
            entry.setdefault("table_ids",   [])

        for field in ("doc_summary", "doc_summary_confidence", "tier_confidence",
                      "outline_summary", "doc_embedding", "outline_summary_confidence"):
            meta.setdefault(field, None)

        return meta

    # ── Video segment normalization ────────────────────────────────────────

    def _normalize_video_segment(
        self,
        seg:      Dict,
        doc_meta: Dict,
        idx:      int,
    ) -> Dict:
        """
        Normalize one video segment dict and add promoted_fields.
        source_modality = video_segment (not text).
        """
        seg.setdefault("chunk_id",          f"{doc_meta.get('doc_id', '')}_seg_{idx:04d}")
        seg.setdefault("doc_id",            doc_meta.get("doc_id", ""))
        seg.setdefault("chunk_index",       idx)
        seg.setdefault("source_modality",   "video_segment")
        seg.setdefault("structure_unit_id", "")
        seg.setdefault("segment_index",     idx)
        seg.setdefault("start_time_s",      0.0)
        seg.setdefault("end_time_s",        0.0)
        seg.setdefault("transcript_text",   "")
        seg.setdefault("text_original_content", seg.get("transcript_text", ""))
        seg.setdefault("keyframe_ids",      [])
        seg.setdefault("contextual_summary", None)
        seg.setdefault("contextual_summary_confidence", None)
        seg.setdefault("detected_codes",    [])
        seg.setdefault("salience_score",    None)
        seg.setdefault("evidence_role",     None)

        raw = {
            "doc_id":          doc_meta.get("doc_id", ""),
            "doc_title":       doc_meta.get("doc_title", ""),
            "tier1":           doc_meta.get("tier1") or "",
            "tier2":           doc_meta.get("tier2") or "",
            "project_id":      doc_meta.get("project_id", ""),
            "document_type":   doc_meta.get("document_type", ""),
            "language":        doc_meta.get("language", "English"),
            "source_modality": "video_segment",
            "page_number":     0,
            "chunk_index":     idx,
            "chunk_strategy":  "transcript",
            "contextual_summary_confidence": _safe_float(
                seg.get("contextual_summary_confidence"), 0.0
            ),
            "salience_score":      _safe_float(seg.get("salience_score"), 0.0),
            "evidence_role":       seg.get("evidence_role") or "unknown",
            "source_id":           doc_meta.get("source_id", ""),
            "source_type":         doc_meta.get("source_type", ""),
            "file_format":         doc_meta.get("file_format", "mp4"),
            "folder_path":         doc_meta.get("folder_path", ""),
            "structure_unit_id":   seg.get("structure_unit_id", ""),
            "structure_unit_type": "scene",
            "sheet_name":          "",
            "slide_index":         0,
            "start_time_s":        _safe_float(seg.get("start_time_s"), 0.0),
        }
        try:
            seg["promoted_fields"] = ExtendedPromotedFields.model_validate(raw).to_dict()
        except Exception as e:
            logger.warning(f"  video_segment promoted_fields failed idx={idx}: {e}")
            seg["promoted_fields"] = {}

        return seg

    # ── Chunk normalization ────────────────────────────────────────────────

    def _normalize_chunk(
        self,
        chunk:      Dict,
        doc_meta:   Dict,
        modality:   str,
        idx:        int,
    ) -> Dict:
        chunk.setdefault("chunk_id",        f"{doc_meta.get('doc_id', '')}_{modality}_{idx:03d}")
        chunk.setdefault("doc_id",          doc_meta.get("doc_id", ""))
        chunk.setdefault("chunk_index",     idx)
        chunk.setdefault("source_modality", modality)
        chunk.setdefault("page_number",     0)

        if modality == "text":
            chunk.setdefault("text_original_content", "")
            chunk.setdefault("local_context",         None)
            chunk.setdefault("section_title",         None)
            chunk.setdefault("chunk_strategy",        doc_meta.get("chunk_strategy", ""))
            chunk.setdefault("token_count",           0)
            chunk.setdefault("element_index",         idx)
            chunk.setdefault("end_element_index",     idx)
            chunk.setdefault("related_figures",       [])
            chunk.setdefault("related_tables",        [])
            chunk.setdefault("section_id",            "")
            chunk.setdefault("structure_unit_id",     "")
            chunk.setdefault("contextual_summary",    None)
            chunk.setdefault("contextual_summary_confidence", None)
            chunk.setdefault("detected_codes",        [])
            chunk.setdefault("salience_score",        0.5)    # neutral default — enrichment writes real value
            chunk.setdefault("evidence_role",         "unknown")
            chunk["related_figures_str"] = json.dumps(chunk.get("related_figures") or [])
            chunk["related_tables_str"]  = json.dumps(chunk.get("related_tables") or [])

        elif modality == "image":
            chunk.setdefault("figure_id",    chunk["chunk_id"])
            chunk.setdefault("image_path",   "")
            chunk.setdefault("image_type",   "unknown")
            chunk.setdefault("related_sections", [])
            chunk.setdefault("section_id",   "")
            chunk.setdefault("structure_unit_id", "")
            chunk.setdefault("image_caption",  None)
            chunk.setdefault("image_caption_confidence", None)
            chunk.setdefault("depicted_component", None)
            chunk.setdefault("depicted_component_confidence", None)
            chunk.setdefault("visible_annotations", None)
            chunk.setdefault("visible_annotations_confidence", None)
            chunk.setdefault("contextual_summary",    None)
            chunk.setdefault("contextual_summary_confidence", None)
            chunk["related_sections_str"] = json.dumps(chunk.get("related_sections") or [])

        elif modality == "table":
            chunk.setdefault("table_html",     None)
            chunk.setdefault("table_csv_path", "")
            chunk.setdefault("html_file_path", "")
            chunk.setdefault("row_count",      0)
            chunk.setdefault("col_count",      0)
            chunk.setdefault("section_id",     "")
            chunk.setdefault("structure_unit_id", "")
            chunk.setdefault("table_caption",  None)
            chunk.setdefault("table_caption_confidence", None)
            chunk.setdefault("table_summary",  None)
            chunk.setdefault("table_summary_confidence", None)
            chunk.setdefault("table_purpose",  None)
            chunk.setdefault("table_purpose_confidence", None)
            chunk.setdefault("markdown",       "")
            chunk.setdefault("html_representation", chunk.get("table_html", ""))

        chunk["promoted_fields"] = self._build_promoted_fields(chunk, doc_meta)
        return chunk

    def _build_promoted_fields(self, chunk: Dict, doc_meta: Dict) -> Dict:
        fmt_spec = chunk.get("format_specific") or {}
        su_id = (
            chunk.get("structure_unit_id")
            or chunk.get("section_id")
            or ""
        )
        raw = {
            "doc_id":          doc_meta.get("doc_id", ""),
            "doc_title":       doc_meta.get("doc_title", ""),
            "tier1":           doc_meta.get("tier1") or "",
            "tier2":           doc_meta.get("tier2") or "",
            "project_id":      doc_meta.get("project_id", ""),
            "document_type":   doc_meta.get("document_type", ""),
            "language":        doc_meta.get("language", "English"),
            "source_modality": chunk.get("source_modality", ""),
            "page_number":     chunk.get("page_number", 0),
            "chunk_index":     chunk.get("chunk_index", 0),
            "chunk_strategy":  chunk.get("chunk_strategy", ""),
            "contextual_summary_confidence": _safe_float(
                chunk.get("contextual_summary_confidence"), 0.0
            ),
            "salience_score":      _safe_float(chunk.get("salience_score"), 0.0),
            "evidence_role":       chunk.get("evidence_role") or "unknown",
            "source_id":           doc_meta.get("source_id", ""),
            "source_type":         doc_meta.get("source_type", ""),
            "file_format":         doc_meta.get("file_format", "pdf"),
            "folder_path":         doc_meta.get("folder_path", ""),
            "structure_unit_id":   su_id,
            "structure_unit_type": doc_meta.get("structure_type", "page"),
            "sheet_name":      fmt_spec.get("sheet_name", ""),
            "slide_index":     chunk.get("slide_index", 0) or fmt_spec.get("slide_index", 0),
            "start_time_s":    chunk.get("start_time_s", 0.0) or fmt_spec.get("start_time_s", 0.0),
        }
        try:
            return ExtendedPromotedFields.model_validate(raw).to_dict()
        except Exception as e:
            logger.warning(f"  promoted_fields build failed: {e}")
            return {}

    # ── Section map linking ────────────────────────────────────────────────

    def _link_section_map(
        self,
        doc_meta:    Dict,
        text_chunks: List[Dict],
        img_chunks:  List[Dict],
        tbl_chunks:  List[Dict],
    ) -> Dict:
        section_map = doc_meta.get("section_map", {})
        if not section_map:
            return doc_meta

        page_index = []
        for section_name, entry in section_map.items():
            if not isinstance(entry, dict):
                continue
            start = _safe_int(entry.get("start_page"), 0)
            end   = _safe_int(entry.get("end_page"),   99999)
            page_index.append((start, end, section_name))
        page_index.sort(key=lambda x: x[0])

        def _find_section(page: int) -> Optional[str]:
            best, best_start = None, -1
            for s, e, name in page_index:
                if s <= page <= e and s > best_start:
                    best, best_start = name, s
            return best

        for chunk in text_chunks:
            section = _find_section(_safe_int(chunk.get("page_number")))
            if section and section in section_map:
                if isinstance(section_map[section], dict):
                    cids = section_map[section].setdefault("chunk_ids", [])
                    if chunk["chunk_id"] not in cids:
                        cids.append(chunk["chunk_id"])

        for chunk in img_chunks:
            section = _find_section(_safe_int(chunk.get("page_number")))
            if section and section in section_map:
                if isinstance(section_map[section], dict):
                    fids = section_map[section].setdefault("figure_ids", [])
                    if chunk["chunk_id"] not in fids:
                        fids.append(chunk["chunk_id"])

        for chunk in tbl_chunks:
            section = _find_section(_safe_int(chunk.get("page_number")))
            if section and section in section_map:
                if isinstance(section_map[section], dict):
                    tids = section_map[section].setdefault("table_ids", [])
                    if chunk["chunk_id"] not in tids:
                        tids.append(chunk["chunk_id"])

        doc_meta["section_map"] = section_map
        return doc_meta

    def _rebuild_index_maps(
        self,
        doc_meta:   Dict,
        img_chunks: List[Dict],
        tbl_chunks: List[Dict],
    ) -> Dict:
        fig_map = {}
        for i, c in enumerate(img_chunks, start=1):
            cid = c.get("chunk_id", "")
            if cid:
                fig_map[str(i)] = cid

        tbl_map = {}
        for i, c in enumerate(tbl_chunks, start=1):
            cid = c.get("chunk_id", "")
            if cid:
                tbl_map[str(i)] = cid

        if fig_map:
            doc_meta["figure_index_map"] = fig_map
        if tbl_map:
            doc_meta["table_index_map"]  = tbl_map

        return doc_meta