from __future__ import annotations

from typing import Dict, Any


def _s(v): return "" if v is None else str(v)
def _i(v): return 0 if v in (None, "") else int(v)
def _f(v): return 0.0 if v in (None, "") else float(v)

def infer_structure_unit_type(doc_meta: Dict[str, Any], chunk: Dict[str, Any]) -> str:
    if chunk.get("structure_unit_type"):
        return _s(chunk.get("structure_unit_type"))
    fmt = _s(doc_meta.get("file_format")).lower()
    mapping = {
        "pdf": "heading",
        "docx": "heading",
        "pptx": "slide",
        "csv": "sheet",
        "xlsx": "sheet",
        "xls": "sheet",
        "xlsm": "sheet",
        "ods": "sheet",
        "mp4": "scene",
        "avi": "scene",
        "mkv": "scene",
        "mov": "scene",
        "webm": "scene",
        "m4v": "scene",
    }
    return mapping.get(fmt, "heading")


def build_promoted_fields(chunk: Dict[str, Any], doc_meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        # ── Document identity ──────────────────────────────────────────
        "doc_id":           _s(doc_meta.get("doc_id")),
        "doc_title":        _s(doc_meta.get("doc_title")),
        "tier1":            _s(doc_meta.get("tier1")),
        "tier2":            _s(doc_meta.get("tier2")),
        "tier_confidence":  _f(doc_meta.get("tier_confidence")),
        "project_id":       _s(doc_meta.get("project_id")),
        "document_type":    _s(doc_meta.get("document_type")),
        "language":         _s(doc_meta.get("language") or "English"),
        "source_id":        _s(chunk.get("source_id") or doc_meta.get("source_id")),
        "file_format":      _s(chunk.get("file_format") or doc_meta.get("file_format")),
        "folder_path":      _s(doc_meta.get("folder_path")),

        # ── Chunk identity ─────────────────────────────────────────────
        "source_modality":  _s(chunk.get("source_modality")),
        "chunk_index":      _i(chunk.get("chunk_index")),
        "chunk_strategy":   _s(chunk.get("chunk_strategy")),
        "page_number":      _i(chunk.get("page_number")),
        "slide_index":      _i(chunk.get("slide_index")),
        "sheet_name":       _s(chunk.get("sheet_name")),
        "start_time_s":     _f(chunk.get("start_time_s")),
        "end_time_s":       _f(chunk.get("end_time_s")),

        # ── Structure navigation ───────────────────────────────────────
        "structure_unit_id":   _s(chunk.get("structure_unit_id")),
        "structure_unit_type": infer_structure_unit_type(doc_meta, chunk),

        # ── Retrieval quality signals ──────────────────────────────────
        # These enable confidence-weighted reranking and filtering.
        # Low confidence scores allow the retrieval layer to deprioritise
        # potentially hallucinated or uncertain LLM-generated fields.
        "evidence_role":                    _s(chunk.get("evidence_role")),
        "evidence_role_confidence":         _f(chunk.get("evidence_role_confidence")),
        "salience_score":                   _f(chunk.get("salience_score")),
        "contextual_summary_confidence":    _f(chunk.get("contextual_summary_confidence")),

        # ── Image-specific quality signals ────────────────────────────
        "image_caption_confidence":         _f(chunk.get("image_caption_confidence")),
        "depicted_component_confidence":    _f(chunk.get("depicted_component_confidence")),
        "visible_annotations_confidence":   _f(chunk.get("visible_annotations_confidence")),
        "frame_role_confidence":            _f(chunk.get("frame_role_confidence")),

        # ── Text-specific quality signals ─────────────────────────────
        "detected_codes_confidence":        _f(chunk.get("detected_codes_confidence")),

        # ── Table-specific quality signals ────────────────────────────
        "table_summary_confidence":         _f(chunk.get("table_summary_confidence")),
        "table_purpose_confidence":         _f(chunk.get("table_purpose_confidence")),
    }