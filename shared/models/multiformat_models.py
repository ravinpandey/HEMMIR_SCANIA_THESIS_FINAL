"""
shared/models/multiformat_models.py

Multi-format data models for HEMMIR ingestion extension.

Design philosophy:
    - All models are pure dataclasses/Pydantic — no I/O, no business logic.
    - Every non-PDF format ADDS new fields; never removes or renames existing ones.
    - SourceRecord and FolderRecord are new Level-1/Level-2 metadata layers.
    - StructuralUnit generalizes PDF's "page" to slide / heading / sheet / scene.
    - All chunk models extend the existing PDF chunk schema exactly.
    - promoted_fields is extended with format/source fields for ChromaDB filters.

Backward compatibility:
    - DocumentMetadata (existing) gains 8 optional fields — all default to None.
    - TextChunkMetadata gains format_specific: Optional[dict] — ignored by PDF path.
    - ImageChunkMetadata gains ocr_text, region_boxes for standalone image files.
    - TableChunkMetadata gains column_schema, sheet_name for XLSX/CSV path.
    - New chunk types: SlideChunkMetadata, VideoSegmentChunkMetadata.
    - New promoted_fields keys: source_type, file_format, structure_unit_type,
      tier1/tier2 are now populated (previously always empty strings).
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union



from pydantic import BaseModel, Field, field_validator, model_validator

# ─────────────────────────────────────────────────────────────────────────────
# SourceType Enum (restored from extension)
# ─────────────────────────────────────────────────────────────────────────────
class SourceType(str, Enum):
    LOCAL = "local"
    S3    = "s3"
    GCS   = "gcs"
    AZURE = "azure"

# ─────────────────────────────────────────────────────────────────────────────
# FileFormat Enum (restored from extension)
# ─────────────────────────────────────────────────────────────────────────────
class FileFormat(str, Enum):
    PDF     = "pdf"
    DOCX    = "docx"
    PPTX    = "pptx"
    CSV     = "csv"
    XLSX    = "xlsx"
    IMAGE   = "image"
    VIDEO   = "video"
    UNKNOWN = "unknown"



# ─────────────────────────────────────────────────────────────────────────────
# StructureUnitType Enum
# ─────────────────────────────────────────────────────────────────────────────
class StructureUnitType(str, Enum):
    PAGE = "page"
    HEADING = "heading"
    SLIDE = "slide"
    SHEET = "sheet"
    ROW_GROUP = "row_group"
    SCENE = "scene"
    SEGMENT = "segment"
    IMAGE = "image"


# ─────────────────────────────────────────────────────────────────────────────
# ChunkStrategy Enum
# ─────────────────────────────────────────────────────────────────────────────
class ChunkStrategy(str, Enum):
    PARAGRAPH = "paragraph"
    TITLE = "title"
    SIZE = "size"
    SLIDE = "slide"
    SCHEMA = "schema"
    ROW_GROUP = "row_group"
    TRANSCRIPT = "transcript"
    WHOLE_FILE = "whole_file"

# ─────────────────────────────────────────────────────────────────────────────
# Level 1 — Source / Repository Record  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

class SourceRecord(BaseModel):
    """
    Registry entry for one ingestion root (local dir or S3 prefix).

    Stored in: source_registry.json (one per ingestion session).
    Read by:   ingestion_layer to stamp every doc_metadata with source_id.
    Read by:   agentic planner to choose which source to search.

    Mandatory fields: source_id, source_type, source_name.
    S3-specific:      bucket_name, root_prefix.
    Local-specific:   local_root.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    source_id:     str  = Field(..., description="Stable slug or UUID — e.g. 'src_alpha_v2'")
    source_type:   SourceType
    source_name:   str  = Field(..., description="Human label — used by agentic planner")

    # ── S3-specific (null for local) ──────────────────────────────────────
    bucket_name:   Optional[str] = None
    root_prefix:   Optional[str] = None   # e.g. "projects/alpha/"
    aws_region:    Optional[str] = None

    # ── Local-specific (null for S3) ──────────────────────────────────────
    local_root:    Optional[str] = None   # absolute path

    # ── Temporal ──────────────────────────────────────────────────────────
    ingestion_date: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    last_sync:      Optional[str] = None

    # ── Optional agent-planning hints ─────────────────────────────────────
    version_tag:   Optional[str] = None   # "v2.1", "2026-Q1"
    access_policy: Optional[str] = None   # "internal", "public", "restricted"
    description:   Optional[str] = None
    tags:          List[str]     = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_source_fields(self) -> "SourceRecord":
        if self.source_type == SourceType.S3:
            if not self.bucket_name:
                raise ValueError("bucket_name is required for S3 sources")
        elif self.source_type == SourceType.LOCAL:
            if not self.local_root:
                raise ValueError("local_root is required for local sources")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_local(cls, root: Union[str, Path], source_name: str,
                   source_id: Optional[str] = None, **kwargs) -> "SourceRecord":
        """Convenience constructor for local directory sources."""
        root = Path(root).resolve()
        sid  = source_id or f"src_{hashlib.md5(str(root).encode()).hexdigest()[:8]}"
        return cls(
            source_id=sid, source_type=SourceType.LOCAL,
            source_name=source_name, local_root=str(root), **kwargs
        )

    @classmethod
    def from_s3(cls, bucket: str, prefix: str, source_name: str,
                source_id: Optional[str] = None, region: str = "eu-west-1",
                **kwargs) -> "SourceRecord":
        """Convenience constructor for S3 bucket sources."""
        sid = source_id or f"src_{hashlib.md5(f'{bucket}/{prefix}'.encode()).hexdigest()[:8]}"
        return cls(
            source_id=sid, source_type=SourceType.S3,
            source_name=source_name, bucket_name=bucket,
            root_prefix=prefix, aws_region=region, **kwargs
        )


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — Folder / Project Record  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

class FolderRecord(BaseModel):
    """
    Represents a folder/prefix within a source.

    Stored:  embedded inside every doc_metadata (not as a separate file).
    Purpose: populates tier1/tier2 (currently always empty in HEMMIR V4),
             gives the agent folder-level filtering capability.
    """

    source_id:        str
    folder_path:      str            # Full relative path from source root
    folder_depth:     int = 0        # 0 = root, 1 = one level deep, etc.
    folder_segments:  List[str] = Field(default_factory=list)  # path.split("/")
    tier1:            Optional[str] = None   # segment[0] — top-level project
    tier2:            Optional[str] = None   # segment[1] — sub-project/topic
    project_label:    Optional[str] = None
    folder_tags:      List[str]     = Field(default_factory=list)

    @classmethod
    def from_path(cls, source_id: str, relative_path: str) -> "FolderRecord":
        """
        Build a FolderRecord from a relative path string.

        Example:
            from_path("src_alpha", "projects/alpha/reports/2026")
            → tier1="projects", tier2="alpha"
        """
        segs  = [s for s in relative_path.replace("\\", "/").split("/") if s]
        depth = len(segs) - 1  # file is last segment; folder depth = segments - 1
        # tier1/tier2 map to the first two folder segments (not file)
        folder_segs = segs[:-1] if segs else []
        return cls(
            source_id       = source_id,
            folder_path     = "/".join(folder_segs),
            folder_depth    = len(folder_segs),
            folder_segments = folder_segs,
            tier1           = folder_segs[0] if len(folder_segs) > 0 else None,
            tier2           = folder_segs[1] if len(folder_segs) > 1 else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Level 4 — Structural Unit  (NEW — generalizes section_map entry)
# ─────────────────────────────────────────────────────────────────────────────

class StructuralUnit(BaseModel):
    """
    A named unit of document structure — slide, heading, sheet, scene, page.

    For PDFs: this mirrors existing section_map entries (backward-compatible).
    For PPTX: one entry per slide.
    For DOCX: one entry per heading.
    For XLSX/CSV: one entry per sheet + optionally per row_group.
    For Video: one entry per scene or transcript segment.

    The doc's structure_map field contains these as values,
    keyed by structure_unit_id.
    """

    structure_unit_id:   str              # e.g. "doc123_slide_007"
    doc_id:              str
    structure_unit_type: StructureUnitType
    unit_index:          int              # position within document (1-based)
    title:               Optional[str]   = None   # slide title, heading text, sheet name
    summary:             Optional[str]   = None   # LLM-generated (enrichment layer)

    # Position anchors — format-specific; only relevant fields are set
    start_page:          Optional[int]   = None   # PDF / DOCX
    end_page:            Optional[int]   = None
    start_element_idx:   Optional[int]   = None   # PDF element counter
    end_element_idx:     Optional[int]   = None
    slide_index:         Optional[int]   = None   # PPTX
    sheet_name:          Optional[str]   = None   # XLSX/CSV
    start_row:           Optional[int]   = None   # XLSX/CSV row groups
    end_row:             Optional[int]   = None
    start_time_s:        Optional[float] = None   # Video scenes (seconds)
    end_time_s:          Optional[float] = None

    # Cross-modal linkage (mirrors PDF section_map structure exactly)
    chunk_ids:           List[str] = Field(default_factory=list)
    figure_ids:          List[str] = Field(default_factory=list)
    table_ids:           List[str] = Field(default_factory=list)
    subsections:         List[str] = Field(default_factory=list)

    # Flags
    synthetic:           bool = False    # True = fallback, not a real heading

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Extended PromotedFields  (backward-compatible addition)
# ─────────────────────────────────────────────────────────────────────────────

class ExtendedPromotedFields(BaseModel):
    """
    ChromaDB-safe flat metadata for every chunk type, all formats.

    Extends the existing PromotedFields model.
    The existing PDF fields (doc_id, doc_title, tier1, tier2, project_id,
    document_type, language, source_modality, page_number, chunk_index,
    chunk_strategy, contextual_summary_confidence) are preserved exactly.

    New fields added (with safe defaults so PDF path is unaffected):
        source_id          — links chunk to SourceRecord
        source_type        — "local" | "s3" | ...
        file_format        — "pdf" | "docx" | "pptx" | "csv" | "xlsx" | "image" | "video"
        structure_unit_type — "page" | "heading" | "slide" | "sheet" | "scene"
        structure_unit_id  — ID of the structural unit this chunk belongs to
        sheet_name         — XLSX/CSV sheet name (empty for non-tabular)
        slide_index        — PPTX slide number (0 for non-PPT)
        start_time_s       — video timestamp start in seconds (0.0 for non-video)
        folder_path        — full folder path relative to source root
    """

    # ── Existing fields (preserved exactly) ──────────────────────────────
    doc_id:                         str   = ""
    doc_title:                      str   = ""
    tier1:                          str   = ""
    tier2:                          str   = ""
    project_id:                     str   = ""
    document_type:                  str   = ""
    language:                       str   = "English"
    source_modality:                str   = ""
    page_number:                    int   = 0
    chunk_index:                    int   = 0
    chunk_strategy:                 str   = ""
    contextual_summary_confidence:  float = 0.0

    # ── New fields (default to safe ChromaDB types) ───────────────────────
    source_id:           str   = ""
    source_type:         str   = ""     # SourceType value
    file_format:         str   = "pdf"  # FileFormat value
    structure_unit_type: str   = "page" # StructureUnitType value
    structure_unit_id:   str   = ""
    sheet_name:          str   = ""
    slide_index:         int   = 0
    start_time_s:        float = 0.0
    folder_path:         str   = ""

    @field_validator("page_number", "chunk_index", "slide_index", mode="before")
    @classmethod
    def _coerce_int(cls, v) -> int:
        try:
            return max(0, int(v or 0))
        except (TypeError, ValueError):
            return 0

    @field_validator("contextual_summary_confidence", "start_time_s", mode="before")
    @classmethod
    def _coerce_float(cls, v) -> float:
        try:
            return float(v or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @field_validator("doc_id", "doc_title", "tier1", "tier2", "project_id",
                     "document_type", "language", "source_modality",
                     "chunk_strategy", "source_id", "source_type",
                     "file_format", "structure_unit_type", "structure_unit_id",
                     "sheet_name", "folder_path", mode="before")
    @classmethod
    def _coerce_str(cls, v) -> str:
        if v is None:
            return ""
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return str(v)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Level 3 extension — DocumentMetadata additions  (backward-compatible)
# ─────────────────────────────────────────────────────────────────────────────

class DocumentMetadataExtension(BaseModel):
    """
    Fields to be merged into the existing DocumentMetadata dict on disk.
    Never replaces existing fields — only adds new ones.

    Usage:
        ext = DocumentMetadataExtension(...)
        doc_meta_dict.update(ext.to_dict())
    """

    # ── Source linkage (NEW) ──────────────────────────────────────────────
    source_id:       Optional[str]        = None
    source_type:     Optional[str]        = None
    folder_path:     Optional[str]        = None
    folder_record:   Optional[Dict]       = None   # FolderRecord.to_dict()

    # ── Format identity (NEW) ─────────────────────────────────────────────
    file_format:     Optional[str]        = None   # FileFormat value
    mime_type:       Optional[str]        = None
    file_size_bytes: Optional[int]        = None
    structure_type:  Optional[str]        = None   # "article" | "slide_deck" | etc.

    # ── Format-specific totals (only one will be non-null per file) ───────
    total_slides:    Optional[int]        = None   # PPTX
    total_sheets:    Optional[int]        = None   # XLSX/CSV
    duration_seconds: Optional[float]    = None   # Video
    width_px:        Optional[int]        = None   # Image
    height_px:       Optional[int]        = None   # Image

    # ── Format-specific maps (mirror section_map / figure_index_map) ─────
    slide_map:       Optional[Dict]       = None   # PPTX: {slide_idx → StructuralUnit}
    sheet_map:       Optional[Dict]       = None   # XLSX: {sheet_name → StructuralUnit}
    column_schemas:  Optional[Dict]       = None   # XLSX/CSV: {sheet_name → [ColSchema]}
    scene_map:       Optional[Dict]       = None   # Video: {scene_idx → StructuralUnit}

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


# ─────────────────────────────────────────────────────────────────────────────
# Column schema for XLSX / CSV
# ─────────────────────────────────────────────────────────────────────────────

class ColumnSchema(BaseModel):
    """
    Schema descriptor for one column in a spreadsheet or CSV.
    Stored in doc_metadata.column_schemas[sheet_name][col_index].
    Used by: agent planner (knows if a column is numeric/categorical),
             enrichment layer (builds schema-aware summaries).
    """
    name:         str
    dtype:        str            # "int64", "float64", "object", "datetime64", etc.
    null_count:   int   = 0
    sample_values: List[Any] = Field(default_factory=list)  # first 3 unique values
    is_numeric:   bool  = False
    is_date:      bool  = False

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Extended chunk types — NEW modalities
# ─────────────────────────────────────────────────────────────────────────────

class SlideChunkMetadata(BaseModel):
    """
    One PPTX slide as a retrieval unit.

    ID scheme:  {doc_id}_slide_{slide_idx:03d}
    Modality:   "slide" (new modality alongside text/image/table)
    Embedding:  text_embedding on slide_text (title + body + notes)
                clip_embedding on slide_thumbnail (if rendered)
    """

    chunk_id:              str
    doc_id:                str
    chunk_index:           int
    source_modality:       str = "slide"
    file_format:           str = "pptx"

    # ── Slide content ─────────────────────────────────────────────────────
    slide_index:           int                    # 1-based
    slide_title:           Optional[str]   = None
    slide_body:            Optional[str]   = None  # all body text blocks joined
    slide_notes:           Optional[str]   = None  # presenter notes
    slide_layout:          Optional[str]   = None  # "Title Slide", "Two Content", etc.

    # ── Visuals on slide ──────────────────────────────────────────────────
    has_image:             bool = False
    has_table:             bool = False
    has_chart:             bool = False
    image_paths:           List[str] = Field(default_factory=list)  # extracted images
    table_ids:             List[str] = Field(default_factory=list)  # table chunk IDs

    # ── Thumbnail (rendered slide image) ─────────────────────────────────
    thumbnail_path:        Optional[str]   = None

    # ── Enrichment fields (filled by enrichment layer) ────────────────────
    contextual_summary:             Optional[str]   = None
    contextual_summary_confidence:  Optional[float] = None

    # ── Embeddings ────────────────────────────────────────────────────────
    text_embedding:        Optional[List[float]] = None   # on slide_text
    clip_embedding:        Optional[List[float]] = None   # on thumbnail

    # ── Back-links ────────────────────────────────────────────────────────
    related_figures:       List[str] = Field(default_factory=list)
    related_tables:        List[str] = Field(default_factory=list)
    section_id:            Optional[str] = None  # structure_unit_id of slide

    # ── Promoted fields for ChromaDB ──────────────────────────────────────
    promoted_fields:       Optional[Dict] = None

    def slide_text(self) -> str:
        """Full text content of the slide for embedding."""
        parts = [p for p in [self.slide_title, self.slide_body, self.slide_notes] if p]
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["slide_text"] = self.slide_text()
        return d


class VideoSegmentChunkMetadata(BaseModel):
    """
    One segment of a video — transcript-based or scene-based.

    Backward compatible with the original schema but extended for the
    three-level VRAG-style retrieval flow:
      1. transcript segment
      2. start/peak/end keyframes
      3. region crops on information-dense peak frames
    """

    chunk_id:              str
    doc_id:                str
    chunk_index:           int
    source_modality:       str = "video"
    file_format:           str = "video"

    # Temporal anchor
    start_time_s:          float
    end_time_s:            float
    segment_index:         int
    scene_index:           Optional[int] = None

    # Transcript
    transcript_text:       Optional[str]   = None
    transcript_confidence: Optional[float] = None
    speaker_label:         Optional[str]   = None

    # Visual anchor (legacy + new fields)
    keyframe_path:         Optional[str]   = None
    keyframe_timestamp_s:  Optional[float] = None
    keyframe_ids:          List[str]       = Field(default_factory=list)
    peak_frame_id:         Optional[str]   = None
    is_information_dense:  bool            = False
    optical_flow_score:    float           = 0.0

    # Enrichment
    contextual_summary:             Optional[str]   = None
    contextual_summary_confidence:  Optional[float] = None

    # Embeddings
    text_embedding:        Optional[List[float]] = None
    clip_embedding:        Optional[List[float]] = None

    # Promoted fields
    promoted_fields:       Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Extensions to EXISTING chunk models
# These are dicts of additional fields — merged into existing chunks at
# normalisation time. Not Pydantic models to avoid rewriting existing code.
# ─────────────────────────────────────────────────────────────────────────────

def make_docx_text_extension(
    heading_level:   Optional[int]   = None,    # 1–6 (H1/H2/...) or None
    heading_text:    Optional[str]   = None,
    paragraph_style: Optional[str]   = None,    # "Normal", "Heading 1", etc.
    list_item:       bool            = False,
    list_depth:      int             = 0,
) -> Dict[str, Any]:
    """
    Extra fields for TextChunkMetadata when source is DOCX.
    Merged into chunk dict under key 'format_specific'.
    """
    return {
        "heading_level":   heading_level,
        "heading_text":    heading_text,
        "paragraph_style": paragraph_style,
        "list_item":       list_item,
        "list_depth":      list_depth,
    }


def make_xlsx_table_extension(
    sheet_name:     str,
    sheet_index:    int,
    header_row:     Optional[List[str]] = None,
    start_row:      Optional[int]       = None,
    end_row:        Optional[int]       = None,
    column_schema:  Optional[List[Dict]] = None,
    row_count:      int = 0,
    col_count:      int = 0,
    is_schema_chunk: bool = False,  # True = the schema-description chunk
) -> Dict[str, Any]:
    """
    Extra fields for TableChunkMetadata when source is XLSX/CSV.
    Merged into chunk dict under key 'format_specific'.
    """
    return {
        "sheet_name":      sheet_name,
        "sheet_index":     sheet_index,
        "header_row":      header_row or [],
        "start_row":       start_row,
        "end_row":         end_row,
        "column_schema":   column_schema or [],
        "row_count":       row_count,
        "col_count":       col_count,
        "is_schema_chunk": is_schema_chunk,
    }


def make_image_standalone_extension(
    width_px:       Optional[int]  = None,
    height_px:      Optional[int]  = None,
    color_mode:     Optional[str]  = None,  # "RGB", "RGBA", "L", etc.
    ocr_text:       Optional[str]  = None,
    ocr_confidence: Optional[float] = None,
    detected_objects: Optional[List[str]] = None,
    region_boxes:   Optional[List[Dict]] = None,  # [{x,y,w,h,label}, ...]
) -> Dict[str, Any]:
    """
    Extra fields for ImageChunkMetadata when source is a standalone image file.
    """
    return {
        "width_px":         width_px,
        "height_px":        height_px,
        "color_mode":       color_mode,
        "ocr_text":         ocr_text,
        "ocr_confidence":   ocr_confidence,
        "detected_objects": detected_objects or [],
        "region_boxes":     region_boxes or [],
    }
