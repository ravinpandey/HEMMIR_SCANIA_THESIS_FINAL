"""
shared/models/metadata_models.py

Pydantic is used here for exactly three things:

1. EMBEDDING DIMENSION VALIDATION
   Wrong-dim vectors cause silent corruption in ChromaDB.
   → clip_embedding         : 512-dim  (CLIP ViT-B-32)
   → text_embedding         : 1536-dim (OpenAI text-embedding-3-small)
   → html_text_embedding    : 1536-dim (OpenAI)
   → doc_embedding          : 1536-dim (OpenAI)

2. CHROMADB METADATA TYPE COERCION
   ChromaDB rejects None, lists, dicts. PromotedFields coerces to safe scalars.

3. DISK→MEMORY BOUNDARY VALIDATION
   validate_*() helpers called once per layer when loading JSON from disk.
   Skips invalid records with a warning rather than crashing the whole run.

All 5 formats + video fully covered:
   PDF / DOCX / PPTX  → TextChunkMetadata, ImageChunkMetadata, TableChunkMetadata
   CSV / XLSX         → SheetTableChunkMetadata  (extends TableChunkMetadata)
   Video              → VideoSegmentChunkMetadata, VideoFrameChunkMetadata
   All formats        → StructureUnitMetadata, DocumentMetadata
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from loguru import logger


# ── Embedding dimension constants ─────────────────────────────────────────────
TEXT_EMBED_DIM = 1536
CLIP_EMBED_DIM = 512
DOC_EMBED_DIM  = 1536


def _check_dim(
    v: Optional[List[float]], expected: int, name: str
) -> Optional[List[float]]:
    if v is None:
        return None
    if not isinstance(v, (list, tuple)):
        raise ValueError(f"{name} must be list[float], got {type(v).__name__}")
    if len(v) != expected:
        raise ValueError(f"{name} must be {expected}-dim, got {len(v)}-dim")
    return list(v)


# ═══════════════════════════════════════════════════════════════════════════════
# Document metadata
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentMetadata(BaseModel):
    """Document-level metadata. Pydantic: doc_embedding dimension validation."""
    doc_id:           str
    doc_title:        str
    source_id:        str            = ""
    project_id:       Optional[str]  = None
    author:           Optional[str]  = None
    file_format:      str            = ""
    document_type:    Optional[str]  = None
    language:         str            = "English"
    total_pages:      int            = 0
    chunk_count:      int            = 0
    last_modified:    Optional[str]  = None
    chunk_strategy:   str            = "paragraph"
    chunk_size:       Optional[int]  = None
    chunk_overlap:    Optional[int]  = None
    folder_path:      str            = ""
    # Maps set at ingestion
    figure_index_map: Dict[str, str]  = Field(default_factory=dict)
    table_index_map:  Dict[str, str]  = Field(default_factory=dict)
    section_map:      Dict[str, Any]  = Field(default_factory=dict)
    related_figures:  List[str]       = Field(default_factory=list)
    related_tables:   List[str]       = Field(default_factory=list)
    # Video-specific
    total_duration_s: Optional[float] = None
    total_segments:   Optional[int]   = None
    total_frames:     Optional[int]   = None
    whisper_model:    Optional[str]   = None
    # Enrichment fields (filled by enrichment layer)
    doc_summary:             Optional[str]   = None
    doc_summary_confidence:  Optional[float] = None
    tier1:                   Optional[str]   = None
    tier2:                   Optional[str]   = None
    tier_confidence:         Optional[float] = None
    # Encoding fields (filled by encoding layer)
    outline_summary:          Any            = None
    outline_summary_text:     Optional[str]  = None
    section_map_text:         Optional[str]  = None
    doc_embedding_input_text: Optional[str]  = None
    # Embedding (dimension validated)
    doc_embedding: Optional[List[float]] = None

    @field_validator("doc_embedding", mode="before")
    @classmethod
    def val_doc_emb(cls, v):
        return _check_dim(v, DOC_EMBED_DIM, "doc_embedding")

    def to_dict(self) -> Dict:
        return self.model_dump()


# ═══════════════════════════════════════════════════════════════════════════════
# Structure Unit
# ═══════════════════════════════════════════════════════════════════════════════

class StructureUnitMetadata(BaseModel):
    """
    One logical unit: section (PDF/DOCX), slide (PPTX), sheet (CSV/XLSX),
    scene (video). Semantic anchor set at ingestion; upgraded by enrichment.
    """
    structure_unit_id:   str
    doc_id:              str
    source_id:           str            = ""
    structure_unit_type: str            = "section"
    unit_index:          int            = 0
    title:               str            = ""
    semantic_anchor:     str            = ""
    # Page / time ranges set at ingestion
    start_page:          Optional[int]   = None
    end_page:            Optional[int]   = None
    start_element_idx:   Optional[int]   = None
    end_element_idx:     Optional[int]   = None
    start_time_s:        Optional[float] = None
    end_time_s:          Optional[float] = None
    # Spreadsheet
    sheet_name:          Optional[str]   = None
    # Chunk membership (filled at ingestion / encoding)
    chunk_ids:           List[str]       = Field(default_factory=list)
    figure_ids:          List[str]       = Field(default_factory=list)
    table_ids:           List[str]       = Field(default_factory=list)
    frame_ids:           List[str]       = Field(default_factory=list)
    # Flags
    synthetic:           bool            = False
    # Enrichment fields (filled by doc_meta_enricher)
    section_summary:     Optional[str]   = None
    keywords:            List[str]       = Field(default_factory=list)
    entities:            List[str]       = Field(default_factory=list)
    subsections:         List[str]       = Field(default_factory=list)

    def to_dict(self) -> Dict:
        return self.model_dump()


# ═══════════════════════════════════════════════════════════════════════════════
# Text chunk (PDF / DOCX / PPTX)
# ═══════════════════════════════════════════════════════════════════════════════

class TextChunkMetadata(BaseModel):
    """Text chunk for PDF/DOCX/PPTX. Pydantic: text_embedding dimension."""
    chunk_id:              str
    doc_id:                str
    source_id:             str                  = ""
    source_modality:       Literal["text"]      = "text"
    file_format:           str                  = ""
    chunk_index:           int                  = 0
    structure_unit_id:     str                  = ""
    structure_unit_type:   str                  = "section"
    structure_unit_title:  Optional[str]        = None
    parent_id:             str                  = ""
    text_original_content: str                  = ""
    section_title:         Optional[str]        = None
    section_id:            Optional[str]        = None
    page_number:           Optional[int]        = None
    element_index:         Optional[int]        = None
    end_element_index:     Optional[int]        = None
    chunk_strategy:        str                  = "paragraph"
    token_count:           Optional[int]        = None
    semantic_anchor:       Optional[str]        = None
    # DOCX-specific
    heading_breadcrumb:    Optional[str]        = None
    # PPTX-specific
    slide_index:           Optional[int]        = None
    slide_title:           Optional[str]        = None
    speaker_notes:         Optional[str]        = None
    # Cross-reference (filled by cross_reference_layer)
    related_figures:       List[str]            = Field(default_factory=list)
    related_tables:        List[str]            = Field(default_factory=list)
    # Enrichment (filled by enrichment_layer)
    local_context:                     Optional[str]   = None
    contextual_summary:                Optional[str]   = None
    contextual_summary_confidence:     Optional[float] = None
    detected_codes:                    List[str]        = Field(default_factory=list)
    entities:                          List[str]        = Field(default_factory=list)
    salience_score:                    Optional[float] = None
    evidence_role:                     Optional[str]   = None
    # Encoding (filled by encoding_layer)
    promoted_fields:                   Dict[str, Any]  = Field(default_factory=dict)
    summary_text:                      Optional[str]   = None
    entity_text:                       Optional[str]   = None
    # Embedding (dimension validated)
    text_embedding: Optional[List[float]] = None

    @field_validator("text_embedding", mode="before")
    @classmethod
    def val_emb(cls, v):
        return _check_dim(v, TEXT_EMBED_DIM, "text_embedding")

    def to_dict(self) -> Dict:
        return self.model_dump()


# ── DOCX-specific subclass ────────────────────────────────────────────────────

class DocxChunkMetadata(TextChunkMetadata):
    """DOCX text chunk. Adds heading_breadcrumb. Inherits all TextChunk fields."""
    file_format: str = "docx"
    chunk_strategy: str = "heading_group"


# ── PPTX-specific subclass ────────────────────────────────────────────────────

class SlideChunkMetadata(TextChunkMetadata):
    """PPTX slide chunk. slide_title and speaker_notes are first-class fields."""
    file_format: str = "pptx"
    chunk_strategy: str = "slide"


# ═══════════════════════════════════════════════════════════════════════════════
# Image chunk (PDF / DOCX / PPTX)
# ═══════════════════════════════════════════════════════════════════════════════

class ImageChunkMetadata(BaseModel):
    """Image chunk. Pydantic: clip_embedding (512) + text_embedding (1536) dims."""
    chunk_id:          str
    doc_id:            str
    source_id:         str                  = ""
    source_modality:   Literal["image"]     = "image"
    file_format:       str                  = ""
    chunk_index:       int                  = 0
    structure_unit_id: str                  = ""
    parent_id:         str                  = ""
    figure_id:         str                  = ""
    image_path:        Optional[str]        = None
    page_number:       Optional[int]        = None
    section_id:        Optional[str]        = None
    # PPTX-specific
    slide_index:       Optional[int]        = None
    slide_title:       Optional[str]        = None
    # Image type
    image_type:        Optional[str]        = None
    # Cross-reference
    related_sections:  List[str]            = Field(default_factory=list)
    # Enrichment
    image_caption:                   Optional[str]   = None
    image_caption_confidence:        Optional[float] = None
    depicted_component:              Optional[str]   = None
    depicted_component_confidence:   Optional[float] = None
    visible_annotations:             Optional[str]   = None
    contextual_summary:              Optional[str]   = None
    contextual_summary_confidence:   Optional[float] = None
    # Encoding
    promoted_fields:     Dict[str, Any] = Field(default_factory=dict)
    # Embedding (dimensions validated)
    clip_embedding: Optional[List[float]] = None
    text_embedding: Optional[List[float]] = None

    @field_validator("clip_embedding", mode="before")
    @classmethod
    def val_clip(cls, v):
        return _check_dim(v, CLIP_EMBED_DIM, "clip_embedding")

    @field_validator("text_embedding", mode="before")
    @classmethod
    def val_text(cls, v):
        return _check_dim(v, TEXT_EMBED_DIM, "image.text_embedding")

    def to_dict(self) -> Dict:
        return self.model_dump()


# ── PPTX thumbnail subclass ───────────────────────────────────────────────────

class SlideImageChunkMetadata(ImageChunkMetadata):
    """PPTX slide thumbnail. Enrichment prompt differs from embedded PDF figure."""
    file_format: str = "pptx"
    image_type: str = "slide_thumbnail"


# ═══════════════════════════════════════════════════════════════════════════════
# Table chunk (PDF / DOCX / PPTX)
# ═══════════════════════════════════════════════════════════════════════════════

class TableChunkMetadata(BaseModel):
    """Table chunk. Pydantic: text_embedding + html_text_embedding dims."""
    chunk_id:          str
    doc_id:            str
    source_id:         str                  = ""
    source_modality:   Literal["table"]     = "table"
    file_format:       str                  = ""
    chunk_index:       int                  = 0
    structure_unit_id: str                  = ""
    parent_id:         str                  = ""
    table_html:        Optional[str]        = None
    table_csv_path:    Optional[str]        = None
    html_file_path:    Optional[str]        = None
    page_number:       Optional[int]        = None
    row_count:         Optional[int]        = None
    col_count:         Optional[int]        = None
    column_names:      List[str]            = Field(default_factory=list)
    markdown:          Optional[str]        = None
    section_id:        Optional[str]        = None
    # Spreadsheet-specific
    sheet_name:        Optional[str]        = None
    sheet_index:       Optional[int]        = None
    row_start:         Optional[int]        = None
    row_end:           Optional[int]        = None
    is_schema_chunk:   bool                 = False
    # Enrichment
    table_caption:              Optional[str]        = None
    table_caption_confidence:   Optional[float]      = None
    table_summary:              Optional[str]        = None
    table_summary_confidence:   Optional[float]      = None
    table_purpose:              Optional[str]        = None
    table_purpose_confidence:   Optional[float]      = None
    column_semantics:           Dict[str, str]       = Field(default_factory=dict)
    # Encoding
    promoted_fields:    Dict[str, Any] = Field(default_factory=dict)
    # Embedding (dimensions validated)
    text_embedding:      Optional[List[float]] = None
    html_text_embedding: Optional[List[float]] = None

    @field_validator("text_embedding", "html_text_embedding", mode="before")
    @classmethod
    def val_embs(cls, v):
        return _check_dim(v, TEXT_EMBED_DIM, "table_embedding")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.table_html:
            warnings.warn(
                f"TableChunk {self.chunk_id}: table_html is None — "
                "html_text_embedding will be skipped",
                stacklevel=2,
            )

    def to_dict(self) -> Dict:
        return self.model_dump()


# ── Spreadsheet-specific subclass ─────────────────────────────────────────────

class SheetTableChunkMetadata(TableChunkMetadata):
    """CSV/XLSX row-group or schema chunk. Adds sheet-level fields."""
    file_format: str = "csv"


# ═══════════════════════════════════════════════════════════════════════════════
# Video chunks
# ═══════════════════════════════════════════════════════════════════════════════

class VideoSegmentChunkMetadata(BaseModel):
    """
    One ASR transcript segment from a video.
    Routed through _encode_text() — consumes contextual_summary + local_context.
    Pydantic: text_embedding dimension.
    """
    chunk_id:              str
    doc_id:                str
    source_id:             str                      = ""
    source_modality:       Literal["video_segment"] = "video_segment"
    file_format:           str                      = "mp4"
    chunk_index:           int                      = 0
    structure_unit_id:     str                      = ""
    parent_id:             str                      = ""
    segment_index:         int                      = 0
    # Temporal
    start_time_s:          float                    = 0.0
    end_time_s:            float                    = 0.0
    duration_s:            float                    = 0.0
    # ASR content
    transcript_text:       str                      = ""
    text_original_content: str                      = ""
    word_timestamps:       Optional[str]            = None  # JSON string
    asr_language:          Optional[str]            = None
    asr_confidence:        Optional[float]          = None
    token_count:           Optional[int]            = None
    # Cross-reference
    keyframe_ids:          List[str]                = Field(default_factory=list)
    scene_sibling_ids:     List[str]                = Field(default_factory=list)
    # Enrichment (filled by video_segment_enricher)
    local_context:                     Optional[str]   = None
    contextual_summary:                Optional[str]   = None
    contextual_summary_confidence:     Optional[float] = None
    detected_codes:                    List[str]        = Field(default_factory=list)
    salience_score:                    Optional[float] = None
    evidence_role:                     Optional[str]   = None
    # Encoding
    promoted_fields:                   Dict[str, Any]  = Field(default_factory=dict)
    # Embedding (dimension validated)
    text_embedding: Optional[List[float]] = None

    @field_validator("text_embedding", mode="before")
    @classmethod
    def val_emb(cls, v):
        return _check_dim(v, TEXT_EMBED_DIM, "video_segment.text_embedding")

    def to_dict(self) -> Dict:
        return self.model_dump()


class VideoFrameChunkMetadata(BaseModel):
    """
    One keyframe extracted from a video segment.
    Routed through _encode_image() — consumes image_caption + contextual_summary.
    Pydantic: clip_embedding (512) + text_embedding (1536) dims.
    """
    chunk_id:          str
    doc_id:            str
    source_id:         str                      = ""
    source_modality:   Literal["video_frame"]   = "video_frame"
    file_format:       str                      = "mp4"
    chunk_index:       int                      = 0
    structure_unit_id: str                      = ""
    parent_id:         str                      = ""   # → parent VideoSegmentChunk
    segment_index:     int                      = 0
    frame_position:    int                      = 1    # 1–4 within segment
    # Temporal
    timestamp_s:       float                    = 0.0
    segment_start_s:   float                    = 0.0
    segment_end_s:     float                    = 0.0
    # Image
    image_path:        Optional[str]            = None
    # Enrichment (filled by video_frame_enricher)
    image_caption:                   Optional[str]   = None
    image_caption_confidence:        Optional[float] = None
    contextual_summary:              Optional[str]   = None
    contextual_summary_confidence:   Optional[float] = None
    frame_role:                      Optional[str]   = None
    # Encoding
    promoted_fields:   Dict[str, Any] = Field(default_factory=dict)
    # Embedding (dimensions validated)
    clip_embedding: Optional[List[float]] = None
    text_embedding: Optional[List[float]] = None

    @field_validator("clip_embedding", mode="before")
    @classmethod
    def val_clip(cls, v):
        return _check_dim(v, CLIP_EMBED_DIM, "video_frame.clip_embedding")

    @field_validator("text_embedding", mode="before")
    @classmethod
    def val_text(cls, v):
        return _check_dim(v, TEXT_EMBED_DIM, "video_frame.text_embedding")

    def to_dict(self) -> Dict:
        return self.model_dump()


# ── Source metadata ───────────────────────────────────────────────────────────

class SourceMetadata(BaseModel):
    source_id:       str
    source_type:     str            = "local"
    source_name:     str            = ""
    local_root:      str            = ""
    folder_path:     str            = ""
    ingestion_date:  str            = ""
    version_tag:     str            = "v1.0"
    description:     str            = ""
    tags:            List[str]      = Field(default_factory=list)

    def to_dict(self) -> Dict:
        return self.model_dump()


# ═══════════════════════════════════════════════════════════════════════════════
# ChromaDB metadata coercion
# ═══════════════════════════════════════════════════════════════════════════════

class PromotedFields(BaseModel):
    """
    Flat key-value metadata for ChromaDB WHERE filters.
    All fields coerced to str/int/float — no None, no lists, no dicts.
    """
    doc_id:           str   = ""
    doc_title:        str   = ""
    tier1:            str   = ""
    tier2:            str   = ""
    project_id:       str   = ""
    document_type:    str   = ""
    language:         str   = "English"
    source_modality:  str   = ""
    file_format:      str   = ""
    page_number:      int   = 0
    chunk_index:      int   = 0
    chunk_strategy:   str   = ""
    structure_unit_id: str  = ""
    structure_unit_type: str = ""
    evidence_role:    str   = ""
    salience_score:   float = 0.0
    slide_index:      int   = 0
    sheet_name:       str   = ""
    start_time_s:     float = 0.0
    end_time_s:       float = 0.0
    folder_path:      str   = ""
    source_id:        str   = ""
    contextual_summary_confidence: float = 0.0

    @field_validator(
        "doc_id","doc_title","tier1","tier2","project_id","document_type",
        "language","source_modality","file_format","chunk_strategy",
        "structure_unit_id","structure_unit_type","evidence_role",
        "sheet_name","folder_path","source_id",
        mode="before",
    )
    @classmethod
    def coerce_str(cls, v):
        return "" if v is None else str(v).strip()

    @field_validator("page_number","chunk_index","slide_index", mode="before")
    @classmethod
    def coerce_int(cls, v):
        if v is None: return 0
        try: return int(v)
        except (TypeError, ValueError): return 0

    @field_validator(
        "salience_score","start_time_s","end_time_s",
        "contextual_summary_confidence",
        mode="before",
    )
    @classmethod
    def coerce_float(cls, v):
        if v is None: return 0.0
        try: return round(float(v), 4)
        except (TypeError, ValueError): return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Disk→memory boundary validators
# Called once per layer when loading JSON. Skips invalid records with warning.
# ═══════════════════════════════════════════════════════════════════════════════

def validate_doc_metadata(raw: Dict) -> DocumentMetadata:
    return DocumentMetadata.model_validate(raw)


def validate_structure_units(raw_list: List[Dict]) -> List[StructureUnitMetadata]:
    out = []
    for i, raw in enumerate(raw_list):
        try:
            out.append(StructureUnitMetadata.model_validate(raw))
        except Exception as e:
            logger.warning(f"[validate_structure_units] idx_{i}: {e}")
    return out


def validate_text_chunks(raw_list: List[Dict]) -> List[TextChunkMetadata]:
    out = []
    for i, raw in enumerate(raw_list):
        try:
            out.append(TextChunkMetadata.model_validate(raw))
        except Exception as e:
            logger.warning(f"[validate_text_chunks] {raw.get('chunk_id', f'idx_{i}')}: {e}")
    return out


def validate_image_chunks(raw_list: List[Dict]) -> List[ImageChunkMetadata]:
    out = []
    for i, raw in enumerate(raw_list):
        try:
            out.append(ImageChunkMetadata.model_validate(raw))
        except Exception as e:
            logger.warning(f"[validate_image_chunks] {raw.get('chunk_id', f'idx_{i}')}: {e}")
    return out


def validate_table_chunks(raw_list: List[Dict]) -> List[TableChunkMetadata]:
    out, no_html = [], []
    for i, raw in enumerate(raw_list):
        try:
            m = TableChunkMetadata.model_validate(raw)
            out.append(m)
            if not m.table_html:
                no_html.append(m.chunk_id)
        except Exception as e:
            logger.warning(f"[validate_table_chunks] {raw.get('chunk_id', f'idx_{i}')}: {e}")
    if no_html:
        logger.warning(f"[validate_table_chunks] {len(no_html)} missing table_html: {no_html}")
    return out


def validate_video_segments(raw_list: List[Dict]) -> List[VideoSegmentChunkMetadata]:
    out = []
    for i, raw in enumerate(raw_list):
        try:
            out.append(VideoSegmentChunkMetadata.model_validate(raw))
        except Exception as e:
            logger.warning(f"[validate_video_segments] {raw.get('chunk_id', f'idx_{i}')}: {e}")
    return out


def validate_video_frames(raw_list: List[Dict]) -> List[VideoFrameChunkMetadata]:
    out = []
    for i, raw in enumerate(raw_list):
        try:
            out.append(VideoFrameChunkMetadata.model_validate(raw))
        except Exception as e:
            logger.warning(f"[validate_video_frames] {raw.get('chunk_id', f'idx_{i}')}: {e}")
    return out
