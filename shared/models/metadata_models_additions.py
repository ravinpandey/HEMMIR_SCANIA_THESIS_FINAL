"""
shared/models/metadata_models_additions.py

Re-exports all format-specific metadata models from metadata_models.py.

The existing HEMMIR_updated extractors import from this exact path:
    from shared.models.metadata_models_additions import (
        StructureUnitMetadata,
        VideoSegmentChunkMetadata,
        VideoFrameChunkMetadata,
        SlideChunkMetadata,
        SlideImageChunkMetadata,
        DocxChunkMetadata,
        SheetTableChunkMetadata,
        SourceMetadata,
    )

This file is a pure re-export shim — all implementations live in
metadata_models.py. Nothing is defined here; everything is imported and
re-exported so both import paths resolve to the same class objects.

No circular imports: metadata_models.py does not import from this file.
"""

from shared.models.metadata_models import (
    # Core models (also available from metadata_models directly)
    DocumentMetadata,
    StructureUnitMetadata,
    TextChunkMetadata,
    ImageChunkMetadata,
    TableChunkMetadata,

    # Format-specific subclasses (the ones extractors specifically need)
    DocxChunkMetadata,
    SlideChunkMetadata,
    SlideImageChunkMetadata,
    SheetTableChunkMetadata,

    # Video-specific models
    VideoSegmentChunkMetadata,
    VideoFrameChunkMetadata,

    # Source / provenance
    SourceMetadata,

    # Embedding dimension constants (used by embedding layer)
    TEXT_EMBED_DIM,
    CLIP_EMBED_DIM,
    DOC_EMBED_DIM,

    # Validation helpers
    validate_doc_metadata,
    validate_structure_units,
    validate_text_chunks,
    validate_image_chunks,
    validate_table_chunks,
    validate_video_segments,
    validate_video_frames,

    # ChromaDB coercion model
    PromotedFields,
)

__all__ = [
    "DocumentMetadata",
    "StructureUnitMetadata",
    "TextChunkMetadata",
    "ImageChunkMetadata",
    "TableChunkMetadata",
    "DocxChunkMetadata",
    "SlideChunkMetadata",
    "SlideImageChunkMetadata",
    "SheetTableChunkMetadata",
    "VideoSegmentChunkMetadata",
    "VideoFrameChunkMetadata",
    "SourceMetadata",
    "TEXT_EMBED_DIM",
    "CLIP_EMBED_DIM",
    "DOC_EMBED_DIM",
    "validate_doc_metadata",
    "validate_structure_units",
    "validate_text_chunks",
    "validate_image_chunks",
    "validate_table_chunks",
    "validate_video_segments",
    "validate_video_frames",
    "PromotedFields",
]
