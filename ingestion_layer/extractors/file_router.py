"""
ingestion_layer/extractors/file_router.py

File Router — dispatches FileRecord objects to the correct extractor.

Design:
    - Registry maps FileFormat → extractor class (not instance).
    - Extractor classes are instantiated lazily per file — avoids loading
      heavy models (Docling, open_clip) until actually needed.
    - PDFExtractor is registered unchanged — zero modification to existing code.
    - All non-PDF extractors implement the same BaseExtractor interface.
    - The router returns a typed ExtractorResult on success, or raises
      ExtractionError with the original exception attached.

Extension pattern:
    To add a new format, create a new extractor class and call:
        FileRouter.register(FileFormat.MY_FORMAT, MyExtractor)
    No other code needs to change.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Type

from loguru import logger

from shared.models.multiformat_models import FileFormat
from ingestion_layer.loaders.source_loaders import FileRecord


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionError(Exception):
    """Raised by the router when an extractor fails."""
    def __init__(self, file_record: FileRecord, cause: Exception):
        super().__init__(str(cause))
        self.file_record = file_record
        self.cause       = cause


class UnsupportedFormatError(ExtractionError):
    """Raised when no extractor is registered for a FileFormat."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# ExtractorResult
# ─────────────────────────────────────────────────────────────────────────────

class ExtractorResult:
    """
    Standardised result from any extractor.

    Mirrors the existing PDFExtractor.extract() return dict:
        {"doc_id": str, "text_count": int, "image_count": int, "table_count": int}
    Adds format-specific counts for new modalities.
    """
    def __init__(
        self,
        doc_id:        str,
        file_format:   FileFormat,
        text_count:    int = 0,
        image_count:   int = 0,
        table_count:   int = 0,
        slide_count:   int = 0,    # PPTX
        segment_count: int = 0,    # Video
        extra:         Optional[Dict[str, Any]] = None,
    ):
        self.doc_id        = doc_id
        self.file_format   = file_format
        self.text_count    = text_count
        self.image_count   = image_count
        self.table_count   = table_count
        self.slide_count   = slide_count
        self.segment_count = segment_count
        self.extra         = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id":        self.doc_id,
            "file_format":   self.file_format.value,
            "text_count":    self.text_count,
            "image_count":   self.image_count,
            "table_count":   self.table_count,
            "slide_count":   self.slide_count,
            "segment_count": self.segment_count,
            **self.extra,
        }

    def __repr__(self) -> str:
        return (
            f"ExtractorResult(doc_id={self.doc_id!r}, format={self.file_format.value}, "
            f"text={self.text_count}, img={self.image_count}, tbl={self.table_count}, "
            f"slides={self.slide_count}, segs={self.segment_count})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# File Router
# ─────────────────────────────────────────────────────────────────────────────

class FileRouter:
    """
    Routes FileRecord objects to the correct extractor class.

    The registry is a class-level dict so it persists across instances
    and can be extended at runtime with FileRouter.register().

    Usage:
        router = FileRouter(output_dir=Path("./output"))
        result = router.route(file_record, chunk_strategy="paragraph", ...)
    """

    # Class-level registry: FileFormat → extractor class
    _registry: Dict[FileFormat, Type] = {}

    def __init__(self, output_dir: Path, **extractor_kwargs):
        """
        Args:
            output_dir:        Root output directory passed to all extractors.
            extractor_kwargs:  Forwarded to every extractor constructor
                               (e.g. chunk_strategy, chunk_size, etc.)
        """
        self.output_dir       = Path(output_dir)
        self.extractor_kwargs = extractor_kwargs
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Registry management ───────────────────────────────────────────────

    @classmethod
    def register(cls, file_format: FileFormat, extractor_class: Type) -> None:
        """
        Register an extractor class for a file format.

        This is called at module import time by each extractor module.
        Can also be called at runtime to override a registered extractor.

        Args:
            file_format:     FileFormat enum value.
            extractor_class: Class with an extract(file_record, **kwargs) method.
        """
        cls._registry[file_format] = extractor_class
        logger.debug(f"FileRouter: registered {extractor_class.__name__} for {file_format.value}")

    @classmethod
    def registered_formats(cls) -> list[FileFormat]:
        """Return the list of currently registered formats."""
        return list(cls._registry.keys())

    # ── Routing ───────────────────────────────────────────────────────────

    def route(self, file_record: FileRecord, **kwargs) -> ExtractorResult:
        """
        Route one FileRecord to the correct extractor and run extraction.

        Args:
            file_record: FileRecord from any source loader.
            **kwargs:    Forwarded to extractor.extract() — override per-file
                         settings (chunk_strategy, skip_images, etc.)

        Returns:
            ExtractorResult on success.

        Raises:
            UnsupportedFormatError if no extractor is registered.
            ExtractionError if the extractor raises any exception.
        """
        fmt = file_record.file_format

        if fmt not in self._registry:
            raise UnsupportedFormatError(
                file_record,
                NotImplementedError(
                    f"No extractor registered for {fmt.value}. "
                    f"Registered: {[f.value for f in self._registry]}"
                )
            )

        extractor_class = self._registry[fmt]

        # Instantiate extractor — heavy model loading happens here (first use)
        try:
            extractor = extractor_class(output_dir=self.output_dir)
        except Exception as e:
            raise ExtractionError(file_record, e)

        # Merge constructor kwargs with per-call overrides
        merged_kwargs = {**self.extractor_kwargs, **kwargs}

        try:
            logger.info(
                f"Routing: {file_record.relative_path!r} → "
                f"{extractor_class.__name__} [{fmt.value}]"
            )
            raw_result = extractor.extract(file_record, **merged_kwargs)

            # Normalise raw dict or ExtractorResult to ExtractorResult
            if isinstance(raw_result, ExtractorResult):
                return raw_result
            elif isinstance(raw_result, dict):
                return ExtractorResult(
                    doc_id      = raw_result.get("doc_id", ""),
                    file_format = fmt,
                    text_count  = raw_result.get("text_count", 0),
                    image_count = raw_result.get("image_count", 0),
                    table_count = raw_result.get("table_count", 0),
                )
            else:
                # Extractor returned something unexpected — wrap in result
                return ExtractorResult(doc_id=str(raw_result), file_format=fmt)

        except Exception as e:
            logger.error(
                f"Extraction failed: {file_record.relative_path!r} "
                f"({fmt.value}) → {type(e).__name__}: {e}"
            )
            raise ExtractionError(file_record, e) from e

    def can_handle(self, file_record: FileRecord) -> bool:
        """Return True if this router has a registered extractor for the file."""
        return file_record.file_format in self._registry
