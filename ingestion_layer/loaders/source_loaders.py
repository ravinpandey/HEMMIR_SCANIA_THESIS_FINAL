"""
ingestion_layer/loaders/source_loaders.py

Source Loaders — discover files from local directories or S3 buckets.

Design:
    - BaseSourceLoader defines the interface: iterate() yields FileRecord objects.
    - LocalSourceLoader walks a nested local directory tree recursively.
    - S3SourceLoader lists all objects under an S3 prefix recursively.
    - Both produce identical FileRecord output — the router/extractor never
      knows whether the file came from disk or S3.
    - S3 files are streamed to a local temp path before extraction
      (avoids streaming binary parsers which require seekable streams).

Backward compatibility:
    - Neither loader changes the existing PDF extraction path.
    - The existing main.py can be left untouched; this replaces its
      `pdf_dir.glob("*.pdf")` call with LocalSourceLoader.iterate().

Usage:
    loader = LocalSourceLoader.from_source(source_record)
    for file_record in loader.iterate(extensions=[".pdf", ".docx"]):
        extractor = router.get_extractor(file_record)
        extractor.extract(file_record)
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional, Set

from loguru import logger

from shared.models.multiformat_models import (
    FolderRecord,
    FileFormat,
    SourceRecord,
    SourceType,
)


# ─────────────────────────────────────────────────────────────────────────────
# FileRecord — canonical representation of one discovered file
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FileRecord:
    """
    Represents one file discovered by a source loader.

    local_path:   absolute path on disk (may be a temp file for S3 objects)
    relative_path: path relative to the source root (e.g. "reports/2026/q1.pdf")
    file_format:  detected FileFormat enum value
    source_record: the SourceRecord this file came from
    folder_record: pre-computed FolderRecord for this file's directory
    is_temp:      True if local_path is a temporary download (must be deleted after use)
    s3_key:       original S3 object key (None for local files)
    file_size_bytes: size in bytes
    last_modified: ISO timestamp string
    """
    local_path:     Path
    relative_path:  str
    file_format:    FileFormat
    source_record:  SourceRecord
    folder_record:  FolderRecord
    is_temp:        bool = False
    s3_key:         Optional[str] = None
    file_size_bytes: int = 0
    last_modified:  Optional[str] = None

    def cleanup(self) -> None:
        """Delete the local temp file if this was a downloaded S3 object."""
        if self.is_temp and self.local_path.exists():
            try:
                self.local_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp file {self.local_path}: {e}")

    @property
    def doc_stem(self) -> str:
        """File name without extension — used as output folder name."""
        return self.local_path.stem

    @property
    def mime_type(self) -> str:
        return _MIME_MAP.get(self.file_format, "application/octet-stream")


# ─────────────────────────────────────────────────────────────────────────────
# Extension → FileFormat mapping
# ─────────────────────────────────────────────────────────────────────────────

_EXT_TO_FORMAT: dict[str, FileFormat] = {
    ".pdf":  FileFormat.PDF,
    ".docx": FileFormat.DOCX,
    ".doc":  FileFormat.DOCX,    # legacy — needs LibreOffice conversion
    ".pptx": FileFormat.PPTX,
    ".ppt":  FileFormat.PPTX,    # legacy — needs LibreOffice conversion
    ".csv":  FileFormat.CSV,
    ".tsv":  FileFormat.CSV,
    ".xlsx": FileFormat.XLSX,
    ".xlsm": FileFormat.XLSX,
    ".xls":  FileFormat.XLSX,    # legacy — use xlrd engine
    ".jpg":  FileFormat.IMAGE,
    ".jpeg": FileFormat.IMAGE,
    ".png":  FileFormat.IMAGE,
    ".gif":  FileFormat.IMAGE,
    ".webp": FileFormat.IMAGE,
    ".tiff": FileFormat.IMAGE,
    ".bmp":  FileFormat.IMAGE,
    ".mp4":  FileFormat.VIDEO,
    ".avi":  FileFormat.VIDEO,
    ".mov":  FileFormat.VIDEO,
    ".mkv":  FileFormat.VIDEO,
    ".webm": FileFormat.VIDEO,
    ".mp3":  FileFormat.VIDEO,   # audio-only treated as video (transcript-only path)
    ".wav":  FileFormat.VIDEO,
    ".m4a":  FileFormat.VIDEO,
}

_MIME_MAP: dict[FileFormat, str] = {
    FileFormat.PDF:    "application/pdf",
    FileFormat.DOCX:   "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    FileFormat.PPTX:   "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    FileFormat.CSV:    "text/csv",
    FileFormat.XLSX:   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    FileFormat.IMAGE:  "image/*",
    FileFormat.VIDEO:  "video/*",
    FileFormat.UNKNOWN: "application/octet-stream",
}

# Files to always skip during traversal
_SKIP_PATTERNS: Set[str] = {
    ".DS_Store", "Thumbs.db", "desktop.ini",
    "__MACOSX", ".git", ".svn", "node_modules",
    "__pycache__", ".pytest_cache",
}

ALL_SUPPORTED_EXTENSIONS: Set[str] = set(_EXT_TO_FORMAT.keys())


def detect_format(path: Path) -> FileFormat:
    """Return FileFormat from file extension. Returns UNKNOWN for unrecognised."""
    return _EXT_TO_FORMAT.get(path.suffix.lower(), FileFormat.UNKNOWN)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Base Loader
# ─────────────────────────────────────────────────────────────────────────────

class BaseSourceLoader(ABC):
    """
    Abstract base for all source loaders.

    Subclasses implement iterate() which yields FileRecord objects.
    The caller is responsible for calling file_record.cleanup() after
    processing each record.
    """

    def __init__(self, source_record: SourceRecord):
        self.source_record = source_record

    @abstractmethod
    def iterate(
        self,
        extensions: Optional[List[str]] = None,
        skip_formats: Optional[List[FileFormat]] = None,
        max_files: Optional[int] = None,
    ) -> Generator[FileRecord, None, None]:
        """
        Yield FileRecord objects for every discoverable file.

        Args:
            extensions:   Whitelist of extensions (e.g. [".pdf", ".docx"]).
                          None means all supported extensions.
            skip_formats: FileFormat values to exclude (e.g. [FileFormat.VIDEO]).
            max_files:    Stop after N files (useful for dry-run / testing).
        """
        ...

    @abstractmethod
    def count(self, extensions: Optional[List[str]] = None) -> int:
        """Return total number of files that would be yielded by iterate()."""
        ...

    def _should_include(
        self,
        path: Path,
        allowed_extensions: Optional[Set[str]],
        skip_formats: Optional[List[FileFormat]],
    ) -> bool:
        """Shared filter logic for both local and S3 loaders."""
        # Skip hidden files and known noise patterns
        if path.name.startswith("."):
            return False
        if any(skip in path.parts for skip in _SKIP_PATTERNS):
            return False
        ext    = path.suffix.lower()
        fmt    = _EXT_TO_FORMAT.get(ext, FileFormat.UNKNOWN)
        if fmt == FileFormat.UNKNOWN:
            return False
        if allowed_extensions and ext not in allowed_extensions:
            return False
        if skip_formats and fmt in skip_formats:
            return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Local Recursive Loader
# ─────────────────────────────────────────────────────────────────────────────

class LocalSourceLoader(BaseSourceLoader):
    """
    Recursively discovers files under a local directory.

    Replaces the existing `pdf_dir.glob("*.pdf")` pattern in main.py.
    Handles nested subdirectories correctly, populating tier1/tier2
    from the folder hierarchy.

    Example:
        source = SourceRecord.from_local("/data/docs", "Research Docs")
        loader = LocalSourceLoader(source)
        for f in loader.iterate(extensions=[".pdf", ".docx"]):
            print(f.relative_path, f.folder_record.tier1)
    """

    def __init__(self, source_record: SourceRecord):
        super().__init__(source_record)
        self.root = Path(source_record.local_root)
        if not self.root.exists():
            raise FileNotFoundError(f"Local root does not exist: {self.root}")

    @classmethod
    def from_source(cls, source_record: SourceRecord) -> "LocalSourceLoader":
        """Convenience factory matching the S3 loader pattern."""
        return cls(source_record)

    def iterate(
        self,
        extensions: Optional[List[str]] = None,
        skip_formats: Optional[List[FileFormat]] = None,
        max_files: Optional[int] = None,
    ) -> Generator[FileRecord, None, None]:
        """
        Walk the directory tree, yielding one FileRecord per discovered file.

        Files are yielded in sorted order (deterministic across runs).
        Subdirectories are recursed automatically.
        Symlinks are followed once (no loop detection for simplicity).
        """
        allowed_exts = {e.lower() for e in extensions} if extensions else None
        count = 0

        for abs_path in sorted(self.root.rglob("*")):
            if not abs_path.is_file():
                continue
            if not self._should_include(abs_path, allowed_exts, skip_formats):
                continue

            rel = abs_path.relative_to(self.root)
            folder_record = FolderRecord.from_path(
                self.source_record.source_id, str(rel)
            )
            file_format = detect_format(abs_path)

            try:
                stat = abs_path.stat()
                size = stat.st_size
                from datetime import datetime
                mtime = datetime.utcfromtimestamp(stat.st_mtime).isoformat()
            except OSError as e:
                logger.warning(f"Cannot stat {abs_path}: {e}")
                size, mtime = 0, None

            yield FileRecord(
                local_path      = abs_path,
                relative_path   = str(rel).replace("\\", "/"),
                file_format     = file_format,
                source_record   = self.source_record,
                folder_record   = folder_record,
                is_temp         = False,
                file_size_bytes = size,
                last_modified   = mtime,
            )

            count += 1
            if max_files and count >= max_files:
                logger.info(f"  LocalLoader: stopped at max_files={max_files}")
                break

    def count(self, extensions: Optional[List[str]] = None) -> int:
        allowed_exts = {e.lower() for e in extensions} if extensions else None
        return sum(
            1 for p in self.root.rglob("*")
            if p.is_file() and self._should_include(p, allowed_exts, None)
        )


# ─────────────────────────────────────────────────────────────────────────────
# S3 Loader
# ─────────────────────────────────────────────────────────────────────────────

class S3SourceLoader(BaseSourceLoader):
    """
    Lists and streams files from an S3 bucket prefix.

    Each file is downloaded to a temporary local file before being yielded.
    The caller MUST call file_record.cleanup() after processing to free disk.

    Requires: boto3, AWS credentials configured (env vars / IAM role / profile).

    Example:
        source = SourceRecord.from_s3("my-bucket", "projects/alpha/", "Alpha")
        loader = S3SourceLoader(source)
        for f in loader.iterate(extensions=[".pdf", ".pptx"]):
            try:
                extractor.extract(f)
            finally:
                f.cleanup()  # always delete temp file
    """

    def __init__(self, source_record: SourceRecord, temp_dir: Optional[str] = None):
        super().__init__(source_record)
        self.bucket    = source_record.bucket_name
        self.prefix    = source_record.root_prefix or ""
        self.temp_dir  = temp_dir or tempfile.gettempdir()
        self._s3_client = None

    def _get_client(self):
        """Lazy init boto3 client — avoids import error when not using S3."""
        if self._s3_client is None:
            try:
                import boto3
                kwargs = {}
                if self.source_record.aws_region:
                    kwargs["region_name"] = self.source_record.aws_region
                self._s3_client = boto3.client("s3", **kwargs)
            except ImportError:
                raise ImportError(
                    "boto3 is required for S3 ingestion. "
                    "Install it: pip install boto3"
                )
        return self._s3_client

    def _list_objects(self) -> Generator[dict, None, None]:
        """
        Paginated S3 ListObjectsV2 — handles buckets with > 1000 objects.
        Yields raw S3 object dicts: {Key, Size, LastModified, ...}
        """
        s3 = self._get_client()
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)

        for page in pages:
            for obj in page.get("Contents", []):
                yield obj

    def _download_to_temp(self, s3_key: str, filename: str) -> Path:
        """
        Download one S3 object to a temp file.
        Returns Path to the temp file.
        """
        s3 = self._get_client()
        tmp_path = Path(self.temp_dir) / f"hemmir_s3_{filename}"
        logger.debug(f"  S3 download: s3://{self.bucket}/{s3_key} → {tmp_path}")
        s3.download_file(self.bucket, s3_key, str(tmp_path))
        return tmp_path

    def iterate(
        self,
        extensions: Optional[List[str]] = None,
        skip_formats: Optional[List[FileFormat]] = None,
        max_files: Optional[int] = None,
    ) -> Generator[FileRecord, None, None]:
        """
        List S3 objects, download each matching file to a temp path,
        and yield FileRecord objects.

        The caller MUST call file_record.cleanup() after each file.
        Wrap in try/finally to guarantee cleanup on extraction errors.
        """
        allowed_exts = {e.lower() for e in extensions} if extensions else None
        count = 0

        for obj in self._list_objects():
            key  = obj["Key"]
            path = Path(key)

            if not self._should_include(path, allowed_exts, skip_formats):
                continue

            # Compute relative path from source prefix
            rel = key[len(self.prefix):].lstrip("/")
            if not rel:
                continue  # skip if key equals prefix (directory marker)

            folder_record = FolderRecord.from_path(
                self.source_record.source_id, rel
            )
            file_format = detect_format(path)

            # Download to temp
            try:
                safe_name = rel.replace("/", "_").replace("\\", "_")
                tmp_path  = self._download_to_temp(key, safe_name)
            except Exception as e:
                logger.error(f"  S3 download failed for {key}: {e}")
                continue

            size  = obj.get("Size", 0)
            mtime = obj.get("LastModified")
            mtime_str = mtime.isoformat() if mtime else None

            yield FileRecord(
                local_path      = tmp_path,
                relative_path   = rel,
                file_format     = file_format,
                source_record   = self.source_record,
                folder_record   = folder_record,
                is_temp         = True,
                s3_key          = key,
                file_size_bytes = size,
                last_modified   = mtime_str,
            )

            count += 1
            if max_files and count >= max_files:
                logger.info(f"  S3Loader: stopped at max_files={max_files}")
                break

    def count(self, extensions: Optional[List[str]] = None) -> int:
        allowed_exts = {e.lower() for e in extensions} if extensions else None
        return sum(
            1 for obj in self._list_objects()
            if self._should_include(Path(obj["Key"]), allowed_exts, None)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory — builds the right loader from a SourceRecord
# ─────────────────────────────────────────────────────────────────────────────

def build_loader(source_record: SourceRecord, **kwargs) -> BaseSourceLoader:
    """
    Factory function — returns the correct loader for a given SourceRecord.

    Args:
        source_record: SourceRecord describing the ingestion root.
        **kwargs:      Passed to the loader constructor (e.g. temp_dir for S3).

    Returns:
        LocalSourceLoader or S3SourceLoader instance.

    Raises:
        NotImplementedError if source_type is not yet supported.
    """
    if source_record.source_type == SourceType.LOCAL:
        return LocalSourceLoader(source_record)
    elif source_record.source_type == SourceType.S3:
        return S3SourceLoader(source_record, **kwargs)
    else:
        raise NotImplementedError(
            f"Source type '{source_record.source_type}' is not yet supported. "
            f"Implement a loader subclass for it."
        )
