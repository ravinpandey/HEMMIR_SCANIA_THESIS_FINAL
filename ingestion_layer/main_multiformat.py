"""
ingestion_layer/main_multiformat.py

HEMMIR Multi-Format Ingestion Orchestrator.

Replaces the PDF-only main.py. Backward compatible:
    python main_multiformat.py --data-dir ./pdfs/ --output-dir ./output/
    works identically to the old main.py for PDF-only directories.

New capabilities:
    --source-type local|s3
    --formats     pdf,docx,pptx,csv,xlsx,image,video
    --skip-formats video
    --row-group-size 100
    --whisper-model base

Usage:
    # Local multi-format
    python main_multiformat.py --data-dir ./docs/ --output-dir ./output/

    # PDF + DOCX only
    python main_multiformat.py --data-dir ./docs/ --output-dir ./output/ \
        --formats pdf,docx

    # S3 source
    python main_multiformat.py \
        --source-type s3 \
        --bucket-name corp-knowledge \
        --s3-prefix projects/alpha/ \
        --output-dir ./output/ \
        --formats pdf,docx,xlsx

    # Dry run
    python main_multiformat.py --data-dir ./docs/ --output-dir ./output/ --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import click
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion_layer.extractors.file_router import FileRouter
from ingestion_layer.extractors.base_extractor import (
    PDFExtractorAdapter,
    DocxExtractor,
    PptxExtractor,
    CsvExtractor,
    XlsxExtractor,
    ImageExtractor,
    VideoExtractor,
    UnknownExtractor,
)
from ingestion_layer.loaders.source_loaders import (
    SourceRecord,
    SourceType,
    FileFormat,
    build_loader,
    ALL_SUPPORTED_EXTENSIONS,
)
from ingestion_layer.utils.metadata_normalizer import MetadataNormalizer

# ── Extractor registry ─────────────────────────────────────────────────────────
FileRouter.register(FileFormat.PDF,     PDFExtractorAdapter)
FileRouter.register(FileFormat.DOCX,    DocxExtractor)
FileRouter.register(FileFormat.PPTX,    PptxExtractor)
FileRouter.register(FileFormat.CSV,     CsvExtractor)
FileRouter.register(FileFormat.XLSX,    XlsxExtractor)
FileRouter.register(FileFormat.IMAGE,   ImageExtractor)
FileRouter.register(FileFormat.VIDEO,   VideoExtractor)
FileRouter.register(FileFormat.UNKNOWN, UnknownExtractor)

logger.remove()
logger.add(
    sys.stdout, level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)

# ── Format helpers ─────────────────────────────────────────────────────────────

_FORMAT_NAME_MAP = {
    "pdf":   FileFormat.PDF,
    "docx":  FileFormat.DOCX,  "doc":  FileFormat.DOCX,
    "pptx":  FileFormat.PPTX,  "ppt":  FileFormat.PPTX,
    "csv":   FileFormat.CSV,   "tsv":  FileFormat.CSV,
    "xlsx":  FileFormat.XLSX,  "xls":  FileFormat.XLSX,
    "image": FileFormat.IMAGE, "img":  FileFormat.IMAGE,
    "video": FileFormat.VIDEO, "vid":  FileFormat.VIDEO, "audio": FileFormat.VIDEO,
}

_FORMAT_TO_EXTENSIONS = {
    FileFormat.PDF:   [".pdf"],
    FileFormat.DOCX:  [".docx", ".doc"],
    FileFormat.PPTX:  [".pptx", ".ppt"],
    FileFormat.CSV:   [".csv", ".tsv"],
    FileFormat.XLSX:  [".xlsx", ".xlsm", ".xls"],
    FileFormat.IMAGE: [".jpg", ".jpeg", ".png", ".gif", ".webp", ".tiff", ".bmp"],
    FileFormat.VIDEO: [".mp4", ".avi", ".mov", ".mkv", ".webm", ".mp3", ".wav", ".m4a"],
}


def _parse_formats(formats_str: Optional[str]) -> Optional[List[str]]:
    if not formats_str:
        return None
    extensions = []
    for name in formats_str.split(","):
        name = name.strip().lower().lstrip(".")
        fmt  = _FORMAT_NAME_MAP.get(name)
        if fmt and fmt in _FORMAT_TO_EXTENSIONS:
            extensions.extend(_FORMAT_TO_EXTENSIONS[fmt])
        else:
            logger.warning(f"Unknown format '{name}' — ignoring")
    return extensions or None


def _parse_skip_formats(skip_str: Optional[str]) -> Optional[List[FileFormat]]:
    if not skip_str:
        return None
    result = []
    for name in skip_str.split(","):
        fmt = _FORMAT_NAME_MAP.get(name.strip().lower())
        if fmt:
            result.append(fmt)
    return result or None


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
# Source
@click.option("--data-dir",     default=None)
@click.option("--source-type",  default="local",
              type=click.Choice(["local", "s3"]))
@click.option("--bucket-name",  default=None)
@click.option("--s3-prefix",    default="")
@click.option("--source-id",    default=None)
@click.option("--source-name",  default=None)
@click.option("--aws-region",   default="eu-west-1", show_default=True)
# Output
@click.option("--output-dir",   default="./output",  show_default=True)
# File selection
@click.option("--formats",      default=None,
              help="Comma-separated whitelist: pdf,docx,pptx,csv,xlsx,image,video")
@click.option("--skip-formats", default=None,
              help="Comma-separated formats to skip: video,image")
@click.option("--max-files",    default=None, type=int)
# Metadata
@click.option("--project-id",    default=None)
@click.option("--tier1",         default=None)
@click.option("--tier2",         default=None)
@click.option("--document-type", default=None)
@click.option("--language",      default="English", show_default=True)
@click.option("--doc-name",      default=None)
# Chunking
@click.option("--chunk-strategy",
              type=click.Choice(
                  ["paragraph", "title", "size", "heading", "slide", "row_group"]
              ),
              default="paragraph", show_default=True)
@click.option("--chunk-size",    default=512,  show_default=True)
@click.option("--chunk-overlap", default=150,  show_default=True)
@click.option("--row-group-size", default=100, show_default=True,
              help="Rows per chunk for CSV/XLSX")
@click.option("--whisper-model",  default="base", show_default=True,
              help="Whisper model for video transcription")
# Flags
@click.option("--skip-images",    is_flag=True, default=False)
@click.option("--skip-tables",    is_flag=True, default=False)
@click.option("--skip-normalize", is_flag=True, default=False,
              help="Skip metadata normalization pass")
@click.option("--dry-run",        is_flag=True, default=False)
@click.option("--verbose",        is_flag=True, default=False)
def run_ingestion(
    data_dir, source_type, bucket_name, s3_prefix, source_id, source_name,
    aws_region, output_dir, formats, skip_formats, max_files,
    project_id, tier1, tier2, document_type, language, doc_name,
    chunk_strategy, chunk_size, chunk_overlap, row_group_size, whisper_model,
    skip_images, skip_tables, skip_normalize, dry_run, verbose,
):
    """HEMMIR Multi-Format Ingestion — PDF, DOCX, PPTX, CSV, XLSX, Image, Video."""

    if verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG",
                   format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/ingestion.log", level="DEBUG", rotation="10 MB", enqueue=True)

    logger.info("=" * 65)
    logger.info("  HEMMIR Multi-Format Ingestion Layer")
    logger.info("=" * 65)

    # Build SourceRecord
    if source_type == "s3":
        if not bucket_name:
            logger.error("--bucket-name required for --source-type s3")
            sys.exit(1)
        source = SourceRecord.from_s3(
            bucket      = bucket_name,
            prefix      = s3_prefix,
            source_name = source_name or f"s3://{bucket_name}/{s3_prefix}",
            source_id   = source_id,
            region      = aws_region,
        )
    else:
        if not data_dir:
            logger.error("--data-dir required for --source-type local")
            sys.exit(1)
        source = SourceRecord.from_local(
            root        = data_dir,
            source_name = source_name or str(data_dir),
            source_id   = source_id,
        )

    logger.info(f"  Source     : {source.source_type.value} | {source.source_name}")
    logger.info(f"  Source ID  : {source.source_id}")
    logger.info(f"  Output Dir : {output_dir}")
    logger.info(f"  Formats    : {formats or 'all'}")
    logger.info(f"  Skip       : {skip_formats or 'none'}")
    logger.info(f"  Strategy   : {chunk_strategy}")
    logger.info(f"  Dry Run    : {dry_run}")
    logger.info("=" * 65)

    loader     = build_loader(source)
    router     = FileRouter(
        output_dir     = Path(output_dir),
        chunk_strategy = chunk_strategy,
        chunk_size     = chunk_size,
        chunk_overlap  = chunk_overlap,
        row_group_size = row_group_size,
        whisper_model  = whisper_model,
        skip_images    = skip_images,
        skip_tables    = skip_tables,
        project_id     = project_id,
        tier1          = tier1,
        tier2          = tier2,
        document_type  = document_type,
        language       = language,
        doc_name       = doc_name,
    )
    normalizer = MetadataNormalizer()

    allowed_exts   = _parse_formats(formats)
    skip_fmt_list  = _parse_skip_formats(skip_formats)

    logger.info("Scanning for files...")
    total = loader.count(extensions=allowed_exts)
    logger.info(f"  Found: {total} file(s)")
    if total == 0:
        logger.warning("No files found.")
        return

    if dry_run:
        logger.info("\n[DRY RUN] Files that would be ingested:")
        for fr in loader.iterate(
            extensions=allowed_exts, skip_formats=skip_fmt_list, max_files=max_files
        ):
            logger.info(f"  {fr.file_format.value:8s}  {fr.relative_path}")
            fr.cleanup()
        return

    stats = {
        "total": 0, "succeeded": 0, "failed": 0, "skipped": 0, "by_format": {}
    }

    for file_record in tqdm(
        loader.iterate(
            extensions=allowed_exts, skip_formats=skip_fmt_list, max_files=max_files
        ),
        desc="Ingesting", unit="file",
    ):
        stats["total"] += 1
        fmt_name = file_record.file_format.value
        try:
            if not router.can_handle(file_record):
                logger.warning(f"No extractor: {file_record.relative_path!r}")
                stats["skipped"] += 1
                continue

            result = router.route(file_record)

            if not skip_normalize:
                doc_dir = Path(output_dir) / file_record.doc_stem
                normalizer.normalize(doc_dir)

            stats["succeeded"] += 1
            stats["by_format"][fmt_name] = stats["by_format"].get(fmt_name, 0) + 1
            logger.success(
                f"✓ {file_record.relative_path!r} → "
                f"text={result.text_count} img={result.image_count} "
                f"tbl={result.table_count}"
            )
        except Exception as e:
            stats["failed"] += 1
            logger.error(f"✗ {file_record.relative_path!r} → {type(e).__name__}: {e}")
        finally:
            file_record.cleanup()

    logger.info("=" * 65)
    logger.info("  Ingestion Complete")
    logger.info(f"  Succeeded : {stats['succeeded']}")
    logger.info(f"  Failed    : {stats['failed']}")
    logger.info(f"  Skipped   : {stats['skipped']}")
    if stats["by_format"]:
        logger.info("  By format:")
        for fmt, count in sorted(stats["by_format"].items()):
            logger.info(f"    {fmt:10s} : {count}")
    logger.info("=" * 65)

    if stats["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    run_ingestion()
