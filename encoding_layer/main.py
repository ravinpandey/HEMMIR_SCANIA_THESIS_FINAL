"""
encoding_layer/main.py

CLI Entry Point — Encoding Layer.

Must run AFTER cross_reference_layer (needs section_id, related_figures).
Must run BEFORE embedding_layer (embedding reads encoded_*_chunks.json).

Correct pipeline order:
  1. ingestion_layer
  2. enrichment_layer
  3. cross_reference_layer
  4. encoding_layer   ← this layer
  5. embedding_layer
  6. indexing_layer

Usage:
  python encoding_layer/main.py --output-dir ./output/
  python encoding_layer/main.py --output-dir ./output/ --doc-name my_doc
  python encoding_layer/main.py --output-dir ./output/ --dry-run
  python encoding_layer/main.py --output-dir ./output/ --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from encoding_layer.encoding_pipeline import EncodingPipeline

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)


@click.command()
@click.option("--output-dir", required=True, help="Output root from previous pipeline layers")
@click.option("--doc-name",   default=None,  help="Encode a specific document only")
@click.option("--dry-run",    is_flag=True,  default=False)
@click.option("--verbose",    is_flag=True,  default=False)
def run_encoding(output_dir, doc_name, dry_run, verbose):
    """Encoding Layer — builds retrieval views, linkage, and promoted_fields."""

    if verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG",
                   format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/encoding.log", level="DEBUG", rotation="10 MB", enqueue=True)

    logger.info("=" * 60)
    logger.info("  HEMMIR Encoding Layer")
    logger.info(f"  Dry run : {dry_run}")
    logger.info("=" * 60)

    output_path = Path(output_dir)
    if doc_name:
        doc_dirs = [output_path / doc_name]
    else:
        doc_dirs = sorted([
            d for d in output_path.iterdir()
            if d.is_dir() and (d / "metadata").exists()
        ])

    if not doc_dirs:
        logger.warning("No document folders found.")
        return

    logger.info(f"  Found {len(doc_dirs)} document(s)")

    pipeline  = EncodingPipeline(dry_run=dry_run)
    summaries = []
    failed    = 0

    for doc_dir in tqdm(doc_dirs, desc="Encoding"):
        try:
            s = pipeline.process_document(doc_dir)
            if s:
                summaries.append(s)
        except Exception as e:
            failed += 1
            logger.error(f"Failed: {doc_dir.name} → {e}")

    logger.info("=" * 60)
    for s in summaries:
        logger.info(
            f"  ✓ {s.get('doc_name',''):<30} [{s.get('format','')}] | "
            f"text={s.get('text_encoded',0)} "
            f"img={s.get('img_encoded',0)} "
            f"tbl={s.get('tbl_encoded',0)}"
        )
    if failed:
        logger.error(f"  Failed: {failed} document(s)")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_encoding()
