"""
cross_reference_layer/main.py

CLI Entry Point — Cross-Reference Layer.

Correct pipeline order:
  1. ingestion_layer
  2. enrichment_layer       ← must run before this (needs section_map page ranges)
  3. cross_reference_layer  ← this layer
  4. encoding_layer         ← must run after this (reads related_figures, section_id)
  5. embedding_layer
  6. indexing_layer

Usage:
  python cross_reference_layer/main.py --output-dir ./output/
  python cross_reference_layer/main.py --output-dir ./output/ --doc-name my_doc
  python cross_reference_layer/main.py --output-dir ./output/ --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from cross_reference_layer.cross_referencer import CrossReferencer

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)


@click.command()
@click.option("--output-dir", required=True)
@click.option("--doc-name",   default=None,       help="Process only this document")
@click.option("--dry-run",    is_flag=True, default=False)
@click.option("--verbose",    is_flag=True, default=False)
def run_cross_reference(output_dir, doc_name, dry_run, verbose):
    """Cross-Reference Layer — links text ↔ images ↔ tables, assigns section_id."""

    if verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG",
                   format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/cross_reference.log", level="DEBUG", rotation="10 MB", enqueue=True)

    logger.info("=" * 60)
    logger.info("  HEMMIR Cross-Reference Layer")
    logger.info(f"  Dry run: {dry_run}")
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

    referencer = CrossReferencer(dry_run=dry_run)
    summaries  = []
    failed     = 0

    for doc_dir in tqdm(doc_dirs, desc="Cross-referencing"):
        try:
            s = referencer.process_document(doc_dir)
            if s:
                summaries.append(s)
        except Exception as e:
            failed += 1
            logger.error(f"Failed: {doc_dir.name} → {e}")

    logger.info("=" * 60)
    for s in summaries:
        doc = s.get("doc_name", "")
        fig = s.get("figure_links", 0)
        tbl = s.get("table_links", 0)
        sid = s.get("section_ids", 0)
        scn = s.get("scene_links", 0)
        logger.info(
            f"  ✓ {doc:<30} | fig={fig} tbl={tbl} "
            f"section_ids={sid} scene_links={scn}"
        )
    if failed:
        logger.error(f"  Failed: {failed} document(s)")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_cross_reference()
