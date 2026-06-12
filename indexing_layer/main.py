"""
indexing_layer/main.py

CLI Entry Point — Indexing Layer.

Modes:
  index  — load processed JSON output into ChromaDB
  stats  — show ChromaDB collection record counts

Usage:
  python indexing_layer/main.py --mode index --output-dir ./output --chroma-dir ./chroma_db
  python indexing_layer/main.py --mode index --output-dir ./output --doc-name my_doc
  python indexing_layer/main.py --mode index --output-dir ./output --dry-run
  python indexing_layer/main.py --mode stats --chroma-dir ./chroma_db
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from indexing_layer.utils.chroma_store  import ChromaStore
from indexing_layer.indexing_pipeline   import IndexingPipeline

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)


@click.command()
@click.option("--mode",       required=True,
              type=click.Choice(["index", "stats"]),
              help="index=upsert to ChromaDB | stats=show collection counts")
@click.option("--output-dir", default="./output",    show_default=True)
@click.option("--chroma-dir", default="./chroma_db", show_default=True)
@click.option("--doc-name",   default=None)
@click.option("--dry-run",    is_flag=True, default=False)
@click.option("--verbose",    is_flag=True, default=False)
def run_indexing(mode, output_dir, chroma_dir, doc_name, dry_run, verbose):
    """Indexing Layer — store document embeddings into ChromaDB."""

    if verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG",
                   format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/indexing.log", level="DEBUG", rotation="10 MB", enqueue=True)

    logger.info("=" * 60)
    logger.info(f"  HEMMIR Indexing Layer — mode={mode}")
    logger.info("=" * 60)

    store = ChromaStore(persist_dir=chroma_dir)

    # Build one shared TextEmbedder and pass it into ChromaStore so it is
    # never re-instantiated per section call. Only needed for index mode.
    text_embedder = None
    if mode == "index":
        try:
            from embedding_layer.embedders.text_embedder import TextEmbedder
            text_embedder = TextEmbedder()
            logger.info("  Shared TextEmbedder ready")
        except Exception as e:
            logger.warning(
                f"  Could not pre-build TextEmbedder: {e} — will lazy-init if needed"
            )

    indexer = IndexingPipeline(store, output_dir=output_dir, text_embedder=text_embedder)

    if mode == "index":
        stats = indexer.index_all(doc_name=doc_name, dry_run=dry_run)
        logger.info(
            f"  Indexing complete — "
            f"succeeded={stats['succeeded']} | "
            f"failed={stats['failed']} | "
            f"skipped={stats['skipped']}"
        )
        col_stats = indexer.get_stats()
        logger.info("  ChromaDB after indexing:")
        for name, count in col_stats.items():
            logger.info(f"    {name:<25} : {count:>8} records")

    elif mode == "stats":
        col_stats = indexer.get_stats()
        logger.info("  ChromaDB collection stats:")
        for name, count in col_stats.items():
            logger.info(f"    {name:<25} : {count:>8} records")


if __name__ == "__main__":
    run_indexing()
