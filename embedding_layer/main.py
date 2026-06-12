"""
embedding_layer/main.py

CLI Entry Point — Embedding Layer.

Usage:
  python embedding_layer/main.py --output-dir ./output/
  python embedding_layer/main.py --output-dir ./output/ --doc-name my_doc
  python embedding_layer/main.py --output-dir ./output/ --text-model text-embedding-3-small
  python embedding_layer/main.py --output-dir ./output/ --clip-model ViT-B-32
  python embedding_layer/main.py --output-dir ./output/ --skip-images --skip-tables
  python embedding_layer/main.py --output-dir ./output/ --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from embedding_layer.embedding_pipeline import EmbeddingPipeline

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)


@click.command()
@click.option("--output-dir",         required=True)
@click.option("--doc-name",           default=None)
@click.option("--text-model",         default="text-embedding-3-small", show_default=True)
@click.option("--clip-model",         default="ViT-B-32",               show_default=True)
@click.option("--clip-pretrained",    default="openai",                  show_default=True)
@click.option("--skip-images",        is_flag=True, default=False)
@click.option("--skip-tables",        is_flag=True, default=False)
@click.option("--skip-video-frames",  is_flag=True, default=False)
@click.option("--dry-run",            is_flag=True, default=False)
@click.option("--verbose",            is_flag=True, default=False)
def run_embedding(
    output_dir, doc_name, text_model, clip_model, clip_pretrained,
    skip_images, skip_tables, skip_video_frames, dry_run, verbose,
):
    """Embedding Layer — text (1536-dim) + CLIP visual (512-dim) vectors."""

    if verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG",
                   format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/embedding.log", level="DEBUG", rotation="10 MB", enqueue=True)

    logger.info("=" * 60)
    logger.info("  HEMMIR Embedding Layer")
    logger.info(f"  Text model  : {text_model}")
    logger.info(f"  CLIP model  : {clip_model} ({clip_pretrained})")
    logger.info(f"  Dry run     : {dry_run}")
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

    pipeline = EmbeddingPipeline(
        text_model        = text_model,
        clip_model        = clip_model,
        clip_pretrained   = clip_pretrained,
        skip_images       = skip_images,
        skip_tables       = skip_tables,
        skip_video_frames = skip_video_frames,
        dry_run           = dry_run,
    )

    summaries = []
    failed    = 0

    for doc_dir in tqdm(doc_dirs, desc="Embedding"):
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
            f"  ✓ {s.get('doc_name',''):<28} [{s.get('format','')}] | "
            f"text={s.get('text_embedded',0)} "
            f"img={s.get('img_embedded',0)} "
            f"tbl={s.get('tbl_embedded',0)} "
            f"seg={s.get('seg_embedded',0)} "
            f"frames={s.get('frame_embedded',0)}"
        )
    if failed:
        logger.error(f"  Failed: {failed} document(s)")
    logger.info("=" * 60)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run_embedding()
