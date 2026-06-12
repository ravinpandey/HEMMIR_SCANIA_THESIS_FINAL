"""
enrichment_layer/main.py

CLI Entry Point — Enrichment Layer.

Usage:
  python enrichment_layer/main.py --output-dir ./output/
  python enrichment_layer/main.py --output-dir ./output/ --doc-name my_manual
  python enrichment_layer/main.py --output-dir ./output/ --provider bedrock --region eu-west-1
  python enrichment_layer/main.py --output-dir ./output/ --skip-images --skip-tables
  python enrichment_layer/main.py --output-dir ./output/ --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from enrichment_layer.enrichment_pipeline import EnrichmentPipeline
from enrichment_layer.utils.llm_client import build_llm_client

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)


@click.command()
@click.option("--output-dir",    required=True,  help="Pipeline output directory")
@click.option("--doc-name",      default=None,   help="Enrich only this document subfolder")
@click.option("--provider",      default="anthropic",
              type=click.Choice(["anthropic", "bedrock"]),
              help="LLM provider")
@click.option("--region",        default="eu-west-1", show_default=True,
              help="AWS region (bedrock only)")
@click.option("--text-model",    default=None,   help="Override text model ID")
@click.option("--vision-model",  default=None,   help="Override vision model ID")
@click.option("--skip-images",   is_flag=True, default=False)
@click.option("--skip-tables",   is_flag=True, default=False)
@click.option("--skip-video-frames", is_flag=True, default=False)
@click.option("--dry-run",       is_flag=True, default=False)
@click.option("--verbose",       is_flag=True, default=False)
def run_enrichment(
    output_dir, doc_name, provider, region, text_model, vision_model,
    skip_images, skip_tables, skip_video_frames, dry_run, verbose,
):
    """LLM Enrichment Layer — contextual chunking, section summaries, image captions."""

    if verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG",
                   format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/enrichment.log", level="DEBUG", rotation="10 MB", enqueue=True)

    logger.info("=" * 60)
    logger.info("  HEMMIR Enrichment Layer")
    logger.info(f"  Provider : {provider}")
    logger.info(f"  Dry run  : {dry_run}")
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

    # Build LLM client
    client_kwargs = {}
    if text_model:
        client_kwargs["text_model"] = text_model
    if vision_model:
        client_kwargs["vision_model"] = vision_model
    if provider == "bedrock":
        client_kwargs["region"] = region

    llm = build_llm_client(provider=provider, **client_kwargs)

    pipeline = EnrichmentPipeline(
        llm               = llm,
        skip_images       = skip_images,
        skip_tables       = skip_tables,
        skip_video_frames = skip_video_frames,
        dry_run           = dry_run,
    )

    summaries = []
    failed    = 0

    for doc_dir in tqdm(doc_dirs, desc="Enriching"):
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
            f"text={s.get('text_enriched',0)} "
            f"img={s.get('image_enriched',0)} "
            f"tbl={s.get('table_enriched',0)} "
            f"seg={s.get('seg_enriched',0)} "
            f"frames={s.get('frame_enriched',0)}"
        )
    if failed:
        logger.error(f"  Failed: {failed} document(s)")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_enrichment()
