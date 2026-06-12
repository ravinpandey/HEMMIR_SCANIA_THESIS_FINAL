"""
retrieval_layer/main.py

CLI Entry Point — Retrieval + Generation.

Usage:
  python retrieval_layer/main.py --query "How to replace the hydraulic pump seal?" \
    --output-dir ./output --chroma-dir ./chroma_db

  python retrieval_layer/main.py --query "Compare XC90 Gen1 vs Gen2 pump specs" \
    --output-dir ./output --chroma-dir ./chroma_db --provider anthropic

  python retrieval_layer/main.py --mode stats --chroma-dir ./chroma_db
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval_layer.retrieval_pipeline import RetrievalPipeline
from generation_layer.generation_pipeline import GenerationPipeline

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)


@click.command()
@click.option("--mode",      default="query",
              type=click.Choice(["query", "stats"]))
@click.option("--query",     default=None,          help="User query")
@click.option("--output-dir",  default="./output",   show_default=True)
@click.option("--chroma-dir",  default="./chroma_db", show_default=True)
@click.option("--provider",  default="anthropic",
              type=click.Choice(["anthropic", "bedrock"]))
@click.option("--region",    default="eu-west-1",   show_default=True)
@click.option("--top-k",     default=10,            show_default=True)
@click.option("--top-k-docs", default=5,            show_default=True)
@click.option("--tier1",     default=None)
@click.option("--tier2",     default=None)
@click.option("--project-id", default=None)
@click.option("--doc-type",  default=None)
@click.option("--no-generate", is_flag=True, default=False)
@click.option("--save-result", is_flag=True, default=False)
@click.option("--verbose",   is_flag=True, default=False)
def run(
    mode, query, output_dir, chroma_dir, provider, region,
    top_k, top_k_docs, tier1, tier2, project_id, doc_type,
    no_generate, save_result, verbose,
):
    """HEMMIR Retrieval + Generation — Explainable Multimodal RAG-Agent."""

    if verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG",
                   format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/retrieval.log", level="DEBUG", rotation="10 MB", enqueue=True)

    # ── Load ChromaDB ──────────────────────────────────────────────────
    from indexing_layer.utils.chroma_store import ChromaStore
    store = ChromaStore(persist_dir=chroma_dir)

    if mode == "stats":
        stats = store.get_collection_stats()
        logger.info("ChromaDB collection stats:")
        for name, count in stats.items():
            logger.info(f"  {name:<25} : {count:>8} records")
        return

    if not query:
        raise click.UsageError("--query is required for mode=query")

    # ── Build embedders ────────────────────────────────────────────────
    from embedding_layer.embedders.text_embedder import TextEmbedder
    text_embedder = TextEmbedder()

    image_embedder = None
    try:
        from embedding_layer.embedders.image_embedder import ImageEmbedder
        image_embedder = ImageEmbedder()
    except Exception as e:
        logger.warning(f"  CLIP image embedder unavailable: {e}")

    # ── Build LLM client ──────────────────────────────────────────────
    from enrichment_layer.utils.llm_client import build_llm_client
    llm_client = None
    if not no_generate:
        try:
            llm_client = build_llm_client(
                provider=provider,
                **({"region": region} if provider == "bedrock" else {})
            )
        except Exception as e:
            logger.warning(f"  LLM client unavailable: {e} — retrieval only")

    # ── Build filters ──────────────────────────────────────────────────
    filters = {k: v for k, v in {
        "tier1": tier1, "tier2": tier2,
        "project_id": project_id, "document_type": doc_type,
    }.items() if v}

    # ── Retrieval ──────────────────────────────────────────────────────
    retrieval_pipeline = RetrievalPipeline(
        store          = store,
        text_embedder  = text_embedder,
        image_embedder = image_embedder,
        llm_client     = llm_client,
        output_dir     = output_dir,
        top_k_docs     = top_k_docs,
        top_k_chunks   = top_k,
    )

    retrieval_result = retrieval_pipeline.retrieve(
        query   = query,
        filters = filters or None,
    )

    if no_generate:
        logger.info(f"Retrieval only: {len(retrieval_result['chunks'])} chunks")
        return

    # ── Generation ─────────────────────────────────────────────────────
    gen_pipeline = GenerationPipeline(
        registry             = retrieval_pipeline.registry,
        retrieval_pipeline   = retrieval_pipeline,
        llm_client           = llm_client,
        evidence_top_n       = top_k,
        max_repair_attempts  = 2,
    )

    answer = gen_pipeline.run(
        question        = query,
        analysis        = retrieval_result["analysis"],
        evidence_pack   = retrieval_result["evidence_pack"],
        retrieval_trace = retrieval_result["retrieval_trace"],
    )

    # ── Display results ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"PATH:  {answer.retrieval_path.upper()}")
    print(f"{'='*70}")
    print(f"\n{answer.answer}\n")
    print(f"{'─'*70}")
    print(f"CONFIDENCE: {answer.uncertainty.score:.3f} ({answer.uncertainty.level})")
    print(f"  {answer.uncertainty.level_explanation}")
    print(f"  → {answer.uncertainty.recommendation}")
    print(f"\nEVIDENCE TRAIL ({len(answer.attributed_claims)} attributed claims):")
    for i, claim in enumerate(answer.attributed_claims[:5], 1):
        direct = "✓ direct" if claim.is_direct else "~ indirect"
        print(
            f"  [{i}] {claim.claim[:70]}\n"
            f"       → {claim.modality} | {claim.location} | "
            f"role={claim.evidence_role} | sal={claim.salience_score:.2f} | {direct}"
        )
    if answer.unsupported_claims:
        print(f"\nUNSUPPORTED CLAIMS ({len(answer.unsupported_claims)}):")
        for uc in answer.unsupported_claims[:3]:
            print(f"  ✗ {uc.claim[:70]}")
    if answer.uncertainty.missing_evidence:
        print(f"\nMISSING EVIDENCE:")
        for gap in answer.uncertainty.missing_evidence[:3]:
            print(f"  • {gap}")
    if answer.retrieval_trace.path == "agent":
        print(f"\nRETRIEVAL TRACE (Agent path):")
        for i, sq in enumerate(answer.retrieval_trace.sub_question_results, 1):
            print(
                f"  [{i}] {sq.question[:60]}\n"
                f"       sections={len(sq.sections_navigated)} "
                f"iter={sq.iterations} suf={sq.sufficiency} "
                f"role_ok={not sq.role_mismatch}"
            )
    print(f"\nDuration: {answer.total_duration_ms:.0f}ms")
    print(f"{'='*70}\n")

    if follow_ups := answer.follow_up_questions:
        print("Follow-up questions:")
        for fq in follow_ups:
            print(f"  • {fq}")
        print()

    # ── Save result ────────────────────────────────────────────────────
    if save_result:
        result_path = Path("retrieval_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(
                answer.model_dump(),
                f, indent=2, default=str,
            )
        logger.info(f"Result saved to {result_path}")


if __name__ == "__main__":
    run()
