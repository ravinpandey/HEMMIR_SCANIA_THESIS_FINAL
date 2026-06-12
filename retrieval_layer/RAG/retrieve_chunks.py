"""
retrieval_layer/RAG/retrieve_chunks.py

Chunk-level retrieval for the RAG path.

Flow:
  1. For each modality in analysis.modalities:
     a. If HyDE enabled: embed hypothetical passage instead of raw query
     b. If multi-query enabled: expand to N variants, retrieve for each, RRF merge
     c. Otherwise: direct embed + retrieve
  2. Set QueryContext on plugin before retrieval
  3. Set section_ids on plugin for Stage 2 section filtering
  4. Collect all chunks across modalities
  5. Return for re-ranking in retrieval_pipeline.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models.pipeline_models import (
    QueryAnalysis,
    QueryContext,
    RetrievedChunk,
)


def retrieve_chunks(
    store,
    registry:        Dict[str, Any],
    analysis:        QueryAnalysis,
    context:         QueryContext,
    doc_ids:         List[str],
    top_k:           int,
    hyde=            None,
    multi_expander=  None,
    section_ids=     None,
) -> List[RetrievedChunk]:
    """
    Retrieve chunks for all modalities in the analysis.

    Args:
        store:          ChromaStore
        registry:       Plugin registry (modality_name → plugin)
        analysis:       QueryAnalysis with modalities, use_hyde flags
        context:        QueryContext with query_text, image_b64 etc.
        doc_ids:        From document-level retrieval (Stage 1 scope limiter)
        top_k:          Chunks per modality
        hyde:           HyDE instance (optional)
        multi_expander: MultiQueryExpander instance (optional)
        section_ids:    Section IDs from Stage 2 retrieval (optional scope limiter)

    Returns:
        Flat list of RetrievedChunk across all modalities (not yet re-ranked)
    """
    all_chunks: List[RetrievedChunk] = []

    for modality_name in analysis.modalities:
        plugin = registry.get(modality_name)
        if plugin is None:
            logger.warning(f"  retrieve_chunks: plugin not found: {modality_name}")
            continue

        plugin.set_query_context(context)

        # Set section_ids on plugin — used by _query_collection in base.py
        plugin._section_ids = section_ids or []

        try:
            chunks = _retrieve_with_strategy(
                plugin         = plugin,
                analysis       = analysis,
                context        = context,
                doc_ids        = doc_ids,
                top_k          = top_k,
                hyde           = hyde,
                multi_expander = multi_expander,
            )
            logger.info(f"  retrieve_chunks [{modality_name}]: {len(chunks)} chunks")
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"  retrieve_chunks [{modality_name}] failed: {e}")

    return all_chunks


def _retrieve_with_strategy(
    plugin,
    analysis:       QueryAnalysis,
    context:        QueryContext,
    doc_ids:        List[str],
    top_k:          int,
    hyde=           None,
    multi_expander= None,
) -> List[RetrievedChunk]:
    """Select retrieval strategy per plugin."""
    query_text = context.query_text or ""

    # ── HyDE strategy ─────────────────────────────────────────────────
    if analysis.use_hyde and hyde is not None:
        try:
            hyde_vec, hyde_text, domain_spec = hyde.embed_query(
                question     = query_text,
                n_hypotheses = 1,
            )
            logger.debug(
                f"  HyDE: domain_spec={domain_spec:.2f} | "
                f"passage[:60]={hyde_text[:60]}"
            )
            return plugin.retrieve(plugin_store(plugin, hyde_vec), hyde_vec, doc_ids, top_k)
        except Exception as e:
            logger.warning(f"  HyDE failed: {e} — falling back to direct embed")

    # ── Multi-query strategy (text_to_text and video only) ────────────
    if (
        multi_expander is not None
        and not analysis.use_hyde
        and plugin.modality_name in ("text_to_text", "video_segment")
        and len(query_text.split()) >= 6
    ):
        try:
            variants = multi_expander.expand_query(query_text, n=3)
            if len(variants) > 1:
                ranked_lists: List[List[Dict]] = []
                for variant in variants:
                    vec    = plugin.embed_query(variant)
                    chunks = plugin.retrieve(None, vec, doc_ids, top_k)
                    ranked_lists.append([
                        {"chunk_id": c.chunk_id, "_chunk": c}
                        for c in chunks
                    ])
                fused = multi_expander.reciprocal_rank_fusion(ranked_lists, top_n=top_k)
                logger.debug(f"  Multi-query: {len(variants)} variants, {len(fused)} fused")
                return [r["_chunk"] for r in fused if "_chunk" in r]
        except Exception as e:
            logger.warning(f"  Multi-query failed: {e} — falling back")

    # ── Direct embed ───────────────────────────────────────────────────
    query_vec = plugin.embed_query(query_text)
    return plugin.retrieve(None, query_vec, doc_ids, top_k)


def plugin_store(plugin, vec):
    """Passthrough — plugins access store through their internal db reference."""
    return None