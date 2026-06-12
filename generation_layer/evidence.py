"""
generation_layer/evidence.py + prompt_builder.py combined

evidence.py: builds final EvidencePack from ranked chunks
prompt_builder.py: formats evidence blocks for LLM generation prompt
"""

from __future__ import annotations

from typing import Any, Dict, List

from shared.models.pipeline_models import (
    EvidenceItem,
    EvidencePack,
    Filters,
    RetrievedChunk,
)


# ── Evidence pack builder ─────────────────────────────────────────────────────

def build_evidence_pack(
    ranked_chunks: List[RetrievedChunk],
    registry:      Dict[str, Any],
    top_n:         int,
) -> EvidencePack:
    """
    Deduplicate ranked chunks and build final EvidencePack.

    Deduplication: keep highest-scoring version of each chunk_id.
    Builds EvidenceItem for each via the relevant modality plugin.
    """
    deduped: Dict[str, RetrievedChunk] = {}
    for chunk in ranked_chunks:
        cid      = chunk.chunk_id
        existing = deduped.get(cid)
        if existing is None or chunk.score_breakdown.final_score > existing.score_breakdown.final_score:
            deduped[cid] = chunk

    selected = list(deduped.values())[:top_n]
    items:   List[EvidenceItem] = []

    for chunk in selected:
        plugin = registry.get(chunk.plugin_name)
        if plugin is None:
            # Fallback: build EvidenceItem without plugin
            items.append(_build_fallback_item(chunk))
            continue
        try:
            items.append(plugin.build_evidence_item(chunk))
        except Exception:
            items.append(_build_fallback_item(chunk))

    return EvidencePack(
        items         = items,
        total_items   = len(items),
        was_truncated = len(deduped) > top_n,
    )


def _build_fallback_item(chunk: RetrievedChunk) -> EvidenceItem:
    meta = chunk.metadata
    # Prefer chunk.content; fall back to metadata text fields
    content = chunk.content or (
        getattr(meta, "text_original_content", None)
        or getattr(meta, "transcript_text", None)
        or getattr(meta, "table_summary", None)
        or getattr(meta, "image_caption", None)
        or getattr(meta, "contextual_summary", None)
        or ""
    )
    return EvidenceItem(
        chunk_id         = meta.chunk_id,
        doc_id           = meta.doc_id,
        chunk_index      = meta.chunk_index,
        source_modality  = meta.source_modality,
        score            = chunk.score_breakdown.final_score,
        score_breakdown  = chunk.score_breakdown,
        metadata         = meta,
        content          = content,
        retrieval_plugin = chunk.plugin_name,
        retrieval_mode   = chunk.retrieval_mode,
        collection_name  = chunk.collection_name,
        extra_payload    = chunk.extra_payload,
    )


# ── Prompt builder ────────────────────────────────────────────────────────────

# Maps source_modality → plugin name for agent-retrieved chunks
_MODALITY_PLUGIN_MAP = {
    "text":          "text_to_text",
    "image":         "text_to_image",
    "table":         "text_to_table",
    "video_segment": "video_segment",
    "video_frame":   "video_frame",
}


def build_evidence_blocks(
    evidence_pack: EvidencePack,
    registry:      Dict[str, Any],
) -> str:
    """
    Format all evidence items into text blocks for the generation prompt.
    Each plugin's format_prompt_block produces modality-appropriate formatting.
    Agent-retrieved chunks (plugin_name='agent') are routed by source_modality.
    """
    blocks = []
    for item in evidence_pack.items:
        plugin = registry.get(item.retrieval_plugin)
        if plugin is None:
            # Route agent-retrieved chunks to the right plugin by modality
            plugin = registry.get(_MODALITY_PLUGIN_MAP.get(item.source_modality, ""))
        if plugin is not None:
            try:
                blocks.append(plugin.format_prompt_block(item))
                continue
            except Exception:
                pass
        # Fallback formatting — extract richest available content
        content = _extract_item_content(item)
        meta    = item.metadata
        summary = getattr(meta, "contextual_summary", "") or ""
        section = (
            getattr(meta, "section_title", "")
            or getattr(meta, "slide_title", "")
            or ""
        )
        block = (
            f"[{item.source_modality.upper()} | chunk_id: {item.chunk_id} "
            f"| doc_id: {item.doc_id}]\n"
            f"Content: {content}"
        )
        if summary:
            block += f"\nSummary: {summary}"
        if section:
            block += f"\nSection: {section}"
        blocks.append(block)

    return "\n\n".join(blocks) if blocks else "No evidence retrieved."


def _extract_item_content(item) -> str:
    """Extract richest available text content from an EvidenceItem."""
    if item.content:
        return item.content
    meta = item.metadata
    return (
        getattr(meta, "text_original_content", None)
        or getattr(meta, "transcript_text", None)
        or getattr(meta, "table_summary", None)
        or getattr(meta, "image_caption", None)
        or getattr(meta, "contextual_summary", None)
        or ""
    )
