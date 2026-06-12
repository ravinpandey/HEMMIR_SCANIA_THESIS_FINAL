"""
generation_layer/fuse.py

Enrichment-boosted chunk fusion and deduplication.

Upgraded from V4:
  - Adds salience_boost and evidence_role_boost to ScoreBreakdown
  - Uses enrichment signals (salience_score, evidence_role) from metadata
  - Role compatibility map: query_intent_type → expected evidence_role(s)
  - Final score formula is the retrieval contribution for the thesis

Changelog:
  - Added table_boost: for comparative and measurement queries, table chunks
    receive a +0.20 score boost before the cross-encoder reranker runs.
  - Salience fallback: when salience_score is None (table chunks), falls back
    to table_summary_confidence from extra_payload so table chunks get a
    meaningful salience signal (0.88-0.98) instead of zero boost.

This function runs AFTER plugin retrieval and BEFORE the cross-encoder
reranker (in retrieval_pipeline). The reranker then further refines using
cross-encoder scores. Both steps use enrichment signals.
"""

from __future__ import annotations

import re
from typing import List

from shared.models.pipeline_models import BoostSignals, RetrievedChunk, ScoreBreakdown

_NOISE_PATTERNS = (
    "hereby grants permission",
    "google hereby",
    "proper attribution",
    "in proceedings of",
    "arxiv preprint",
    "all rights reserved",
    "copyright notice",
)

_ROLE_COMPAT = {
    "procedural":  ["procedure", "procedural"],
    "measurement": ["measurement"],
    "definition":  ["definition"],
    "comparative": ["procedure", "measurement", "definition"],
    "diagnostic":  ["measurement", "procedure"],
    "exploratory": ["definition", "context", "procedure"],
    "visual":      ["context", "definition"],
    "temporal":    ["context", "procedure"],
}

ROLE_BOOST   = 0.12
SALIENCE_W   = 0.10
_TABLE_BOOST_INTENTS = {"comparative", "measurement"}
TABLE_BOOST = 0.20


def fuse_chunks(
    chunks:           List[RetrievedChunk],
    boost_signals:    BoostSignals,
    query_intent_type: str = "definition",
) -> List[RetrievedChunk]:
    exact_codes      = [c.lower() for c in boost_signals.exact_code_match]
    preferred_type   = (boost_signals.document_type_preference or "").lower()
    preferred_modal  = (boost_signals.modality_preference or "").lower()
    compatible_roles = _ROLE_COMPAT.get(query_intent_type, [])
    apply_table_boost = query_intent_type in _TABLE_BOOST_INTENTS

    scored: List[RetrievedChunk] = []
    for chunk in chunks:
        vector_score = chunk.score_breakdown.vector_score
        content      = _searchable_text(chunk).lower()
        meta         = chunk.metadata

        code_boost  = 1.0 if exact_codes and any(c in content for c in exact_codes) else 0.0
        chunk_type  = str(chunk.extra_payload.get("document_type") or "").lower()
        type_boost  = 1.0 if preferred_type and chunk_type == preferred_type else 0.0
        modal_boost = 1.0 if preferred_modal and chunk.source_modality == preferred_modal else 0.0

        # Salience — text chunks have salience_score from TextEnricher.
        # Table chunks fall back to table_summary_confidence from extra_payload.
        salience = getattr(meta, "salience_score", None)
        if salience is None or salience == 0.0:
            _tsc = chunk.extra_payload.get("table_summary_confidence")
            if _tsc is not None:
                salience = _tsc
            elif chunk.source_modality == 'image':
                salience = float(chunk.extra_payload.get('contextual_summary_confidence') or chunk.extra_payload.get('image_caption_confidence') or 0.0)
        sal_boost = SALIENCE_W * float(salience) if salience is not None else 0.0

        role = (getattr(meta, "evidence_role", "") or "").lower()
        if not role and chunk.source_modality == 'image':
            _itype = (chunk.extra_payload.get('image_type') or '').lower()
            role = {'chart':'measurement','plot':'measurement','diagram':'definition','schematic':'procedure','graph':'comparative'}.get(_itype, 'context')
        role_boost = ROLE_BOOST if role in compatible_roles else 0.0

        table_boost = 0.0
        if apply_table_boost and chunk.source_modality == "table":
            table_boost = TABLE_BOOST

        noise = -0.20 if any(p in content for p in _NOISE_PATTERNS) else 0.0

        final = round(
            vector_score
            + 0.20 * code_boost
            + 0.10 * type_boost
            + 0.10 * modal_boost
            + sal_boost
            + role_boost
            + table_boost
            + noise,
            4,
        )
        final = max(0.0, min(1.5, final))

        scored.append(chunk.model_copy(update={
            "score_breakdown": ScoreBreakdown(
                vector_score        = vector_score,
                cross_encoder_score = chunk.score_breakdown.cross_encoder_score,
                code_boost          = code_boost,
                type_boost          = type_boost,
                modality_boost      = modal_boost,
                salience_boost      = round(sal_boost, 4),
                evidence_role_boost = round(role_boost, 4),
                noise_penalty       = noise,
                final_score         = final,
            )
        }))

    scored.sort(key=lambda c: c.score_breakdown.final_score, reverse=True)
    return scored


def _searchable_text(chunk: RetrievedChunk) -> str:
    meta  = chunk.metadata
    parts = [chunk.content or ""]
    for attr in (
        "text_original_content", "contextual_summary", "transcript_text",
        "table_summary", "table_purpose", "markdown",
        "image_caption", "depicted_component", "visible_annotations",
        "section_title", "local_context",
    ):
        if hasattr(meta, attr):
            val = getattr(meta, attr, None)
            if val:
                parts.append(str(val))
    parts.extend(
        str(v) for v in chunk.extra_payload.values()
        if isinstance(v, (str, int, float))
    )
    return re.sub(r"\s+", " ", " ".join(parts)).strip()
