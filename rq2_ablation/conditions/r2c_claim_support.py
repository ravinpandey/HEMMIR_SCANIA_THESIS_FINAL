"""
rq2_ablation/conditions/r2c_claim_support.py

R2-C: Claim + Support Scoring
───────────────────────────────
Decomposes the answer into atomic claims, checks each for evidence support,
removes unsupported claims, synthesizes from supported claims only.

No attack detection — contradictions are not handled.

Purpose: test whether support-based claim filtering reduces hallucination
         compared to unfiltered decomposition (R2-B).

Lexicographic decision rule:
  1. Unsupported claims are removed (support_score < SUPPORT_THRESHOLD).
  2. Remaining claims are kept regardless of attack (no attack check here).
  3. If all claims are unsupported → abstain with "Insufficient evidence."
"""

from __future__ import annotations

from typing import List

from loguru import logger

from .base import (
    ConditionOutput,
    format_evidence,
    generate_claims,
    check_support,
    synthesize_answer,
    claims_to_text,
)

SUPPORT_THRESHOLD = 0.5   # claim must score >= this to be kept


def run(
    llm,
    question:   str,
    texts:      List[str],
    chunk_ids:  List[str],
    modalities: List[str],
) -> ConditionOutput:
    evidence_text = format_evidence(texts, chunk_ids, modalities)

    # Step 1 — Decompose
    claims = generate_claims(llm, question, evidence_text)
    logger.info(f"  R2-C: {len(claims)} claims generated")

    if not claims:
        return ConditionOutput(
            condition="R2-C",
            answer="Insufficient evidence to answer.",
            abstained=True,
            error="claim generation returned empty",
        )

    # Step 2 — Support check for each claim
    for c in claims:
        is_sup, score, reason = check_support(llm, c.text, evidence_text)
        c.support_score  = score
        c.support_reason = reason
        c.status         = "supported" if (is_sup and score >= SUPPORT_THRESHOLD) else "unsupported"

    supported   = [c for c in claims if c.status == "supported"]
    unsupported = [c for c in claims if c.status == "unsupported"]
    logger.info(f"  R2-C: {len(supported)} supported, {len(unsupported)} unsupported")

    # Step 3 — Lexicographic rule: remove unsupported, synthesize from supported
    if not supported:
        return ConditionOutput(
            condition="R2-C",
            answer="Insufficient evidence to answer.",
            claims=claims,
            unsupported_count=len(unsupported),
            contradicted_count=0,
            coverage_score=0.0,
            abstained=True,
        )

    coverage = len(supported) / max(1, len(claims))
    answer = synthesize_answer(
        llm, question,
        claims_to_text(supported),
        instruction="Answer using only these evidence-supported claims.",
    )
    abstained = answer.strip().lower().startswith("insufficient evidence")

    return ConditionOutput(
        condition="R2-C",
        answer=answer,
        claims=claims,
        unsupported_count=len(unsupported),
        contradicted_count=0,
        coverage_score=round(coverage, 4),
        abstained=abstained,
    )
