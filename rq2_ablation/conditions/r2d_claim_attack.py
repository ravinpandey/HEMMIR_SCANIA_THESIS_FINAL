"""
rq2_ablation/conditions/r2d_claim_attack.py

R2-D: Claim + Support + Attack Detection
──────────────────────────────────────────
Extends R2-C by also detecting contradictions.
Claims that are contradicted by evidence are removed even if supported.

Purpose: test whether attack detection on top of support scoring
         further improves faithfulness by removing contested claims.

Lexicographic decision rule (in order):
  1. Contradicted claims are removed first (attack_score >= ATTACK_THRESHOLD).
  2. Unsupported claims are removed second (support_score < SUPPORT_THRESHOLD).
  3. Remaining claims are synthesized into the final answer.
  4. If no defensible claims remain → abstain.

No coverage scoring — that is reserved for R2-E (Full HEMMIR).
"""

from __future__ import annotations

from typing import List

from loguru import logger

from .base import (
    ConditionOutput,
    format_evidence,
    generate_claims,
    check_support,
    check_attack,
    synthesize_answer,
    claims_to_text,
)

SUPPORT_THRESHOLD = 0.5
ATTACK_THRESHOLD  = 0.5   # claim must score < this to avoid removal


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
    logger.info(f"  R2-D: {len(claims)} claims generated")

    if not claims:
        return ConditionOutput(
            condition="R2-D",
            answer="Insufficient evidence to answer.",
            abstained=True,
            error="claim generation returned empty",
        )

    # Step 2 — Support + attack check for every claim
    for c in claims:
        is_sup, s_score, s_reason = check_support(llm, c.text, evidence_text)
        is_att, a_score, a_reason = check_attack(llm,  c.text, evidence_text)

        c.support_score  = s_score
        c.support_reason = s_reason
        c.attack_score   = a_score
        c.attack_reason  = a_reason

        # Lexicographic rule: contradicted > unsupported > supported
        if is_att and a_score >= ATTACK_THRESHOLD:
            c.status = "contradicted"
        elif not (is_sup and s_score >= SUPPORT_THRESHOLD):
            c.status = "unsupported"
        else:
            c.status = "supported"

    contradicted = [c for c in claims if c.status == "contradicted"]
    unsupported  = [c for c in claims if c.status == "unsupported"]
    supported    = [c for c in claims if c.status == "supported"]

    logger.info(
        f"  R2-D: {len(supported)} supported, "
        f"{len(unsupported)} unsupported, {len(contradicted)} contradicted"
    )

    # Step 3 — Synthesize from defensible claims only
    if not supported:
        return ConditionOutput(
            condition="R2-D",
            answer="Insufficient evidence to answer.",
            claims=claims,
            unsupported_count=len(unsupported),
            contradicted_count=len(contradicted),
            coverage_score=0.0,
            abstained=True,
        )

    coverage = len(supported) / max(1, len(claims))
    answer = synthesize_answer(
        llm, question,
        claims_to_text(supported),
        instruction=(
            "Answer using only these defensible claims. "
            "Contradicted and unsupported claims have been removed."
        ),
    )
    abstained = answer.strip().lower().startswith("insufficient evidence")

    return ConditionOutput(
        condition="R2-D",
        answer=answer,
        claims=claims,
        unsupported_count=len(unsupported),
        contradicted_count=len(contradicted),
        coverage_score=round(coverage, 4),
        abstained=abstained,
    )
