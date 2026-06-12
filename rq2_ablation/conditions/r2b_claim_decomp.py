"""
rq2_ablation/conditions/r2b_claim_decomp.py

R2-B: Claim Decomposition Only
────────────────────────────────
Decomposes the expected answer into atomic claims, then re-synthesizes.
No support or attack scoring — claims are NOT filtered.

Purpose: isolate whether structured decomposition + synthesis alone
         improves faithfulness over direct generation (R2-A).
         If R2-B ≈ R2-A, decomposition without verification adds no value.
"""

from __future__ import annotations

from typing import List

from loguru import logger

from .base import (
    ConditionOutput,
    ClaimRecord,
    format_evidence,
    generate_claims,
    synthesize_answer,
    claims_to_text,
)


def run(
    llm,
    question:   str,
    texts:      List[str],
    chunk_ids:  List[str],
    modalities: List[str],
) -> ConditionOutput:
    evidence_text = format_evidence(texts, chunk_ids, modalities)

    # Step 1 — Decompose into atomic claims (no filtering)
    claims = generate_claims(llm, question, evidence_text)
    logger.info(f"  R2-B: {len(claims)} claims generated")

    if not claims:
        return ConditionOutput(
            condition="R2-B",
            answer="Insufficient evidence to answer.",
            abstained=True,
            error="claim generation returned empty",
        )

    # All claims kept — status = unscored (no verification)
    for c in claims:
        c.status = "unscored"

    # Step 2 — Synthesize answer from all claims (unfiltered)
    answer = synthesize_answer(
        llm, question,
        claims_to_text(claims),
        instruction="Answer concisely using all the claims above.",
    )
    abstained = answer.strip().lower().startswith("insufficient evidence")

    return ConditionOutput(
        condition="R2-B",
        answer=answer,
        claims=claims,
        unsupported_count=0,
        contradicted_count=0,
        coverage_score=1.0,
        abstained=abstained,
    )
