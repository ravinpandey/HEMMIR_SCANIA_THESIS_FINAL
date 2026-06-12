"""
rq2_ablation/conditions/r2e_full_hemmir.py

R2-E: Full HEMMIR Argumentation Scoring
─────────────────────────────────────────
Implements the complete argumentation pipeline:

  Step 1 — Claim decomposition
  Step 2 — Support scoring   (α = 0.5)
  Step 3 — Attack scoring    (β = 0.3)
  Step 4 — Consistency bonus (γ = 0.2)
  Step 5 — Claim strength:   S(Ci) = α×support - β×attack + γ×consistency
  Step 6 — Claim selection:  strong (S≥0.65) + contested (has both S+A)
  Step 7 — Coverage check:   if coverage < MIN_COVERAGE → include uncertainty note
  Step 8 — Synthesis from defensible claims only

This mirrors the ArgRAGReasoner logic but operates on plain-text evidence
(checkpoint retrieved_texts) rather than EvidencePack objects — keeping
it directly comparable to R2-A through R2-D.

Claim status labels:
  strong      — S(Ci) >= 0.65, no attack → use with full confidence
  moderate    — 0.35 <= S(Ci) < 0.65, no attack → use as is
  contested   — has both support AND attack evidence → use with note
  weak        — S(Ci) < 0.35 → exclude
  contradicted — attack_score >= ATTACK_THRESHOLD and no support → exclude

Final answer status:
  Faithful and complete     — coverage >= 0.7
  Faithful but incomplete   — coverage 0.3-0.7
  Insufficient evidence     — coverage < 0.3 or no defensible claims
"""

from __future__ import annotations

from statistics import mean
from typing import List

from loguru import logger

from .base import (
    ConditionOutput,
    ClaimRecord,
    format_evidence,
    generate_claims,
    check_support,
    check_attack,
    synthesize_answer,
    claims_to_text,
)

# ── Claim strength weights (from ArgRAGReasoner) ──────────────────────────────
ALPHA = 0.5   # support weight
BETA  = 0.3   # attack penalty
GAMMA = 0.2   # inter-claim consistency bonus

STRONG_THRESHOLD    = 0.65
WEAK_THRESHOLD      = 0.35
ATTACK_THRESHOLD    = 0.5
MIN_COVERAGE        = 0.3   # below this → express uncertainty


def run(
    llm,
    question:   str,
    texts:      List[str],
    chunk_ids:  List[str],
    modalities: List[str],
) -> ConditionOutput:
    evidence_text = format_evidence(texts, chunk_ids, modalities)

    # ── Step 1: Claim decomposition ───────────────────────────────────────────
    claims = generate_claims(llm, question, evidence_text)
    logger.info(f"  R2-E: {len(claims)} claims generated")

    if not claims:
        return ConditionOutput(
            condition="R2-E",
            answer="Insufficient evidence to answer.",
            abstained=True,
            error="claim generation returned empty",
        )

    # ── Steps 2-3: Support and attack scoring per claim ───────────────────────
    for c in claims:
        is_sup, s_score, s_reason = check_support(llm, c.text, evidence_text)
        is_att, a_score, a_reason = check_attack(llm,  c.text, evidence_text)
        c.support_score      = s_score if is_sup else 0.0
        c.attack_score       = a_score if is_att else 0.0
        c.raw_support_score  = c.support_score
        c.raw_attack_score   = c.attack_score
        c.support_reason = s_reason
        c.attack_reason  = a_reason

    # ── Step 4: Consistency bonus ─────────────────────────────────────────────
    def _consistency(c: ClaimRecord) -> float:
        if c.raw_support_score > 0 and c.raw_attack_score > 0:
            return 0.0   # both sides — disputed
        if c.raw_support_score > 0:
            return 1.0
        return 0.5

    # ── Step 5: Claim strength  S(Ci) = α×s - β×a + γ×cons ──────────────────
    for c in claims:
        strength = (
            ALPHA * c.raw_support_score
            - BETA  * c.raw_attack_score
            + GAMMA * _consistency(c)
        )
        c.support_score = round(max(0.0, min(1.0, strength)), 4)

    # ── Step 6: Claim selection ───────────────────────────────────────────────
    for c in claims:
        has_support = c.raw_support_score > 0.0
        has_attack  = c.raw_attack_score  > ATTACK_THRESHOLD

        if has_attack and not has_support:
            c.status = "contradicted"
        elif c.support_score >= STRONG_THRESHOLD:
            c.status = "strong"
        elif c.support_score >= WEAK_THRESHOLD:
            c.status = "moderate"
        elif has_support and has_attack:
            c.status = "contested"
        else:
            c.status = "weak"

    strong     = [c for c in claims if c.status in ("strong", "moderate")]
    contested  = [c for c in claims if c.status == "contested"]
    weak       = [c for c in claims if c.status == "weak"]
    contradict = [c for c in claims if c.status == "contradicted"]
    defensible = strong + contested   # used for synthesis

    logger.info(
        f"  R2-E: strong/moderate={len(strong)} contested={len(contested)} "
        f"weak={len(weak)} contradicted={len(contradict)}"
    )

    # ── Step 7: Coverage check ────────────────────────────────────────────────
    coverage = len(defensible) / max(1, len(claims))

    if not defensible:
        return ConditionOutput(
            condition="R2-E",
            answer="Insufficient evidence to answer.",
            claims=claims,
            unsupported_count=len(weak),
            contradicted_count=len(contradict),
            coverage_score=0.0,
            abstained=True,
        )

    # ── Step 8: Synthesis from defensible claims ──────────────────────────────
    contested_note = ""
    if contested:
        contested_note = (
            " Note: the following claims have conflicting evidence and are "
            "marked uncertain: " + "; ".join(c.text[:60] for c in contested[:2])
        )

    uncertainty_note = ""
    if coverage < MIN_COVERAGE:
        uncertainty_note = (
            " Warning: coverage is low — the answer may be incomplete "
            "due to limited supporting evidence."
        )

    instruction = (
        "Answer using only these defensible claims."
        + (f" Flag contested claims: {[c.text[:40] for c in contested]}"
           if contested else "")
        + (" Express uncertainty about completeness." if coverage < MIN_COVERAGE else "")
    )

    answer = synthesize_answer(
        llm, question,
        claims_to_text(defensible),
        instruction=instruction,
    )

    # Append notes to answer
    if uncertainty_note:
        answer = answer + uncertainty_note
    if contested_note:
        answer = answer + contested_note

    abstained = answer.strip().lower().startswith("insufficient evidence")

    # Map to claim status labels for logging
    for c in claims:
        if c.status in ("strong", "moderate"):
            c.status = "supported"
        elif c.status == "contested":
            c.status = "supported"   # used but flagged
        elif c.status in ("weak",):
            c.status = "unsupported"
        elif c.status == "contradicted":
            c.status = "contradicted"

    return ConditionOutput(
        condition="R2-E",
        answer=answer,
        claims=claims,
        unsupported_count=len(weak),
        contradicted_count=len(contradict),
        coverage_score=round(coverage, 4),
        abstained=abstained,
    )
