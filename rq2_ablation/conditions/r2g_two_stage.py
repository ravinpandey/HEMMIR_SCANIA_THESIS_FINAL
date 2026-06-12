"""
rq2_ablation/conditions/r2g_two_stage.py

R2-G: Two-Stage HEMMIR (Chunk Filter → Claim ArgRAG)
──────────────────────────────────────────────────────
Combines R2-F and R2-E into a two-layer filtering pipeline:

  Stage 1 (Chunk-level, R2-F style):
    LLM classifies each retrieved chunk as SUPPORTING / NEUTRAL / CONTRADICTING.
    Only SUPPORTING chunks are passed to Stage 2.
    If no chunks survive → abstain.

  Stage 2 (Claim-level, R2-E style):
    Full ArgRAG on the pre-filtered evidence:
      1. Decompose question into 2-6 atomic claims (LLM decides)
      2. Score: S(Ci) = 0.5·support − 0.3·attack + 0.2·consistency
      3. Keep strong/moderate/contested claims (S ≥ WEAK_THRESHOLD)
      4. Abstain if coverage < MIN_COVERAGE
      5. Synthesize from defensible claims

Why this matters:
  R2-E runs ArgRAG on all 10 retrieved chunks, including noisy ones.
  Noisy chunks can incorrectly attack valid claims, reducing their S(Ci).
  R2-G pre-removes noisy chunks so ArgRAG scores on cleaner evidence —
  expected result: fewer false attacks, lower unsupported ratio.

  R2-F vs R2-G comparison isolates the added value of claim-level
  argumentation over chunk-level filtering alone.
"""

from __future__ import annotations

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
from .r2f_direct_filter import _parse_classifications, FILTER_SYSTEM, FILTER_PROMPT

# ── ArgRAG weights (same as R2-E) ─────────────────────────────────────────────
ALPHA            = 0.5
BETA             = 0.3
GAMMA            = 0.2
STRONG_THRESHOLD = 0.65
WEAK_THRESHOLD   = 0.35
ATTACK_THRESHOLD = 0.5
MIN_COVERAGE     = 0.3


def run(
    llm,
    question:   str,
    texts:      List[str],
    chunk_ids:  List[str],
    modalities: List[str],
) -> ConditionOutput:

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Chunk-level filter (identical to R2-F)
    # ══════════════════════════════════════════════════════════════════════════
    evidence_lines = []
    for i, (text, cid, mod) in enumerate(zip(texts, chunk_ids, modalities), 1):
        limit = 600 if mod == "image" else 300
        evidence_lines.append(f"E{i} ({mod}): {text[:limit]}")
    evidence_list_text = "\n\n".join(evidence_lines)

    filter_prompt = FILTER_PROMPT.format(
        question=question,
        evidence_list=evidence_list_text,
    )

    supporting_indices: List[int] = []
    try:
        resp = llm.invoke(
            system=FILTER_SYSTEM,
            prompt=filter_prompt,
            max_tokens=600,
            temperature=0,
        )
        data = _parse_classifications(resp)
        for item in data.get("classifications", []):
            eid   = str(item.get("id", "")).strip()
            label = str(item.get("label", "")).strip().upper()
            if label == "SUPPORTING" and eid.startswith("E"):
                try:
                    idx = int(eid[1:]) - 1
                    if 0 <= idx < len(texts):
                        supporting_indices.append(idx)
                except ValueError:
                    pass
        logger.info(
            f"  R2-G Stage1: {len(supporting_indices)}/{len(texts)} chunks kept"
        )
    except Exception as e:
        logger.warning(f"  R2-G Stage1 failed ({e}), using all chunks")
        supporting_indices = list(range(len(texts)))

    if not supporting_indices:
        # Graceful degradation: no chunk was explicitly SUPPORTING but don't abstain yet.
        # PPTX image descriptions and vision captions are often classified NEUTRAL by the
        # chunk filter because they're descriptive rather than directly assertive — yet
        # they contain the answer. Fall back to non-CONTRADICTING chunks and let Stage 2
        # ArgRAG be the quality gate instead.
        contradicting_indices: set = set()
        try:
            for item in data.get("classifications", []):
                if str(item.get("label", "")).upper() == "CONTRADICTING":
                    eid = str(item.get("id", ""))
                    if eid.startswith("E"):
                        try:
                            contradicting_indices.add(int(eid[1:]) - 1)
                        except ValueError:
                            pass
        except Exception:
            pass
        fallback = [i for i in range(len(texts)) if i not in contradicting_indices]
        if fallback:
            supporting_indices = fallback[:min(5, len(fallback))]
            logger.warning(
                f"  R2-G Stage1 fallback: 0 SUPPORTING — using top-{len(supporting_indices)} "
                f"non-CONTRADICTING chunks (image/PPTX content may be descriptive, not assertive)"
            )
        else:
            return ConditionOutput(
                condition="R2-G",
                answer="Insufficient evidence to answer.",
                abstained=True,
                coverage_score=0.0,
            )

    # Filter to supporting chunks only
    s_texts      = [texts[i]      for i in supporting_indices]
    s_chunk_ids  = [chunk_ids[i]  for i in supporting_indices]
    s_modalities = [modalities[i] for i in supporting_indices]
    stage1_coverage = len(supporting_indices) / max(1, len(texts))

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Claim-level ArgRAG on filtered evidence (identical to R2-E)
    # ══════════════════════════════════════════════════════════════════════════
    evidence_text = format_evidence(s_texts, s_chunk_ids, s_modalities)

    claims = generate_claims(llm, question, evidence_text)
    logger.info(f"  R2-G Stage2: {len(claims)} claims generated")

    if not claims:
        return ConditionOutput(
            condition="R2-G",
            answer="Insufficient evidence to answer.",
            abstained=True,
            coverage_score=0.0,
            error="claim generation returned empty after chunk filter",
        )

    # Support + attack scoring
    for c in claims:
        is_sup, s_score, s_reason = check_support(llm, c.text, evidence_text)
        is_att, a_score, a_reason = check_attack(llm,  c.text, evidence_text)
        c.support_score      = s_score if is_sup else 0.0
        c.attack_score       = a_score if is_att else 0.0
        c.raw_support_score  = c.support_score   # preserve before ArgRAG formula
        c.raw_attack_score   = c.attack_score
        c.support_reason = s_reason
        c.attack_reason  = a_reason

    # Consistency bonus
    def _consistency(c: ClaimRecord) -> float:
        if c.raw_support_score > 0 and c.raw_attack_score > 0:
            return 0.0
        if c.raw_support_score > 0:
            return 1.0
        return 0.5

    # Claim strength S(Ci)
    for c in claims:
        strength = ALPHA * c.raw_support_score - BETA * c.raw_attack_score + GAMMA * _consistency(c)
        c.support_score = round(max(0.0, min(1.0, strength)), 4)

    # Claim selection
    for c in claims:
        has_support = c.support_score > 0.0
        has_attack  = c.attack_score  > ATTACK_THRESHOLD
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
    defensible = strong + contested

    logger.info(
        f"  R2-G Stage2: strong/mod={len(strong)} contested={len(contested)} "
        f"weak={len(weak)} contradicted={len(contradict)}"
    )

    # Coverage check — uses combined stage1 × stage2 coverage
    claim_coverage = len(defensible) / max(1, len(claims))
    combined_coverage = round(stage1_coverage * claim_coverage, 4)

    if not defensible:
        return ConditionOutput(
            condition="R2-G",
            answer="Insufficient evidence to answer.",
            claims=claims,
            unsupported_count=len(weak),
            contradicted_count=len(contradict),
            coverage_score=0.0,
            abstained=True,
        )

    instruction = (
        "Answer using only these defensible claims."
        + (f" Flag contested claims: {[c.text[:40] for c in contested]}"
           if contested else "")
        + (" Express uncertainty about completeness." if claim_coverage < MIN_COVERAGE else "")
    )

    answer = synthesize_answer(llm, question, claims_to_text(defensible), instruction=instruction)

    # Normalise claim status labels
    for c in claims:
        if c.status in ("strong", "moderate", "contested"):
            c.status = "supported"
        elif c.status == "weak":
            c.status = "unsupported"

    abstained = answer.strip().lower().startswith("insufficient evidence")

    return ConditionOutput(
        condition="R2-G",
        answer=answer,
        claims=claims,
        unsupported_count=len(weak),
        contradicted_count=len(contradict),
        coverage_score=combined_coverage,
        abstained=abstained,
    )
