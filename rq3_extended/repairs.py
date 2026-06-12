"""
rq3_extended/repairs.py

Four repair actions for HEMMIR self-correction (RQ3 Extended).

faith_repair      — Re-synthesize from existing evidence with stricter
                    STRONG_THRESHOLD (0.70 vs normal 0.65). No new retrieval.
                    Targets: low faithfulness caused by weakly-supported claims
                    passing the normal threshold.

evidence_repair   — MultiQuery re-retrieval: generate 4 query variants,
                    retrieve from Chroma, RRF merge, append new chunks,
                    re-run full R2-G. Targets: weak evidence pool.

gap_repair        — Extract weak/unsupported claims from the initial R2-G
                    output, generate targeted gap queries for those aspects,
                    retrieve, merge, re-run R2-G. Targets: incomplete answers.

full_escalation   — MultiQuery + expanded K (15) + full R2-G on all merged
                    evidence. Fallback when all three components are very low.

All repairs return a ConditionOutput (same schema as R2-G run).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from loguru import logger

from rq2_ablation.conditions.base import (
    ConditionOutput,
    ClaimRecord,
    format_evidence,
    generate_claims,
    check_support,
    check_attack,
    synthesize_answer,
    claims_to_text,
)
from rq2_ablation.conditions.r2f_direct_filter import _parse_classifications, FILTER_SYSTEM, FILTER_PROMPT
from .config import (
    FAITH_REPAIR_STRONG_THR,
    NORMAL_STRONG_THR,
    DEFAULT_N_VARIANTS,
    DEFAULT_TOP_K,
    DEFAULT_EXPANDED_K,
)
from .multiquery import multiquery_retrieve

# ── Shared ArgRAG constants (mirrors r2g_two_stage.py) ────────────────────────
ALPHA            = 0.5
BETA             = 0.3
GAMMA            = 0.2
WEAK_THRESHOLD   = 0.35
ATTACK_THRESHOLD = 0.5
MIN_COVERAGE     = 0.3


def _run_argrag(
    llm,
    question:        str,
    texts:           List[str],
    chunk_ids:       List[str],
    modalities:      List[str],
    strong_threshold: float = NORMAL_STRONG_THR,
    condition_label: str = "repair",
) -> ConditionOutput:
    """
    Full ArgRAG pass identical to R2-G Stage 2 (claim-level scoring).
    strong_threshold is parameterised so faith_repair can tighten it.
    """
    evidence_text = format_evidence(texts, chunk_ids, modalities)
    claims = generate_claims(llm, question, evidence_text)
    logger.info(f"  [{condition_label}] {len(claims)} claims generated")

    if not claims:
        return ConditionOutput(
            condition=condition_label,
            answer="Insufficient evidence to answer.",
            abstained=True,
            coverage_score=0.0,
            error="claim generation returned empty",
        )

    for c in claims:
        is_sup, s_score, s_reason = check_support(llm, c.text, evidence_text)
        is_att, a_score, a_reason = check_attack(llm,  c.text, evidence_text)
        c.support_score      = s_score if is_sup else 0.0
        c.attack_score       = a_score if is_att else 0.0
        c.raw_support_score  = c.support_score
        c.raw_attack_score   = c.attack_score
        c.support_reason = s_reason
        c.attack_reason  = a_reason

    def _consistency(c: ClaimRecord) -> float:
        if c.raw_support_score > 0 and c.raw_attack_score > 0:
            return 0.0
        return 1.0 if c.raw_support_score > 0 else 0.5

    for c in claims:
        strength = ALPHA * c.raw_support_score - BETA * c.raw_attack_score + GAMMA * _consistency(c)
        c.support_score = round(max(0.0, min(1.0, strength)), 4)

    for c in claims:
        has_support = c.support_score > 0.0
        has_attack  = c.attack_score > ATTACK_THRESHOLD
        if has_attack and not has_support:
            c.status = "contradicted"
        elif c.support_score >= strong_threshold:
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

    claim_coverage = len(defensible) / max(1, len(claims))

    if not defensible:
        return ConditionOutput(
            condition=condition_label,
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

    for c in claims:
        if c.status in ("strong", "moderate", "contested"):
            c.status = "supported"
        elif c.status == "weak":
            c.status = "unsupported"

    abstained = answer.strip().lower().startswith("insufficient evidence")

    return ConditionOutput(
        condition=condition_label,
        answer=answer,
        claims=claims,
        unsupported_count=len(weak),
        contradicted_count=len(contradict),
        coverage_score=round(claim_coverage, 4),
        abstained=abstained,
    )


# ── Repair 1: Faith repair ────────────────────────────────────────────────────

def faith_repair(
    llm,
    question:   str,
    texts:      List[str],
    chunk_ids:  List[str],
    modalities: List[str],
) -> ConditionOutput:
    """
    Re-synthesize using only STRONG claims (threshold 0.70 vs normal 0.65).
    No new retrieval — tightens the claim acceptance filter on existing evidence.
    """
    logger.info("  [faith_repair] Re-synthesizing with STRONG_THR=0.70")
    return _run_argrag(
        llm, question, texts, chunk_ids, modalities,
        strong_threshold=FAITH_REPAIR_STRONG_THR,
        condition_label="faith_repair",
    )


# ── Repair 2: Evidence repair ─────────────────────────────────────────────────

def evidence_repair(
    llm,
    text_embedder,
    store,
    question:   str,
    arxiv_id:   str,
    texts:      List[str],
    chunk_ids:  List[str],
    modalities: List[str],
    n_variants: int = DEFAULT_N_VARIANTS,
    top_k:      int = DEFAULT_TOP_K,
) -> ConditionOutput:
    """
    MultiQuery re-retrieval: generate n_variants query reformulations,
    retrieve from Chroma, RRF merge, append new chunks, re-run R2-G.
    """
    logger.info(f"  [evidence_repair] MultiQuery re-retrieval ({n_variants} variants)")
    new_texts, new_ids, new_mods = multiquery_retrieve(
        llm, text_embedder, store, question, arxiv_id,
        existing_chunk_ids=chunk_ids,
        n_variants=n_variants,
        top_k=top_k,
    )

    merged_texts = list(texts) + new_texts
    merged_ids   = list(chunk_ids) + new_ids
    merged_mods  = list(modalities) + new_mods

    logger.info(
        f"  [evidence_repair] {len(texts)} original + {len(new_texts)} new "
        f"= {len(merged_texts)} total chunks"
    )
    # If no new chunks found, re-running ArgRAG on identical evidence won't help
    if not new_texts:
        logger.warning(
            "  [evidence_repair] No new chunks found — document may not contain "
            "the requested information. Returning original evidence with re-run."
        )
    return _run_argrag(
        llm, question, merged_texts, merged_ids, merged_mods,
        condition_label="evidence_repair",
    )


# ── Repair 3: Gap repair ──────────────────────────────────────────────────────

def gap_repair(
    llm,
    text_embedder,
    store,
    question:   str,
    arxiv_id:   str,
    texts:      List[str],
    chunk_ids:  List[str],
    modalities: List[str],
    weak_claim_texts: List[str],
    n_variants: int = DEFAULT_N_VARIANTS,
    top_k:      int = DEFAULT_TOP_K,
) -> ConditionOutput:
    """
    Targeted gap retrieval: use weak/unsupported claims as query seeds
    to find specific missing evidence, then re-run R2-G on merged evidence.
    """
    logger.info(
        f"  [gap_repair] Targeted gap retrieval for "
        f"{len(weak_claim_texts)} weak claims"
    )
    if not weak_claim_texts:
        weak_claim_texts = [question]

    new_texts, new_ids, new_mods = multiquery_retrieve(
        llm, text_embedder, store, question, arxiv_id,
        existing_chunk_ids=chunk_ids,
        n_variants=n_variants,
        top_k=top_k,
        weak_claims=weak_claim_texts,
    )

    merged_texts = list(texts) + new_texts
    merged_ids   = list(chunk_ids) + new_ids
    merged_mods  = list(modalities) + new_mods

    logger.info(
        f"  [gap_repair] {len(texts)} original + {len(new_texts)} new gap chunks"
    )
    return _run_argrag(
        llm, question, merged_texts, merged_ids, merged_mods,
        condition_label="gap_repair",
    )


# ── Repair 4: Full escalation ─────────────────────────────────────────────────

def full_escalation(
    llm,
    text_embedder,
    store,
    question:   str,
    arxiv_id:   str,
    texts:      List[str],
    chunk_ids:  List[str],
    modalities: List[str],
    n_variants: int = DEFAULT_N_VARIANTS,
    expanded_k: int = DEFAULT_EXPANDED_K,
) -> ConditionOutput:
    """
    Full escalation: MultiQuery with expanded K + full R2-G.
    Used when all three C components are simultaneously very low.
    """
    logger.info(
        f"  [full_escalation] MultiQuery + expanded K={expanded_k}"
    )
    new_texts, new_ids, new_mods = multiquery_retrieve(
        llm, text_embedder, store, question, arxiv_id,
        existing_chunk_ids=chunk_ids,
        n_variants=n_variants,
        top_k=expanded_k,
    )

    merged_texts = list(texts) + new_texts
    merged_ids   = list(chunk_ids) + new_ids
    merged_mods  = list(modalities) + new_mods

    logger.info(
        f"  [full_escalation] {len(texts)} original + {len(new_texts)} new "
        f"= {len(merged_texts)} total chunks"
    )
    return _run_argrag(
        llm, question, merged_texts, merged_ids, merged_mods,
        condition_label="full_escalation",
    )
