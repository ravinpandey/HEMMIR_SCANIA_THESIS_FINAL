"""
generation_layer/uncertainty.py

Confidence-first scoring for HEMMIR — the primary research contribution.

Formula (Confidence-first, Additive):
  C = W_FAITHFULNESS(0.45) × faithfulness
    + W_EVIDENCE(0.25)     × evidence_strength
    + W_COMPLETENESS(0.20) × completeness
    + W_DIRECT(0.10)       × direct_evidence_bonus

  Where:
    evidence_strength    = 0.5 × mean_salience + 0.5 × mean_cross_encoder
    completeness         = f(sufficiency, unsupported_ratio, gap_ratio)
    direct_evidence_bonus = 1.0 if faithfulness >= 0.70 and sufficiency >= partial

  Uncertainty = 1 - Confidence (derived, not primary)

Why this is superior to penalty-stacking:
  The previous U = base - penalties caused "death by a thousand cuts":
  partial evidence + two unsupported claims → U=0.05 even when faith=0.92.
  The confidence-first approach rewards what IS present, not punishes gaps.

Confidence thresholds for agentic routing:
  C > 0.75 → HIGH:   Deliver response with citations
  C > 0.50 → MEDIUM: Self-correction loop to fill gaps
  C < 0.50 → LOW:    Re-retrieval, context insufficient

Aligns with RAG Triad (Faithfulness, Relevance, Groundedness).
Each component is independently ablatable — key thesis experiment.

Added in this version:
  weakest_component() — returns which of the 4 signals is lowest,
  used by generation_pipeline to decide which repair action to take.
  Exported constants: repair action names + thresholds + weights
  so generation_pipeline can import them without circular dependency.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Any, List, Optional, Tuple

from loguru import logger

from shared.models.pipeline_models import (
    EvidencePack,
    FaithfulnessReport,
    RetrievalTrace,
    RoleMismatch,
    UncertaintyReport,
)

# ── Confidence weights (sum = 1.0) ────────────────────────────────────────────
W_FAITHFULNESS  = 0.45
W_EVIDENCE      = 0.25
W_COMPLETENESS  = 0.20
W_DIRECT        = 0.10

SALIENCE_MIX    = 0.50
CROSS_MIX       = 0.50

SUFFICIENCY_MAP = {
    "full":         1.0,
    "partial":      0.7,
    "insufficient": 0.3,
}
UNSUPPORTED_PENALTY = 0.08
GAP_PENALTY         = 0.04
HALLUCINATION_CAP   = 0.35

HIGH_THRESHOLD   = 0.75
MEDIUM_THRESHOLD = 0.50

# Repair action names — imported by generation_pipeline routing logic
REPAIR_LOW_FAITHFULNESS = "tighten_to_supported_claims"
REPAIR_LOW_EVIDENCE     = "re_retrieve_broader"
REPAIR_LOW_COMPLETENESS = "re_retrieve_missing_gaps"
REPAIR_LOW_DIRECT       = "escalate_to_agent"


def compute_uncertainty(
    evidence_pack:       EvidencePack,
    faithfulness_report: FaithfulnessReport,
    retrieval_trace:     RetrievalTrace,
    llm_client=          None,
) -> UncertaintyReport:
    """
    Compute confidence score and derive uncertainty.
    score field = CONFIDENCE (higher = better).
    uncertainty = 1 - score (derived).
    """
    items = evidence_pack.items

    # ── Faithfulness ──────────────────────────────────────────────────
    faithfulness_val = max(0.0, min(1.0, faithfulness_report.overall_faithfulness))

    # ── Evidence Strength ─────────────────────────────────────────────
    salience_vals = []
    role_conf_vals = []
    for item in items:
        _ep = item.extra_payload if isinstance(item.extra_payload, dict) else {}

        # Salience — check metadata first, then extra_payload (ChromaDB passthrough)
        sal = getattr(item.metadata, "salience_score", None)
        if sal is None or sal == 0.0:
            sal = _ep.get("salience_score")
        if sal is None or sal == 0.0:
            sal = _ep.get("table_summary_confidence")
        if sal is not None:
            try:
                salience_vals.append(max(0.0, min(1.0, float(sal))))
            except (TypeError, ValueError):
                pass

        # evidence_role_confidence — from extra_payload (ChromaDB metadata passthrough)
        rc = getattr(item.metadata, "evidence_role_confidence", None)
        if rc is None:
            rc = _ep.get("evidence_role_confidence")
        if rc is not None:
            try:
                role_conf_vals.append(max(0.0, min(1.0, float(rc))))
            except (TypeError, ValueError):
                pass

    mean_salience   = mean(salience_vals)   if salience_vals   else 0.5
    mean_role_conf  = mean(role_conf_vals)  if role_conf_vals  else 0.5

    cross_vals = [
        item.score_breakdown.cross_encoder_score
        for item in items
        if item.score_breakdown.cross_encoder_score > 0
    ]
    mean_cross = mean(cross_vals) if cross_vals else 0.0
    # Include role_confidence in evidence strength — dampens uncertain enrichment signals
    evidence_strength = SALIENCE_MIX * mean_salience * mean_role_conf + CROSS_MIX * mean_cross

    # ── Mean contextual confidence (for logging) ───────────────────────
    conf_vals = []
    for item in items:
        conf = getattr(item.metadata, "contextual_summary_confidence", None)
        if conf is not None:
            try:
                conf_vals.append(max(0.0, min(1.0, float(conf))))
            except (TypeError, ValueError):
                pass
    mean_conf = mean(conf_vals) if conf_vals else 0.5

    # ── Completeness ──────────────────────────────────────────────────
    suf_value     = SUFFICIENCY_MAP.get(retrieval_trace.overall_sufficiency, 0.5)
    n_unsupported = len(faithfulness_report.unsupported_claims)
    n_missing     = len(faithfulness_report.missing_evidence)

    completeness = max(0.0, min(1.0,
        suf_value
        - UNSUPPORTED_PENALTY * n_unsupported
        - min(0.30, GAP_PENALTY * n_missing)
    ))

    # ── Direct Evidence Bonus ─────────────────────────────────────────
    direct_bonus = 1.0 if (faithfulness_val >= 0.70 and suf_value >= 0.5) else 0.0

    # ── Confidence score ──────────────────────────────────────────────
    confidence = (
        W_FAITHFULNESS   * faithfulness_val
        + W_EVIDENCE     * evidence_strength
        + W_COMPLETENESS * completeness
        + W_DIRECT       * direct_bonus
    )
    confidence = max(0.0, min(1.0, confidence))

    # Hallucination hard cap
    hallucination = faithfulness_report.contains_hallucination
    if hallucination:
        confidence = min(confidence, HALLUCINATION_CAP)

    # ── Level ─────────────────────────────────────────────────────────
    if confidence >= HIGH_THRESHOLD:
        level = "HIGH_CONFIDENCE"
    elif confidence >= MEDIUM_THRESHOLD:
        level = "MEDIUM_CONFIDENCE"
    else:
        level = "LOW_CONFIDENCE"

    # ── Modality signals ──────────────────────────────────────────────
    asr_conf: Optional[float] = None
    cap_conf: Optional[float] = None
    dominant_modality = _dominant_modality(items)

    for item in items:
        if item.source_modality == "video_segment":
            asr = getattr(item.metadata, "asr_confidence", None)
            if asr is not None:
                asr_conf = float(asr)
                break
        if item.source_modality in ("image", "video_frame"):
            cap = getattr(item.metadata, "image_caption_confidence", None)
            if cap is not None:
                cap_conf = float(cap)
                break

    # ── Missing evidence ──────────────────────────────────────────────
    missing_evidence = list(faithfulness_report.missing_evidence)
    for sq in retrieval_trace.sub_question_results:
        for gap in sq.missing_aspects:
            if gap and gap not in missing_evidence:
                missing_evidence.append(gap)

    # ── Narrative ─────────────────────────────────────────────────────
    explanation = ""
    recommendation = ""
    if llm_client:
        try:
            explanation, recommendation = _generate_narrative(
                llm_client    = llm_client,
                score         = confidence,
                level         = level,
                mean_salience = mean_salience,
                mean_cross    = mean_cross,
                suf_value     = suf_value,
                n_unsupported = n_unsupported,
                n_missing     = n_missing,
                missing_gaps  = missing_evidence[:3],
                n_role_mm     = len(faithfulness_report.role_mismatches),
                path          = retrieval_trace.path,
            )
        except Exception as e:
            logger.warning(f"  Confidence narrative failed: {e}")

    if not explanation:
        explanation = _rule_based_explanation(level, n_unsupported, n_missing, suf_value)
    if not recommendation:
        recommendation = _rule_based_recommendation(level, missing_evidence)

    suf_raw = SUFFICIENCY_MAP.get(retrieval_trace.overall_sufficiency, 0.5)
    logger.info(
        f"  Confidence: score={confidence:.3f} level={level} "
        f"faith={faithfulness_val:.2f} evid={evidence_strength:.2f} "
        f"complete={completeness:.2f} direct={direct_bonus:.1f} "
        f"salience={mean_salience:.2f} cross={mean_cross:.2f} "
        f"suf={suf_raw:.1f} unsupported={n_unsupported}"
    )

    return UncertaintyReport(
        score               = round(confidence, 4),
        level               = level,
        mean_salience       = round(mean_salience, 4),
        mean_cross_enc      = round(mean_cross, 4),
        mean_conf           = round(mean_conf, 4),
        sufficiency_value   = round(suf_raw, 4),
        faithfulness        = round(faithfulness_report.overall_faithfulness, 4),
        uncertainty_prior   = retrieval_trace.uncertainty_prior,
        unsupported_count   = n_unsupported,
        missing_gap_count   = n_missing,
        role_mismatch_count = len(faithfulness_report.role_mismatches),
        hallucination_flag  = hallucination,
        dominant_modality   = dominant_modality,
        asr_confidence      = asr_conf,
        caption_confidence  = cap_conf,
        level_explanation   = explanation,
        recommendation      = recommendation,
        missing_evidence    = missing_evidence,
        role_mismatches     = faithfulness_report.role_mismatches,
    )


def weakest_component(
    faithfulness_val:  float,
    evidence_strength: float,
    completeness:      float,
    direct_bonus:      float,
) -> Tuple[str, str]:
    """
    Identify the weakest confidence component and return the repair action.

    Returns:
        (component_name, repair_action)

    Used by generation_pipeline to decide which targeted repair to apply
    during the self-correction loop (MEDIUM_CONFIDENCE) or re-retrieval
    escalation (LOW_CONFIDENCE).

    Repair actions:
      tighten_to_supported_claims → strip unsupported claims, regenerate
      re_retrieve_broader         → re-retrieve with HyDE / more modalities
      re_retrieve_missing_gaps    → target missing sub-question evidence
      escalate_to_agent           → switch RAG→Agent path
    """
    scores = {
        "faithfulness": faithfulness_val  * W_FAITHFULNESS,
        "evidence":     evidence_strength * W_EVIDENCE,
        "completeness": completeness      * W_COMPLETENESS,
        "direct":       direct_bonus      * W_DIRECT,
    }
    weakest = min(scores, key=lambda k: scores[k])
    repair_map = {
        "faithfulness": REPAIR_LOW_FAITHFULNESS,
        "evidence":     REPAIR_LOW_EVIDENCE,
        "completeness": REPAIR_LOW_COMPLETENESS,
        "direct":       REPAIR_LOW_DIRECT,
    }
    return weakest, repair_map[weakest]


# ── LLM narrative ─────────────────────────────────────────────────────────────

NARRATIVE_SYSTEM = """\
You write plain-language confidence assessments for industrial engineers.
Be honest, specific, and practical. No jargon.\
"""

NARRATIVE_PROMPT = """\
Write a 2-sentence confidence explanation for a Scania engineer.

Query answered via: {path} path
Confidence score: {score:.2f} ({level})
Faithfulness (claims grounded in evidence): {faith:.0%}
Evidence strength: {evid:.2f}
Evidence sufficiency: {suf} ({suf_pct:.0%} complete)
Unsupported claims: {n_unsupported}
Missing information: {n_missing} gaps
Missing topics: {missing_gaps}
Evidence role mismatches: {n_role_mm}

Write:
EXPLANATION: <2 plain sentences — what was found and how confident>
RECOMMENDATION: <one action — e.g. "Check Section 3.2 directly" or "none needed">\
"""


def _generate_narrative(
    llm_client, score, level, mean_salience, mean_cross,
    suf_value, n_unsupported, n_missing, missing_gaps, n_role_mm, path,
) -> tuple[str, str]:
    evidence_strength = SALIENCE_MIX * mean_salience + CROSS_MIX * mean_cross
    prompt = NARRATIVE_PROMPT.format(
        path          = path,
        score         = score,
        level         = level,
        faith         = score,
        evid          = evidence_strength,
        suf           = "full" if suf_value >= 1.0 else "partial" if suf_value >= 0.5 else "insufficient",
        suf_pct       = suf_value,
        n_unsupported = n_unsupported,
        n_missing     = n_missing,
        missing_gaps  = ", ".join(missing_gaps) if missing_gaps else "none",
        n_role_mm     = n_role_mm,
    )
    response = llm_client.invoke(
        system=NARRATIVE_SYSTEM, prompt=prompt, max_tokens=200
    )
    exp_match = re.search(r"EXPLANATION:\s*(.+?)(?=RECOMMENDATION:|$)", response, re.S)
    rec_match = re.search(r"RECOMMENDATION:\s*(.+?)$", response, re.S)
    explanation    = exp_match.group(1).strip() if exp_match else ""
    recommendation = rec_match.group(1).strip() if rec_match else ""
    return explanation, recommendation


def _rule_based_explanation(
    level: str, n_unsupported: int, n_missing: int, suf_value: float
) -> str:
    if level == "HIGH_CONFIDENCE":
        return "Evidence is well-grounded and directly supports all major claims."
    if level == "MEDIUM_CONFIDENCE":
        parts = []
        if n_unsupported:
            parts.append(f"{n_unsupported} claim(s) could not be fully verified")
        if n_missing:
            parts.append(f"{n_missing} information gap(s) remain")
        return ". ".join(parts) + "." if parts else "Evidence is partially complete."
    return (
        "Evidence is insufficient to fully answer this query. "
        "Key information was not found in the corpus."
    )


def _rule_based_recommendation(level: str, missing_evidence: List[str]) -> str:
    if level == "HIGH_CONFIDENCE":
        return "No further verification needed."
    if missing_evidence:
        return f"Verify directly: {missing_evidence[0]}"
    return "Cross-check against original documents before acting on this answer."


def _dominant_modality(items: List[Any]) -> str:
    counts: dict[str, int] = {}
    for item in items:
        m = item.source_modality
        counts[m] = counts.get(m, 0) + 1
    if not counts:
        return "text"
    return max(counts, key=lambda k: counts[k])