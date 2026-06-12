"""
rq3_extended/evaluator.py

Re-evaluates a ConditionOutput from a repair action:
  - Calls judge_faithfulness + judge_correctness (same as rq2_runner.py)
  - Computes the C_score using the RQ3 formula
  - Returns a flat dict matching the rq2_runner condition record schema
    so results slot directly into the existing analysis pipeline.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from loguru import logger

from rq2_ablation.conditions.base import ConditionOutput, format_evidence
from rq2_ablation.judge import judge_faithfulness, judge_correctness
from .config import (
    W_FAITH, W_EVSTR, W_COMP, W_DIRECT,
    HAL_CAP, COVERAGE_DIRECT_THR,
)


def _evidence_strength_from_texts(texts: List[str], chunk_ids: List[str]) -> float:
    """
    Proxy evidence strength from chunk count and deduplication ratio.
    Used for re-retrieved evidence that lacks R1-G ranking_trace scores.
    Heuristic: more chunks from the paper ≈ stronger evidence pool.
    Capped at 0.90 to avoid over-confidence.
    """
    if not texts:
        return 0.30
    unique_arxiv_ids = len({cid.split("_")[0] for cid in chunk_ids if "_" in cid})
    n = min(len(texts), 15)
    base = 0.50 + 0.03 * n
    cross_doc_bonus = 0.05 if unique_arxiv_ids > 1 else 0.0
    return round(min(0.90, base + cross_doc_bonus), 4)


def _c_score(
    faithfulness: float,
    evstr:        float,
    comp:         float,
    coverage:     float,
    abstained:    bool,
    unsupported:  int,
    total_claims: int,
) -> float:
    direct = W_DIRECT if coverage >= COVERAGE_DIRECT_THR else 0.0
    raw    = W_FAITH * faithfulness + W_EVSTR * evstr + W_COMP * comp + direct

    cap_triggered = abstained or (
        total_claims > 0 and (unsupported / total_claims) >= 0.5
    )
    if cap_triggered:
        raw = min(raw, HAL_CAP)
    return round(max(0.0, min(1.0, raw)), 4)


def evaluate_repair(
    llm,
    question:      str,
    gold_answer:   str,
    texts:         List[str],
    chunk_ids:     List[str],
    modalities:    List[str],
    output:        ConditionOutput,
    prior_evstr:   float,
    repair_type:   str,
    latency_ms:    float,
) -> Dict:
    """
    Evaluate a repaired ConditionOutput.

    Returns a record compatible with rq2_runner condition schema,
    plus extra fields for RQ3 Extended analysis:
      repair_type, c_score, evidence_strength
    """
    evidence_text = format_evidence(texts, chunk_ids, modalities)

    faith_result   = judge_faithfulness(llm, question, evidence_text, output.answer)
    correct_result = judge_correctness(llm, question, gold_answer, output.answer)

    # EvidenceStrength: use prior R1-G value if retrieval unchanged,
    # otherwise estimate from merged evidence pool
    evstr = (
        prior_evstr
        if repair_type == "faith_repair"
        else _evidence_strength_from_texts(texts, chunk_ids)
    )

    comp = (
        round(correct_result["key_facts_matched"] / max(1, correct_result["key_facts_total"]), 4)
        if correct_result.get("key_facts_total", 0) > 0
        else round(float(output.coverage_score), 4)
    )

    total_claims = len(output.claims)
    cs = _c_score(
        faithfulness  = faith_result["faithfulness"],
        evstr         = evstr,
        comp          = comp,
        coverage      = output.coverage_score,
        abstained     = output.abstained,
        unsupported   = output.unsupported_count,
        total_claims  = total_claims,
    )

    claim_records = [
        {
            "text":          c.text,
            "type":          c.claim_type,
            "status":        c.status,
            "support_score": c.support_score,
            "attack_score":  c.attack_score,
        }
        for c in output.claims
    ]

    logger.info(
        f"  [{repair_type}] faith={faith_result['faithfulness']:.3f} "
        f"correct={correct_result['correctness']:.3f} "
        f"C={cs:.3f} abstained={output.abstained}"
    )

    return {
        # Standard RQ2 fields
        "answer":                output.answer,
        "abstained":             output.abstained,
        "claims":                claim_records,
        "unsupported_count":     output.unsupported_count,
        "contradicted_count":    output.contradicted_count,
        "coverage_score":        output.coverage_score,
        "faithfulness":          faith_result["faithfulness"],
        "unsupported_sentences": faith_result["unsupported_sentences"],
        "faithfulness_verdict":  faith_result["verdict"],
        "is_faithful":           faith_result["is_faithful"],
        "correctness":           correct_result["correctness"],
        "correctness_verdict":   correct_result["verdict"],
        "key_facts_matched":     correct_result["key_facts_matched"],
        "key_facts_total":       correct_result["key_facts_total"],
        "latency_ms":            latency_ms,
        "error":                 output.error,
        # RQ3 Extended extra fields
        "repair_type":           repair_type,
        "c_score":               cs,
        "evidence_strength":     evstr,
        "completeness":          comp,
    }
