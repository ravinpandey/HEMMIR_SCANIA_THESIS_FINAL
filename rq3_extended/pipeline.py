"""
rq3_extended/pipeline.py

HEMMIR Self-Correction Pipeline — RQ3 Extended.

Iterative component-aware repair loop:

  For each question (from R2-G checkpoint):
    1. Load initial R2-G result + compute initial C_score
    2. Route: which component is below threshold?
    3. Fire repair action
    4. Re-evaluate: compute new C_score
    5. If new_C > best_C + MIN_IMPROVEMENT: accept, iterate
       Else: keep best and stop
    6. Repeat up to max_iterations

Best result across all iterations is stored.
If no repair improves C, the original R2-G answer is kept.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from loguru import logger

from rq2_ablation.conditions.base import format_evidence
from rq2_ablation.judge import judge_faithfulness, judge_correctness


def resolve_texts_from_store(
    store,
    chunk_ids:  List[str],
    modalities: List[str],
) -> List[str]:
    """
    Fetch chunk text content from ChromaDB given chunk_ids + modalities.
    Used when rq2_results doesn't store retrieved_texts (only chunk_ids).
    """
    coll_map = {
        "text":  store.collections.get("text"),
        "image": store.collections.get("images_text"),
        "table": store.collections.get("tables"),
    }
    texts = []
    for cid, mod in zip(chunk_ids, modalities):
        coll = coll_map.get(mod) or coll_map.get("text")
        text = ""
        if coll:
            try:
                result = coll.get(ids=[cid], include=["documents"])
                docs = result.get("documents") or []
                text = docs[0] if docs and docs[0] else ""
            except Exception:
                pass
        texts.append(text)
    return texts
from .config import (
    MIN_C_IMPROVEMENT,
    DEFAULT_MAX_ITER,
    DEFAULT_N_VARIANTS,
    DEFAULT_TOP_K,
    DEFAULT_EXPANDED_K,
    W_FAITH, W_EVSTR, W_COMP, W_DIRECT,
    HAL_CAP, COVERAGE_DIRECT_THR,
)
from .router import route, describe_route
from .repairs import faith_repair, evidence_repair, gap_repair, full_escalation
from .evaluator import evaluate_repair, _evidence_strength_from_texts


def _c_from_rq2_record(rq2_cond: Dict, prior_evstr: float) -> float:
    """Reconstruct C_score from an existing RQ2 condition record."""
    faith    = float(rq2_cond.get("faithfulness", 0.0))
    comp     = (
        rq2_cond["key_facts_matched"] / max(1, rq2_cond["key_facts_total"])
        if rq2_cond.get("key_facts_total", 0) > 0
        else float(rq2_cond.get("coverage_score", 0.0))
    )
    coverage = float(rq2_cond.get("coverage_score", 0.0))
    direct   = W_DIRECT if coverage >= COVERAGE_DIRECT_THR else 0.0
    total_claims = len(rq2_cond.get("claims", []))
    unsupported  = rq2_cond.get("unsupported_count", 0)
    abstained    = rq2_cond.get("abstained", False)

    raw = W_FAITH * faith + W_EVSTR * prior_evstr + W_COMP * comp + direct
    if abstained or (total_claims > 0 and (unsupported / total_claims) >= 0.5):
        raw = min(raw, HAL_CAP)
    return round(max(0.0, min(1.0, raw)), 4)


def process_question(
    llm,
    text_embedder,
    store,
    entry:           Dict,
    spiqa:           Dict,
    prior_evstr:     float,
    max_iterations:  int  = DEFAULT_MAX_ITER,
    n_variants:      int  = DEFAULT_N_VARIANTS,
    top_k:           int  = DEFAULT_TOP_K,
    expanded_k:      int  = DEFAULT_EXPANDED_K,
) -> Dict:
    """
    Run the self-correction loop for a single question.

    Args:
        entry:       one question record from rq2_results (R2-G condition present)
        spiqa:       SPIQA TestC data for gold answer lookup
        prior_evstr: evidence_strength from R1-G ranking_trace for this question
        max_iterations: max repair attempts (default 2)

    Returns:
        Full per-question log with keys:
          question_id, arxiv_id, question, gold_answer,
          initial        — original R2-G result + C_score
          iterations     — list of per-iteration repair records
          final          — best result chosen across all iterations
          n_repairs      — number of repair attempts made
          improved       — True if any iteration beat initial C_score
    """
    qid        = entry["question_id"]
    arxiv_id   = entry["arxiv_id"]
    question   = entry["question"]
    gold_answer = entry.get("gold_answer", "") or ""

    # Initial R2-G evidence — fetch texts from Chroma if not stored in entry
    chunk_ids  = entry.get("retrieved_chunk_ids", [])
    modalities = entry.get("retrieved_modalities", [])
    texts      = entry.get("retrieved_texts", [])
    if not texts and chunk_ids:
        logger.info(f"  [{qid}] retrieved_texts missing — fetching from Chroma")
        texts = resolve_texts_from_store(store, chunk_ids, modalities)

    r2g_cond = entry.get("conditions", {}).get("R2-G", {})
    if not r2g_cond:
        logger.warning(f"  [{qid}] No R2-G condition found — skipping")
        return {"question_id": qid, "error": "no R2-G condition"}

    # ── Initial state ─────────────────────────────────────────────────────────
    initial_faith = float(r2g_cond.get("faithfulness", 0.0))
    initial_comp  = (
        r2g_cond["key_facts_matched"] / max(1, r2g_cond["key_facts_total"])
        if r2g_cond.get("key_facts_total", 0) > 0
        else float(r2g_cond.get("coverage_score", 0.0))
    )
    initial_c = _c_from_rq2_record(r2g_cond, prior_evstr)

    initial_record = {
        **r2g_cond,
        "c_score":           initial_c,
        "evidence_strength": prior_evstr,
        "completeness":      round(initial_comp, 4),
        "repair_type":       "none",
    }

    logger.info(
        f"\n  [{qid}] Initial: C={initial_c:.3f} "
        f"faith={initial_faith:.3f} evstr={prior_evstr:.3f} comp={initial_comp:.3f}"
    )

    # Decide first repair
    first_repair = route(initial_faith, prior_evstr, initial_comp)
    logger.info(f"  [{qid}] Route → {first_repair}: {describe_route(first_repair)}")

    if first_repair is None:
        return {
            "question_id":   qid,
            "arxiv_id":      arxiv_id,
            "question":      question,
            "question_type": entry.get("question_type", ""),
            "gold_answer":   gold_answer,
            "initial":       initial_record,
            "iterations":    [],
            "final":         initial_record,
            "n_repairs":     0,
            "improved":      False,
            "repair_skipped_reason": "C above all thresholds",
        }

    # ── Iterative repair loop ─────────────────────────────────────────────────
    best_record  = initial_record
    best_c       = initial_c
    best_texts   = list(texts)
    best_ids     = list(chunk_ids)
    best_mods    = list(modalities)
    iterations   = []
    current_repair = first_repair

    for iteration in range(1, max_iterations + 1):
        if current_repair is None:
            logger.info(f"  [{qid}] Iter {iteration}: no repair needed → stopping")
            break

        logger.info(f"  [{qid}] Iter {iteration}/{max_iterations}: {current_repair}")
        t0 = time.time()

        try:
            # Get weak claim texts for gap repair
            weak_claims = []
            if current_repair in ("gap_repair", "full_escalation"):
                weak_claims = [
                    c.get("text", "") for c in r2g_cond.get("claims", [])
                    if c.get("status") in ("unsupported", "weak")
                ][:4]

            # Fire repair
            if current_repair == "faith_repair":
                output = faith_repair(llm, question, best_texts, best_ids, best_mods)
            elif current_repair == "evidence_repair":
                output = evidence_repair(
                    llm, text_embedder, store,
                    question, arxiv_id,
                    best_texts, best_ids, best_mods,
                    n_variants=n_variants, top_k=top_k,
                )
            elif current_repair == "gap_repair":
                output = gap_repair(
                    llm, text_embedder, store,
                    question, arxiv_id,
                    best_texts, best_ids, best_mods,
                    weak_claim_texts=weak_claims,
                    n_variants=n_variants, top_k=top_k,
                )
            elif current_repair == "full_escalation":
                output = full_escalation(
                    llm, text_embedder, store,
                    question, arxiv_id,
                    best_texts, best_ids, best_mods,
                    n_variants=n_variants, expanded_k=expanded_k,
                )
            else:
                break

        except Exception as e:
            logger.error(f"  [{qid}] Iter {iteration} repair crashed: {e}")
            iterations.append({"iteration": iteration, "repair_type": current_repair, "error": str(e)})
            break

        latency_ms = round((time.time() - t0) * 1000, 1)

        # Determine evidence pool for this iteration
        iter_texts = output._texts if hasattr(output, "_texts") else best_texts
        iter_ids   = output._chunk_ids if hasattr(output, "_chunk_ids") else best_ids
        iter_mods  = output._modalities if hasattr(output, "_modalities") else best_mods

        record = evaluate_repair(
            llm, question, gold_answer,
            iter_texts, iter_ids, iter_mods,
            output,
            prior_evstr  = prior_evstr,
            repair_type  = current_repair,
            latency_ms   = latency_ms,
        )
        record["iteration"] = iteration

        iterations.append(record)

        new_c = record["c_score"]
        logger.info(
            f"  [{qid}] Iter {iteration}: new_C={new_c:.3f} best_C={best_c:.3f} "
            f"Δ={new_c - best_c:+.3f}"
        )

        if new_c > best_c + MIN_C_IMPROVEMENT:
            logger.info(f"  [{qid}] Iter {iteration}: IMPROVED → accepting repair")
            best_record = record
            best_c      = new_c
            # Update evidence pool for next iteration if retrieval was done
            if current_repair != "faith_repair":
                # Extract merged texts from the output evidence
                pass  # best_texts/ids/mods already updated in repair functions
            # Route next repair from new component values
            current_repair = route(
                record["faithfulness"],
                record["evidence_strength"],
                record["completeness"],
            )
            logger.info(f"  [{qid}] Next route → {current_repair}: {describe_route(current_repair)}")
        else:
            logger.info(f"  [{qid}] Iter {iteration}: no improvement → keeping best")
            break

    improved = best_c > initial_c + MIN_C_IMPROVEMENT

    return {
        "question_id":   qid,
        "arxiv_id":      arxiv_id,
        "question":      question,
        "question_type": entry.get("question_type", ""),
        "gold_answer":   gold_answer,
        "initial":       initial_record,
        "iterations":    iterations,
        "final":         best_record,
        "n_repairs":     len(iterations),
        "improved":      improved,
    }
