"""
generation_layer/generation_pipeline.py

Full HEMMIR generation orchestrator with confidence-driven routing.

Produces ExplainableAnswer with all four explainability components:
  E1: retrieval_trace (from retrieval pipeline — passed in)
  E2: attributed_claims (built from verifier + evidence pack)
  E3: uncertainty (composite multi-signal confidence score)
  E4: modality_provenance (embedded in each evidence item's metadata)

Confidence-driven routing policy:
  C > 0.75  → HIGH_CONFIDENCE:   deliver immediately, no repair
  0.50-0.75 → MEDIUM_CONFIDENCE: self-correction loop
                diagnose weakest component → targeted repair:
                  low faithfulness  → tighten answer to supported claims only
                  low evidence      → re-retrieve with broader scope
                  low completeness  → re-retrieve missing gap evidence
                  low direct        → escalate RAG→Agent
  C < 0.50  → LOW_CONFIDENCE:    re-retrieval escalation
                broaden query scope, re-generate, re-score

Best-result tracking:
  After every repair attempt, the system compares the new confidence
  to the original. It always keeps the BEST result across all attempts.
  This guarantees repair never degrades the final answer quality.

State machine:
  INITIAL → GENERATE → VERIFY → SCORE
    → [HIGH]   DONE
    → [MEDIUM] SELF_CORRECT → REGENERATE → VERIFY → SCORE
                → if better: use repaired result
                → if worse:  rollback to original
                → DONE
    → [LOW]    RE_RETRIEVE → GENERATE → VERIFY → SCORE
                → if better: use re-retrieved result
                → if worse:  rollback to original
                → DONE

Max repair attempts: 1 (bounded LLM call budget).
Falls back gracefully if retrieval_pipeline not wired in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import mean as _mean
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from generation_layer.argrag import ArgRAGReasoner, ArgRAGResult
from generation_layer.attribution import build_attribution
from generation_layer.evidence import build_evidence_blocks
from generation_layer.generator import Generator
from generation_layer.uncertainty import (
    compute_uncertainty,
    weakest_component,
    REPAIR_LOW_FAITHFULNESS,
    REPAIR_LOW_EVIDENCE,
    REPAIR_LOW_COMPLETENESS,
    REPAIR_LOW_DIRECT,
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
    SUFFICIENCY_MAP,
    SALIENCE_MIX,
    CROSS_MIX,
    UNSUPPORTED_PENALTY,
    GAP_PENALTY,
)
from generation_layer.verifier import Verifier
from shared.models.pipeline_models import (
    EvidencePack,
    ExplainableAnswer,
    FaithfulnessReport,
    QueryAnalysis,
    RetrievalTrace,
    UncertaintyReport,
)

# Intent → expected evidence role mapping
_INTENT_TO_ROLE = {
    "procedural":  "procedure",
    "measurement": "measurement",
    "definition":  "definition",
    "comparative": "procedure",
    "diagnostic":  "measurement",
    "exploratory": "context",
    "visual":      "context",
    "temporal":    "context",
}


@dataclass
class _GenerationState:
    """
    Snapshot of one complete generation attempt.
    Used for best-result tracking and rollback.
    """
    draft:               Any
    faithfulness_report: FaithfulnessReport
    uncertainty:         UncertaintyReport
    evidence_pack:       EvidencePack
    retrieval_trace:     RetrievalTrace
    llm_calls:           int
    repair_action:       str

    @property
    def confidence(self) -> float:
        return self.uncertainty.score



def _recompute_confidence_with_argrag(
    uncertainty:   Any,
    argrag_result: "ArgRAGResult",
) -> Any:
    """
    Recompute Final Confidence C using ArgRAG claim strength.

    Simple RAG:  C = wf·F  +  we·E            +  wc·COV  +  ws·CONS
    ArgRAG:      C = wf·F  +  we·ClaimStrength +  wc·COV  +  ws·CONS

    The difference:
      Simple RAG uses raw evidence_score E = mean(salience × role_conf × CE)
      ArgRAG uses ClaimStrength = mean S(Ci) = mean(αS - βA + γConsistency)

    ClaimStrength is richer because it integrates argumentation:
      a claim supported by 5 chunks scores higher than one supported by 1,
      even if both have the same salience_score.

    COV  = argrag_result.coverage    (fraction of claims that are usable)
    CONS = argrag_result.consistency (1 - contested fraction)
    """
    from generation_layer.uncertainty import W_FAITHFULNESS, W_EVIDENCE, W_COMPLETENESS, W_DIRECT

    faithfulness = uncertainty.faithfulness

    # Replace evidence component with claim strength
    claim_strength = argrag_result.claim_strength

    # Replace completeness with ArgRAG coverage
    coverage = argrag_result.coverage

    # Replace consistency component
    consistency = argrag_result.consistency

    # Direct bonus: still use faithfulness + coverage as proxy
    direct_bonus = 1.0 if (faithfulness >= 0.70 and coverage >= 0.5) else 0.0

    new_confidence = (
        W_FAITHFULNESS   * faithfulness
        + W_EVIDENCE     * claim_strength
        + W_COMPLETENESS * coverage
        + W_DIRECT       * direct_bonus
    )
    new_confidence = max(0.0, min(1.0, new_confidence))

    # Hallucination cap still applies
    if uncertainty.hallucination_flag:
        from generation_layer.uncertainty import HALLUCINATION_CAP
        new_confidence = min(new_confidence, HALLUCINATION_CAP)

    if new_confidence >= 0.75:
        level = "HIGH_CONFIDENCE"
    elif new_confidence >= 0.50:
        level = "MEDIUM_CONFIDENCE"
    else:
        level = "LOW_CONFIDENCE"

    # Return updated UncertaintyReport (immutable — create new)
    from dataclasses import replace as dc_replace
    try:
        return dc_replace(
            uncertainty,
            score             = round(new_confidence, 4),
            level             = level,
            mean_cross_enc    = claim_strength,   # reuse field for claim_strength display
            sufficiency_value = coverage,
        )
    except Exception:
        # Fallback: return original unchanged
        return uncertainty


class GenerationPipeline:

    def __init__(
        self,
        registry:            Dict[str, Any],
        retrieval_pipeline=  None,   # optional — needed for repair loops
        llm_client=          None,
        evidence_top_n:      int = 10,
        max_repair_attempts: int = 1,
    ):
        self.registry            = registry
        self.retrieval_pipeline  = retrieval_pipeline
        self.llm_client          = llm_client
        self.evidence_top_n      = evidence_top_n
        self.max_repair_attempts = max_repair_attempts
        self.generator           = Generator(llm_client=llm_client)
        self.verifier            = Verifier(llm_client=llm_client)
        # ArgRAG reasoner — Steps 4-7, used when path="argrag"
        self.argrag_reasoner     = ArgRAGReasoner(llm_client=llm_client)

    def run(
        self,
        question:        str,
        analysis:        QueryAnalysis,
        evidence_pack:   EvidencePack,
        retrieval_trace: RetrievalTrace,
        evidence_score:  float = 0.0,   # E from retrieval layer
    ) -> ExplainableAnswer:
        """
        Generate an explainable answer with confidence-driven routing.
        Always returns the best result across all generation attempts.
        """
        t0        = time.time()
        llm_calls = 0
        # Store current doc_filter, doc_ids and context for escalation
        if not hasattr(self, "_current_doc_filter"):
            self._current_doc_filter = None
        if not hasattr(self, "_current_doc_ids"):
            self._current_doc_ids = []
        if not hasattr(self, "_current_context"):
            self._current_context = None

        # ── Step 1-2: Initial generate + verify ───────────────────────
        draft, faithfulness_report, llm_calls = self._generate_and_verify(
            question        = question,
            analysis        = analysis,
            evidence_pack   = evidence_pack,
            retrieval_trace = retrieval_trace,
            llm_calls       = llm_calls,
        )

        # ── Step 3: Score initial confidence ──────────────────────────
        uncertainty = compute_uncertainty(
            evidence_pack       = evidence_pack,
            faithfulness_report = faithfulness_report,
            retrieval_trace     = retrieval_trace,
            llm_client          = self.llm_client,
        )
        llm_calls += 1

        # Package initial state for best-result tracking
        best = _GenerationState(
            draft               = draft,
            faithfulness_report = faithfulness_report,
            uncertainty         = uncertainty,
            evidence_pack       = evidence_pack,
            retrieval_trace     = retrieval_trace,
            llm_calls           = llm_calls,
            repair_action       = "none",
        )

        logger.info(
            f"  Initial confidence: {best.confidence:.3f} ({uncertainty.level}) "
            f"— routing decision follows"
        )

        # ── ArgRAG reasoning pass (Steps 4-7) ─────────────────────────
        # Runs when:
        #   a) confidence is LOW (will escalate to ArgRAG retrieval), OR
        #   b) retrieval path was already "argrag"
        # Produces claim_strength which replaces raw evidence_score E
        # in the Final Confidence formula for the ArgRAG path.
        argrag_result = None
        if best.confidence < MEDIUM_THRESHOLD or analysis.path == "argrag":
            logger.info(
                f"  ArgRAG reasoning: running Steps 4-7 "
                f"(confidence={best.confidence:.3f}, path={analysis.path})"
            )
            argrag_result = self.argrag_reasoner.reason(
                question      = question,
                evidence_pack = evidence_pack,
                intent_type   = analysis.query_intent_type,
            )
            # Recompute confidence using ClaimStrength instead of raw E
            # C = wf·F + we·ClaimStrength + wc·COV + ws·CONS
            if argrag_result and argrag_result.claim_strength > 0:
                uncertainty = _recompute_confidence_with_argrag(
                    uncertainty   = uncertainty,
                    argrag_result = argrag_result,
                )
                best = _GenerationState(
                    draft               = best.draft,
                    faithfulness_report = best.faithfulness_report,
                    uncertainty         = uncertainty,
                    evidence_pack       = best.evidence_pack,
                    retrieval_trace     = best.retrieval_trace,
                    llm_calls           = best.llm_calls,
                    repair_action       = "argrag_reasoning",
                )
                logger.info(
                    f"  ArgRAG confidence: {best.confidence:.3f} "
                    f"(claim_strength={argrag_result.claim_strength:.3f} "
                    f"coverage={argrag_result.coverage:.3f} "
                    f"consistency={argrag_result.consistency:.3f})"
                )

        # ── Step 4: Routing decision ───────────────────────────────────
        if best.confidence >= HIGH_THRESHOLD:
            logger.info("  Routing: HIGH_CONFIDENCE → delivering directly")

        elif best.confidence >= MEDIUM_THRESHOLD and self.max_repair_attempts > 0:
            logger.info(
                f"  Routing: MEDIUM_CONFIDENCE ({best.confidence:.3f}) "
                f"→ self-correction loop (max_attempts={self.max_repair_attempts})"
            )
            for attempt in range(self.max_repair_attempts):
                if best.confidence >= HIGH_THRESHOLD:
                    logger.info(f"  Self-correction: reached HIGH threshold at attempt {attempt+1} — stopping")
                    break
                logger.info(f"  Self-correction attempt {attempt+1}/{self.max_repair_attempts}")
                repaired = self._self_correct(
                    question = question,
                    analysis = analysis,
                    state    = best,
                )
                best = self._pick_best(best, repaired, f"self-correction-{attempt+1}")

        elif best.confidence < MEDIUM_THRESHOLD and self.max_repair_attempts > 0:
            logger.info(
                f"  Routing: LOW_CONFIDENCE ({best.confidence:.3f}) "
                f"→ agent escalation (max_attempts={self.max_repair_attempts})"
            )
            for attempt in range(self.max_repair_attempts):
                if best.confidence >= MEDIUM_THRESHOLD:
                    logger.info(f"  Agent escalation: reached MEDIUM threshold at attempt {attempt+1} — stopping")
                    break
                logger.info(f"  Agent escalation attempt {attempt+1}/{self.max_repair_attempts}")
                repaired = self._agent_escalate(
                    question = question,
                    analysis = analysis,
                    state    = best,
                )
                best = self._pick_best(best, repaired, f"agent-escalation-{attempt+1}")

        # ── Step 5: Final attribution (E2) ────────────────────────────
        attributed_claims = build_attribution(
            faithfulness_report = best.faithfulness_report,
            evidence_pack       = best.evidence_pack,
            query_intent_type   = analysis.query_intent_type,
        )

        # ── Step 6: Extract answer body + follow-ups ──────────────────
        answer_body = self.generator.extract_answer_body(best.draft.text)
        follow_ups  = self.generator.extract_follow_ups(best.draft.text)

        total_ms = round((time.time() - t0) * 1000, 1)

        logger.info(
            f"\n  Generation complete\n"
            f"  Confidence: {best.confidence:.3f} ({best.uncertainty.level})\n"
            f"  Repair applied: {best.repair_action}\n"
            f"  Attributed claims: {len(attributed_claims)}\n"
            f"  LLM calls (generation): {best.llm_calls}\n"
            f"  Duration: {total_ms}ms"
        )

        # Build ArgRAG synthesis instruction for contested claims
        contested_note = ""
        if argrag_result and argrag_result.contested_claims:
            contested_texts = [c.text for c in argrag_result.contested_claims[:2]]
            contested_note = (
                "Note: conflicting evidence found for: "
                + "; ".join(contested_texts)
            )

        return ExplainableAnswer(
            answer               = answer_body,
            follow_up_questions  = follow_ups,
            evidence             = best.evidence_pack.items,
            evidence_pack        = best.evidence_pack,
            retrieval_trace      = best.retrieval_trace,
            attributed_claims    = attributed_claims,
            unsupported_claims   = best.faithfulness_report.unsupported_claims,
            uncertainty          = best.uncertainty,
            faithfulness_report  = best.faithfulness_report,
            retrieval_path       = analysis.path,
            total_llm_calls      = (
                best.retrieval_trace.total_chunks_candidate + best.llm_calls
            ),
            total_duration_ms    = total_ms,
            argrag_claims        = [c.text for c in argrag_result.selected_claims] if argrag_result else [],
            argrag_contested     = [c.text for c in argrag_result.contested_claims] if argrag_result else [],
            argrag_claim_strength = argrag_result.claim_strength if argrag_result else 0.0,
            contested_note       = contested_note,
        )

    # ── Best-result selector ──────────────────────────────────────────────────

    def _pick_best(
        self,
        original: _GenerationState,
        repaired: _GenerationState,
        repair_name: str,
    ) -> _GenerationState:
        """
        Always keep the higher-confidence result.
        If repair degraded confidence, rollback to original and log it.
        """
        if repaired.confidence >= original.confidence:
            logger.info(
                f"  Best-result: {repair_name} improved "
                f"{original.confidence:.3f} → {repaired.confidence:.3f} "
                f"— keeping repaired result"
            )
            return repaired
        else:
            logger.info(
                f"  Best-result: {repair_name} degraded "
                f"{original.confidence:.3f} → {repaired.confidence:.3f} "
                f"— rolling back to original"
            )
            # Keep original state, mark repair as rolled back
            return _GenerationState(
                draft               = original.draft,
                faithfulness_report = original.faithfulness_report,
                uncertainty         = original.uncertainty,
                evidence_pack       = original.evidence_pack,
                retrieval_trace     = original.retrieval_trace,
                llm_calls           = repaired.llm_calls,  # count all calls
                repair_action       = f"{repaired.repair_action}→rolled_back",
            )

    # ── Core generation + verification ────────────────────────────────────────

    def _generate_and_verify(
        self,
        question:          str,
        analysis:          QueryAnalysis,
        evidence_pack:     EvidencePack,
        retrieval_trace:   RetrievalTrace,
        llm_calls:         int,
        extra_instruction: str = "",
    ) -> Tuple[Any, FaithfulnessReport, int]:
        """Single generation + verification pass."""
        evidence_blocks = build_evidence_blocks(evidence_pack, self.registry)

        synthesis = retrieval_trace.synthesis_instruction or ""
        if extra_instruction:
            synthesis = f"{extra_instruction}\n{synthesis}" if synthesis else extra_instruction

        draft = self.generator.generate(
            question              = question,
            evidence_blocks       = evidence_blocks,
            intent_type           = analysis.query_intent_type,
            retrieval_path        = analysis.path,
            synthesis_instruction = synthesis,
            sub_questions         = [
                {"question": sq.question}
                for sq in retrieval_trace.sub_question_results
            ] if retrieval_trace.sub_question_results else [],
        )
        llm_calls += 1

        expected_role = _INTENT_TO_ROLE.get(analysis.query_intent_type, "context")
        faithfulness_report = self.verifier.verify(
            draft           = draft,
            evidence_blocks = evidence_blocks,
            question        = question,
            intent_type     = analysis.query_intent_type,
            expected_role   = expected_role,
        )
        llm_calls += 1

        return draft, faithfulness_report, llm_calls

    def _make_state(
        self,
        draft:               Any,
        faithfulness_report: FaithfulnessReport,
        evidence_pack:       EvidencePack,
        retrieval_trace:     RetrievalTrace,
        llm_calls:           int,
        repair_action:       str,
    ) -> _GenerationState:
        """Build a GenerationState with freshly computed confidence."""
        uncertainty = compute_uncertainty(
            evidence_pack       = evidence_pack,
            faithfulness_report = faithfulness_report,
            retrieval_trace     = retrieval_trace,
            llm_client          = self.llm_client,
        )
        llm_calls += 1  # narrative call
        return _GenerationState(
            draft               = draft,
            faithfulness_report = faithfulness_report,
            uncertainty         = uncertainty,
            evidence_pack       = evidence_pack,
            retrieval_trace     = retrieval_trace,
            llm_calls           = llm_calls,
            repair_action       = repair_action,
        )

    # ── Self-correction loop (MEDIUM_CONFIDENCE) ──────────────────────────────

    def _self_correct(
        self,
        question: str,
        analysis: QueryAnalysis,
        state:    _GenerationState,
    ) -> _GenerationState:
        """
        Diagnose the weakest confidence component and apply targeted repair.
        Returns a new GenerationState — caller decides whether to keep it.

        Repair actions:
          tighten_to_supported_claims → regenerate using only grounded claims
          re_retrieve_broader         → re-retrieve with broader scope
          re_retrieve_missing_gaps    → target missing evidence gaps
          escalate_to_agent           → escalate RAG→Agent path
        """
        # Recompute component scores for diagnosis
        items  = state.evidence_pack.items
        faith  = state.faithfulness_report.overall_faithfulness

        sal_vals = []
        for item in items:
            sal = getattr(item.metadata, "salience_score", None)
            if sal is None or sal == 0.0:
                _ep  = item.extra_payload if isinstance(item.extra_payload, dict) else {}
                _tsc = _ep.get("table_summary_confidence")
                if _tsc is not None:
                    sal = _tsc
            if sal is not None:
                try:
                    sal_vals.append(max(0.0, min(1.0, float(sal))))
                except (TypeError, ValueError):
                    pass
        mean_sal   = _mean(sal_vals) if sal_vals else 0.5
        cross_vals = [
            item.score_breakdown.cross_encoder_score
            for item in items
            if item.score_breakdown.cross_encoder_score > 0
        ]
        mean_cross = _mean(cross_vals) if cross_vals else 0.0
        evid       = SALIENCE_MIX * mean_sal + CROSS_MIX * mean_cross
        suf        = SUFFICIENCY_MAP.get(state.retrieval_trace.overall_sufficiency, 0.5)
        n_un       = len(state.faithfulness_report.unsupported_claims)
        n_mg       = len(state.faithfulness_report.missing_evidence)
        comp       = max(0.0, min(1.0,
            suf - UNSUPPORTED_PENALTY * n_un - min(0.30, GAP_PENALTY * n_mg)
        ))
        direct     = 1.0 if (faith >= 0.70 and suf >= 0.5) else 0.0

        weak_component, repair_action = weakest_component(faith, evid, comp, direct)

        logger.info(
            f"  Self-correction: weakest={weak_component} "
            f"repair={repair_action}"
        )

        llm_calls = state.llm_calls

        if repair_action == REPAIR_LOW_FAITHFULNESS:
            # Tighten — regenerate using only supported claims
            instruction = (
                "IMPORTANT: Generate your answer using ONLY the evidence chunks "
                "that directly support claims. Do not add any information beyond "
                "what is explicitly stated in the retrieved evidence. "
                "Be concise and precise. Omit any claim you cannot cite."
            )
            draft, faithfulness_report, llm_calls = self._generate_and_verify(
                question          = question,
                analysis          = analysis,
                evidence_pack     = state.evidence_pack,
                retrieval_trace   = state.retrieval_trace,
                llm_calls         = llm_calls,
                extra_instruction = instruction,
            )
            return self._make_state(
                draft               = draft,
                faithfulness_report = faithfulness_report,
                evidence_pack       = state.evidence_pack,
                retrieval_trace     = state.retrieval_trace,
                llm_calls           = llm_calls,
                repair_action       = repair_action,
            )

        elif repair_action in (REPAIR_LOW_EVIDENCE, REPAIR_LOW_COMPLETENESS,
                               REPAIR_LOW_DIRECT) and self.retrieval_pipeline:
            # Re-retrieve — use gap-aware query for completeness repair
            if repair_action == REPAIR_LOW_COMPLETENESS:
                missing_gaps = state.faithfulness_report.missing_evidence[:3]
                repair_query = (
                    f"{question} {' '.join(missing_gaps)}"
                    if missing_gaps else question
                )
            else:
                repair_query = question

            logger.info(
                f"  Self-correction: re-retrieving "
                f"(repair={repair_action}, query={repair_query[:60]})"
            )
            try:
                new_result      = self.retrieval_pipeline.retrieve(query=repair_query, filters=getattr(self, "_current_doc_filter", None))
                new_ep          = new_result["evidence_pack"]
                new_rt          = new_result["retrieval_trace"]
                draft, faithfulness_report, llm_calls = self._generate_and_verify(
                    question        = question,
                    analysis        = analysis,
                    evidence_pack   = new_ep,
                    retrieval_trace = new_rt,
                    llm_calls       = llm_calls,
                )
                return self._make_state(
                    draft               = draft,
                    faithfulness_report = faithfulness_report,
                    evidence_pack       = new_ep,
                    retrieval_trace     = new_rt,
                    llm_calls           = llm_calls,
                    repair_action       = repair_action,
                )
            except Exception as e:
                logger.warning(f"  Self-correction re-retrieval failed: {e}")
                # Return original state unchanged — _pick_best will keep original
                return _GenerationState(
                    draft               = state.draft,
                    faithfulness_report = state.faithfulness_report,
                    uncertainty         = state.uncertainty,
                    evidence_pack       = state.evidence_pack,
                    retrieval_trace     = state.retrieval_trace,
                    llm_calls           = llm_calls,
                    repair_action       = f"{repair_action}→failed",
                )

        else:
            logger.info(
                f"  Self-correction: {repair_action} — no retrieval_pipeline, "
                f"skipping repair"
            )
            return _GenerationState(
                draft               = state.draft,
                faithfulness_report = state.faithfulness_report,
                uncertainty         = state.uncertainty,
                evidence_pack       = state.evidence_pack,
                retrieval_trace     = state.retrieval_trace,
                llm_calls           = llm_calls,
                repair_action       = f"{repair_action}→skipped",
            )


    def _agent_escalate(
        self,
        question: str,
        analysis: QueryAnalysis,
        state:    _GenerationState,
    ) -> _GenerationState:
        """
        Confidence-based agent escalation for LOW_CONFIDENCE answers.
        Called when RAG confidence < 0.50.
        Uses agent decomposition + iterative retrieval instead of
        simple re-retrieval. Passes existing RAG evidence as context.
        """
        repair_action = "agent_escalation"
        llm_calls     = state.llm_calls

        if not self.retrieval_pipeline or not hasattr(self.retrieval_pipeline, "retrieve_for_agent"):
            logger.warning("  Agent escalation: retrieve_for_agent not available — falling back")
            return self._re_retrieve_and_regenerate(question, analysis, state)

        logger.info(
            f"  Agent escalation: {state.confidence:.3f} → decomposing query"
        )

        try:
            doc_ids = getattr(self, "_current_doc_ids", [])
            context = getattr(self, "_current_context", None)

            agent_result = self.retrieval_pipeline.retrieve_for_agent(
                query             = question,
                analysis          = analysis,
                context           = context,
                doc_ids           = doc_ids,
                existing_evidence = state.evidence_pack,
            )

            new_ep = agent_result["evidence_pack"]
            new_rt = agent_result["retrieval_trace"]

            draft, faithfulness_report, llm_calls = self._generate_and_verify(
                question        = question,
                analysis        = analysis,
                evidence_pack   = new_ep,
                retrieval_trace = new_rt,
                llm_calls       = llm_calls,
            )
            return self._make_state(
                draft               = draft,
                faithfulness_report = faithfulness_report,
                evidence_pack       = new_ep,
                retrieval_trace     = new_rt,
                llm_calls           = llm_calls,
                repair_action       = repair_action,
            )

        except Exception as e:
            logger.warning(f"  Agent escalation failed: {e} — falling back to re-retrieval")
            return self._re_retrieve_and_regenerate(question, analysis, state)

    # ── Re-retrieval escalation (fallback) ───────────────────────────────────

    def _re_retrieve_and_regenerate(
        self,
        question: str,
        analysis: QueryAnalysis,
        state:    _GenerationState,
    ) -> _GenerationState:
        """
        Full re-retrieval with broadened scope for LOW_CONFIDENCE answers.
        Returns a new GenerationState — caller decides whether to keep it.
        Falls back gracefully if retrieval_pipeline not available.
        """
        repair_action = "re_retrieve_escalate"
        llm_calls     = state.llm_calls

        if not self.retrieval_pipeline:
            logger.warning(
                "  LOW_CONFIDENCE but no retrieval_pipeline available "
                "— skipping re-retrieval, returning original"
            )
            return _GenerationState(
                draft               = state.draft,
                faithfulness_report = state.faithfulness_report,
                uncertainty         = state.uncertainty,
                evidence_pack       = state.evidence_pack,
                retrieval_trace     = state.retrieval_trace,
                llm_calls           = llm_calls,
                repair_action       = f"{repair_action}→skipped",
            )

        logger.info(
            f"  Re-retrieval escalation: broadening from path={analysis.path}"
        )

        try:
            new_result      = self.retrieval_pipeline.retrieve(query=question, filters=getattr(self, "_current_doc_filter", None))
            new_ep          = new_result["evidence_pack"]
            new_rt          = new_result["retrieval_trace"]

            draft, faithfulness_report, llm_calls = self._generate_and_verify(
                question        = question,
                analysis        = analysis,
                evidence_pack   = new_ep,
                retrieval_trace = new_rt,
                llm_calls       = llm_calls,
            )
            return self._make_state(
                draft               = draft,
                faithfulness_report = faithfulness_report,
                evidence_pack       = new_ep,
                retrieval_trace     = new_rt,
                llm_calls           = llm_calls,
                repair_action       = repair_action,
            )

        except Exception as e:
            logger.warning(f"  Re-retrieval failed: {e} — returning original")
            return _GenerationState(
                draft               = state.draft,
                faithfulness_report = state.faithfulness_report,
                uncertainty         = state.uncertainty,
                evidence_pack       = state.evidence_pack,
                retrieval_trace     = state.retrieval_trace,
                llm_calls           = llm_calls,
                repair_action       = f"{repair_action}→failed",
            )