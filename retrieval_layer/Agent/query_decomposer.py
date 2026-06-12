"""
retrieval_layer/Agent/query_decomposer.py

Decomposes complex queries into atomic sub-questions for structured retrieval.

Each sub-question carries:
  - question text
  - evidence_role: what type of chunk answers it (procedure/measurement/etc.)
  - preferred_modality: which collection to search
  - section_hint: keyword to guide sections_collection navigation
  - is_prerequisite_for: ordering dependencies

The decomposition is structured so the Section Navigator can immediately
use each sub-question's section_hint without further LLM calls.

Maximum 4 sub-questions — keeps agent LLM call budget bounded.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from loguru import logger


DECOMPOSER_SYSTEM = """\
You are an expert at decomposing complex industrial document queries into \
atomic sub-questions for structured retrieval.

Each sub-question must be:
  1. Independently answerable from a single document section or scene
  2. Specific enough to retrieve a particular type of evidence
  3. Tagged with the evidence type it requires

Document structure context:
  - Technical manuals: numbered sections (e.g. "3.2 Hydraulic Pump Maintenance")
  - Training videos: scenes (e.g. "Scene 3: Engine Assembly Demonstration")
  - Spreadsheets: sheets (e.g. "FaultCodes", "MaintenanceLog")
  - Presentations: slides (e.g. "Slide 7: Safety Requirements")

You are precise and conservative — never generate more sub-questions than necessary.\
"""

DECOMPOSER_PROMPT = """\
Decompose this complex query into atomic sub-questions.

Original query: "{query}"
Query intent type: {intent_type}
Section hint from analysis: {section_hint}

For each sub-question specify exactly:
  - The question (answerable from ONE section/scene/sheet)
  - evidence_role: what type of chunk answers it
  - preferred_modality: which index to search first
  - section_hint: keyword a human expert would use to find the section
  - is_prerequisite_for: index (0-based) of sub-question this must answer first

Respond in EXACTLY this JSON (no markdown):
{{
  "sub_questions": [
    {{
      "question": "<specific atomic sub-question>",
      "evidence_role": "procedural|measurement|definition|comparative|diagnostic",
      "preferred_modality": "text|image|table|video_segment|video_frame",
      "section_hint": "<keyword>",
      "is_prerequisite_for": null
    }}
  ],
  "synthesis_instruction": "<one sentence: how to combine sub-answers>",
  "requires_cross_document": false,
  "decomposition_confidence": 0.0
}}

Rules:
- Maximum 4 sub-questions
- Order prerequisites first
- section_hint must be a natural keyword (not a section number)
- decomposition_confidence: 0.9=clean decomposition, 0.5=some ambiguity
- preferred_modality MUST be "table" when sub-question asks for numerical values, complexity formulas, Big-O notation, performance metrics, BLEU scores, benchmark results, specifications, tolerances, or any quantitative comparison between approaches
- preferred_modality MUST be "image" when asking for diagrams, figures, or visualizations
- preferred_modality MUST be "text" for definitions, explanations, and procedures\
"""


class QueryDecomposer:
    """
    Decomposes a complex query into structured sub-questions.
    Falls back to using sub_questions from QueryAnalysis if LLM fails
    (LLM analysis may have already produced decomposition).
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def decompose(
        self,
        query:            str,
        intent_type:      str,
        section_hint:     Optional[str]          = None,
        existing_sub_qs:  Optional[List[Dict]]   = None,
    ) -> tuple[List[Dict[str, Any]], str, float]:
        """
        Decompose query into sub-questions.

        Args:
            query:           Original user query
            intent_type:     From QueryAnalysis.query_intent_type
            section_hint:    Navigation hint from QueryAnalysis
            existing_sub_qs: Pre-generated sub-questions from QueryAnalysis LLM call

        Returns:
            (sub_questions, synthesis_instruction, decomposition_confidence)
        """
        # Prefer sub-questions from the query analyser if they are rich enough
        if existing_sub_qs and len(existing_sub_qs) >= 1:
            all_have_role = all(sq.get("evidence_role") for sq in existing_sub_qs)
            if all_have_role:
                logger.debug(
                    f"  Decomposer: using {len(existing_sub_qs)} sub-questions "
                    "from query analyser"
                )
                return (
                    existing_sub_qs,
                    "Synthesise all sub-answers into a unified response.",
                    0.85,
                )

        # Fresh decomposition LLM call
        try:
            return self._llm_decompose(query, intent_type, section_hint)
        except Exception as e:
            logger.warning(f"  Decomposer LLM failed: {e} — using fallback")
            return self._fallback_decompose(query, intent_type, section_hint)

    def _llm_decompose(
        self,
        query:        str,
        intent_type:  str,
        section_hint: Optional[str],
    ) -> tuple[List[Dict], str, float]:
        prompt = DECOMPOSER_PROMPT.format(
            query        = query,
            intent_type  = intent_type,
            section_hint = section_hint or "none",
        )
        response = self.llm.invoke(
            system     = DECOMPOSER_SYSTEM,
            prompt     = prompt,
            max_tokens = 600,
        )
        data = _extract_json(response)
        if not data or not data.get("sub_questions"):
            raise ValueError("No sub_questions in decomposer response")

        sub_qs     = data["sub_questions"][:4]   # enforce max 4
        synthesis  = data.get("synthesis_instruction", "Synthesise all sub-answers.")
        confidence = float(data.get("decomposition_confidence", 0.7))

        logger.info(f"  Decomposer: {len(sub_qs)} sub-questions generated")
        for i, sq in enumerate(sub_qs):
            logger.debug(
                f"    [{i}] {sq.get('question','')[:60]} "
                f"| role={sq.get('evidence_role','')} "
                f"| hint={sq.get('section_hint','')}"
            )

        return sub_qs, synthesis, confidence

    def _fallback_decompose(
        self,
        query:        str,
        intent_type:  str,
        section_hint: Optional[str],
    ) -> tuple[List[Dict], str, float]:
        """Rule-based fallback when LLM is unavailable."""
        sub_qs = [{
            "question":          query,
            "evidence_role":     _intent_to_role(intent_type),
            "preferred_modality": "text",
            "section_hint":      section_hint or _extract_keywords(query),
            "is_prerequisite_for": None,
        }]
        return sub_qs, "Answer directly from retrieved evidence.", 0.3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _intent_to_role(intent_type: str) -> str:
    mapping = {
        "procedural":  "procedure",
        "measurement": "measurement",
        "definition":  "definition",
        "comparative": "procedure",
        "diagnostic":  "measurement",
        "exploratory": "context",
        "visual":      "context",
        "temporal":    "context",
    }
    return mapping.get(intent_type, "context")


def _extract_keywords(query: str) -> str:
    stop = {"what","is","are","the","a","an","how","for","in","of","to"}
    words = [w for w in re.findall(r'\b\w+\b', query.lower()) if w not in stop]
    return " ".join(words[:4])


def _extract_json(text: str):
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    candidate = fenced.group(1).strip() if fenced else text
    start = candidate.find("{")
    if start == -1:
        return None
    depth = 0
    end   = -1
    for i, ch in enumerate(candidate[start:], start=start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    try:
        result = json.loads(candidate[start:end+1])
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None
