"""
generation_layer/verifier.py

Role-aware faithfulness verifier.

Upgraded from V4:
  - supported_claims now carry evidence_role and is_direct flag
  - unsupported_claims carry reason and suggested_search
  - role_mismatches: NEW — flags when claim is supported by wrong evidence type
  - contains_hallucination: NEW — explicit flag for generation pipeline
  - overall_faithfulness: weighted combination of supported ratio + role quality

The role_mismatches array feeds directly into the uncertainty layer —
  a significant role mismatch lowers the uncertainty score.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from loguru import logger

from shared.models.pipeline_models import (
    DraftAnswer,
    FaithfulnessReport,
    RoleMismatch,
    SupportedClaim,
    UnsupportedClaim,
)

VERIFIER_SYSTEM = """\
You are a faithfulness auditor for an industrial document QA system.
Verify every claim in the generated answer is:
  1. Directly stated in the retrieved evidence (not hallucinated or inferred beyond evidence)
  2. Supported by the RIGHT TYPE of evidence (procedure chunk for procedures, etc.)
  3. Correctly cited to the chunk that contains the information

Be strict — a claim is unsupported if it requires inference beyond what evidence states.\
"""

VERIFIER_PROMPT = """\
Verify the faithfulness of this answer against the evidence.

Question: {question}
Query intent type: {intent_type}
Expected evidence role: {expected_role}

Generated answer:
{draft_answer}

Evidence chunks:
{evidence_blocks}

For each factual claim in the answer:
1. Is it directly stated in the evidence? (not inferred)
2. Is the cited chunk_id correct?
3. Is the evidence the right TYPE for this claim?

Respond in EXACTLY this JSON (no markdown):
{{
  "supported_claims": [
    {{
      "claim": "<exact claim text>",
      "supporting_chunk_id": "<chunk_id>",
      "evidence_role": "<role of supporting chunk>",
      "is_direct": true,
      "citation_correct": true
    }}
  ],
  "unsupported_claims": [
    {{
      "claim": "<claim text>",
      "reason": "<why unsupported>",
      "suggested_search": "<what to search for>"
    }}
  ],
  "role_mismatches": [
    {{
      "claim": "<claim text>",
      "expected_role": "<needed role>",
      "actual_role": "<chunk role>",
      "impact": "minor|significant"
    }}
  ],
  "missing_evidence": ["<information not in evidence but needed>"],
  "overall_faithfulness": 0.0,
  "contains_hallucination": false
}}\
"""


class Verifier:

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def verify(
        self,
        draft:           DraftAnswer,
        evidence_blocks: str,
        question:        str        = "",
        intent_type:     str        = "definition",
        expected_role:   str        = "context",
    ) -> FaithfulnessReport:
        """
        Verify draft answer faithfulness against evidence.

        Args:
            draft:           Generated answer draft
            evidence_blocks: Formatted evidence text from plugins
            question:        Original user question (for context)
            intent_type:     From QueryAnalysis — for role checking
            expected_role:   What evidence role was expected

        Returns:
            FaithfulnessReport with structured verification results
        """
        if draft.abstained:
            return FaithfulnessReport(
                overall_faithfulness = 0.0,
                missing_evidence     = ["Generation abstained — no evidence retrieved"],
            )

        if not self.llm:
            return self._heuristic_verify(draft)

        prompt = VERIFIER_PROMPT.format(
            question       = question or "Not provided",
            intent_type    = intent_type,
            expected_role  = expected_role,
            draft_answer   = draft.text[:3500],
            evidence_blocks = evidence_blocks[:5000],
        )
        try:
            response = self.llm.invoke(
                system     = VERIFIER_SYSTEM,
                prompt     = prompt,
                max_tokens = 1500,
            )
            result = self._parse_response(response)
            # Retry with simpler prompt if JSON parse failed
            if not result.supported_claims and not result.unsupported_claims:
                logger.info("  Verifier: retrying with simplified prompt")
                simple_prompt = (
                    f"Question: {question}\n\n"
                    f"Answer to verify:\n{draft.text[:2000]}\n\n"
                    f"Evidence:\n{evidence_blocks[:3000]}\n\n"
                    "List each cited claim as supported or unsupported. "
                    "Respond in EXACTLY this JSON (no markdown, no extra text):\n"
                    '{"supported_claims": [{"claim": "...", "supporting_chunk_id": "...", '
                    '"evidence_role": "context", "is_direct": true, "citation_correct": true}], '
                    '"unsupported_claims": [], "role_mismatches": [], "missing_evidence": [], '
                    '"overall_faithfulness": 0.8, "contains_hallucination": false}'
                )
                response2 = self.llm.invoke(
                    system     = VERIFIER_SYSTEM,
                    prompt     = simple_prompt,
                    max_tokens = 1200,
                )
                result = self._parse_response(response2)
            return result
        except Exception as e:
            logger.warning(f"  Verifier LLM failed: {e} — using heuristic")
            return self._heuristic_verify(draft)

    def _parse_response(self, response: str) -> FaithfulnessReport:
        data = _extract_json(response)
        if not data:
            logger.warning("  Verifier: no JSON in response — heuristic fallback")
            return FaithfulnessReport(overall_faithfulness=0.5)

        supported = [
            SupportedClaim(
                claim               = item.get("claim", ""),
                supporting_chunk_id = item.get("supporting_chunk_id", ""),
                evidence_role       = item.get("evidence_role", ""),
                is_direct           = bool(item.get("is_direct", True)),
                citation_correct    = bool(item.get("citation_correct", True)),
            )
            for item in (data.get("supported_claims") or [])
            if isinstance(item, dict) and item.get("claim")
        ]

        unsupported = [
            UnsupportedClaim(
                claim            = item.get("claim", ""),
                reason           = item.get("reason", ""),
                suggested_search = item.get("suggested_search", ""),
            )
            for item in (data.get("unsupported_claims") or [])
            if isinstance(item, dict) and item.get("claim")
        ]

        role_mismatches = [
            RoleMismatch(
                claim         = item.get("claim", ""),
                expected_role = item.get("expected_role", ""),
                actual_role   = item.get("actual_role", ""),
                impact        = item.get("impact", "minor"),
            )
            for item in (data.get("role_mismatches") or [])
            if isinstance(item, dict) and item.get("claim")
        ]

        faithfulness = float(data.get("overall_faithfulness", 0.5))
        hallucination = bool(data.get("contains_hallucination", False))

        logger.info(
            f"  Verifier: supported={len(supported)} unsupported={len(unsupported)} "
            f"role_mm={len(role_mismatches)} hallucination={hallucination} "
            f"faithfulness={faithfulness:.2f}"
        )

        return FaithfulnessReport(
            supported_claims    = supported,
            unsupported_claims  = unsupported,
            role_mismatches     = role_mismatches,
            missing_evidence    = list(data.get("missing_evidence") or []),
            overall_faithfulness = faithfulness,
            contains_hallucination = hallucination,
        )

    def _heuristic_verify(self, draft: DraftAnswer) -> FaithfulnessReport:
        """Simple heuristic when LLM is unavailable."""
        n_cited = len(draft.cited_chunk_ids)
        has_insufficient = "INSUFFICIENT EVIDENCE" in (draft.text or "").upper()

        if has_insufficient:
            return FaithfulnessReport(
                overall_faithfulness = 0.1,
                missing_evidence     = ["LLM flagged insufficient evidence"],
            )

        faithfulness = min(0.9, 0.4 + 0.1 * n_cited) if n_cited else 0.3
        return FaithfulnessReport(
            overall_faithfulness = faithfulness,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

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