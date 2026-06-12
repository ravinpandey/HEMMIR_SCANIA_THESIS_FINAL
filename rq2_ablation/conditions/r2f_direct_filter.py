"""
rq2_ablation/conditions/r2f_direct_filter.py

R2-F: Direct LLM Evidence Filter
─────────────────────────────────
Addresses the reviewer challenge: "why not just ask the LLM which evidence
supports the query and generate from that?"

Pipeline:
  Step 1 — LLM classifies each retrieved chunk as SUPPORTING / NEUTRAL / CONTRADICTING
  Step 2 — Keep only SUPPORTING chunks
  Step 3 — Generate answer from filtered evidence (same prompt as R2-A)

Key difference from R2-E (ArgRAG):
  - Operates at chunk level, not claim level
  - No claim decomposition — whole-answer filtering, not atomic-claim filtering
  - Single LLM call for filtering (vs per-claim support + attack calls in R2-E)

This condition isolates whether chunk-level LLM filtering provides similar
benefits to ArgRAG's claim-level argumentation scoring.
"""

from __future__ import annotations

import json
import re
from typing import List

from loguru import logger

from .base import ConditionOutput, format_evidence

# ── Prompts ───────────────────────────────────────────────────────────────────

FILTER_SYSTEM = """\
You are a precise evidence relevance classifier for technical and industrial document QA.
Documents include scientific papers, PPTX presentations, service manuals, and industrial reports.
Classify each evidence chunk's relationship to the given question.
IMPORTANT: Image descriptions, vision captions, and diagram annotations count as SUPPORTING
when they contain visual information that helps answer the question — even if phrased descriptively.
Respond ONLY in valid JSON as instructed. No markdown, no preamble.\
"""

FILTER_PROMPT = """\
For the question below, classify each evidence chunk as one of:
  SUPPORTING   — directly and specifically helps answer the question
  NEUTRAL      — related to the topic but does not answer the question
  CONTRADICTING — contradicts or undermines a likely answer

Question: {question}

Evidence chunks:
{evidence_list}

Return ONLY this JSON (no markdown):
{{"classifications": [{{"id": "E1", "label": "SUPPORTING|NEUTRAL|CONTRADICTING", "reason": "<one sentence>"}}]}}\
"""

GENERATE_SYSTEM = """\
You are a precise scientific QA assistant.
Answer questions using ONLY the provided supporting evidence.
Be concise and factual. Do not add information not present in the evidence.\
"""

GENERATE_PROMPT = """\
Answer this question using ONLY the supporting evidence below.

Question: {question}

Supporting evidence:
{evidence}

Requirements:
- Use only information present in the evidence
- Be concise (2-5 sentences)
- If evidence is insufficient, begin with "Insufficient evidence:"

Answer:\
"""


def _parse_classifications(text: str) -> dict:
    clean = re.sub(r"```(?:json)?", "", text).strip()
    start = clean.find("{")
    end   = clean.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(clean[start:end])
        except Exception:
            pass
    return {}


def run(
    llm,
    question:   str,
    texts:      List[str],
    chunk_ids:  List[str],
    modalities: List[str],
) -> ConditionOutput:

    # ── Step 1: Build numbered evidence list for classification ───────────────
    evidence_lines = []
    for i, (text, cid, mod) in enumerate(zip(texts, chunk_ids, modalities), 1):
        evidence_lines.append(f"E{i} ({mod}): {text[:300]}")
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
        classifications = data.get("classifications", [])

        for item in classifications:
            eid   = str(item.get("id", "")).strip()   # "E1", "E2", ...
            label = str(item.get("label", "")).strip().upper()
            if label == "SUPPORTING" and eid.startswith("E"):
                try:
                    idx = int(eid[1:]) - 1   # E1 → index 0
                    if 0 <= idx < len(texts):
                        supporting_indices.append(idx)
                except ValueError:
                    pass

        logger.info(
            f"  R2-F: {len(supporting_indices)}/{len(texts)} chunks classified SUPPORTING"
        )
    except Exception as e:
        logger.warning(f"  R2-F: classification failed ({e}), falling back to all chunks")
        supporting_indices = list(range(len(texts)))

    # ── Step 2: Filter to supporting chunks only ──────────────────────────────
    if not supporting_indices:
        # No supporting evidence found — abstain
        return ConditionOutput(
            condition="R2-F",
            answer="Insufficient evidence to answer.",
            abstained=True,
            coverage_score=0.0,
        )

    sup_texts      = [texts[i]      for i in supporting_indices]
    sup_chunk_ids  = [chunk_ids[i]  for i in supporting_indices]
    sup_modalities = [modalities[i] for i in supporting_indices]

    coverage = len(supporting_indices) / max(1, len(texts))

    # ── Step 3: Generate from supporting evidence only ────────────────────────
    filtered_evidence = format_evidence(sup_texts, sup_chunk_ids, sup_modalities)
    gen_prompt = GENERATE_PROMPT.format(
        question=question,
        evidence=filtered_evidence[:4000],
    )

    try:
        answer = llm.invoke(
            system=GENERATE_SYSTEM,
            prompt=gen_prompt,
            max_tokens=400,
            temperature=0,
        ).strip()
        abstained = answer.lower().startswith("insufficient evidence")
        return ConditionOutput(
            condition="R2-F",
            answer=answer,
            abstained=abstained,
            coverage_score=round(coverage, 4),
        )
    except Exception as e:
        logger.error(f"  R2-F generation failed: {e}")
        return ConditionOutput(
            condition="R2-F",
            answer="",
            error=str(e),
        )
