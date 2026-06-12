"""
rq2_ablation/conditions/r2a_direct_rag.py

R2-A: Direct RAG Baseline
─────────────────────────
Generates an answer directly from retrieved evidence.
No claim decomposition. No support/attack/coverage scoring.
This is the baseline all other conditions are compared against.
"""

from __future__ import annotations

from typing import List

from loguru import logger

from .base import ConditionOutput, format_evidence

DIRECT_RAG_SYSTEM = """\
You are a precise scientific QA assistant.
Answer questions using ONLY the provided evidence from scientific papers.
Be concise and factual. Do not add information not present in the evidence.
If the evidence is insufficient, say so explicitly.\
"""

DIRECT_RAG_PROMPT = """\
Answer this question using ONLY the evidence below.

Question: {question}

Evidence:
{evidence}

Requirements:
- Use only information present in the evidence
- Be concise (2-5 sentences)
- If evidence is insufficient, begin with "Insufficient evidence:"

Answer:\
"""


def run(
    llm,
    question:   str,
    texts:      List[str],
    chunk_ids:  List[str],
    modalities: List[str],
) -> ConditionOutput:
    evidence_text = format_evidence(texts, chunk_ids, modalities)
    prompt = DIRECT_RAG_PROMPT.format(
        question=question,
        evidence=evidence_text[:4000],
    )
    try:
        answer = llm.invoke(
            system=DIRECT_RAG_SYSTEM,
            prompt=prompt,
            max_tokens=400,
            temperature=0,
        ).strip()
        abstained = answer.lower().startswith("insufficient evidence")
        return ConditionOutput(
            condition="R2-A",
            answer=answer,
            abstained=abstained,
            coverage_score=1.0,
        )
    except Exception as e:
        logger.error(f"  R2-A failed: {e}")
        return ConditionOutput(
            condition="R2-A",
            answer="",
            error=str(e),
        )
