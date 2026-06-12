"""
rq2_ablation/judge.py

LLM-as-judge for RQ2 evaluation.

Two independent judges run at temperature=0 on every generated answer:

  1. faithfulness_judge:
     Scores how well every claim in the answer is supported by
     the retrieved evidence chunks (0-1). Independent of correctness.
     An answer can be faithful (grounded in evidence) but still wrong
     if the retrieved evidence itself was irrelevant.

  2. correctness_judge:
     Scores how well the generated answer matches the gold free-form
     answer from SPIQA (0-1). Captures factual accuracy.

Binary threshold for McNemar test: FAITHFUL_THRESHOLD = 0.7
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

from loguru import logger

FAITHFUL_THRESHOLD = 0.7   # >= this → faithful for McNemar binary test


# ── System prompts ────────────────────────────────────────────────────────────

FAITH_SYSTEM = """\
You are a strict scientific QA faithfulness evaluator.
Your job is to determine whether every factual claim in a generated answer
is directly supported by the provided evidence passages.
You respond ONLY in valid JSON. No markdown, no preamble.\
"""

CORRECT_SYSTEM = """\
You are a scientific QA correctness evaluator.
Your job is to determine how well a generated answer matches a gold reference answer.
You respond ONLY in valid JSON. No markdown, no preamble.\
"""

# ── Faithfulness judge ────────────────────────────────────────────────────────

FAITH_PROMPT = """\
Evaluate the faithfulness of this generated answer to the retrieved evidence.

Question: {question}

Retrieved evidence:
{evidence}

Generated answer:
{answer}

Definition of faithfulness:
- A claim is FAITHFUL if it is directly and specifically supported by the evidence above.
- A claim is UNFAITHFUL if it adds information not present in the evidence,
  contradicts the evidence, or makes claims the evidence does not confirm.
- Partial support counts as partial faithfulness (score 0.3-0.6).

Score 1.0 = every sentence is grounded in the evidence.
Score 0.0 = the answer is entirely hallucinated or contradicted by evidence.

Also count sentences/claims that are unfaithful.

Return ONLY this JSON (no markdown):
{{
  "faithfulness": <0.0-1.0>,
  "unsupported_sentences": <integer count>,
  "verdict": "<one sentence explaining the score>",
  "is_faithful": <true if faithfulness >= {threshold}>
}}\
"""

def judge_faithfulness(
    llm,
    question:   str,
    evidence:   str,
    answer:     str,
) -> Dict:
    """
    Score how well the answer is grounded in the evidence.

    Returns dict with:
      faithfulness       float 0-1
      unsupported_sentences  int
      verdict            str
      is_faithful        bool  (faithfulness >= FAITHFUL_THRESHOLD)
    """
    if not answer or not answer.strip():
        return {
            "faithfulness": 0.0,
            "unsupported_sentences": 0,
            "verdict": "empty answer",
            "is_faithful": False,
        }

    # Abstained answers: "Insufficient evidence to answer." — treat as faithful
    if answer.strip().lower().startswith("insufficient evidence"):
        return {
            "faithfulness": 1.0,
            "unsupported_sentences": 0,
            "verdict": "abstained — no claims to evaluate",
            "is_faithful": True,
        }

    prompt = FAITH_PROMPT.format(
        question=question,
        evidence=evidence[:4000],
        answer=answer[:1500],
        threshold=FAITHFUL_THRESHOLD,
    )
    try:
        resp = llm.invoke(
            system=FAITH_SYSTEM,
            prompt=prompt,
            max_tokens=300,
            temperature=0,
        )
        data = _parse_json(resp)
        faith = float(data.get("faithfulness", 0.5))
        faith = max(0.0, min(1.0, faith))
        return {
            "faithfulness":           round(faith, 4),
            "unsupported_sentences":  int(data.get("unsupported_sentences", 0)),
            "verdict":                str(data.get("verdict", "")),
            "is_faithful":            faith >= FAITHFUL_THRESHOLD,
        }
    except Exception as e:
        logger.warning(f"  faithfulness_judge failed: {e}")
        return {
            "faithfulness": 0.5,
            "unsupported_sentences": 0,
            "verdict": f"judge error: {e}",
            "is_faithful": False,
        }


# ── Correctness judge ─────────────────────────────────────────────────────────

CORRECT_PROMPT = """\
Evaluate how well the generated answer matches the gold reference answer.

Question: {question}

Gold reference answer:
{gold_answer}

Generated answer:
{answer}

Definition of correctness:
- Score 1.0: the generated answer contains all key information from the gold answer
- Score 0.7: most key information is present, minor omissions or slight differences
- Score 0.4: partial match — some correct facts, some missing or incorrect
- Score 0.1: very little overlap with the gold answer
- Score 0.0: completely wrong or contradicts the gold answer

Note: the generated answer does not need to be verbatim — semantic equivalence counts.
If the generated answer says "Insufficient evidence to answer" but the gold answer
has content, score 0.0 for correctness.

Return ONLY this JSON (no markdown):
{{
  "correctness": <0.0-1.0>,
  "key_facts_matched": <integer>,
  "key_facts_total": <integer>,
  "verdict": "<one sentence explaining the score>"
}}\
"""

def judge_correctness(
    llm,
    question:    str,
    gold_answer: str,
    answer:      str,
) -> Dict:
    """
    Score how well the answer matches the SPIQA gold free-form answer.

    Returns dict with:
      correctness        float 0-1
      key_facts_matched  int
      key_facts_total    int
      verdict            str
    """
    if not answer or not answer.strip():
        return {
            "correctness": 0.0,
            "key_facts_matched": 0,
            "key_facts_total": 0,
            "verdict": "empty answer",
        }

    if not gold_answer or not gold_answer.strip():
        return {
            "correctness": 0.5,
            "key_facts_matched": 0,
            "key_facts_total": 0,
            "verdict": "no gold answer available",
        }

    prompt = CORRECT_PROMPT.format(
        question=question,
        gold_answer=gold_answer[:800],
        answer=answer[:1500],
    )
    try:
        resp = llm.invoke(
            system=CORRECT_SYSTEM,
            prompt=prompt,
            max_tokens=300,
            temperature=0,
        )
        data = _parse_json(resp)
        corr = float(data.get("correctness", 0.5))
        corr = max(0.0, min(1.0, corr))
        return {
            "correctness":       round(corr, 4),
            "key_facts_matched": int(data.get("key_facts_matched", 0)),
            "key_facts_total":   int(data.get("key_facts_total", 0)),
            "verdict":           str(data.get("verdict", "")),
        }
    except Exception as e:
        logger.warning(f"  correctness_judge failed: {e}")
        return {
            "correctness": 0.5,
            "key_facts_matched": 0,
            "key_facts_total": 0,
            "verdict": f"judge error: {e}",
        }


# ── Helper ────────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    clean = re.sub(r"```(?:json)?", "", text).strip()
    start = clean.find("{")
    end   = clean.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(clean[start:end])
    return {}
