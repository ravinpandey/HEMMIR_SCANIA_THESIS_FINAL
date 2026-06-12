"""
rq2_ablation/conditions/base.py

Shared data structures, evidence formatting, and LLM helper functions
used by all five R2 conditions.

All LLM calls use temperature=0 for deterministic, reproducible ablation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Tuple

from loguru import logger


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ClaimRecord:
    text:              str
    claim_type:        str   = "context"   # measurement | definition | procedure | context
    status:            str   = "unscored"  # supported | unsupported | contradicted | unscored
    support_score:     float = 0.0         # ArgRAG composite strength (post-formula)
    attack_score:      float = 0.0
    raw_support_score: float = 0.0         # raw LLM confidence before ArgRAG formula
    raw_attack_score:  float = 0.0         # raw LLM confidence before ArgRAG formula
    support_reason:    str   = ""
    attack_reason:     str   = ""


@dataclass
class ConditionOutput:
    condition:          str
    answer:             str
    claims:             List[ClaimRecord] = field(default_factory=list)
    unsupported_count:  int   = 0
    contradicted_count: int   = 0
    coverage_score:     float = 1.0   # supported_claims / total_claims
    abstained:          bool  = False
    error:              str   = ""


# ── Evidence formatting ───────────────────────────────────────────────────────

def format_evidence(
    texts:      List[str],
    chunk_ids:  List[str],
    modalities: List[str],
    max_chars_per_chunk: int = 400,
    table_chars: int = 2500,
    image_chars: int = 1500,
) -> str:
    lines = []
    for i, (text, cid, mod) in enumerate(zip(texts, chunk_ids, modalities), 1):
        if mod == "table":
            limit = table_chars
        elif mod == "image":
            limit = image_chars
        else:
            limit = max_chars_per_chunk
        lines.append(f"[E{i}] ({mod}) id={cid}:\n{text[:limit]}")
    return "\n\n".join(lines)


# ── LLM helper functions ──────────────────────────────────────────────────────

def generate_claims(llm, question: str, evidence_text: str) -> List[ClaimRecord]:
    """Decompose expected answer into atomic, verifiable claims (LLM decides count 2-6)."""
    prompt = CLAIM_GEN_PROMPT.format(
        question=question, evidence=evidence_text[:5000]
    )
    try:
        resp = llm.invoke(
            system=ARGRAG_SYSTEM, prompt=prompt,
            max_tokens=600, temperature=0,
        )
        claims = _parse_claims(resp)
    except Exception as e:
        logger.warning(f"  generate_claims failed: {e}")
        return []

    # Filter out meta-claims that describe evidence limitations rather than
    # stating facts from the evidence. These become the sole "defensible" claim
    # and force an abstention even when real evidence exists.
    filtered = [
        c for c in claims
        if not any(p in c.text.lower() for p in _META_CLAIM_PATTERNS)
    ]
    if filtered:
        return filtered

    # All claims were meta-claims — return originals so Stage 2 can still attempt
    # synthesis (better than returning nothing and triggering empty-claim abstention).
    logger.warning(
        "  generate_claims: all %d claims were meta-claims — returning unfiltered", len(claims)
    )
    return claims


def check_support(llm, claim: str, evidence_text: str) -> Tuple[bool, float, str]:
    """Returns (is_supported, score 0-1, reason)."""
    prompt = SUPPORT_PROMPT.format(claim=claim, evidence=evidence_text[:5000])
    try:
        resp = llm.invoke(
            system=ARGRAG_SYSTEM, prompt=prompt,
            max_tokens=200, temperature=0,
        )
        data = _parse_json(resp)
        return (
            bool(data.get("supported", False)),
            float(data.get("score", 0.0)),
            str(data.get("reason", "")),
        )
    except Exception as e:
        logger.debug(f"  check_support failed: {e}")
        return False, 0.0, ""


def check_attack(llm, claim: str, evidence_text: str) -> Tuple[bool, float, str]:
    """Returns (is_contradicted, score 0-1, reason)."""
    prompt = ATTACK_PROMPT.format(claim=claim, evidence=evidence_text[:5000])
    try:
        resp = llm.invoke(
            system=ARGRAG_SYSTEM, prompt=prompt,
            max_tokens=200, temperature=0,
        )
        data = _parse_json(resp)
        return (
            bool(data.get("contradicted", False)),
            float(data.get("score", 0.0)),
            str(data.get("reason", "")),
        )
    except Exception as e:
        logger.debug(f"  check_attack failed: {e}")
        return False, 0.0, ""


def synthesize_answer(
    llm,
    question:    str,
    claims_text: str,
    instruction: str = "",
) -> str:
    """Generate final answer from verified claims."""
    prompt = SYNTHESIS_PROMPT.format(
        question=question,
        claims=claims_text or "No verified claims available.",
        instruction=instruction or "Answer concisely and precisely.",
    )
    try:
        resp = llm.invoke(
            system=SYNTHESIS_SYSTEM, prompt=prompt,
            max_tokens=800, temperature=0,
        )
        return resp.strip()
    except Exception as e:
        logger.warning(f"  synthesize_answer failed: {e}")
        return "Insufficient evidence to answer."


def claims_to_text(claims: List[ClaimRecord]) -> str:
    lines = []
    for i, c in enumerate(claims, 1):
        lines.append(f"{i}. [{c.claim_type}] {c.text}")
    return "\n".join(lines) if lines else ""


# ── Prompts ───────────────────────────────────────────────────────────────────

ARGRAG_SYSTEM = """\
You are a precise scientific QA reasoning assistant.
You decompose questions into verifiable claims and assess evidence relationships.
Respond ONLY in valid JSON as instructed. No markdown, no preamble.\
"""

SYNTHESIS_SYSTEM = """\
You are a precise scientific QA assistant.
Answer questions using only the provided verified claims.
Be factual. Do not add information not present in the claims.
For questions asking to "explain in detail", "describe", or "list" — enumerate ALL specific
items, stations, steps, components, and values mentioned in the claims. Do not summarise.\
"""

CLAIM_GEN_PROMPT = """\
Decompose the expected answer to this question into the minimum number of atomic, \
verifiable claims needed to fully answer it. Use between 2 and 8 claims — no more \
than the question complexity requires.

Question: {question}

Evidence:
{evidence}

Rules:
- Each claim must be a SINGLE verifiable statement (not a question)
- Claims must be grounded in the evidence above
- Simple factual questions need 2-3 claims; layout, process, or descriptive questions need 5-8
- Keep each claim under 50 words
- For layout/process questions: each station, step, or component gets its own claim
- CRITICAL: Claims must state what the evidence SAYS — never what it lacks.
  FORBIDDEN claim patterns (these are meta-claims, not evidence claims):
    ✗ "The evidence does not contain..."
    ✗ "No information about X is provided..."
    ✗ "The evidence shows tables but no definition of..."
    ✗ "X is not mentioned in the evidence..."
  If you cannot find a claim that states something positive from the evidence,
  generate a claim about the closest related fact the evidence DOES contain.

Claim types:
  measurement: states a specific value, metric, or number
  definition:  defines what something is
  procedure:   describes how something works or is done
  context:     provides background or supporting information

Return ONLY this JSON (no markdown):
{{"claims": [{{"text": "<claim>", "type": "<measurement|definition|procedure|context>"}}]}}\
"""

# Patterns that identify meta-claims about evidence limitations — filtered out
# before ArgRAG scoring so they cannot become the sole defensible claim.
_META_CLAIM_PATTERNS = (
    "does not contain", "does not include", "not found in",
    "no explicit", "no information", "not mentioned", "not specified",
    "evidence does", "evidence contains", "evidence shows", "evidence only",
    "evidence lacks", "not provided", "not present",
)

SUPPORT_PROMPT = """\
Does the evidence below SUPPORT this claim?

Claim: {claim}

Evidence:
{evidence}

A claim is SUPPORTED if the evidence directly and specifically confirms it.
A claim is NOT SUPPORTED if the evidence is silent, vague, or only tangentially related.

Return ONLY this JSON (no markdown):
{{"supported": <true|false>, "score": <0.0-1.0>, "reason": "<one sentence>"}}\
"""

ATTACK_PROMPT = """\
Does any evidence below CONTRADICT or WEAKEN this claim?

Claim: {claim}

Evidence:
{evidence}

A claim is CONTRADICTED if the evidence directly opposes it or makes it factually incorrect.
Being unrelated does NOT count as contradicting.

Return ONLY this JSON (no markdown):
{{"contradicted": <true|false>, "score": <0.0-1.0>, "reason": "<one sentence>"}}\
"""

SYNTHESIS_PROMPT = """\
Answer the question using ONLY the verified claims below.
If the claims list is empty or says "No verified claims available", respond with exactly:
"Insufficient evidence to answer."

Question: {question}

Verified claims:
{claims}

Instruction: {instruction}

Answer:\
"""


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    clean = re.sub(r"```(?:json)?", "", text).strip()
    start = clean.find("{")
    end   = clean.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(clean[start:end])
    return {}


def _parse_claims(text: str) -> List[ClaimRecord]:
    try:
        data = _parse_json(text)
        claims = []
        for c in data.get("claims", []):
            txt = str(c.get("text", "")).strip()
            typ = str(c.get("type", "context")).strip().lower()
            if txt:
                claims.append(ClaimRecord(text=txt, claim_type=typ))
        return claims
    except Exception as e:
        logger.debug(f"  _parse_claims failed: {e}")
        return []
