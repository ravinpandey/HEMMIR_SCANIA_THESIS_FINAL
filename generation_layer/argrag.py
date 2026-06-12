"""
generation_layer/argrag.py

ArgRAG Reasoning Layer — Steps 4-7 from the system diagram.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESEARCH CONTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This module implements the argumentation-based reasoning layer on top
of retrieved evidence. It transforms RAG from a retrieval-generation
pipeline into a structured reasoning system.

The four steps implemented here:

  Step 4 — Sub-claim generation
      Decomposes the answer into discrete verifiable claims.
      Each claim is typed by the evidence it requires
      (definition / procedure / measurement / context).
      C = f(Query Intent + Evidence)

  Step 5 — Relation establishment
      For each claim, determines whether each evidence chunk
      supports, attacks, or is neutral to the claim.
      Support:  evidence directly confirms the claim
      Attack:   evidence contradicts or qualifies the claim
      Neutral:  evidence is related but neither confirms nor denies

  Step 6 — Claim strength scoring  [MATHEMATICAL CORE]
      S(Ci) = α × Support_count - β × Attack_count + γ × Consistency
      α = 0.5  (support weight)
      β = 0.3  (attack penalty)
      γ = 0.2  (inter-claim consistency bonus)
      Produces a score ∈ [0, 1] per claim.

  Step 7 — Claim selection
      Classifies each claim as:
        strong   → S(Ci) >= STRONG_THRESHOLD (0.65)
        weak     → WEAK_THRESHOLD <= S(Ci) < STRONG_THRESHOLD
        contested → has both support AND attack evidence
      Only strong + contested claims are used for answer generation.
      Contested claims are flagged for the user.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONNECTION TO FINAL CONFIDENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The claim strength S(Ci) replaces the raw evidence score E in the
Final Confidence formula for the ArgRAG path:

  Simple RAG:  C = wf·F  +  we·E           +  wc·COV  +  ws·CONS
  ArgRAG:      C = wf·F  +  we·ClaimStrength +  wc·COV  +  ws·CONS

  Where:
    ClaimStrength = mean S(Ci) over all selected claims
    COV (Coverage)    = strong_claims / total_claims
    CONS (Consistency) = 1 - (contested_claims / total_claims)

This is stronger than raw E because it integrates argumentation —
a claim supported by 5 chunks and attacked by 0 gets a higher
ClaimStrength than one supported by 1 chunk, even if both have
the same salience_score.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Called from generation_pipeline.py when path = "argrag":

    from generation_layer.argrag import ArgRAGReasoner

    reasoner = ArgRAGReasoner(llm_client=llm_client)
    result   = reasoner.reason(
        question     = query,
        evidence_pack = evidence_pack,
        intent_type  = analysis.query_intent_type,
    )

    # result.claim_strength    → replaces E in confidence formula
    # result.selected_claims   → used for answer generation
    # result.contested_claims  → flagged in final output
    # result.coverage          → COV in confidence formula
    # result.consistency       → CONS in confidence formula
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from shared.models.pipeline_models import EvidencePack

# ── Claim strength weights ────────────────────────────────────────────────────
# S(Ci) = α × Support - β × Attack + γ × Consistency
# Tunable hyperparameters — report in thesis
ALPHA = 0.5   # support weight
BETA  = 0.3   # attack penalty
GAMMA = 0.2   # inter-claim consistency bonus

STRONG_THRESHOLD    = 0.65   # S(Ci) >= this → strong claim
WEAK_THRESHOLD      = 0.35   # S(Ci) < this  → weak claim, excluded
CONTESTED_THRESHOLD = 0.10   # attack_count >= this fraction → contested


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Claim:
    """A single decomposed sub-claim with its argumentation state."""
    text:          str
    claim_type:    str           # definition / procedure / measurement / context
    supports:      List[str] = field(default_factory=list)   # chunk_ids
    attacks:       List[str] = field(default_factory=list)   # chunk_ids
    neutrals:      List[str] = field(default_factory=list)   # chunk_ids
    strength:      float     = 0.0
    status:        str       = "unscored"  # strong / weak / contested / unscored


@dataclass
class ArgRAGResult:
    """
    Output of the ArgRAG reasoning pass.

    claim_strength  → replaces E in Final Confidence formula
    coverage        → COV component: strong_claims / total_claims
    consistency     → CONS component: 1 - contested_ratio
    selected_claims → strong + contested claims for answer generation
    contested_claims → flagged for user (have conflicting evidence)
    all_claims      → complete reasoning trace for explainability
    """
    all_claims:       List[Claim]
    selected_claims:  List[Claim]
    contested_claims: List[Claim]
    weak_claims:      List[Claim]
    claim_strength:   float         # replaces E in ArgRAG confidence
    coverage:         float         # COV in confidence formula
    consistency:      float         # CONS in confidence formula


# ── ArgRAG Reasoner ───────────────────────────────────────────────────────────

class ArgRAGReasoner:
    """
    Implements Steps 4-7 of the ArgRAG reasoning path.

    Steps:
        4. Sub-claim generation — LLM decomposes answer into claims
        5. Relation establishment — LLM assigns support/attack/neutral
        6. Claim strength scoring — S(Ci) = αS - βA + γC
        7. Claim selection — strong / weak / contested classification
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def reason(
        self,
        question:      str,
        evidence_pack: EvidencePack,
        intent_type:   str = "definition",
    ) -> ArgRAGResult:
        """
        Run full ArgRAG reasoning pipeline on retrieved evidence.

        Args:
            question:      Original user query
            evidence_pack: Retrieved and reranked evidence from HEMMIR
            intent_type:   From QueryAnalysis — shapes claim generation

        Returns:
            ArgRAGResult with claim_strength, coverage, consistency,
            selected_claims for answer generation.
        """
        if not evidence_pack.items:
            logger.warning("  ArgRAG: empty evidence pack — returning null result")
            return _null_result()

        # ── Step 4: Sub-claim generation ──────────────────────────────────────
        claims = self._generate_claims(question, evidence_pack, intent_type)
        logger.info(f"  ArgRAG Step 4: {len(claims)} claims generated")

        if not claims:
            logger.warning("  ArgRAG: no claims generated — returning null result")
            return _null_result()

        # ── Step 5: Relation establishment ────────────────────────────────────
        claims = self._establish_relations(claims, evidence_pack)
        logger.info(
            f"  ArgRAG Step 5: relations established | "
            f"total support links: {sum(len(c.supports) for c in claims)} | "
            f"total attack links: {sum(len(c.attacks) for c in claims)}"
        )

        # ── Step 6: Claim strength scoring ────────────────────────────────────
        claims = self._score_claims(claims)
        logger.info(
            f"  ArgRAG Step 6: scores computed | "
            f"mean S(Ci)={mean(c.strength for c in claims):.3f}"
        )

        # ── Step 7: Claim selection ───────────────────────────────────────────
        result = self._select_claims(claims)
        logger.info(
            f"  ArgRAG Step 7: strong={len(result.selected_claims)} | "
            f"contested={len(result.contested_claims)} | "
            f"weak={len(result.weak_claims)} | "
            f"claim_strength={result.claim_strength:.3f} | "
            f"coverage={result.coverage:.3f} | "
            f"consistency={result.consistency:.3f}"
        )

        return result

    # ── Step 4: Sub-claim generation ─────────────────────────────────────────

    def _generate_claims(
        self,
        question:      str,
        evidence_pack: EvidencePack,
        intent_type:   str,
    ) -> List[Claim]:
        """
        Step 4 — Decompose the expected answer into discrete verifiable claims.

        Each claim is:
          - A single verifiable statement (not a question)
          - Typed by the evidence it requires
          - Derived from the query intent and available evidence

        Returns list of Claim objects with text and claim_type.
        """
        if not self.llm:
            # Fallback: create one claim per evidence item (no decomposition)
            return _fallback_claims(question, evidence_pack)

        # Build evidence summary for LLM
        evidence_summary = _build_evidence_summary(evidence_pack, max_items=8)

        prompt = CLAIM_GENERATION_PROMPT.format(
            question       = question,
            intent_type    = intent_type,
            evidence_summary = evidence_summary,
        )

        try:
            response = self.llm.invoke(
                system     = ARGRAG_SYSTEM,
                prompt     = prompt,
                max_tokens = 800,
            )
            claims = _parse_claims(response)
            if claims:
                return claims
        except Exception as e:
            logger.warning(f"  ArgRAG claim generation failed: {e}")

        return _fallback_claims(question, evidence_pack)

    # ── Step 5: Relation establishment ───────────────────────────────────────

    def _establish_relations(
        self,
        claims:        List[Claim],
        evidence_pack: EvidencePack,
    ) -> List[Claim]:
        """
        Step 5 — Assign support/attack/neutral relations between
        each claim and each evidence chunk.

        For each (claim, evidence_chunk) pair:
          support  → chunk directly confirms the claim
          attack   → chunk contradicts or qualifies the claim
          neutral  → chunk is related but neither confirms nor denies

        Uses LLM for relation classification if available.
        Falls back to keyword-overlap heuristic when LLM unavailable.
        """
        if not self.llm:
            return _heuristic_relations(claims, evidence_pack)

        # Build evidence chunks list (chunk_id → content)
        evidence_items = {
            item.chunk_id: _item_text(item)
            for item in evidence_pack.items
        }

        for claim in claims:
            prompt = RELATION_PROMPT.format(
                claim            = claim.text,
                claim_type       = claim.claim_type,
                evidence_summary = _build_evidence_summary(evidence_pack, max_items=6),
            )
            try:
                response = self.llm.invoke(
                    system     = ARGRAG_SYSTEM,
                    prompt     = prompt,
                    max_tokens = 500,
                )
                _parse_relations(response, claim, evidence_items)
            except Exception as e:
                logger.debug(f"  ArgRAG relation establishment failed for claim: {e}")
                # Fallback: assign all chunks as neutral
                claim.neutrals = list(evidence_items.keys())

        return claims

    # ── Step 6: Claim strength scoring ───────────────────────────────────────

    def _score_claims(self, claims: List[Claim]) -> List[Claim]:
        """
        Step 6 — Compute S(Ci) = α × Support - β × Attack + γ × Consistency

        α = 0.5 (support weight)
        β = 0.3 (attack penalty)
        γ = 0.2 (inter-claim consistency bonus)

        Support and attack counts are normalised by total evidence count.
        Consistency bonus: 1.0 if no other claim conflicts with this claim,
        0.0 if another claim attacks the same evidence this claim supports.

        Score is clamped to [0, 1].
        """
        # Build consistency context — which chunks are disputed
        # A chunk is disputed if it appears in supports of one claim
        # and attacks of another claim
        chunk_in_supports: Dict[str, List[int]] = {}
        chunk_in_attacks:  Dict[str, List[int]] = {}
        for i, claim in enumerate(claims):
            for cid in claim.supports:
                chunk_in_supports.setdefault(cid, []).append(i)
            for cid in claim.attacks:
                chunk_in_attacks.setdefault(cid, []).append(i)

        disputed_chunks = set(chunk_in_supports.keys()) & set(chunk_in_attacks.keys())

        for claim in claims:
            n_total   = max(1, len(claim.supports) + len(claim.attacks) + len(claim.neutrals))
            n_support = len(claim.supports)
            n_attack  = len(claim.attacks)

            # Normalised counts
            support_ratio = n_support / n_total
            attack_ratio  = n_attack  / n_total

            # Consistency bonus: reduced if any supporting chunk is also disputed
            n_disputed = sum(1 for cid in claim.supports if cid in disputed_chunks)
            consistency = 1.0 - (n_disputed / max(1, n_support)) if n_support > 0 else 0.5

            # S(Ci) = α × Support - β × Attack + γ × Consistency
            strength = (
                ALPHA * support_ratio
                - BETA  * attack_ratio
                + GAMMA * consistency
            )
            claim.strength = round(max(0.0, min(1.0, strength)), 4)

        return claims

    # ── Step 7: Claim selection ───────────────────────────────────────────────

    def _select_claims(self, claims: List[Claim]) -> ArgRAGResult:
        """
        Step 7 — Classify each claim as strong / weak / contested.

        strong    → S(Ci) >= STRONG_THRESHOLD (0.65)
                    Used for answer generation without warning.

        contested → has both support AND attack evidence
                    Used for answer generation WITH a flag:
                    "Note: this claim has conflicting evidence."

        weak      → S(Ci) < WEAK_THRESHOLD (0.35)
                    Excluded from answer generation.

        Computes:
            claim_strength → mean S(Ci) over selected claims
                             replaces E in Final Confidence formula
            coverage       → selected_claims / total_claims (COV)
            consistency    → 1 - contested_ratio (CONS)
        """
        strong    = []
        contested = []
        weak      = []

        for claim in claims:
            is_contested = (len(claim.attacks) > 0 and len(claim.supports) > 0)

            if is_contested:
                claim.status = "contested"
                contested.append(claim)
            elif claim.strength >= STRONG_THRESHOLD:
                claim.status = "strong"
                strong.append(claim)
            elif claim.strength < WEAK_THRESHOLD:
                claim.status = "weak"
                weak.append(claim)
            else:
                # Between weak and strong threshold, not contested → include
                claim.status = "moderate"
                strong.append(claim)  # include as usable

        selected = strong + contested

        n_total    = len(claims)
        n_selected = len(selected)
        n_contested = len(contested)

        # claim_strength: mean S(Ci) over selected claims
        # Used as E replacement in: C = wf·F + we·ClaimStrength + wc·COV + ws·CONS
        if selected:
            claim_strength = mean(c.strength for c in selected)
        else:
            claim_strength = 0.0

        # COV: coverage = what fraction of claims are usable
        coverage = n_selected / max(1, n_total)

        # CONS: consistency = how few claims are contested
        # High consistency → low contested ratio → high CONS
        consistency = 1.0 - (n_contested / max(1, n_selected)) if n_selected > 0 else 0.0

        return ArgRAGResult(
            all_claims       = claims,
            selected_claims  = selected,
            contested_claims = contested,
            weak_claims      = weak,
            claim_strength   = round(claim_strength, 4),
            coverage         = round(coverage, 4),
            consistency      = round(consistency, 4),
        )


# ── Prompts ───────────────────────────────────────────────────────────────────

ARGRAG_SYSTEM = """\
You are a precise reasoning assistant for industrial and scientific document analysis.
You decompose questions into verifiable claims and assess evidence relationships.
You respond ONLY in valid JSON as instructed. No markdown, no preamble.\
"""

CLAIM_GENERATION_PROMPT = """\
Decompose the expected answer to this question into discrete verifiable claims.

Question: {question}
Query intent: {intent_type}

Available evidence:
{evidence_summary}

Generate 3-6 atomic claims that together answer the question.
Each claim must be:
  - A single verifiable statement (not a question or instruction)
  - Typed by what evidence would confirm it
  - Grounded in the available evidence above

Claim types:
  definition  → defines what something is
  procedure   → describes how to do something
  measurement → states a specific value, metric, or quantity
  context     → provides background or related information

Respond in EXACTLY this JSON (no markdown):
{{
  "claims": [
    {{
      "text": "<single verifiable statement>",
      "claim_type": "<definition|procedure|measurement|context>"
    }}
  ]
}}\
"""

RELATION_PROMPT = """\
Assess the relationship between this claim and each evidence chunk.

Claim: {claim}
Claim type: {claim_type}

Evidence chunks:
{evidence_summary}

For each evidence chunk, determine:
  support  → chunk directly confirms the claim with specific details
  attack   → chunk contradicts, qualifies, or limits the claim
  neutral  → chunk is related but neither confirms nor denies

Respond in EXACTLY this JSON (no markdown):
{{
  "relations": [
    {{
      "chunk_id": "<chunk_id>",
      "relation": "<support|attack|neutral>",
      "reason": "<one sentence why>"
    }}
  ]
}}\
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_evidence_summary(evidence_pack: EvidencePack, max_items: int = 8) -> str:
    """Build a concise evidence summary for LLM prompts."""
    lines = []
    for item in evidence_pack.items[:max_items]:
        meta    = item.metadata
        summary = (
            getattr(meta, "contextual_summary", "") or
            item.content or ""
        )[:200]
        role = getattr(meta, "evidence_role", "") or ""
        lines.append(
            f"[{item.chunk_id}] ({item.source_modality}"
            f"{', ' + role if role else ''}): {summary}"
        )
    return "\n".join(lines)


def _item_text(item: Any) -> str:
    """Extract text from an evidence item."""
    meta = item.metadata
    return (
        item.content or
        getattr(meta, "contextual_summary", "") or
        getattr(meta, "text_original_content", "") or
        getattr(meta, "table_summary", "") or
        getattr(meta, "image_caption", "") or
        ""
    )[:300]


def _parse_claims(response: str) -> List[Claim]:
    """Parse LLM JSON response into Claim objects."""
    try:
        clean = response.strip()
        if "```" in clean:
            clean = re.sub(r"```(?:json)?", "", clean).strip()
        data   = json.loads(clean)
        claims = []
        for c in data.get("claims", []):
            text       = str(c.get("text", "")).strip()
            claim_type = str(c.get("claim_type", "context")).strip().lower()
            if text:
                claims.append(Claim(text=text, claim_type=claim_type))
        return claims
    except Exception as e:
        logger.debug(f"  ArgRAG claim parse failed: {e}")
        return []


def _parse_relations(
    response:       str,
    claim:          Claim,
    evidence_items: Dict[str, str],
) -> None:
    """Parse LLM relation response and update claim in-place."""
    try:
        clean = response.strip()
        if "```" in clean:
            clean = re.sub(r"```(?:json)?", "", clean).strip()
        data = json.loads(clean)
        for rel in data.get("relations", []):
            chunk_id = str(rel.get("chunk_id", "")).strip()
            relation = str(rel.get("relation", "neutral")).strip().lower()
            if chunk_id not in evidence_items:
                continue
            if relation == "support":
                claim.supports.append(chunk_id)
            elif relation == "attack":
                claim.attacks.append(chunk_id)
            else:
                claim.neutrals.append(chunk_id)
    except Exception as e:
        logger.debug(f"  ArgRAG relation parse failed: {e}")
        # All neutral fallback
        claim.neutrals = list(evidence_items.keys())


def _fallback_claims(question: str, evidence_pack: EvidencePack) -> List[Claim]:
    """
    Fallback when LLM is unavailable.
    Creates one claim per top-3 evidence item using its contextual_summary.
    """
    claims = []
    for item in evidence_pack.items[:3]:
        meta    = item.metadata
        summary = (
            getattr(meta, "contextual_summary", "") or
            item.content or ""
        )[:200]
        if summary:
            role = getattr(meta, "evidence_role", "context") or "context"
            claims.append(Claim(text=summary, claim_type=role))

    if not claims:
        claims.append(Claim(
            text       = f"Evidence retrieved for: {question}",
            claim_type = "context",
        ))
    return claims


def _heuristic_relations(
    claims:        List[Claim],
    evidence_pack: EvidencePack,
) -> List[Claim]:
    """
    Fallback relation establishment using keyword overlap.
    Each claim-evidence pair is scored by token overlap.
    Top overlapping chunks → support. Remaining → neutral.
    """
    evidence_texts = {
        item.chunk_id: _item_text(item).lower()
        for item in evidence_pack.items
    }

    for claim in claims:
        claim_tokens = set(re.findall(r'\w{3,}', claim.text.lower()))
        scored = []
        for chunk_id, text in evidence_texts.items():
            chunk_tokens = set(re.findall(r'\w{3,}', text))
            overlap = len(claim_tokens & chunk_tokens) / max(1, len(claim_tokens))
            scored.append((chunk_id, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Top 2 by overlap → support, rest → neutral
        for i, (chunk_id, score) in enumerate(scored):
            if i < 2 and score > 0.15:
                claim.supports.append(chunk_id)
            else:
                claim.neutrals.append(chunk_id)

    return claims


def _null_result() -> ArgRAGResult:
    """Return null ArgRAGResult when reasoning cannot proceed."""
    return ArgRAGResult(
        all_claims       = [],
        selected_claims  = [],
        contested_claims = [],
        weak_claims      = [],
        claim_strength   = 0.0,
        coverage         = 0.0,
        consistency      = 0.0,
    )