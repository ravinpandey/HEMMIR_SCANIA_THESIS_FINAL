"""
retrieval_layer/utils/hyde.py

HyDE (Hypothetical Document Embedding) and Multi-Query Expansion.

HyDE: generates a hypothetical document passage and embeds it instead of
the raw query. Dramatically improves recall when query vocabulary differs
from document vocabulary — common in industrial domains where users ask
in plain English but documents use technical jargon.

MultiQueryExpander: generates N rewritten query variants, retrieves
for each, merges with Reciprocal Rank Fusion (RRF).

Both are used by the RAG path before chunk retrieval.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# ── Research-grade prompts ────────────────────────────────────────────────────

HYDE_SYSTEM = """\
You are a technical document expert specialising in industrial vehicle \
manufacturing documentation. You write as if quoting directly from \
Scania technical manuals, maintenance procedures, and engineering reports.\
"""

HYDE_PROMPT = """\
A user has asked: "{question}"

Write a hypothetical document passage that would perfectly answer this question \
if it existed in a Scania technical document.

Requirements:
- Write as a direct excerpt from a technical document — NOT as an answer
- Use precise technical language: exact values, units, part numbers, procedure steps
- Be specific — include the kind of detail a trained technician would need
- Length: 3-5 sentences only
- Do not hedge — write as authoritative technical fact
- Match the vocabulary and style of maintenance manuals / engineering reports

Then rate:
DOMAIN_SPECIFICITY: <0.0-1.0>
(1.0 = very specific to heavy vehicle manufacturing, 0.0 = general knowledge)

Hypothetical passage:\
"""

EXPAND_SYSTEM = """\
You generate diverse search query variants for industrial document retrieval.
Each variant should rephrase the question using different technical vocabulary \
that might appear in Scania manuals, engineering reports, or training materials.\
"""

EXPAND_PROMPT = """\
Generate {n} different search queries to find documents answering this question.
Use different phrasing and technical vocabulary for each.

Original question: "{question}"

Requirements:
- Use technical terminology a Scania engineer would use
- Vary the vocabulary: some formal (as in a manual), some diagnostic, some procedural
- Keep each query concise (5-15 words)

Return ONLY a JSON array (no markdown):
["query 1", "query 2", "query 3"]\
"""


# ── Domain-adaptive prompt templates ─────────────────────────────────────────

DOMAIN_CONFIGS = {
    "scania": {
        "system": (
            "You are a technical document expert specialising in industrial vehicle "
            "manufacturing documentation. You write as if quoting directly from "
            "Scania technical manuals, maintenance procedures, and engineering reports."
        ),
        "domain_desc": "Scania technical manual, maintenance procedure, or engineering report",
        "vocab_hint":  "Use technical terminology a Scania engineer would use",
    },
    "spiqa": {
        "system": (
            "You are a technical retrieval assistant. Write a concise factual passage from a scientific paper that answers the question. No preamble. "
            "as if quoting directly from academic research papers, including methods, "
            "results, and experimental descriptions."
        ),
        "domain_desc": "academic research paper or scientific publication",
        "vocab_hint":  "Use academic and scientific terminology from the relevant field",
    },
    "fintech": {
        "system": (
            "You are a technical retrieval assistant. Write a concise factual passage from a financial document that answers the question. No preamble. "
            "as if quoting directly from financial reports, regulatory documents, "
            "or investment research."
        ),
        "domain_desc": "financial report, regulatory document, or investment research",
        "vocab_hint":  "Use financial and regulatory terminology",
    },
}

HYDE_SYSTEM_TEMPLATE = """You are a technical information retrieval assistant. Given a question, write a concise factual passage that would appear in a {domain_desc} and directly answers the question. Write only the passage — no preamble, no refusal, no explanation. This is for search indexing only."""

HYDE_PROMPT_TEMPLATE = """A user has asked: "{question}"

Write a hypothetical document passage that would perfectly answer this question if it existed in a {domain_desc}.

Requirements:
- Write as a direct excerpt from a document — NOT as an answer to the user
- {vocab_hint}
- Be specific — include concrete details, values, and terminology
- Length: 3-5 sentences only
- Do not hedge — write as authoritative fact
- Match the vocabulary and style of the relevant domain documentation

Then rate:
DOMAIN_SPECIFICITY: <0.0-1.0>
(1.0 = very domain-specific technical content, 0.0 = general knowledge)

Hypothetical passage:"""

EXPAND_SYSTEM_TEMPLATE = """You generate diverse search query variants for {domain_desc} retrieval.
Each variant should rephrase the question using different technical vocabulary that might appear in {domain_desc}s."""

EXPAND_PROMPT_TEMPLATE = """Generate {n} different search queries to find documents answering this question.
Use different phrasing and technical vocabulary for each.

Original question: "{question}"

Requirements:
- {vocab_hint}
- Vary the vocabulary: some formal (as in a document), some analytical, some descriptive
- Keep each query concise (5-15 words)

Return ONLY a JSON array (no markdown):
["query 1", "query 2", "query 3"]"""

def get_domain_prompts(domain: str = "scania") -> dict:
    """Get system/prompt strings for the given domain."""
    cfg = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS["scania"])
    return {
        "hyde_system":   HYDE_SYSTEM_TEMPLATE.format(domain_desc=cfg["domain_desc"]),
        "hyde_prompt":   HYDE_PROMPT_TEMPLATE.format,   # call with question=, domain_desc=
        "expand_system": EXPAND_SYSTEM_TEMPLATE.format(domain_desc=cfg["domain_desc"]),
        "expand_prompt": EXPAND_PROMPT_TEMPLATE.format,  # call with question=, n=, vocab_hint=
        "domain_desc":   cfg["domain_desc"],
        "vocab_hint":    cfg["vocab_hint"],
    }


# ── HyDE ─────────────────────────────────────────────────────────────────────

class HyDE:
    """
    Hypothetical Document Embedding.

    Generates a domain-appropriate hypothetical passage and embeds it
    in the same space as the indexed chunks. Bridges the vocabulary gap
    between user queries and technical document language.
    """

    def __init__(self, llm_client, text_embedder, domain: str = "scania"):
        self.llm      = llm_client
        self.embedder = text_embedder
        self.domain   = domain
        self._prompts = get_domain_prompts(domain)

    def embed_query(
        self,
        question:         str,
        n_hypotheses:     int  = 1,
        fallback_on_fail: bool = True,
    ) -> Tuple[List[float], str, float]:
        """
        Generate hypothetical passage(s), embed, return averaged vector.

        Returns:
            (embedding_vector, hypothetical_text, domain_specificity)
        """
        hypotheses:    List[str]         = []
        specificities: List[float]       = []

        for _ in range(n_hypotheses):
            try:
                _cfg = DOMAIN_CONFIGS.get(self.domain, DOMAIN_CONFIGS["scania"])
                response = self.llm.invoke(
                    system      = self._prompts["hyde_system"],
                    prompt      = self._prompts["hyde_prompt"](
                        question    = question,
                        domain_desc = _cfg["domain_desc"],
                        vocab_hint  = _cfg["vocab_hint"],
                    ),
                    max_tokens  = 250,
                    temperature = 0,
                )
                hyp, spec = self._parse_hyde_response(response)
                if hyp:
                    hypotheses.append(hyp)
                    specificities.append(spec)
            except Exception as e:
                logger.warning(f"  HyDE generation failed: {e}")

        if not hypotheses:
            if fallback_on_fail:
                logger.debug("  HyDE fallback to raw query")
                emb = self._embed(question)
                return emb, question, 0.5
            raise RuntimeError("HyDE: all attempts failed")

        # Embed each hypothesis
        embeddings: List[List[float]] = []
        for hyp in hypotheses:
            try:
                emb = self._embed(hyp)
                if emb:
                    embeddings.append(emb)
            except Exception as e:
                logger.warning(f"  HyDE embedding failed: {e}")

        if not embeddings:
            emb = self._embed(question)
            return emb, question, 0.5

        # Average and L2-normalise
        dim = len(embeddings[0])
        avg = [
            sum(e[i] for e in embeddings) / len(embeddings)
            for i in range(dim)
        ]
        norm = sum(x**2 for x in avg) ** 0.5
        if norm > 0:
            avg = [x / norm for x in avg]

        mean_spec = sum(specificities) / len(specificities) if specificities else 0.5
        return avg, " | ".join(hypotheses), mean_spec

    # Refusal patterns — LLM safety / content-filter rejections
    _REFUSAL_PATTERNS = re.compile(
        r"^(i cannot|i can'?t|i appreciate|i'm unable|i am unable|"
        r"i do not|i don'?t|i must decline|i need to|i would not|"
        r"i'm sorry|i am sorry|sorry,? i|as an ai|as a language model)",
        re.I,
    )

    def _parse_hyde_response(self, text: str) -> Tuple[str, float]:
        """Extract hypothetical passage and domain specificity."""
        spec_match = re.search(r"DOMAIN_SPECIFICITY:\s*([\d.]+)", text)
        specificity = float(spec_match.group(1)) if spec_match else 0.5
        specificity = max(0.0, min(1.0, specificity))

        # Remove the DOMAIN_SPECIFICITY line to get the passage
        passage = re.sub(r"DOMAIN_SPECIFICITY:.*", "", text).strip()
        # Remove any "Hypothetical passage:" prefix
        passage = re.sub(r"^Hypothetical passage:\s*", "", passage, flags=re.I).strip()

        if len(passage) < 20:
            return "", specificity

        # Detect LLM refusal — fall back to raw query rather than embed garbage
        first_line = passage.split("\n")[0].strip()
        if self._REFUSAL_PATTERNS.match(first_line):
            logger.warning(f"  HyDE: refusal detected, skipping passage")
            return "", specificity

        return passage, specificity

    def _embed(self, text: str) -> List[float]:
        return self.embedder.encode([text])[0]


# ── Multi-Query Expander ──────────────────────────────────────────────────────

class MultiQueryExpander:
    """
    Generates multiple query variants and merges retrieved results
    using Reciprocal Rank Fusion (RRF).

    Each variant is embedded and searched independently. Chunks that
    appear highly ranked across multiple variants score higher — a
    signal of robust relevance independent of vocabulary.
    """

    def __init__(self, llm_client, domain: str = "scania"):
        self.llm      = llm_client
        self.domain   = domain
        self._prompts = get_domain_prompts(domain)

    def expand_query(self, question: str, n: int = 3) -> List[str]:
        """Generate n rewritten query variants. Always includes original."""
        try:
            _cfg = DOMAIN_CONFIGS.get(self.domain, DOMAIN_CONFIGS["scania"])
            response = self.llm.invoke(
                system      = self._prompts["expand_system"],
                prompt      = self._prompts["expand_prompt"](
                    question    = question,
                    n           = n,
                    vocab_hint  = _cfg["vocab_hint"],
                ),
                max_tokens  = 300,
                temperature = 0,
            )
            clean  = re.sub(r"```json|```", "", response).strip()
            start  = clean.find("[")
            end    = clean.rfind("]") + 1
            if start >= 0 and end > start:
                queries = json.loads(clean[start:end])
                if isinstance(queries, list):
                    variants = [question] + [
                        q for q in queries
                        if isinstance(q, str) and q.strip()
                    ][:n]
                    logger.debug(f"  MultiQuery expanded to {len(variants)} variants")
                    return variants
        except Exception as e:
            logger.warning(f"  Query expansion failed: {e}")
        return [question]

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: List[List[Dict[str, Any]]],
        k:            int = 60,
        top_n:        int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Merge multiple ranked chunk lists using RRF.

        RRF score = Σ 1/(k + rank)
        Chunks appearing consistently high across variants score highest.

        Args:
            ranked_lists: list of ranked chunk dicts (each has chunk_id)
            k:            RRF constant (default 60, from original RRF paper)
            top_n:        number of results to return
        """
        from collections import defaultdict
        scores: Dict[str, float]        = defaultdict(float)
        chunks: Dict[str, Dict]         = {}

        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked, start=1):
                cid = chunk.get("chunk_id") or chunk.get("id", "")
                if cid:
                    scores[cid] += 1.0 / (k + rank)
                    if cid not in chunks:
                        chunks[cid] = chunk

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        result     = []
        for cid in sorted_ids[:top_n]:
            c = chunks[cid].copy()
            c["rrf_score"]   = round(scores[cid], 6)
            c["final_score"] = round(scores[cid], 6)
            result.append(c)

        return result