"""
retrieval_layer/reranker/cross_encoder_reranker.py

Cross-encoder re-ranking with enrichment signal boosting.

The core thesis contribution in the retrieval scoring:
  final_score = w1 * vector_score
              + w2 * cross_encoder_score
              + w3 * salience_score        (from enrichment)
              + w4 * evidence_role_boost   (intent-role match)
              + w5 * confidence_score      (contextual_summary_confidence)
              - noise_penalty

This composite score is what distinguishes HEMMIR from RAG-Fusion and
standard retrieval systems that use only cosine similarity.

Cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 6-layer MiniLM, fast enough for production
  - Scores (query, passage) pairs directly
  - Score range: approximately -10 to +10 → normalised to 0-1

Evidence role boost:
  Maps query_intent_type to expected evidence_role.
  If chunk's evidence_role matches query intent → +boost
  If mismatch (e.g. context chunk for procedural query) → 0
  This directly uses the enrichment layer's role tags.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models.pipeline_models import RetrievedChunk, ScoreBreakdown

# Weights for composite score — tunable, reported in thesis
# Validated on SPIQA TestB and Scania corpus
W_VECTOR        = 0.40   # cosine similarity from dense embedding
W_CROSS_ENC     = 0.30   # cross-encoder pointwise score
W_SALIENCE      = 0.12   # from enrichment layer salience_score
W_EVIDENCE_ROLE = 0.08   # intent-role compatibility boost
W_CONFIDENCE    = 0.05   # contextual_summary_confidence
W_ROLE_CONF     = 0.05   # evidence_role_confidence — new signal
                          # downweights role boost when LLM was uncertain

# Evidence role → query intent type compatibility map
# Maps query_intent_type → expected evidence_role(s)
ROLE_COMPATIBILITY: Dict[str, List[str]] = {
    "procedural":  ["procedure", "procedural"],
    "measurement": ["measurement"],
    "definition":  ["definition"],
    "comparative": ["procedure", "measurement", "definition"],
    "diagnostic":  ["measurement", "procedure"],
    "exploratory": ["context", "definition", "procedure"],
    "visual":      ["context", "definition"],
    "temporal":    ["context", "procedure"],
}

ROLE_BOOST_VALUE   = 0.15   # applied when role matches
ROLE_PENALTY_VALUE = 0.0    # no penalty for mismatch — just no boost


class CrossEncoderReranker:
    """
    Re-ranks retrieved chunks using a cross-encoder model combined
    with enrichment signals from the indexing pipeline.

    The composite score is the primary novel contribution:
    enrichment-grounded retrieval scoring for explainable IR.
    """

    def __init__(
        self,
        model_name: str  = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device:     str  = "cpu",
    ):
        self._model = None
        self._model_name = model_name
        self._device     = device
        self._available  = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._model     = CrossEncoder(self._model_name, device=self._device)
            self._available = True
            logger.info(f"CrossEncoder loaded: {self._model_name} on {self._device}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — "
                "cross-encoder reranking disabled. "
                "pip install sentence-transformers"
            )
        except Exception as e:
            logger.warning(f"CrossEncoder load failed: {e} — falling back to vector-only")

    def set_weights(
        self,
        w_vector:        float = 0.40,
        w_cross_enc:     float = 0.30,
        w_salience:      float = 0.12,
        w_evidence_role: float = 0.08,
        w_confidence:    float = 0.05,
        w_role_conf:     float = 0.05,
    ) -> None:
        """Set composite scoring weights for ablation experiments."""
        self._w_vector        = w_vector
        self._w_cross_enc     = w_cross_enc
        self._w_salience      = w_salience
        self._w_evidence_role = w_evidence_role
        self._w_confidence    = w_confidence
        self._w_role_conf     = w_role_conf

    def rerank(
        self,
        query:            str,
        chunks:           List[RetrievedChunk],
        query_intent_type: str = "definition",
        top_k:            int  = 20,
    ) -> List[RetrievedChunk]:
        """
        Re-rank chunks using cross-encoder + enrichment signals.

        Args:
            query:             Original user query
            chunks:            Retrieved chunks from modality plugins
            query_intent_type: From QueryAnalysis — used for role matching
            top_k:             Maximum chunks to return

        Returns:
            Re-ranked list, highest final_score first.
        """
        if not chunks:
            return []

        # Cross-encoder scores (0 if model unavailable → uses vector only)
        cross_scores = self._cross_encode(query, chunks)

        reranked = []
        for chunk, cross_score in zip(chunks, cross_scores):
            meta          = chunk.metadata
            vector_score  = chunk.score_breakdown.vector_score

            # Enrichment signals — read from chunk metadata
            salience      = _get_float(meta, "salience_score", 0.5)
            conf          = _get_float(meta, "contextual_summary_confidence", 0.5)
            role          = _get_str(meta, "evidence_role", "context")

            # Evidence role boost — weighted by role confidence
            # If LLM was uncertain about role classification (low role_conf),
            # the role boost is dampened — prevents incorrect role tags from
            # inflating scores. This is the evidence_role_confidence contribution.
            role_conf = _get_float(meta, "evidence_role_confidence", 0.5)
            compatible_roles = ROLE_COMPATIBILITY.get(query_intent_type, [])
            role_boost = (ROLE_BOOST_VALUE * role_conf) if role in compatible_roles else ROLE_PENALTY_VALUE

            # Noise penalty from extra_payload
            content      = (chunk.content or "").lower()
            noise_patterns = [
                "hereby grants permission", "google hereby",
                "in proceedings of", "arxiv preprint", "all rights reserved",
                "copyright notice",
            ]
            noise_penalty = -0.15 if any(p in content for p in noise_patterns) else 0.0

            # Composite score — research contribution
            # Combines dense retrieval, cross-encoder, and enrichment signals
            final = round(
                getattr(self,'_w_vector',W_VECTOR)          * vector_score
                + getattr(self,'_w_cross_enc',W_CROSS_ENC)     * cross_score
                + getattr(self,'_w_salience',W_SALIENCE)       * salience
                + getattr(self,'_w_evidence_role',W_EVIDENCE_ROLE) * role_boost
                + W_CONFIDENCE    * conf
                + W_ROLE_CONF     * role_conf
                + noise_penalty,
                4,
            )
            final = max(0.0, min(1.0, final))

            reranked.append(
                chunk.model_copy(
                    update={
                        "score_breakdown": ScoreBreakdown(
                            vector_score       = vector_score,
                            cross_encoder_score = cross_score,
                            code_boost         = chunk.score_breakdown.code_boost,
                            type_boost         = chunk.score_breakdown.type_boost,
                            modality_boost     = chunk.score_breakdown.modality_boost,
                            salience_boost     = round(W_SALIENCE * salience, 4),
                            evidence_role_boost = round(W_EVIDENCE_ROLE * role_boost, 4),
                            noise_penalty      = noise_penalty,
                            final_score        = final,
                        )
                    }
                )
            )

        reranked.sort(key=lambda c: c.score_breakdown.final_score, reverse=True)
        return reranked[:top_k]

    def _cross_encode(
        self,
        query:  str,
        chunks: List[RetrievedChunk],
    ) -> List[float]:
        """
        Score all (query, chunk_text) pairs with the cross-encoder.
        Returns normalised 0-1 scores. Falls back to 0.5 if model unavailable.
        """
        if not self._available or not self._model:
            return [0.5] * len(chunks)

        pairs  = [(query, _chunk_text(c)[:512]) for c in chunks]
        try:
            raw_scores = self._model.predict(pairs)
            # Normalise sigmoid(raw) to 0-1
            return [
                round(float(1.0 / (1.0 + math.exp(-s))), 4)
                for s in raw_scores
            ]
        except Exception as e:
            logger.warning(f"  CrossEncoder predict failed: {e}")
            return [0.5] * len(chunks)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_text(chunk: RetrievedChunk) -> str:
    """Build the text representation for cross-encoder input."""
    meta  = chunk.metadata
    parts = [chunk.content or ""]
    for attr in (
        "contextual_summary", "text_original_content", "transcript_text",
        "image_caption", "table_summary", "table_purpose", "section_title",
    ):
        if hasattr(meta, attr):
            val = getattr(meta, attr, None)
            if val:
                parts.append(str(val))
    return " ".join(p for p in parts if p)[:512]


def _get_float(meta: Any, attr: str, default: float) -> float:
    val = getattr(meta, attr, None) or meta.__dict__.get(attr)
    if val is None:
        return default
    try:
        return max(0.0, min(1.0, float(val)))
    except (TypeError, ValueError):
        return default


def _get_str(meta: Any, attr: str, default: str) -> str:
    val = getattr(meta, attr, None)
    if val is None:
        return default
    return str(val).lower().strip()