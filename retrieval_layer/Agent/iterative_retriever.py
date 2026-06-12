"""
retrieval_layer/Agent/iterative_retriever.py

The core Agent retrieval loop.

For each sub-question:
  1. Navigator finds relevant sections (structure_unit_ids)
  2. Retrieve chunks filtered to those section_ids from ChromaDB
  3. Sufficiency checker evaluates whether evidence is complete
  4. If partial/insufficient AND iterations remain: refine query and repeat
  5. Collect SubQuestionResult for the RetrievalTrace

Maximum iterations per sub-question: 2 (bounded LLM call budget)
Maximum sub-questions: 4 (from decomposer)
Maximum total LLM calls for Agent path: ~12 (decompose + 4×navigate + 4×2×sufficiency)

The sufficiency checker produces:
  - sufficiency: full | partial | insufficient
  - missing_aspects: structured gap list → feeds UncertaintyReport
  - evidence_role_match: bool → key signal for explainability layer
  - refinement_query: next search if still insufficient
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from shared.models.pipeline_models import (
    RetrievedChunk,
    ScoreBreakdown,
    SubQuestionResult,
)


MAX_ITERATIONS   = 2    # per sub-question
MAX_SUB_QUESTIONS = 4   # enforced by decomposer, checked here too


SUFFICIENCY_SYSTEM = """\
You are a critical evidence evaluator for an industrial document retrieval system.
Determine whether retrieved evidence is sufficient to answer a sub-question.

Be conservative — mark as insufficient if ANY important aspect cannot be answered.
Missing evidence in a safety-critical industrial context is worse than admitting uncertainty.\
"""

SUFFICIENCY_PROMPT = """\
Evaluate whether this evidence sufficiently answers the sub-question.

Sub-question: "{sub_question}"
Evidence role needed: {evidence_role}

Retrieved evidence:
{evidence_text}

Evaluate:
1. Does evidence DIRECTLY answer the sub-question or only partially?
2. Is evidence the RIGHT TYPE? (e.g. procedure chunk for procedural sub-question)
3. What specific information is still missing?
4. Could a trained technician answer the sub-question from this evidence alone?

Respond in EXACTLY this JSON (no markdown):
{{
  "sufficiency": "full|partial|insufficient",
  "answered_aspects": ["<aspect answered>"],
  "missing_aspects": ["<aspect still missing>"],
  "evidence_role_match": true,
  "role_mismatch_explanation": "",
  "refinement_query": null,
  "sufficiency_score": 0.0,
  "confidence_in_evaluation": 0.0
}}\
"""


class IterativeRetriever:
    """
    Executes the Agent retrieval loop for all sub-questions.

    Returns a list of RetrievedChunk (deduplicated, from all sub-questions)
    and a list of SubQuestionResult (for the RetrievalTrace).
    """

    def __init__(
        self,
        llm_client,
        text_embedder,
        store,
        navigator,
        registry:      Dict[str, Any],
        top_k_chunks:  int = 8,
    ):
        self.llm          = llm_client
        self.embedder     = text_embedder
        self.store        = store
        self.navigator    = navigator
        self.registry     = registry
        self.top_k_chunks = top_k_chunks

    def retrieve(
        self,
        sub_questions:  List[Dict[str, Any]],
        doc_ids:        List[str],
        query_context,
    ) -> Tuple[List[RetrievedChunk], List[SubQuestionResult]]:
        """
        Execute retrieval for all sub-questions.

        Args:
            sub_questions:  From QueryDecomposer
            doc_ids:        From document-level retrieval
            query_context:  QueryContext with query_text

        Returns:
            (all_chunks, sub_question_results)
        """
        all_chunks:  Dict[str, RetrievedChunk] = {}   # chunk_id → chunk (deduplicated)
        sq_results:  List[SubQuestionResult]   = []

        for i, sq in enumerate(sub_questions[:MAX_SUB_QUESTIONS]):
            logger.info(f"  Agent sub-question [{i+1}]: {sq.get('question','')[:70]}")
            chunks, result = self._retrieve_one(sq, doc_ids, query_context)

            # Merge — keep highest-scoring version of each chunk
            for chunk in chunks:
                cid      = chunk.chunk_id
                existing = all_chunks.get(cid)
                if existing is None or chunk.score_breakdown.final_score > existing.score_breakdown.final_score:
                    all_chunks[cid] = chunk

            sq_results.append(result)

        # Sort by final_score descending
        final_chunks = sorted(
            all_chunks.values(),
            key=lambda c: c.score_breakdown.final_score,
            reverse=True,
        )
        return final_chunks, sq_results

    def _retrieve_one(
        self,
        sq:            Dict[str, Any],
        doc_ids:       List[str],
        query_context,
    ) -> Tuple[List[RetrievedChunk], SubQuestionResult]:
        question    = sq.get("question", "")
        role        = sq.get("evidence_role", "context")
        preferred   = sq.get("preferred_modality", "text")
        hint        = sq.get("section_hint", "")

        result = SubQuestionResult(
            question           = question,
            evidence_role      = role,
            preferred_modality = preferred,
            section_hint       = hint,
        )

        # ── Section navigation ─────────────────────────────────────────
        selected_sections, nav_conf, completeness = self.navigator.navigate(
            sq, self.store, doc_ids
        )
        result.sections_navigated  = [s["structure_unit_id"] for s in selected_sections]
        result.navigation_confidence = nav_conf

        section_ids = result.sections_navigated

        # ── Iterative chunk retrieval ──────────────────────────────────
        current_query  = question
        chunks:        List[RetrievedChunk] = []
        iteration      = 0

        while iteration < MAX_ITERATIONS:
            iteration += 1
            new_chunks = self._fetch_chunks(
                query        = current_query,
                section_ids  = section_ids,
                doc_ids      = doc_ids,
                preferred    = preferred,
                query_context = query_context,
            )
            # Merge new chunks with existing
            chunk_map: Dict[str, RetrievedChunk] = {c.chunk_id: c for c in chunks}
            for c in new_chunks:
                existing = chunk_map.get(c.chunk_id)
                if not existing or c.score_breakdown.final_score > existing.score_breakdown.final_score:
                    chunk_map[c.chunk_id] = c
            chunks = sorted(chunk_map.values(), key=lambda c: c.score_breakdown.final_score, reverse=True)

            if not chunks:
                result.sufficiency     = "insufficient"
                result.sufficiency_score = 0.0
                result.iterations      = iteration
                result.chunks_retrieved = 0
                break

            # Sufficiency check
            try:
                suf, missing, role_ok, refinement = self._check_sufficiency(
                    question  = question,
                    role      = role,
                    chunks    = chunks[:5],  # top 5 for evaluation
                )
            except Exception as e:
                logger.warning(f"  Sufficiency check failed: {e} — assuming partial")
                suf, missing, role_ok, refinement = "partial", [], True, None

            result.sufficiency       = suf
            result.missing_aspects   = missing
            result.role_mismatch     = not role_ok
            result.iterations        = iteration
            result.chunks_retrieved  = len(chunks)

            if suf == "full":
                break

            if refinement and iteration < MAX_ITERATIONS:
                logger.info(f"  Agent refining query (iter {iteration}): {refinement[:60]}")
                current_query = refinement
            else:
                break

        return chunks, result

    def _fetch_chunks(
        self,
        query:         str,
        section_ids:   List[str],
        doc_ids:       List[str],
        preferred:     str,
        query_context,
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks from the preferred collection, filtered by section_id.
        Falls back to doc_id filter if no section_ids available.
        """
        query_vec = self.embedder.embed_query(query)
        if not query_vec:
            return []

        # Map preferred modality to collection
        collection_map = {
            "text":          "text",
            "text_to_table": "tables",
            "text_to_text":  "text",
            "text_to_image": "images_text",
            "image":         "images_text",
            "table":         "tables",
            "video_segment": "video_segs",
            "video_frame":   "video_frames",
        }
        coll_key = collection_map.get(preferred, "text")
        collection = self.store.collections.get(coll_key)
        if not collection:
            return []

        count = max(0, int(collection.count()))
        if count == 0:
            return []

        # Build WHERE filter — section_id preferred, doc_id fallback
        # For table modalities: skip section filter — tables float to any page
        # and the navigator rarely finds the exact table section. Use doc_id only.
        table_modalities = {"table", "text_to_table"}
        if preferred in table_modalities:
            where = self._build_where([], doc_ids)
        else:
            where = self._build_where(section_ids, doc_ids)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vec],
            "n_results":        min(self.top_k_chunks, count),
            "include":          ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            results = collection.query(**kwargs)
        except Exception as e:
            logger.warning(f"  Agent chunk fetch failed: {e}")
            return []

        return self._parse_results(results, coll_key, preferred)

    def _build_where(
        self, section_ids: List[str], doc_ids: List[str]
    ) -> Optional[Dict]:
        """Build ChromaDB WHERE clause — section_id first, doc_id fallback."""
        if section_ids:
            if len(section_ids) == 1:
                return {"structure_unit_id": {"$eq": section_ids[0]}}
            return {"structure_unit_id": {"$in": section_ids}}
        if doc_ids:
            if len(doc_ids) == 1:
                return {"doc_id": {"$eq": doc_ids[0]}}
            return {"doc_id": {"$in": doc_ids}}
        return None

    def _parse_results(
        self, results: Dict, coll_key: str, preferred: str
    ) -> List[RetrievedChunk]:
        """Parse ChromaDB results into RetrievedChunk objects."""
        from shared.models.metadata_models import (
            TextChunkMetadata, ImageChunkMetadata,
            TableChunkMetadata, VideoSegmentChunkMetadata, VideoFrameChunkMetadata,
        )

        chunks  = []
        ids     = results.get("ids",       [[]])[0]
        docs    = results.get("documents", [[]])[0]
        metas   = results.get("metadatas", [[]])[0]
        dists   = results.get("distances", [[]])[0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            score = round(max(0.0, 1.0 - float(dist)), 4)
            meta  = meta or {}

            try:
                if preferred in ("video_segment",):
                    metadata = VideoSegmentChunkMetadata(
                        chunk_id              = chunk_id,
                        doc_id                = str(meta.get("doc_id", "")),
                        chunk_index           = int(meta.get("chunk_index", 0)),
                        structure_unit_id     = str(meta.get("structure_unit_id", "")),
                        transcript_text       = doc or "",
                        text_original_content = doc or "",
                        start_time_s          = float(meta.get("start_time_s", 0.0)),
                        end_time_s            = float(meta.get("end_time_s", 0.0)),
                        contextual_summary    = str(meta.get("contextual_summary", "")),
                    )
                elif preferred in ("video_frame",):
                    metadata = VideoFrameChunkMetadata(
                        chunk_id          = chunk_id,
                        doc_id            = str(meta.get("doc_id", "")),
                        chunk_index       = int(meta.get("chunk_index", 0)),
                        structure_unit_id = str(meta.get("structure_unit_id", "")),
                        timestamp_s       = float(meta.get("timestamp_s", 0.0)),
                        contextual_summary = str(meta.get("contextual_summary", "")),
                        image_caption     = str(meta.get("image_caption", "")),
                        frame_role        = str(meta.get("frame_role", "")),
                    )
                elif preferred == "image":
                    metadata = ImageChunkMetadata(
                        chunk_id          = chunk_id,
                        doc_id            = str(meta.get("doc_id", "")),
                        chunk_index       = int(meta.get("chunk_index", 0)),
                        structure_unit_id = str(meta.get("structure_unit_id", "")),
                        image_caption     = str(meta.get("image_caption", "")),
                        contextual_summary = str(meta.get("contextual_summary", "")),
                    )
                elif preferred == "table":
                    _tsc = float(meta.get("table_summary_confidence") or 0.0) or None
                    metadata = TableChunkMetadata(
                        chunk_id                  = chunk_id,
                        doc_id                    = str(meta.get("doc_id", "")),
                        chunk_index               = int(meta.get("chunk_index", 0)),
                        structure_unit_id         = str(meta.get("structure_unit_id", "")),
                        table_summary             = str(meta.get("table_summary", "")),
                        table_summary_confidence  = _tsc,
                        table_purpose             = str(meta.get("table_purpose", "")),
                        table_html                = meta.get("table_html") or None,
                        page_number               = int(meta.get("page_number") or 0) or None,
                        salience_score            = _tsc,
                    )
                else:
                    metadata = TextChunkMetadata(
                        chunk_id              = chunk_id,
                        doc_id                = str(meta.get("doc_id", "")),
                        chunk_index           = int(meta.get("chunk_index", 0)),
                        structure_unit_id     = str(meta.get("structure_unit_id", "")),
                        text_original_content = doc or "",
                        section_id            = str(meta.get("section_id", "")),
                        section_title         = str(meta.get("section_title", "")),
                        contextual_summary    = str(meta.get("contextual_summary", "")),
                        contextual_summary_confidence = float(meta.get("contextual_summary_confidence") or 0.5),
                        salience_score        = float(meta.get("salience_score", 0.5)),
                        evidence_role         = str(meta.get("evidence_role", "context")),
                        page_number           = int(meta.get("page_number") or 0) or None,
                    )
            except Exception as e:
                logger.debug(f"  Chunk parse error {chunk_id}: {e}")
                continue

            chunks.append(RetrievedChunk(
                plugin_name      = "agent",
                retrieval_mode   = "agent",
                collection_name  = coll_key,
                metadata         = metadata,
                score_breakdown  = ScoreBreakdown(
                    vector_score = score,
                    final_score  = score,
                ),
                content          = doc or "",
                extra_payload    = dict(meta),
            ))

        return chunks

    def _check_sufficiency(
        self,
        question: str,
        role:     str,
        chunks:   List[RetrievedChunk],
    ) -> Tuple[str, List[str], bool, Optional[str]]:
        """
        Ask LLM if retrieved evidence is sufficient.

        Returns:
            (sufficiency, missing_aspects, role_match, refinement_query)
        """
        # Build evidence text for evaluation
        evidence_parts = []
        for i, c in enumerate(chunks, 1):
            meta    = c.metadata
            content = c.content or getattr(meta, "text_original_content", "") or ""
            summary = getattr(meta, "contextual_summary", "") or ""
            e_role  = getattr(meta, "evidence_role", "") or ""
            evidence_parts.append(
                f"[{i}] role={e_role} | {summary[:100]} | {content[:200]}"
            )

        prompt = SUFFICIENCY_PROMPT.format(
            sub_question    = question,
            evidence_role   = role,
            evidence_text   = "\n".join(evidence_parts),
        )
        response = self.llm.invoke(
            system     = SUFFICIENCY_SYSTEM,
            prompt     = prompt,
            max_tokens = 400,
        )
        data = _extract_json(response)
        if not data:
            return "partial", [], True, None

        sufficiency   = data.get("sufficiency", "partial")
        missing       = data.get("missing_aspects", []) or []
        role_match    = bool(data.get("evidence_role_match", True))
        refinement    = data.get("refinement_query")

        logger.debug(
            f"  Sufficiency: {sufficiency} | role_match={role_match} "
            f"| missing={len(missing)}"
        )
        return sufficiency, missing, role_match, refinement


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
