"""
retrieval_layer/modality_plugin/video/video_frame_plugin.py

Video keyframe retrieval plugin.

Dual-index search:
  1. video_frames_clip (512-dim CLIP) — visual similarity
  2. image_chunks_text filtered to source_modality=video_frame — caption/summary text

Results merged by chunk_id, best score wins.

Returns VideoFrameChunkMetadata with:
  - timestamp_s: exact frame timestamp for citation
  - frame_role: diagram/demo/talking_head/equipment etc.
  - parent_id: → parent VideoSegmentChunk (for temporal context)
  - image_caption + contextual_summary: from video_frame_enricher
  - contextual_summary includes: what is visible + what speaker is saying

format_prompt_block produces:
  [VIDEO FRAME | chunk_id | doc_id]
  Timestamp: 4:32
  Frame role: demonstration
  Caption: <what is visible>
  Context: <what speaker is saying at this moment>
"""

from __future__ import annotations

from typing import Any, Dict, List

from shared.models.metadata_models import VideoFrameChunkMetadata
from shared.models.pipeline_models import EvidenceItem, RetrievedChunk, ScoreBreakdown
from retrieval_layer.modality_plugin.base import ModalityPlugin


class VideoFramePlugin(ModalityPlugin):
    """
    Retrieves video keyframes for visual temporal queries.
    Searches CLIP collection (visual) + text collection (caption/summary).
    Merges results — chunks appearing in both get combined score.
    """
    modality_name = "video_frame"

    def embed_query(self, query: str) -> List[float]:
        """CLIP text embedding for visual search."""
        text = (self.query_context.query_text if self.query_context else query) or query
        return self._embed_clip_text(text)

    def retrieve(
        self,
        db,
        query_vector: List[float],
        doc_ids:      List[str],
        top_k:        int,
    ) -> List[RetrievedChunk]:
        candidates: Dict[str, RetrievedChunk] = {}

        # ── Branch 1: CLIP visual search ──────────────────────────────
        if query_vector:
            clip_results = self._query_collection(
                db, "video_frames", query_vector, doc_ids, top_k
            )
            self._parse_into(clip_results, candidates, "video_frames", "clip")

        # ── Branch 2: Text search on caption + summary ─────────────────
        if self.text_embedder and self.query_context:
            text_vec = self.text_embedder.embed_query(
                self.query_context.query_text or ""
            )
            if text_vec:
                # Filter image_chunks_text to video frames only
                text_results = self._query_collection_with_filter(
                    db, "images_text", text_vec, doc_ids, top_k,
                    extra_filter={"source_modality": {"$eq": "video_frame"}},
                )
                self._parse_into(text_results, candidates, "images_text", "text")

        # Sort by final_score
        return sorted(
            candidates.values(),
            key=lambda c: c.score_breakdown.final_score,
            reverse=True,
        )[:top_k]

    def _parse_into(
        self,
        results:     Dict,
        candidates:  Dict[str, RetrievedChunk],
        coll_name:   str,
        embed_type:  str,
    ) -> None:
        ids   = results.get("ids",       [[]])[0]
        docs  = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for chunk_id, content, meta, dist in zip(ids, docs, metas, dists):
            meta  = meta or {}
            score = round(max(0.0, 1.0 - float(dist)), 4)

            try:
                metadata = VideoFrameChunkMetadata(
                    chunk_id          = chunk_id,
                    doc_id            = str(meta.get("doc_id", "")),
                    chunk_index       = int(meta.get("chunk_index", 0)),
                    structure_unit_id = str(meta.get("structure_unit_id", "")),
                    file_format       = str(meta.get("file_format", "mp4")),
                    parent_id         = str(meta.get("parent_id", "")),
                    segment_index     = int(meta.get("segment_index", 0)),
                    frame_position    = int(meta.get("frame_position", 1)),
                    timestamp_s       = float(meta.get("timestamp_s", 0.0)),
                    segment_start_s   = float(meta.get("segment_start_s", 0.0)),
                    segment_end_s     = float(meta.get("segment_end_s", 0.0)),
                    image_path        = str(meta.get("image_path", "")),
                    image_caption     = str(meta.get("image_caption", "")),
                    image_caption_confidence = _safe_float(meta.get("image_caption_confidence")),
                    contextual_summary = str(meta.get("contextual_summary", "")),
                    contextual_summary_confidence = _safe_float(meta.get("contextual_summary_confidence")),
                    frame_role        = str(meta.get("frame_role", "other")),
                )
            except Exception:
                continue

            existing = candidates.get(chunk_id)
            if existing is None:
                # First time seeing this chunk
                candidates[chunk_id] = RetrievedChunk(
                    plugin_name      = self.modality_name,
                    retrieval_mode   = "rag",
                    collection_name  = coll_name,
                    metadata         = metadata,
                    score_breakdown  = ScoreBreakdown(
                        vector_score = score,
                        final_score  = score,
                    ),
                    content          = content or "",
                    extra_payload    = dict(meta),
                )
            else:
                # Chunk found in both branches — take max score
                existing_score = existing.score_breakdown.vector_score
                best_score     = max(existing_score, score)
                candidates[chunk_id] = existing.model_copy(update={
                    "score_breakdown": ScoreBreakdown(
                        vector_score = best_score,
                        final_score  = best_score,
                    )
                })

    def _query_collection_with_filter(
        self,
        db,
        collection_name: str,
        query_vector:    List[float],
        doc_ids:         List[str],
        top_k:           int,
        extra_filter:    Dict,
    ) -> Dict:
        """Query collection with both doc_id and extra filters combined."""
        count = self._collection_count(db, collection_name)
        if count == 0 or not query_vector:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        where_parts = [extra_filter]
        if doc_ids:
            if len(doc_ids) == 1:
                where_parts.append({"doc_id": {"$eq": doc_ids[0]}})
            else:
                where_parts.append({"doc_id": {"$in": doc_ids}})

        where = {"$and": where_parts} if len(where_parts) > 1 else where_parts[0]

        try:
            return db.collections[collection_name].query(
                query_embeddings=[query_vector],
                n_results=min(top_k, count),
                where=where,
            )
        except Exception:
            # If filter fails, fall back to no doc_id filter
            try:
                return db.collections[collection_name].query(
                    query_embeddings=[query_vector],
                    n_results=min(top_k, count),
                    where=extra_filter,
                )
            except Exception:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def build_evidence_item(self, chunk: RetrievedChunk) -> EvidenceItem:
        meta = chunk.metadata
        return EvidenceItem(
            chunk_id         = meta.chunk_id,
            doc_id           = meta.doc_id,
            chunk_index      = meta.chunk_index,
            source_modality  = meta.source_modality,
            score            = chunk.score_breakdown.final_score,
            score_breakdown  = chunk.score_breakdown,
            metadata         = meta,
            content          = chunk.content,
            retrieval_plugin = chunk.plugin_name,
            retrieval_mode   = chunk.retrieval_mode,
            collection_name  = chunk.collection_name,
            extra_payload    = chunk.extra_payload,
        )

    def format_prompt_block(self, item: EvidenceItem) -> str:
        meta      = item.metadata
        timestamp = getattr(meta, "timestamp_s", 0.0)
        role      = getattr(meta, "frame_role", "") or ""
        caption   = getattr(meta, "image_caption", "") or ""
        summary   = getattr(meta, "contextual_summary", "") or ""

        return "\n".join([
            f"[VIDEO FRAME | chunk_id: {item.chunk_id} | doc_id: {item.doc_id}]",
            f"Timestamp: {_fmt_time(timestamp)}",
            f"Frame role: {role}",
            f"Caption: {caption}",
            f"Context (what speaker is saying): {summary[:300]}",
        ])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _safe_float(v: Any, default: float = 0.5) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
