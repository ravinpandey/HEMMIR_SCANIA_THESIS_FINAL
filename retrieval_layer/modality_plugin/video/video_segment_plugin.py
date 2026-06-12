"""
retrieval_layer/modality_plugin/video/video_segment_plugin.py

Video segment retrieval plugin.

Searches the video_segments collection using text embeddings of
contextual_summary + transcript_text (built by embedding layer).

Returns VideoSegmentChunkMetadata with:
  - start_time_s, end_time_s: for temporal citation in generation
  - keyframe_ids: links to associated video frames
  - scene_sibling_ids: other segments in the same scene
  - contextual_summary: enriched description of what is being discussed
  - evidence_role: set by video_segment_enricher
  - salience_score: set by video_segment_enricher

format_prompt_block includes timestamp so generation layer can produce
citations like: "At 4:32 [seg_0012], the instructor demonstrates..."
"""

from __future__ import annotations

from typing import Any, Dict, List

from shared.models.metadata_models import VideoSegmentChunkMetadata
from shared.models.pipeline_models import EvidenceItem, RetrievedChunk, ScoreBreakdown
from retrieval_layer.modality_plugin.base import ModalityPlugin


class VideoSegmentPlugin(ModalityPlugin):
    """
    Retrieves video transcript segments for temporal queries.
    Uses text embeddings (1536-dim) of enriched segment content.
    """
    modality_name = "video_segment"

    def embed_query(self, query: str) -> List[float]:
        if not self.text_embedder:
            return []
        text = (self.query_context.query_text if self.query_context else query) or query
        return self.text_embedder.embed_query(text)

    def retrieve(
        self,
        db,
        query_vector: List[float],
        doc_ids:      List[str],
        top_k:        int,
    ) -> List[RetrievedChunk]:
        results = self._query_collection(db, "video_segs", query_vector, doc_ids, top_k)
        items:  List[RetrievedChunk] = []

        ids   = results.get("ids",       [[]])[0]
        docs  = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for chunk_id, content, meta, dist in zip(ids, docs, metas, dists):
            meta  = meta or {}
            score = round(max(0.0, 1.0 - float(dist)), 4)

            try:
                metadata = VideoSegmentChunkMetadata(
                    chunk_id              = chunk_id,
                    doc_id                = str(meta.get("doc_id", "")),
                    chunk_index           = int(meta.get("chunk_index", 0)),
                    structure_unit_id     = str(meta.get("structure_unit_id", "")),
                    file_format           = str(meta.get("file_format", "mp4")),
                    segment_index         = int(meta.get("segment_index", 0)),
                    start_time_s          = float(meta.get("start_time_s", 0.0)),
                    end_time_s            = float(meta.get("end_time_s", 0.0)),
                    duration_s            = float(meta.get("end_time_s", 0.0)) - float(meta.get("start_time_s", 0.0)),
                    transcript_text       = content or "",
                    text_original_content = content or "",
                    asr_language          = str(meta.get("asr_language", "en")),
                    asr_confidence        = _safe_float(meta.get("asr_confidence")),
                    contextual_summary    = str(meta.get("contextual_summary", "")),
                    contextual_summary_confidence = _safe_float(meta.get("contextual_summary_confidence")),
                    keyframe_ids          = self._decode_json_list(meta.get("keyframe_ids")),
                    scene_sibling_ids     = self._decode_json_list(meta.get("scene_sibling_ids")),
                    evidence_role         = str(meta.get("evidence_role", "context")),
                    salience_score        = _safe_float(meta.get("salience_score")),
                    detected_codes        = self._decode_json_list(meta.get("detected_codes")),
                )
            except Exception:
                continue

            items.append(RetrievedChunk(
                plugin_name      = self.modality_name,
                retrieval_mode   = "rag",
                collection_name  = "video_segs",
                metadata         = metadata,
                score_breakdown  = ScoreBreakdown(vector_score=score, final_score=score),
                content          = content or "",
                extra_payload    = dict(meta),
            ))

        return items

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
        meta   = item.metadata
        start  = getattr(meta, "start_time_s", 0.0)
        end    = getattr(meta, "end_time_s", 0.0)
        ts     = f"{_fmt_time(start)} – {_fmt_time(end)}"
        summary = getattr(meta, "contextual_summary", "") or ""
        role    = getattr(meta, "evidence_role", "") or ""
        sal     = getattr(meta, "salience_score", 0.0) or 0.0

        return "\n".join([
            f"[VIDEO SEGMENT | chunk_id: {item.chunk_id} | doc_id: {item.doc_id}]",
            f"Timestamp: {ts}",
            f"Evidence role: {role} | Salience: {sal:.2f}",
            f"Context: {summary}",
            f"Transcript: {item.content[:400]}",
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