from __future__ import annotations

from retrieval_layer.modality_plugin.base import ModalityPlugin
from shared.models.pipeline_models import EvidenceItem, RetrievedChunk, ScoreBreakdown
from shared.models.metadata_models import ImageChunkMetadata


class ImageToImagePlugin(ModalityPlugin):
    modality_name = "image_to_image"

    def embed_query(self, query: str) -> list[float]:
        if not self.query_context or not self.query_context.query_image_b64:
            raise ValueError("image_to_image requires query_image_b64 in QueryContext")
        return self._embed_clip_image(self.query_context.query_image_b64)

    def retrieve(self, db, query_vector: list[float], doc_ids: list[str], top_k: int) -> list[RetrievedChunk]:
        results = self._query_collection(db, "images_clip", query_vector, doc_ids, top_k)
        items: list[RetrievedChunk] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for chunk_id, content, meta, dist in zip(ids, docs, metas, dists):
            metadata = ImageChunkMetadata(
                chunk_id=chunk_id,
                doc_id=str(meta.get("doc_id", "")),
                chunk_index=int(meta.get("chunk_index", 0)),
                source_modality="image",
                figure_id=str(meta.get("figure_id") or ""),
                page_number=int(meta.get("page_number") or 0),
                image_type=str(meta.get("image_type") or ""),
                related_sections=self._decode_json_list(meta.get("related_sections")),
                image_caption=str(meta.get("image_caption") or ""),
                depicted_component=str(meta.get("depicted_component") or ""),
                contextual_summary=str(meta.get("contextual_summary") or ""),
                contextual_summary_confidence=float(meta.get("contextual_summary_confidence") or 0.0),
                related_sections_str=meta.get("related_sections"),
            )
            score = round(max(0.0, 1.0 - float(dist)), 4)
            items.append(
                RetrievedChunk(
                    plugin_name=self.modality_name,
                    retrieval_mode="rag",
                    collection_name="images_clip",
                    metadata=metadata,
                    score_breakdown=ScoreBreakdown(vector_score=score, final_score=score),
                    content=content or "",
                    extra_payload=dict(meta or {}),
                )
            )
        return items

    def build_evidence_item(self, chunk: RetrievedChunk) -> EvidenceItem:
        metadata = chunk.metadata
        return EvidenceItem(
            chunk_id=metadata.chunk_id,
            doc_id=metadata.doc_id,
            chunk_index=metadata.chunk_index,
            source_modality=metadata.source_modality,
            score=chunk.score_breakdown.final_score,
            score_breakdown=chunk.score_breakdown,
            metadata=metadata,
            content=chunk.content,
            retrieval_plugin=chunk.plugin_name,
            retrieval_mode=chunk.retrieval_mode,
            collection_name=chunk.collection_name,
            extra_payload=chunk.extra_payload,
        )

    def format_prompt_block(self, item: EvidenceItem) -> str:
        meta = item.metadata
        return "\n".join(
            [
                f"[IMAGE | chunk_id: {item.chunk_id} | doc_id: {item.doc_id}]",
                f"Caption: {meta.image_caption or ''}",
                f"Component: {meta.depicted_component or ''}",
                f"Context: {meta.contextual_summary or ''}",
            ]
        )
