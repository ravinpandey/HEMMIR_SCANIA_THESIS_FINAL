from __future__ import annotations

from retrieval_layer.modality_plugin.base import ModalityPlugin
from retrieval_layer.modality_plugin.same_modality.image_to_image import ImageToImagePlugin
from retrieval_layer.modality_plugin.same_modality.text_to_text import TextToTextPlugin
from shared.models.metadata_models import ImageChunkMetadata
from shared.models.pipeline_models import EvidenceItem, RetrievedChunk, ScoreBreakdown


class TextToImagePlugin(ModalityPlugin):
    modality_name = "text_to_image"

    def __init__(self, text_embedder=None, image_embedder=None, bedrock_client=None, store=None):
        super().__init__(text_embedder=text_embedder, image_embedder=image_embedder, bedrock_client=bedrock_client, store=store)
        self._text_plugin = TextToTextPlugin(
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            bedrock_client=bedrock_client,
        )
        self._image_prompt_plugin = ImageToImagePlugin(
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            bedrock_client=bedrock_client,
        )

    def set_query_context(self, context) -> None:
        super().set_query_context(context)
        self._text_plugin.set_query_context(context)
        self._image_prompt_plugin.set_query_context(context)

    def embed_query(self, query: str) -> list[float]:
        text = (self.query_context.query_text if self.query_context else query) or query
        return self._embed_clip_text(text)

    def retrieve(self, db, query_vector: list[float], doc_ids: list[str], top_k: int) -> list[RetrievedChunk]:
        query_text = (self.query_context.query_text if self.query_context else "") or ""

        text_query_vector = self._text_plugin.embed_query(query_text)
        text_support = self._text_plugin.retrieve(db, text_query_vector, doc_ids, top_k)
        annotated_text = self._annotate_cross_modal_results(
            text_support,
            branch="support_text",
            branch_plugin=self._text_plugin.modality_name,
            branch_query=query_text,
        )

        target_query = self._augment_query_from_text(query_text, text_support)
        image_hits = self._retrieve_image_branch(db, target_query, doc_ids, top_k)
        annotated_images = self._annotate_cross_modal_results(
            image_hits,
            branch="target_image",
            branch_plugin=self._image_prompt_plugin.modality_name,
            branch_query=target_query,
        )

        return self._dedupe_cross_modal_results(annotated_text + annotated_images, top_k * 2)

    def build_evidence_item(self, chunk: RetrievedChunk) -> EvidenceItem:
        source_modality = chunk.source_modality
        if source_modality == "text":
            item = self._text_plugin.build_evidence_item(chunk.model_copy(update={"plugin_name": self._text_plugin.modality_name}))
        else:
            item = self._image_prompt_plugin.build_evidence_item(
                chunk.model_copy(update={"plugin_name": self._image_prompt_plugin.modality_name})
            )
        item.retrieval_plugin = self.modality_name
        item.extra_payload.update(chunk.extra_payload)
        return item

    def format_prompt_block(self, item: EvidenceItem) -> str:
        if item.source_modality == "text":
            routed = item.model_copy(update={"retrieval_plugin": self._text_plugin.modality_name})
            return self._text_plugin.format_prompt_block(routed)
        routed = item.model_copy(update={"retrieval_plugin": self._image_prompt_plugin.modality_name})
        return self._image_prompt_plugin.format_prompt_block(routed)

    def _augment_query_from_text(self, query_text: str, text_support: list[RetrievedChunk]) -> str:
        support_parts: list[str] = []
        for chunk in text_support[:3]:
            meta = chunk.metadata
            section = getattr(meta, "section_title", "") or chunk.extra_payload.get("section_title", "")
            summary = getattr(meta, "contextual_summary", "") or ""
            if section:
                support_parts.append(f"section: {section}")
            if summary:
                support_parts.append(summary[:160])
        support_text = " | ".join(part for part in support_parts if part)
        if not support_text:
            return query_text
        return f"{query_text}\nSupport context: {support_text}"

    def _retrieve_image_branch(self, db, target_query: str, doc_ids: list[str], top_k: int) -> list[RetrievedChunk]:
        candidates: dict[str, RetrievedChunk] = {}
        clip_vector = self._embed_clip_text(target_query)
        text_vector = self.text_embedder.embed_query(target_query) if self.text_embedder else []

        for collection_name, vector in (("images_clip", clip_vector), ("images_text", text_vector)):
            results = self._query_collection(db, collection_name, vector, doc_ids, top_k)
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
                    visible_annotations=str(meta.get("visible_annotations") or ""),
                    contextual_summary=str(meta.get("contextual_summary") or ""),
                    contextual_summary_confidence=float(meta.get("contextual_summary_confidence") or 0.0),
                    related_sections_str=meta.get("related_sections"),
                )
                score = round(max(0.0, 1.0 - float(dist)), 4)
                candidate = RetrievedChunk(
                    plugin_name=self.modality_name,
                    retrieval_mode="rag",
                    collection_name=collection_name,
                    metadata=metadata,
                    score_breakdown=ScoreBreakdown(vector_score=score, final_score=score),
                    content=content or "",
                    extra_payload=dict(meta or {}),
                )
                existing = candidates.get(chunk_id)
                if existing is None or candidate.score_breakdown.vector_score > existing.score_breakdown.vector_score:
                    candidates[chunk_id] = candidate
        return sorted(candidates.values(), key=lambda item: item.score_breakdown.vector_score, reverse=True)[:top_k]