from __future__ import annotations

from retrieval_layer.modality_plugin.base import ModalityPlugin
from retrieval_layer.modality_plugin.same_modality.image_to_image import ImageToImagePlugin
from retrieval_layer.modality_plugin.same_modality.text_to_text import TextToTextPlugin
from shared.models.pipeline_models import EvidenceItem, RetrievedChunk


class ImageToTextPlugin(ModalityPlugin):
    modality_name = "image_to_text"

    def __init__(self, text_embedder=None, image_embedder=None, bedrock_client=None, store=None):
        super().__init__(text_embedder=text_embedder, image_embedder=image_embedder, bedrock_client=bedrock_client, store=store)
        self._image_plugin = ImageToImagePlugin(
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            bedrock_client=bedrock_client,
        )
        self._text_plugin = TextToTextPlugin(
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            bedrock_client=bedrock_client,
        )

    def set_query_context(self, context) -> None:
        super().set_query_context(context)
        self._image_plugin.set_query_context(context)
        self._text_plugin.set_query_context(context)

    def embed_query(self, query: str) -> list[float]:
        return self._image_plugin.embed_query(query)

    def retrieve(self, db, query_vector: list[float], doc_ids: list[str], top_k: int) -> list[RetrievedChunk]:
        image_support = self._image_plugin.retrieve(db, query_vector, doc_ids, top_k)
        annotated_images = self._annotate_cross_modal_results(
            image_support,
            branch="support_image",
            branch_plugin=self._image_plugin.modality_name,
            branch_query="[query_image_b64]",
        )

        target_query = self._derive_text_query(image_support)
        if not target_query:
            return self._dedupe_cross_modal_results(annotated_images, top_k)

        text_context = self._copy_context(query_text=target_query)
        self._text_plugin.set_query_context(text_context)
        text_vector = self._text_plugin.embed_query(target_query)
        text_hits = self._text_plugin.retrieve(db, text_vector, doc_ids, top_k)
        self._text_plugin.set_query_context(self.query_context)

        annotated_text = self._annotate_cross_modal_results(
            text_hits,
            branch="target_text",
            branch_plugin=self._text_plugin.modality_name,
            branch_query=target_query,
        )

        return self._dedupe_cross_modal_results(annotated_images + annotated_text, top_k * 2)

    def build_evidence_item(self, chunk: RetrievedChunk) -> EvidenceItem:
        if chunk.source_modality == "image":
            item = self._image_plugin.build_evidence_item(chunk.model_copy(update={"plugin_name": self._image_plugin.modality_name}))
        else:
            item = self._text_plugin.build_evidence_item(chunk.model_copy(update={"plugin_name": self._text_plugin.modality_name}))
        item.retrieval_plugin = self.modality_name
        item.extra_payload.update(chunk.extra_payload)
        return item

    def format_prompt_block(self, item: EvidenceItem) -> str:
        if item.source_modality == "image":
            routed = item.model_copy(update={"retrieval_plugin": self._image_plugin.modality_name})
            return self._image_plugin.format_prompt_block(routed)
        routed = item.model_copy(update={"retrieval_plugin": self._text_plugin.modality_name})
        return self._text_plugin.format_prompt_block(routed)

    def _derive_text_query(self, image_support: list[RetrievedChunk]) -> str:
        image_b64 = self.query_context.query_image_b64 if self.query_context else None
        if image_b64 and self.bedrock_client:
            try:
                prompt = "Describe the key technical content of this image in 2 concise sentences for document retrieval."
                return self.bedrock_client.invoke_with_image(
                    prompt=prompt,
                    image_b64=image_b64,
                    media_type="image/png",
                    max_tokens=120,
                ).strip()
            except Exception:
                pass

        support_parts: list[str] = []
        for chunk in image_support[:3]:
            meta = chunk.metadata
            for value in (
                getattr(meta, "image_caption", "") or "",
                getattr(meta, "depicted_component", "") or "",
                getattr(meta, "visible_annotations", "") or "",
                getattr(meta, "contextual_summary", "") or "",
            ):
                if value:
                    support_parts.append(value[:180])
        return " | ".join(dict.fromkeys(part for part in support_parts if part))
