from __future__ import annotations

from retrieval_layer.modality_plugin.base import ModalityPlugin
from retrieval_layer.modality_plugin.same_modality.table_to_table import TableToTablePlugin
from retrieval_layer.modality_plugin.same_modality.text_to_text import TextToTextPlugin
from shared.models.pipeline_models import EvidenceItem, RetrievedChunk


class TextToTablePlugin(ModalityPlugin):
    modality_name = "text_to_table"

    def __init__(self, text_embedder=None, image_embedder=None, bedrock_client=None, store=None):
        super().__init__(
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            bedrock_client=bedrock_client,
            store=store,
        )
        # FIX: pass store to inner plugins so _query_collection has a
        # resolved ChromaStore when db=None is passed from the pipeline.
        # Previously store was never forwarded, causing _collection_count
        # to return 0 and retrieve() to return 0 chunks every time.
        self._text_plugin = TextToTextPlugin(
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            bedrock_client=bedrock_client,
            store=store,
        )
        self._table_plugin = TableToTablePlugin(
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            bedrock_client=bedrock_client,
            store=store,
        )

    def set_query_context(self, context) -> None:
        super().set_query_context(context)
        self._text_plugin.set_query_context(context)
        self._table_plugin.set_query_context(context)

    def embed_query(self, query: str) -> list[float]:
        return self._text_plugin.embed_query(query)

    def retrieve(self, db, query_vector: list[float], doc_ids: list[str], top_k: int) -> list[RetrievedChunk]:
        query_text = (self.query_context.query_text if self.query_context else "") or ""

        text_support = self._text_plugin.retrieve(db, query_vector, doc_ids, top_k)
        annotated_text = self._annotate_cross_modal_results(
            text_support,
            branch="support_text",
            branch_plugin=self._text_plugin.modality_name,
            branch_query=query_text,
        )

        target_query = self._augment_query_from_text(query_text, text_support)
        table_context = self._copy_context(query_table_text=target_query)
        self._table_plugin.set_query_context(table_context)
        table_vector = self._table_plugin.embed_query(target_query)
        table_hits = self._table_plugin.retrieve(db, table_vector, doc_ids, top_k)
        self._table_plugin.set_query_context(self.query_context)

        annotated_tables = self._annotate_cross_modal_results(
            table_hits,
            branch="target_table",
            branch_plugin=self._table_plugin.modality_name,
            branch_query=target_query,
        )

        return self._dedupe_cross_modal_results(annotated_text + annotated_tables, top_k * 2)

    def build_evidence_item(self, chunk: RetrievedChunk) -> EvidenceItem:
        if chunk.source_modality == "text":
            item = self._text_plugin.build_evidence_item(
                chunk.model_copy(update={"plugin_name": self._text_plugin.modality_name})
            )
        else:
            item = self._table_plugin.build_evidence_item(
                chunk.model_copy(update={"plugin_name": self._table_plugin.modality_name})
            )
        item.retrieval_plugin = self.modality_name
        item.extra_payload.update(chunk.extra_payload)
        return item

    def format_prompt_block(self, item: EvidenceItem) -> str:
        if item.source_modality == "text":
            routed = item.model_copy(
                update={"retrieval_plugin": self._text_plugin.modality_name}
            )
            return self._text_plugin.format_prompt_block(routed)
        routed = item.model_copy(
            update={"retrieval_plugin": self._table_plugin.modality_name}
        )
        return self._table_plugin.format_prompt_block(routed)

    def _augment_query_from_text(
        self, query_text: str, text_support: list[RetrievedChunk]
    ) -> str:
        support_parts: list[str] = []
        for chunk in text_support[:3]:
            meta    = chunk.metadata
            summary = getattr(meta, "contextual_summary", "") or ""
            section = (
                getattr(meta, "section_title", "")
                or chunk.extra_payload.get("section_title", "")
            )
            if section:
                support_parts.append(f"section: {section}")
            if summary:
                support_parts.append(summary[:180])
        support_text = " | ".join(part for part in support_parts if part)
        if not support_text:
            return query_text
        return f"{query_text}\nRelevant text support: {support_text}"
