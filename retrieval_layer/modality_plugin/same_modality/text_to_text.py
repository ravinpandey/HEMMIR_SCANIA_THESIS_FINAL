from __future__ import annotations

from retrieval_layer.modality_plugin.base import ModalityPlugin
from shared.models.pipeline_models import EvidenceItem, RetrievedChunk, ScoreBreakdown
from shared.models.metadata_models import TextChunkMetadata


class TextToTextPlugin(ModalityPlugin):
    modality_name = "text_to_text"

    def embed_query(self, query: str) -> list[float]:
        if not self.text_embedder:
            return []
        text = (self.query_context.query_text if self.query_context else query) or query
        return self.text_embedder.embed_query(text)

    def retrieve(self, db, query_vector: list[float], doc_ids: list[str], top_k: int) -> list[RetrievedChunk]:
        results = self._query_collection(db, "text", query_vector, doc_ids, top_k)
        items: list[RetrievedChunk] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for chunk_id, content, meta, dist in zip(ids, docs, metas, dists):
            metadata = TextChunkMetadata(
                chunk_id=chunk_id,
                doc_id=str(meta.get("doc_id", "")),
                chunk_index=int(meta.get("chunk_index", 0)),
                source_modality="text",
                text_original_content=content or "",
                local_context=str(meta.get("section_path") or ""),
                page_number=int(meta.get("page_number") or 0),
                section_title=str(meta.get("section_title") or ""),
                chunk_strategy=str(meta.get("chunk_strategy") or "paragraph"),
                token_count=int(meta.get("token_count") or 0),
                related_figures=self._decode_json_list(meta.get("related_figures")),
                related_tables=self._decode_json_list(meta.get("related_tables")),
                section_id=str(meta.get("section_id") or ""),
                contextual_summary=str(meta.get("contextual_summary") or ""),
                contextual_summary_confidence=float(meta.get("contextual_summary_confidence") or 0.0),
                related_figures_str=meta.get("related_figures"),
                related_tables_str=meta.get("related_tables"),
            )
            score = round(max(0.0, 1.0 - float(dist)), 4)
            items.append(
                RetrievedChunk(
                    plugin_name=self.modality_name,
                    retrieval_mode="rag",
                    collection_name="text",
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
                f"[TEXT | chunk_id: {item.chunk_id} | doc_id: {item.doc_id}]",
                f"Content: {item.content}",
                f"Summary: {meta.contextual_summary or ''}",
                f"Section: {meta.section_title or item.extra_payload.get('section_title', '')}",
            ]
        )