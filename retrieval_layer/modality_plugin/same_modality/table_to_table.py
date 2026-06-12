from __future__ import annotations

from retrieval_layer.modality_plugin.base import ModalityPlugin
from shared.models.pipeline_models import EvidenceItem, RetrievedChunk, ScoreBreakdown
from shared.models.metadata_models import TableChunkMetadata


class TableToTablePlugin(ModalityPlugin):
    modality_name = "table_to_table"

    def embed_query(self, query: str) -> list[float]:
        if not self.text_embedder:
            return []
        text = ""
        if self.query_context:
            text = self.query_context.query_table_text or self.query_context.query_text
        return self.text_embedder.embed_query(text or query)

    def retrieve(self, db, query_vector: list[float], doc_ids: list[str], top_k: int) -> list[RetrievedChunk]:
        results = self._query_collection(db, "tables", query_vector, doc_ids, top_k * 2)
        items: dict[str, RetrievedChunk] = {}
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for chunk_id, content, meta, dist in zip(ids, docs, metas, dists):
            base_chunk_id = str(chunk_id).replace("_html", "")
            metadata = TableChunkMetadata(
                chunk_id=base_chunk_id,
                doc_id=str(meta.get("doc_id", "")),
                chunk_index=int(meta.get("chunk_index", 0)),
                source_modality="table",
                table_html=meta.get("table_html"),
                page_number=int(meta.get("page_number") or 0),
                row_count=int(meta.get("row_count") or 0),
                col_count=int(meta.get("col_count") or 0),
                table_caption=str(meta.get("table_caption") or ""),
                table_summary=str(meta.get("table_summary") or ""),
                table_summary_confidence=float(meta.get("table_summary_confidence") or 0.0),
                table_purpose=str(meta.get("table_purpose") or ""),
                markdown=meta.get("markdown"),
            )
            score = round(max(0.0, 1.0 - float(dist)), 4)
            candidate = RetrievedChunk(
                plugin_name=self.modality_name,
                retrieval_mode="rag",
                collection_name="tables",
                metadata=metadata,
                score_breakdown=ScoreBreakdown(vector_score=score, final_score=score),
                content=content or "",
                extra_payload=dict(meta or {}),
            )
            existing = items.get(base_chunk_id)
            if existing is None or candidate.score_breakdown.vector_score > existing.score_breakdown.vector_score:
                items[base_chunk_id] = candidate
        return list(items.values())[:top_k]

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
        table_body = meta.markdown or item.content
        return "\n".join(
            [
                f"[TABLE | chunk_id: {item.chunk_id} | doc_id: {item.doc_id}]",
                f"Summary: {meta.table_summary or ''}",
                f"Purpose: {meta.table_purpose or ''}",
                f"Data: {table_body}",
            ]
        )
