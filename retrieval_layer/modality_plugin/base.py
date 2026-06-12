"""
retrieval_layer/modality_plugin/base.py

Updated to support section_ids filter in _build_where() and _query_collection().
This enables Stage 3 chunk retrieval filtered by both doc_id AND structure_unit_id.
"""
from __future__ import annotations

import abc
import base64
import io
import json
from typing import Any, List, Optional

from shared.models.pipeline_models import EvidenceItem, QueryContext, RetrievedChunk


class ModalityPlugin(abc.ABC):
    modality_name: str

    def __init__(self, text_embedder=None, image_embedder=None, bedrock_client=None, store=None):
        self.text_embedder  = text_embedder
        self.image_embedder = image_embedder
        self.bedrock_client = bedrock_client
        self.db             = store
        self.query_context: Optional[QueryContext] = None

    def set_query_context(self, context: QueryContext) -> None:
        self.query_context = context

    @abc.abstractmethod
    def embed_query(self, query: str) -> list[float]: ...

    @abc.abstractmethod
    def retrieve(self, db, query_vector: list[float], doc_ids: list[str], top_k: int) -> list[RetrievedChunk]: ...

    @abc.abstractmethod
    def build_evidence_item(self, chunk: RetrievedChunk) -> EvidenceItem: ...

    @abc.abstractmethod
    def format_prompt_block(self, item: EvidenceItem) -> str: ...

    def _build_where(
        self,
        doc_ids:     list[str],
        section_ids: Optional[List[str]] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Build ChromaDB where filter.

        Args:
            doc_ids:     Filter by document IDs (Stage 1 output)
            section_ids: Filter by section IDs (Stage 2 output) — optional

        Returns:
            ChromaDB where clause or None.
        """
        conditions = []

        # Doc filter
        if doc_ids:
            if len(doc_ids) == 1:
                conditions.append({"doc_id": {"$eq": doc_ids[0]}})
            else:
                conditions.append({"doc_id": {"$in": doc_ids}})

        # Section filter — uses structure_unit_id field in chunk metadata
        if section_ids:
            if len(section_ids) == 1:
                conditions.append({"structure_unit_id": {"$eq": section_ids[0]}})
            else:
                conditions.append({"structure_unit_id": {"$in": section_ids}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _collection_count(self, db, collection_name: str) -> int:
        resolved = db if db is not None else self.db
        try:
            return max(0, int(resolved.collections[collection_name].count()))
        except Exception:
            return 0

    def _query_collection(
        self,
        db,
        collection_name: str,
        query_vector:    list[float],
        doc_ids:         list[str],
        top_k:           int,
        section_ids:     Optional[List[str]] = None,
    ) -> dict[str, Any]:
        """
        Query a ChromaDB collection with doc + section filters.

        Args:
            section_ids: Optional Stage 2 section filter.
                         When provided, restricts search to chunks within
                         those sections only.
        """
        resolved = db if db is not None else self.db
        count    = self._collection_count(resolved, collection_name)
        if count == 0 or not query_vector:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results":        min(top_k, count),
        }
        # Use section_ids from parameter or from plugin instance attribute
        effective_section_ids = section_ids or getattr(self, "_section_ids", None) or []
        where = self._build_where(doc_ids, section_ids=effective_section_ids if effective_section_ids else None)
        if where:
            kwargs["where"] = where

        try:
            return resolved.collections[collection_name].query(**kwargs)
        except Exception as e:
            # Section filter may fail if structure_unit_id not indexed
            # Fall back to doc-only filter
            from loguru import logger
            logger.warning(f"  _query_collection with section filter failed: {e} — retrying without section filter")
            kwargs["where"] = self._build_where(doc_ids)
            if kwargs["where"] is None:
                del kwargs["where"]
            return resolved.collections[collection_name].query(**kwargs)

    def _decode_json_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if not value:
            return []
        try:
            decoded = json.loads(value)
            if isinstance(decoded, list):
                return [str(item) for item in decoded]
        except Exception:
            pass
        return []

    def _embed_clip_text(self, text: str) -> list[float]:
        if not self.image_embedder:
            return []
        try:
            import torch
            tokens = self.image_embedder._clip_tokenizer([text]).to(self.image_embedder._device)
            with torch.no_grad():
                features = self.image_embedder._clip_model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            return features[0].cpu().tolist()
        except Exception:
            return []

    def _embed_clip_image(self, image_b64: str) -> list[float]:
        if not self.image_embedder or not image_b64:
            return []
        try:
            import torch
            from PIL import Image as PILImage
            image_bytes = base64.b64decode(image_b64)
            image  = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self.image_embedder._preprocess(image).unsqueeze(0).to(self.image_embedder._device)
            with torch.no_grad():
                features = self.image_embedder._clip_model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            return features[0].cpu().tolist()
        except Exception:
            return []

    def _copy_context(self, **updates) -> QueryContext:
        if self.query_context is None:
            raise ValueError(f"{self.modality_name} requires QueryContext before retrieval")
        return self.query_context.model_copy(update=updates)

    def _annotate_cross_modal_results(
        self,
        chunks: list[RetrievedChunk],
        *,
        branch:         str,
        branch_plugin:  str,
        branch_query:   str,
    ) -> list[RetrievedChunk]:
        annotated: list[RetrievedChunk] = []
        for chunk in chunks:
            payload = dict(chunk.extra_payload)
            payload.update({
                "cross_modal_parent":        self.modality_name,
                "cross_modal_branch":        branch,
                "cross_modal_branch_plugin": branch_plugin,
                "cross_modal_query":         branch_query,
            })
            annotated.append(chunk.model_copy(update={
                "plugin_name":   self.modality_name,
                "extra_payload": payload,
            }))
        return annotated

    def _dedupe_cross_modal_results(self, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        deduped: dict[tuple[str, str], RetrievedChunk] = {}
        for chunk in chunks:
            key      = (chunk.source_modality, chunk.chunk_id)
            existing = deduped.get(key)
            if existing is None or chunk.score_breakdown.vector_score > existing.score_breakdown.vector_score:
                deduped[key] = chunk
        ordered = sorted(deduped.values(), key=lambda c: c.score_breakdown.vector_score, reverse=True)
        return ordered[:top_k]