"""
retrieval_layer/RAG/retrieve_docs.py

3-Stage retrieval — Document → Section → Chunk.

Stage 1: retrieve_documents() — finds relevant documents
Stage 2: retrieve_sections() — finds relevant sections within those documents
Stage 3: chunk retrieval uses doc_ids + section_ids as filters (in base.py)
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger

from shared.models.metadata_models import DocumentMetadata
from shared.models.pipeline_models import (
    DocCandidate,
    Filters,
    QueryAnalysis,
)


# ── Stage 1: Document retrieval ───────────────────────────────────────────────

def retrieve_documents(
    store,
    text_embedder,
    analysis: QueryAnalysis,
    top_k:    int,
) -> List[DocCandidate]:
    """Find top-k most relevant documents for the query."""
    # Fast path: when Filters.doc_id is set, skip vector search entirely.
    # Used by RQ1 evaluation where the target paper's Chroma item-ID is known.
    _f = analysis.filters if hasattr(analysis, "filters") else None
    if _f and getattr(_f, "doc_id", None):
        logger.info(f"  retrieve_docs: doc_id override — returning [{_f.doc_id}] directly")
        return [DocCandidate(doc_id=_f.doc_id, score=1.0)]

    if not text_embedder:
        logger.warning("  retrieve_docs: no text_embedder — returning all documents")
        return _fallback_all_documents(store, top_k)

    query_vec = text_embedder.embed_query(analysis.rewritten_query)
    if not query_vec:
        logger.warning("  retrieve_docs: embed_query returned empty — fallback")
        return _fallback_all_documents(store, top_k)

    collection = store.collections.get("documents")
    if not collection:
        logger.error("  retrieve_docs: documents collection not found")
        return []

    count = max(0, int(collection.count()))
    if count == 0:
        logger.warning("  retrieve_docs: documents collection is empty")
        return []

    kwargs = {
        "query_embeddings": [query_vec],
        "n_results":        min(top_k, count),
    }
    where = _build_doc_where(analysis.filters)
    if where:
        kwargs["where"] = where

    try:
        results = collection.query(**kwargs)
    except Exception as e:
        logger.warning(f"  retrieve_docs: query failed: {e} — fallback")
        return _fallback_all_documents(store, top_k)

    ids   = results.get("ids",       [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    candidates: List[DocCandidate] = []
    for doc_id, meta, dist in zip(ids, metas, dists):
        score = round(max(0.0, 1.0 - float(dist)), 4)
        candidates.append(DocCandidate(
            doc_id   = doc_id,
            score    = score,
            metadata = _doc_metadata_from_meta(doc_id, meta or {}),
        ))

    if candidates:
        MIN_DOC_SCORE = 0.20
        filtered = [c for c in candidates if c.score >= MIN_DOC_SCORE]
        if not filtered:
            filtered = candidates[:1]
        if len(filtered) < len(candidates):
            logger.info(
                f"  retrieve_docs: filtered {len(candidates)} → {len(filtered)} "
                f"documents (min_score={MIN_DOC_SCORE})"
            )
        logger.info(f"  retrieve_docs: {len(filtered)} documents found")
        return filtered

    _f = analysis.filters if hasattr(analysis, "filters") else None
    if _f and getattr(_f, "doc_title", None):
        logger.info("  retrieve_docs: no vector match but doc_title filter active — using filtered fallback")
        return _fallback_all_documents(store, top_k, where={"doc_title": {"$eq": _f.doc_title}})

    logger.info("  retrieve_docs: no vector match — using all documents")
    return _fallback_all_documents(store, top_k)


def extract_doc_ids(candidates: List[DocCandidate]) -> List[str]:
    return [c.doc_id for c in candidates]


# ── Stage 2: Section retrieval (NEW) ─────────────────────────────────────────

def retrieve_sections(
    store,
    text_embedder,
    query:     str,
    doc_ids:   List[str],
    top_k:     int = 10,
) -> List[str]:
    """
    Find top-k most relevant section IDs within the given documents.

    Searches sections_collection filtered by doc_id.
    Returns section IDs (structure_unit_ids) for use as chunk filter.

    Args:
        store:         ChromaStore with sections_collection
        text_embedder: TextEmbedder for query embedding
        query:         Rewritten query text
        doc_ids:       Document IDs to search within (from Stage 1)
        top_k:         Max sections to return

    Returns:
        List[str] of section IDs sorted by relevance.
    """
    if not text_embedder or not doc_ids:
        return []

    collection = store.collections.get("sections")
    if not collection:
        logger.debug("  retrieve_sections: sections collection not found — skipping Stage 2")
        return []

    count = max(0, int(collection.count()))
    if count == 0:
        return []

    query_vec = text_embedder.embed_query(query)
    if not query_vec:
        return []

    # Filter sections to only those belonging to retrieved documents
    if len(doc_ids) == 1:
        where = {"doc_id": {"$eq": doc_ids[0]}}
    else:
        where = {"doc_id": {"$in": doc_ids}}

    try:
        results = collection.query(
            query_embeddings = [query_vec],
            n_results        = min(top_k, count),
            where            = where,
            include          = ["metadatas", "distances"],
        )
    except Exception as e:
        logger.warning(f"  retrieve_sections: query failed: {e}")
        return []

    ids   = results.get("ids",       [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    # Only keep sections with good relevance score
    MIN_SECTION_SCORE = 0.30
    section_ids = []
    for sec_id, meta, dist in zip(ids, metas, dists):
        score = round(max(0.0, 1.0 - float(dist)), 4)
        if score >= MIN_SECTION_SCORE:
            section_ids.append(sec_id)

    # Always keep at least top-3 sections even if below threshold
    if not section_ids and ids:
        section_ids = list(ids[:3])

    logger.info(
        f"  retrieve_sections: {len(section_ids)} sections found "
        f"(from {len(doc_ids)} docs)"
    )
    return section_ids


# ── Shared utilities ──────────────────────────────────────────────────────────

def _build_doc_where(filters: Filters) -> Optional[dict]:
    conditions = []
    if filters.document_type:
        conditions.append({"document_type": {"$eq": filters.document_type}})
    if filters.tier1:
        conditions.append({"tier1": {"$eq": filters.tier1}})
    if filters.tier2:
        conditions.append({"tier2": {"$eq": filters.tier2}})
    if filters.project_id:
        conditions.append({"project_id": {"$eq": filters.project_id}})
    if filters.doc_title:
        conditions.append({"doc_title": {"$eq": filters.doc_title}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _fallback_all_documents(store, top_k: int, where: dict = None) -> List[DocCandidate]:
    """Return all documents with score=0.0 when vector search fails."""
    try:
        collection = store.collections.get("documents")
        if not collection:
            return []
        get_kwargs = {"include": ["metadatas"]}
        if where:
            get_kwargs["where"] = where
        payload = collection.get(**get_kwargs)
        candidates = []
        for doc_id, meta in zip(
            payload.get("ids", []),
            payload.get("metadatas", []),
        ):
            candidates.append(DocCandidate(
                doc_id   = doc_id,
                score    = 0.0,
                metadata = _doc_metadata_from_meta(doc_id, meta or {}),
            ))
        return candidates[:top_k]
    except Exception as e:
        logger.error(f"  retrieve_docs fallback failed: {e}")
        return []


def _doc_metadata_from_meta(doc_id: str, meta: dict) -> DocumentMetadata:
    return DocumentMetadata(
        doc_id           = doc_id,
        doc_title        = str(meta.get("doc_title", "")),
        source_id        = str(meta.get("source_id", "")),
        project_id       = meta.get("project_id"),
        author           = meta.get("author"),
        file_format      = str(meta.get("file_format", "")),
        document_type    = meta.get("document_type"),
        language         = str(meta.get("language") or "English"),
        total_pages      = int(meta.get("total_pages") or 0),
        chunk_count      = int(meta.get("chunk_count") or 0),
        doc_summary      = str(meta.get("doc_summary", ""))[:500],
        tier1            = meta.get("tier1"),
        tier2            = meta.get("tier2"),
        tier_confidence  = float(meta.get("tier_confidence") or 0.0),
        total_duration_s = float(meta.get("total_duration_s") or 0.0),
    )