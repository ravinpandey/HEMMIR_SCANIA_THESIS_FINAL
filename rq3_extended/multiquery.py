"""
rq3_extended/multiquery.py

MultiQuery re-retrieval for HEMMIR self-correction.

Generates N query variants with the LLM (academic domain prompts),
retrieves from all three Chroma collections per variant, merges
using Reciprocal Rank Fusion, and returns the merged evidence in
the (texts, chunk_ids, modalities) format expected by R2-G.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from loguru import logger

EXPAND_SYSTEM = """\
You are an expert at reformulating questions for retrieval from technical documents.
Documents may include service manuals, technical reports, research papers, maintenance
procedures, specifications, and industrial guides.
Generate distinct query variants that cover different angles, vocabulary, and sub-aspects.
Respond ONLY with a JSON array of strings. No markdown, no explanation.\
"""

EXPAND_PROMPT = """\
Original question: {question}

Generate {n} different retrieval queries that together improve the chances of finding
all evidence needed to answer the original question. Each variant should:
- Use different vocabulary or framing (technical synonyms, abbreviations, full terms)
- Target a different sub-aspect of the question
- Be specific enough for technical document retrieval (include units, codes, or
  component names if present in the original question)

Return ONLY a JSON array: ["variant1", "variant2", ...]\
"""

GAP_EXPAND_PROMPT = """\
The following aspects were not covered in an initial answer:
{weak_claims}

Original question: {question}

Generate {n} targeted retrieval queries that specifically search for evidence
covering these missing aspects. Each query should focus on one missing aspect
and use terminology likely to appear in technical documentation.

Return ONLY a JSON array: ["query1", "query2", ...]\
"""


def generate_query_variants(
    llm,
    question: str,
    n: int = 4,
    weak_claims: Optional[List[str]] = None,
) -> List[str]:
    """Generate n query variants. Always includes the original question."""
    prompt = (
        GAP_EXPAND_PROMPT.format(
            question=question,
            n=n,
            weak_claims="\n".join(f"- {c}" for c in (weak_claims or [])),
        )
        if weak_claims
        else EXPAND_PROMPT.format(question=question, n=n)
    )
    try:
        resp = llm.invoke(
            system=EXPAND_SYSTEM,
            prompt=prompt,
            max_tokens=400,
            temperature=0,
        )
        clean = re.sub(r"```(?:json)?", "", resp).strip()
        start = clean.find("[")
        end   = clean.rfind("]") + 1
        if start >= 0 and end > start:
            variants = json.loads(clean[start:end])
            if isinstance(variants, list):
                all_variants = [question] + [
                    v for v in variants if isinstance(v, str) and v.strip()
                ][:n]
                logger.info(f"  MultiQuery: {len(all_variants)} variants generated")
                return all_variants
    except Exception as e:
        logger.warning(f"  MultiQuery expansion failed: {e}")
    return [question]


def _rrf_merge(
    ranked_lists: List[List[Tuple[str, str, str, float]]],
    k: int = 60,
    top_n: int = 15,
    min_score_percentile: float = 0.5,
) -> List[Tuple[str, str, str]]:
    """
    Reciprocal Rank Fusion over multiple ranked chunk lists.

    Each item in ranked_lists is a list of (text, chunk_id, modality, score).
    Returns top_n (text, chunk_id, modality) tuples by RRF score,
    filtered to only chunks above min_score_percentile of RRF scores
    (prevents low-relevance chunks from polluting evidence pool).
    """
    scores: Dict[str, float] = {}
    meta:   Dict[str, Tuple[str, str, str]] = {}

    for ranked in ranked_lists:
        for rank, (text, chunk_id, modality, _score) in enumerate(ranked, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in meta:
                meta[chunk_id] = (text, chunk_id, modality)

    if not scores:
        return []

    # Apply minimum score percentile filter — drop bottom chunks
    all_scores = sorted(scores.values())
    min_threshold = all_scores[int(len(all_scores) * min_score_percentile)]

    sorted_ids = sorted(
        (cid for cid, s in scores.items() if s >= min_threshold),
        key=lambda x: scores[x],
        reverse=True,
    )
    return [meta[cid] for cid in sorted_ids[:top_n]]


def _extract_doc_id(chunk_ids: List[str]) -> Optional[str]:
    """
    Extract doc_id from chunk IDs. Handles multiple pipeline formats:
      - {doc_id}_chunk_{N}          (encoding_layer format)
      - {doc_id}_text_{N:04d}       (app _chunk_id format)
      - {doc_id}_slide{N:03d}_*     (pptx research format)
      - {doc_id}_{modality}_{idx}   (generic ingest format)
    Returns the longest common prefix across chunk_ids (most reliable).
    """
    if not chunk_ids:
        return None

    # Try known separator patterns first
    for cid in chunk_ids[:5]:
        for sep in ("_chunk_", "_text_", "_table_", "_image_", "_slide"):
            if sep in cid:
                candidate = cid.split(sep)[0]
                if len(candidate) >= 8:  # minimum hash length
                    return candidate

    # Fallback: find longest common prefix across first few chunk_ids
    if len(chunk_ids) >= 2:
        prefix = chunk_ids[0]
        for cid in chunk_ids[1:4]:
            while not cid.startswith(prefix) and prefix:
                prefix = prefix[:-1]
            if not prefix:
                break
        # Trim at last underscore to get clean doc_id
        if "_" in prefix:
            prefix = prefix[:prefix.rfind("_")]
        if len(prefix) >= 8:
            return prefix

    # Last resort: split first chunk on first underscore
    if "_" in chunk_ids[0]:
        candidate = chunk_ids[0].split("_")[0]
        if len(candidate) >= 8:
            return candidate

    return None


def multiquery_retrieve(
    llm,
    text_embedder,
    store,
    question: str,
    arxiv_id: str,
    existing_chunk_ids: List[str],
    n_variants: int = 4,
    top_k: int = 10,
    weak_claims: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    MultiQuery re-retrieval returning (texts, chunk_ids, modalities).

    Steps:
      1. Generate n_variants query reformulations
      2. For each variant, embed + query text/image/table Chroma collections
      3. RRF-merge all results
      4. Union with existing_chunk_ids (new chunks appended after existing)
      5. Return merged (texts, chunk_ids, modalities) list

    New chunks are appended after existing so R2-G sees the original
    evidence first (positional recency bias is avoided in ArgRAG
    since it scores each claim individually).
    """
    variants = generate_query_variants(llm, question, n=n_variants, weak_claims=weak_claims)

    collections = {
        "text":        store.collections.get("text"),
        "images_text": store.collections.get("images_text"),
        "tables":      store.collections.get("tables"),
    }
    modality_map = {
        "text":        "text",
        "images_text": "image",
        "tables":      "table",
    }

    all_ranked: List[List[Tuple[str, str, str, float]]] = []

    # Extract doc_id hash from existing chunk_ids (format: <hash>_chunk_<N>)
    doc_id = _extract_doc_id(existing_chunk_ids)
    doc_filter = {"doc_id": {"$eq": doc_id}} if doc_id else None
    logger.debug(f"  MultiQuery: doc_filter={doc_filter}")

    for variant in variants:
        embedding = text_embedder.embed_query(variant)
        if embedding is None:
            logger.warning(f"  MultiQuery: embed failed for variant '{variant[:40]}'")
            continue

        variant_results: List[Tuple[str, str, str, float]] = []

        for coll_name, collection in collections.items():
            if collection is None:
                continue
            modality = modality_map[coll_name]
            try:
                where_filter = doc_filter if doc_filter else None
                result = collection.query(
                    query_embeddings=[embedding],
                    n_results=top_k,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                )
                ids   = result.get("ids",       [[]])[0]
                docs  = result.get("documents", [[]])[0]
                dists = result.get("distances", [[]])[0]

                for cid, doc, dist in zip(ids, docs, dists):
                    if not doc or not doc.strip():
                        continue
                    score = max(0.0, 1.0 - dist)
                    variant_results.append((doc, cid, modality, score))

            except Exception as e:
                logger.debug(f"  MultiQuery collection '{coll_name}': {e}")

        if variant_results:
            variant_results.sort(key=lambda x: x[3], reverse=True)
            all_ranked.append(variant_results)

    if not all_ranked:
        logger.warning("  MultiQuery: no results from any variant — returning existing evidence")
        return [], [], []

    # Merge with RRF — limit to top_k candidates, filter bottom 50% by RRF score
    # (prevents adding low-quality chunks that degrade Stage 1 classification)
    merged = _rrf_merge(all_ranked, top_n=top_k, min_score_percentile=0.5)

    # Only add genuinely NEW chunks — limit to top 5 to avoid polluting evidence pool
    # with lower-ranked chunks that the original retrieval already deprioritised
    MAX_NEW_CHUNKS = 5
    new_texts, new_chunk_ids, new_modalities = [], [], []
    seen = set(existing_chunk_ids)
    for text, chunk_id, modality in merged:
        if len(new_chunk_ids) >= MAX_NEW_CHUNKS:
            break
        if chunk_id not in seen and text and text.strip():
            new_texts.append(text)
            new_chunk_ids.append(chunk_id)
            new_modalities.append(modality)
            seen.add(chunk_id)

    logger.info(
        f"  MultiQuery: {len(new_chunk_ids)} new chunks added from "
        f"{len(variants)} variants (RRF top-{MAX_NEW_CHUNKS}, bottom-50% filtered)"
    )
    return new_texts, new_chunk_ids, new_modalities
