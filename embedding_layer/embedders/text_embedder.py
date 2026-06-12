"""
embedding_layer/embedders/text_embedder.py

Text embedding using OpenAI text-embedding-3-small (1536-dim).

Three embed functions:

embed_text_chunks(chunks)
    Input: contextual_summary + "\\n\\n" + text_original_content
    Now that enrichment has filled contextual_summary, this input is
    dramatically richer than the raw text alone.
    Output: text_embedding (1536-dim)

embed_table_chunks(chunks)
    Semantic pass: table_caption + table_summary + table_purpose → text_embedding
    Content pass:  cleaned HTML cell values                       → html_text_embedding
    Two vectors per table chunk — one for "what does this table mean" queries,
    one for "find rows where column X = Y" queries.

embed_video_segments(chunks)
    Input: contextual_summary + "\\n\\n" + transcript_text
    Same path as text chunks — video segments are routed through text collections.
    Output: text_embedding (1536-dim)

embed_doc(doc_meta)
    Input: doc_title + doc_summary + outline_summary + section_map_text
    All filled by enrichment — much richer than title-only fallback.
    Output: doc_embedding (1536-dim) written into doc_meta dict
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

from loguru import logger
from tqdm import tqdm

from shared.models.metadata_models import TEXT_EMBED_DIM

DEFAULT_MODEL = "text-embedding-3-small"


def _build_text_input(chunk: Dict[str, Any]) -> str:
    # Prefer encoding layer's fused retrieval text (richest representation)
    fused = (chunk.get("fused_retrieval_text") or "").strip()
    if fused:
        return fused
    # Fall back to retrieval_text from encoding views
    retrieval = (chunk.get("retrieval_text") or "").strip()
    if retrieval:
        return retrieval
    # Fall back to enrichment output: situated context + raw content
    summary = (chunk.get("contextual_summary") or "").strip()
    content = (
        chunk.get("text_original_content")
        or chunk.get("transcript_text")
        or ""
    ).strip()
    if summary:
        return f"{summary}\n\n{content}"
    return content


def _build_table_semantic(chunk: Dict[str, Any]) -> str:
    parts = [
        chunk.get("table_caption")  or "",
        chunk.get("table_summary")  or "",
        chunk.get("table_purpose")  or "",
    ]
    return " ".join(p.strip() for p in parts if p.strip())


def _build_table_content(chunk: Dict[str, Any]) -> str:
    html = chunk.get("table_html") or chunk.get("markdown") or ""
    if not html:
        return _build_table_semantic(chunk)
    # Strip HTML tags, preserve cell separators
    cleaned = re.sub(r"</t[dh]>", " | ", html, flags=re.IGNORECASE)
    cleaned = re.sub(r"</tr>",     "\n",  cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>",   " ",   cleaned)
    cleaned = re.sub(r"\s+",        " ",   cleaned).strip()
    caption = (chunk.get("table_caption") or "").strip()
    result  = f"{caption}\n{cleaned}" if caption else cleaned
    return result[:2000]


def _is_embeddable(text: str) -> bool:
    return bool(text and len(text.strip()) >= 8)


class TextEmbedder:

    def __init__(self, model: str = DEFAULT_MODEL):
        logger.info(f"Loading OpenAI text embedder: {model}")
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            self._model  = model
            self._dim    = TEXT_EMBED_DIM
            logger.success(f"OpenAI embedder ready: {model} ({self._dim}-dim)")
        except ImportError:
            raise ImportError("pip install openai")

    @property
    def dim(self) -> int:
        return self._dim

    # ── Public embed functions ─────────────────────────────────────────

    def embed_text_chunks(
        self, chunks: List[Dict], batch_size: int = 100
    ) -> List[Dict]:
        return self._embed(chunks, _build_text_input, "text_embedding", "text", batch_size)

    def embed_table_chunks(
        self, chunks: List[Dict], batch_size: int = 100
    ) -> List[Dict]:
        chunks = self._embed(chunks, _build_table_semantic, "text_embedding",      "table-semantic", batch_size)
        chunks = self._embed(chunks, _build_table_content,  "html_text_embedding", "table-content",  batch_size)
        return chunks

    def embed_video_segments(
        self, chunks: List[Dict], batch_size: int = 100
    ) -> List[Dict]:
        """Video segments use the same text path as text chunks."""
        return self._embed(chunks, _build_text_input, "text_embedding", "video-seg", batch_size)

    def embed_doc(self, doc_meta: Dict) -> Dict:
        """Embed the document-level summary. Writes doc_embedding in-place."""
        text = self._build_doc_input(doc_meta)
        if not _is_embeddable(text):
            logger.warning(f"  doc_embedding: no embeddable text for {doc_meta.get('doc_id')}")
            return doc_meta
        emb = self.embed_query(text)
        if emb:
            doc_meta["doc_embedding"] = emb
            logger.info(f"  doc_embedding: {len(emb)}-dim from {len(text)} chars")
        return doc_meta

    def embed_query(self, text: str) -> Optional[List[float]]:
        """Embed a single query string. Used for doc_embedding and query-time."""
        text = (text or "").strip()
        if not text:
            return None
        results = self._call_api([text])
        return results[0] if results else None

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Batch encode interface for retrieval layer HyDE."""
        return self._call_api([t for t in texts if t.strip()])

    # ── Internal helpers ───────────────────────────────────────────────

    def _embed(
        self,
        chunks:     List[Dict],
        builder:    Any,
        field:      str,
        label:      str,
        batch_size: int,
    ) -> List[Dict]:
        to_embed = [
            (i, builder(c))
            for i, c in enumerate(chunks)
            if not c.get(field) and _is_embeddable(builder(c))
        ]
        if not to_embed:
            return chunks

        indices = [i for i, _ in to_embed]
        texts   = [t for _, t in to_embed]

        logger.info(f"  TextEmbedder [{label}]: {len(texts)} chunks → {field}")
        all_embs: List[List[float]] = []

        for start in tqdm(range(0, len(texts), batch_size), desc=f"{label}→{field}"):
            batch = texts[start : start + batch_size]
            embs  = self._call_api(batch)
            all_embs.extend(embs)

        for idx, emb in zip(indices, all_embs):
            chunks[idx][field] = emb

        logger.success(f"  TextEmbedder [{label}]: {len(all_embs)} × {self._dim}-dim")
        return chunks

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(3):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"  OpenAI retry {attempt + 1}/3: {e}")
                time.sleep(2 ** attempt)
        return []

    def _build_doc_input(self, doc_meta: Dict) -> str:
        # Use pre-built doc_embedding_input_text from encoding layer if available.
        # It contains doc_title + doc_summary + outline_summary + section_map
        # already assembled — 8000+ chars vs ~500 chars without it.
        pre_built = doc_meta.get("doc_embedding_input_text")
        if pre_built and isinstance(pre_built, str) and pre_built.strip():
            return pre_built.strip()
        # Fallback: build from parts — handles both old and new key names
        parts = []
        for key in ("doc_title", "doc_summary", "outline_summary",
                    "outline_summary_text", "section_map_text"):
            val = doc_meta.get(key)
            if val and isinstance(val, str) and val.strip():
                parts.append(val.strip())
        return "\n\n".join(parts)