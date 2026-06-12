"""
indexing_layer/utils/chroma_store.py

ChromaDB storage — 8 collections.

All research fixes applied:
  1. doc_embedding uses doc_embedding_input_text (8000+ chars)
  2. sections embed semantic_anchor + summary + keywords
  3. bm25_text + title_text stored on all chunks
  4. evidence_role_confidence + all confidence fields stored
  5. related_section_ids + sibling_chunk_ids stored
  6. visible_annotations + depicted_component_confidence stored
  7. table_html stored in metadata
  8. section_map reads correct keys (summary + section_summary)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger

COLLECTION_NAMES = {
    "documents":   "documents_collection",
    "sections":    "sections_collection",
    "text":        "text_chunks",
    "tables":      "table_chunks",
    "images_clip": "image_chunks_clip",
    "images_text": "image_chunks_text",
    "video_segs":  "video_segments",
    "video_frames":"video_frames_clip",
}


def _s(v):  return "" if v is None else str(v).strip()
def _i(v):
    if v is None: return 0
    try: return int(v)
    except: return 0
def _f(v):
    if v is None: return 0.0
    try: return round(float(v), 4)
    except: return 0.0
def _j(v):
    if v is None: return "[]"
    if isinstance(v, (list, dict)): return json.dumps(v)
    return str(v)


class ChromaStore:

    def __init__(self, persist_dir="./chroma_db", text_embedder=None):
        try:
            import chromadb
        except ImportError:
            raise ImportError("pip install chromadb")
        self.client = chromadb.PersistentClient(path=persist_dir)
        logger.info(f"ChromaDB initialized at: {persist_dir}")
        self.collections = {
            name: self.client.get_or_create_collection(
                name=cname, metadata={"hnsw:space": "cosine"},
            )
            for name, cname in COLLECTION_NAMES.items()
        }
        self._embedder = text_embedder

    # ── Document ──────────────────────────────────────────────────────

    def upsert_document(self, doc_meta: Dict[str, Any]) -> None:
        doc_id    = _s(doc_meta.get("doc_id"))
        embedding = doc_meta.get("doc_embedding")
        if not embedding:
            logger.warning(f"  doc_embedding missing for {doc_id} — generating")
            embedding = self._generate_doc_embedding(doc_meta)

        title  = _s(doc_meta.get("doc_title"))
        author = _s(doc_meta.get("author"))
        header = " | ".join(p for p in [title, author] if p)
        doc_text = (
            _s(doc_meta.get("doc_embedding_input_text")) or
            _s(doc_meta.get("doc_summary")) or title or doc_id
        )
        if header:
            doc_text = header + "\n" + doc_text

        outline = doc_meta.get("outline_summary")
        if isinstance(outline, list):
            outline_str = " | ".join(
                f"{e.get('section','')}: {e.get('summary','')}"
                for e in outline[:10] if isinstance(e, dict)
            )
        else:
            outline_str = _s(outline)

        metadata = {
            "doc_title":              title,
            "author":                 author,
            "tier1":                  _s(doc_meta.get("tier1")),
            "tier2":                  _s(doc_meta.get("tier2")),
            "project_id":             _s(doc_meta.get("project_id")),
            "document_type":          _s(doc_meta.get("document_type")),
            "language":               _s(doc_meta.get("language") or "English"),
            "file_format":            _s(doc_meta.get("file_format")),
            "source_id":              _s(doc_meta.get("source_id")),
            "chunk_count":            _i(doc_meta.get("chunk_count")),
            "total_pages":            _i(doc_meta.get("total_pages")),
            "total_segments":         _i(doc_meta.get("total_segments")),
            "doc_summary":            _s(doc_meta.get("doc_summary"))[:500],
            "doc_summary_confidence": _f(doc_meta.get("doc_summary_confidence")),
            "tier_confidence":        _f(doc_meta.get("tier_confidence")),
            "outline_summary":        outline_str[:500],
            "section_map_text":       _s(doc_meta.get("section_map_text"))[:500],
        }

        kwargs: Dict[str, Any] = dict(
            ids=[doc_id], documents=[doc_text[:2000]], metadatas=[metadata]
        )
        if embedding:
            kwargs["embeddings"] = [embedding]
        self.collections["documents"].upsert(**kwargs)
        logger.info(f"  Upserted document: {doc_id}")
        self._upsert_sections(doc_id, doc_meta)

    def _upsert_sections(self, doc_id: str, doc_meta: Dict[str, Any]) -> None:
        section_map = doc_meta.get("section_map", {})
        if not isinstance(section_map, dict) or not section_map:
            return

        ids, docs, metas, embs = [], [], [], []
        order = 0

        for heading, entry in section_map.items():
            if not isinstance(entry, dict):
                continue
            order += 1

            # Read correct keys — section_map uses "summary" OR "section_summary"
            summary = _s(
                entry.get("summary") or
                entry.get("section_summary") or ""
            )
            # semantic_anchor from DocMetaEnricher (merged in by indexing_pipeline)
            semantic_anchor = _s(
                entry.get("semantic_anchor") or
                entry.get("semantic_anchor_text") or ""
            )
            keywords    = entry.get("keywords", [])
            entities    = entry.get("entities", [])
            subsections = entry.get("subsections", [])

            # Build rich section_text: heading → anchor → summary → keywords
            section_parts = [heading]
            if semantic_anchor and semantic_anchor != heading:
                section_parts.append(semantic_anchor)
            if summary:
                section_parts.append(summary)
            if keywords:
                section_parts.append(" ".join(keywords[:8]))
            section_text = ". ".join(p.strip() for p in section_parts if p.strip())

            section_emb = self._embed_text(section_text)
            if not section_emb:
                continue

            # Use structure_unit_id when available (matches the section_id /
            # structure_unit_id stored in chunks — avoids _sec_ vs _slide_ mismatch
            # for PPT documents). Falls back to _sec_NNN for legacy PDF ingestion.
            section_id = entry.get("structure_unit_id") or f"{doc_id}_sec_{order:03d}"
            ids.append(section_id)
            docs.append(section_text[:1000])
            metas.append({
                "doc_id":                     doc_id,
                "section_name":               heading[:200],
                "section_order":              order,
                "semantic_anchor":            semantic_anchor[:300],
                "section_summary":            summary[:400],
                "section_summary_confidence": _f(entry.get("summary_confidence")),
                "keywords_str":               _j(keywords),
                "keywords_confidence":        _f(entry.get("keywords_confidence")),
                "entities_str":               _j(entities),
                "subsections_str":            _j(subsections),
                "start_page":                 _i(entry.get("start_page")),
                "end_page":                   _i(entry.get("end_page")),
                "chunk_ids_str":              _j(entry.get("chunk_ids", [])),
                "figure_ids_str":             _j(entry.get("figure_ids", [])),
            })
            embs.append(section_emb)

        if ids:
            self._batch_upsert(self.collections["sections"], ids, docs, metas, embs)
            logger.info(f"  Upserted {len(ids)} sections for {doc_id}")

    # ── Text chunks ───────────────────────────────────────────────────

    def upsert_text_chunks(self, chunks: List[Dict[str, Any]], doc_meta: Dict[str, Any]) -> None:
        valid = [c for c in chunks if c.get("text_embedding")]
        if not valid:
            logger.warning("  No text chunks with embeddings to upsert")
            return
        ids, docs, metas, embs = [], [], [], []
        for c in valid:
            ids.append(_s(c["chunk_id"]))
            docs.append(_s(c.get("text_original_content"))[:2000])
            embs.append(c["text_embedding"])
            metas.append({
                # Identity
                "doc_id":           _s(c.get("doc_id")),
                "source_id":        _s(c.get("source_id") or doc_meta.get("source_id")),
                "source_modality":  "text",
                "file_format":      _s(c.get("file_format") or doc_meta.get("file_format")),
                "chunk_index":      _i(c.get("chunk_index")),
                "page_number":      _i(c.get("page_number")),
                "slide_index":      _i(c.get("slide_index")),
                "chunk_strategy":   _s(c.get("chunk_strategy")),
                "token_count":      _i(c.get("token_count")),
                # Structure
                "structure_unit_id":  _s(c.get("structure_unit_id")),
                "section_title":      _s(c.get("section_title") or c.get("slide_title")),
                "section_id":         _s(c.get("section_id")),
                "heading_breadcrumb": _s(c.get("heading_breadcrumb")),
                # Enrichment signals
                "contextual_summary":            _s(c.get("contextual_summary"))[:1200],
                "contextual_summary_confidence": _f(c.get("contextual_summary_confidence")),
                "evidence_role":                 _s(c.get("evidence_role")),
                "evidence_role_confidence":      _f(c.get("evidence_role_confidence")),
                "salience_score":                _f(c.get("salience_score")),
                "detected_codes":                _j(c.get("detected_codes", [])),
                "detected_codes_confidence":     _f(c.get("detected_codes_confidence")),
                # Cross-reference linkage
                "related_figures":    _j(c.get("related_figures", [])),
                "related_tables":     _j(c.get("related_tables", [])),
                "related_section_ids": _j(c.get("related_section_ids", [])),
                "sibling_chunk_ids":  _j(c.get("sibling_chunk_ids", [])),
                "neighbor_chunk_ids": _j(c.get("neighbor_chunk_ids", [])),
                # Retrieval views (hybrid retrieval)
                "bm25_text":   _s(c.get("bm25_text"))[:500],
                "title_text":  _s(c.get("title_text"))[:200],
                # Document context
                "tier1":         _s(doc_meta.get("tier1")),
                "tier2":         _s(doc_meta.get("tier2")),
                "project_id":    _s(doc_meta.get("project_id")),
                "document_type": _s(doc_meta.get("document_type")),
                "language":      _s(doc_meta.get("language") or "English"),
            })
        self._batch_upsert(self.collections["text"], ids, docs, metas, embs)
        logger.info(f"  Upserted {len(valid)} text chunks")

    # ── Table chunks ──────────────────────────────────────────────────

    def upsert_table_chunks(self, chunks: List[Dict[str, Any]], doc_meta: Dict[str, Any]) -> None:
        semantic = [c for c in chunks if c.get("text_embedding")]
        if semantic:
            ids, docs, metas, embs = [], [], [], []
            for c in semantic:
                ids.append(_s(c["chunk_id"]))
                docs.append(f"{_s(c.get('table_caption'))} {_s(c.get('table_summary'))} {_s(c.get('table_purpose'))}")
                embs.append(c["text_embedding"])
                metas.append(self._table_meta(c, doc_meta, "semantic"))
            self._batch_upsert(self.collections["tables"], ids, docs, metas, embs)
            logger.info(f"  Upserted {len(semantic)} table chunks (semantic)")
        else:
            logger.warning("  No table chunks with semantic embeddings")

        html_valid = [c for c in chunks if c.get("html_text_embedding")]
        if html_valid:
            ids, docs, metas, embs = [], [], [], []
            for c in html_valid:
                ids.append(_s(c["chunk_id"]) + "_html")
                docs.append(_s(c.get("table_html"))[:1000])
                embs.append(c["html_text_embedding"])
                metas.append(self._table_meta(c, doc_meta, "content"))
            self._batch_upsert(self.collections["tables"], ids, docs, metas, embs)
            logger.info(f"  Upserted {len(html_valid)} table chunks (HTML content)")

    def _table_meta(self, c: Dict, doc_meta: Dict, embed_type: str) -> Dict[str, Any]:
        return {
            "doc_id":           _s(c.get("doc_id")),
            "source_id":        _s(c.get("source_id") or doc_meta.get("source_id")),
            "source_modality":  "table",
            "file_format":      _s(c.get("file_format") or doc_meta.get("file_format")),
            "embed_type":       embed_type,
            "chunk_index":      _i(c.get("chunk_index")),
            "page_number":      _i(c.get("page_number")),
            "sheet_name":       _s(c.get("sheet_name")),
            "sheet_index":      _i(c.get("sheet_index")),
            "row_count":        _i(c.get("row_count")),
            "col_count":        _i(c.get("col_count")),
            "column_names_str": _j(c.get("column_names", [])),
            "table_html":       _s(c.get("table_html"))[:500],
            "table_caption":    _s(c.get("table_caption"))[:200],
            "table_summary":    _s(c.get("table_summary"))[:400],
            "table_purpose":    _s(c.get("table_purpose"))[:300],
            "table_summary_confidence": _f(c.get("table_summary_confidence")),
            "table_purpose_confidence": _f(c.get("table_purpose_confidence")),
            "section_id":       _s(c.get("section_id")),
            "structure_unit_id": _s(c.get("structure_unit_id")),
            "bm25_text":        _s(c.get("bm25_text"))[:500],
            "title_text":       _s(c.get("title_text"))[:200],
            "tier1":            _s(doc_meta.get("tier1")),
            "tier2":            _s(doc_meta.get("tier2")),
            "project_id":       _s(doc_meta.get("project_id")),
            "document_type":    _s(doc_meta.get("document_type")),
        }

    # ── Image chunks ──────────────────────────────────────────────────

    def upsert_image_chunks(self, chunks: List[Dict[str, Any]], doc_meta: Dict[str, Any]) -> None:
        self._upsert_clip(chunks, doc_meta, "images_clip", "image")
        self._upsert_img_text(chunks, doc_meta, "image")

    def _upsert_clip(self, chunks, doc_meta, coll_key, modality):
        valid = [c for c in chunks if c.get("clip_embedding")]
        if not valid: return
        ids, docs, metas, embs = [], [], [], []
        for c in valid:
            ids.append(_s(c["chunk_id"]))
            docs.append(
                f"{_s(c.get('image_caption'))} "
                f"{_s(c.get('depicted_component'))} "
                f"{_s(c.get('visible_annotations'))} "
                f"{_s(c.get('contextual_summary'))}"
            )
            embs.append(c["clip_embedding"])
            metas.append(self._image_meta(c, doc_meta, "clip", modality))
        self._batch_upsert(self.collections[coll_key], ids, docs, metas, embs)
        logger.info(f"  Upserted {len(valid)} {modality} clips → {coll_key}")

    def _upsert_img_text(self, chunks, doc_meta, modality):
        valid = [c for c in chunks if c.get("text_embedding")]
        if not valid: return
        ids, docs, metas, embs = [], [], [], []
        for c in valid:
            ids.append(_s(c["chunk_id"]))
            docs.append(
                f"{_s(c.get('image_caption'))} "
                f"{_s(c.get('depicted_component'))} "
                f"{_s(c.get('visible_annotations'))} "
                f"{_s(c.get('contextual_summary'))}"
            )
            embs.append(c["text_embedding"])
            metas.append(self._image_meta(c, doc_meta, "text", modality))
        self._batch_upsert(self.collections["images_text"], ids, docs, metas, embs)
        logger.info(f"  Upserted {len(valid)} {modality} text embeds → images_text")

    def _image_meta(self, c, doc_meta, embed_type, modality):
        return {
            "doc_id":          _s(c.get("doc_id")),
            "source_id":       _s(c.get("source_id") or doc_meta.get("source_id")),
            "source_modality": modality,
            "file_format":     _s(c.get("file_format") or doc_meta.get("file_format")),
            "embed_type":      embed_type,
            "chunk_index":     _i(c.get("chunk_index")),
            "page_number":     _i(c.get("page_number")),
            "slide_index":     _i(c.get("slide_index")),
            "figure_id":       _s(c.get("figure_id")),
            "image_type":      _s(c.get("image_type")),
            "frame_role":      _s(c.get("frame_role")),
            # Enrichment signals
            "image_caption":                 _s(c.get("image_caption"))[:300],
            "image_caption_confidence":      _f(c.get("image_caption_confidence")),
            "depicted_component":            _s(c.get("depicted_component"))[:200],
            "depicted_component_confidence": _f(c.get("depicted_component_confidence")),
            "visible_annotations":           _s(c.get("visible_annotations"))[:300],
            "visible_annotations_confidence": _f(c.get("visible_annotations_confidence")),
            "contextual_summary":            _s(c.get("contextual_summary"))[:1200],
            "contextual_summary_confidence": _f(c.get("contextual_summary_confidence")),
            "frame_role_confidence":         _f(c.get("frame_role_confidence")),
            # Structure
            "section_id":        _s(c.get("section_id")),
            "structure_unit_id": _s(c.get("structure_unit_id")),
            "related_sections":  _j(c.get("related_sections", [])),
            # Retrieval views
            "bm25_text":  _s(c.get("bm25_text"))[:500],
            "title_text": _s(c.get("title_text"))[:200],
            # Document context
            "tier1":      _s(doc_meta.get("tier1")),
            "tier2":      _s(doc_meta.get("tier2")),
            "project_id": _s(doc_meta.get("project_id")),
        }

    # ── Video segments ────────────────────────────────────────────────

    def upsert_video_segments(self, segments: List[Dict[str, Any]], doc_meta: Dict[str, Any]) -> None:
        valid = [s for s in segments if s.get("text_embedding")]
        if not valid:
            logger.warning("  No video segments with embeddings to upsert")
            return
        ids, docs, metas, embs = [], [], [], []
        for s in valid:
            ids.append(_s(s["chunk_id"]))
            docs.append(_s(s.get("contextual_summary") or s.get("transcript_text"))[:2000])
            embs.append(s["text_embedding"])
            metas.append({
                "doc_id":          _s(s.get("doc_id")),
                "source_id":       _s(s.get("source_id") or doc_meta.get("source_id")),
                "source_modality": "video_segment",
                "file_format":     _s(s.get("file_format") or doc_meta.get("file_format")),
                "chunk_index":     _i(s.get("chunk_index")),
                "segment_index":   _i(s.get("segment_index")),
                "start_time_s":    _f(s.get("start_time_s")),
                "end_time_s":      _f(s.get("end_time_s")),
                "duration_s":      _f(s.get("duration_s")),
                "structure_unit_id": _s(s.get("structure_unit_id")),
                "transcript_text":  _s(s.get("transcript_text"))[:400],
                "asr_language":     _s(s.get("asr_language")),
                "asr_confidence":   _f(s.get("asr_confidence")),
                "contextual_summary":            _s(s.get("contextual_summary"))[:400],
                "contextual_summary_confidence": _f(s.get("contextual_summary_confidence")),
                "evidence_role":                 _s(s.get("evidence_role")),
                "evidence_role_confidence":      _f(s.get("evidence_role_confidence")),
                "salience_score":                _f(s.get("salience_score")),
                "detected_codes":                _j(s.get("detected_codes", [])),
                "keyframe_ids":      _j(s.get("keyframe_ids", [])),
                "scene_sibling_ids": _j(s.get("scene_sibling_ids", [])),
                "bm25_text":  _s(s.get("bm25_text"))[:500],
                "title_text": _s(s.get("title_text"))[:200],
                "tier1":      _s(doc_meta.get("tier1")),
                "tier2":      _s(doc_meta.get("tier2")),
                "project_id": _s(doc_meta.get("project_id")),
            })
        self._batch_upsert(self.collections["video_segs"], ids, docs, metas, embs)
        logger.info(f"  Upserted {len(valid)} video segments")

    # ── Video frames ──────────────────────────────────────────────────

    def upsert_video_frames(self, frames: List[Dict[str, Any]], doc_meta: Dict[str, Any]) -> None:
        self._upsert_clip(frames, doc_meta, "video_frames", "video_frame")
        self._upsert_img_text(frames, doc_meta, "video_frame")

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from embedding_layer.embedders.text_embedder import TextEmbedder
                self._embedder = TextEmbedder()
                logger.info("  ChromaStore: created shared TextEmbedder (lazy init)")
            except Exception as e:
                logger.error(f"  ChromaStore: cannot create TextEmbedder: {e}")
                return None
        return self._embedder

    def _embed_text(self, text: str):
        embedder = self._get_embedder()
        if not embedder: return None
        try: return embedder.embed_query(text)
        except Exception as e:
            logger.error(f"  Inline embed failed: {e}")
            return None

    def _generate_doc_embedding(self, doc_meta: Dict):
        # Use pre-built doc_embedding_input_text from encoding layer (8000+ chars)
        pre_built = doc_meta.get("doc_embedding_input_text")
        if pre_built and isinstance(pre_built, str) and pre_built.strip():
            return self._embed_text(pre_built[:8000])
        # Fallback
        parts = []
        outline = doc_meta.get("outline_summary")
        if isinstance(outline, list):
            outline_text = " | ".join(
                f"{e.get('section','')}: {e.get('summary','')}"
                for e in outline[:10] if isinstance(e, dict)
            )
            if outline_text: parts.append(outline_text)
        elif outline and isinstance(outline, str):
            parts.append(outline)
        for key in ("doc_title", "doc_summary", "section_map_text"):
            val = doc_meta.get(key)
            if val and isinstance(val, str) and val.strip():
                parts.append(val.strip())
        text = "\n\n".join(parts)
        return self._embed_text(text) if text else None

    def _batch_upsert(self, collection, ids, docs, metas, embeddings, batch_size=500):
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.upsert(
                ids=ids[start:end], documents=docs[start:end],
                metadatas=metas[start:end], embeddings=embeddings[start:end],
            )

    def get_collection_stats(self) -> Dict[str, int]:
        return {name: col.count() for name, col in self.collections.items()}