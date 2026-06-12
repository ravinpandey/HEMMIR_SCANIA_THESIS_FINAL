"""
encoding_layer/retrieval_text_builders.py

Builds multiple specialised text views from enriched chunk dicts.

Research design — why multiple views:

A single chunk needs to be findable by fundamentally different query types:

  "show me the attention architecture diagram"     → visual, semantic
  "what is d_k in multi-head attention"            → exact term
  "F3-447 fault code procedure"                    → exact code (Scania)
  "section 3.2 encoder decoder"                    → structural/citation
  "28.4 BLEU WMT 2014 English German"              → exact metric

Dense embeddings excel at semantic matching but fail on exact codes.
BM25 excels at exact matching but fails on paraphrase queries.
Hybrid retrieval (dense + sparse) consistently outperforms either alone
by 5-15% on recall in literature (Karpukhin et al., Lin et al., 2021).

View design:

  retrieval_text    Dense primary: doc_title + section + anchor + summary +
                    content + context. Capped at 350 words (~450 tokens for
                    technical text) to stay within 512-token embedding window.

  title_text        Document routing: doc_title + tier + section + anchor only.
                    No chunk content. Used in Layer 1 structural retrieval to
                    route queries to the right document/section before chunk search.
                    For Scania: routes "coolant fault" to the correct manual section.

  bm25_text         BM25 sparse index: detected_codes + entities + keywords +
                    annotations. Preserved exact for token-level matching.
                    For Scania: "F3-447" "45 Nm" "SDP3" "E404" "D13K"
                    For SPIQA: "BLEU=28.4" "d_k=64" "h=8" "WMT2014"

  entity_text       Hybrid keyword: entities + keywords + codes. Used for both
                    sparse keyword index and secondary dense embedding.

  rerank_text       Cross-encoder input (700 words). Includes evidence_role
                    so reranker prefers definition/procedure over reference.

Token budget:
  512 tokens ≈ 350 words for technical text (1.3-1.5 tokens/word subword).
  retrieval_text capped at 350 words.
  rerank_text capped at 700 words (cross-encoders handle longer inputs).
"""

from __future__ import annotations

import re
from typing import Dict, Any, Iterable, List


STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at","with","by","from","as","is","are","was","were",
    "be","been","being","this","that","these","those","it","its","into","about","than","such","their","there","which","who","whom",
    "what","when","where","why","how","can","could","should","would","will","may","might","do","does","did","done","not","no","yes",
    "also","we","our","you","your","they","them","he","she","his","her","i","me","my","mine","us"
}

_RETRIEVAL_MAX_WORDS  = 350   # ~450 tokens for technical text — safe for 512-token window
_RERANK_MAX_WORDS     = 700   # cross-encoders handle longer inputs
_SUMMARY_MAX_WORDS    = 150
_SECTION_CTX_WORDS    = 200
_BM25_MAX_WORDS       = 200
_TITLE_MAX_WORDS      = 80    # pure routing signal, ~120 tokens


def clean_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]


def summarize(text: str, max_sentences: int = 2, max_chars: int = 320) -> str:
    sents = split_sentences(text)
    if not sents:
        return ""
    return " ".join(sents[:max_sentences])[:max_chars].strip()


def extract_keywords(text: str, max_terms: int = 15) -> List[str]:
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-_/]{2,}\b", clean_text(text).lower())
    freq = {}
    for w in words:
        if w in STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1
    ordered = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ordered[:max_terms]]


def extract_entities(text: str, max_terms: int = 15) -> List[str]:
    vals = re.findall(r"\b[A-Z][A-Za-z0-9\-_/]{2,}\b", clean_text(text))
    seen = []
    for v in vals:
        if v not in seen:
            seen.append(v)
    return seen[:max_terms]


def join_nonempty(parts: Iterable[str], sep: str = " | ") -> str:
    vals = [clean_text(p) for p in parts if clean_text(p)]
    return sep.join(vals)


def cap_words(text: str, max_words: int = 512) -> str:
    words = clean_text(text).split()
    return " ".join(words[:max_words])


def build_bm25_text(
    codes:    List[str],
    entities: List[str],
    keywords: List[str],
    annots:   str = "",
) -> str:
    """
    Build BM25 sparse-index text from high-precision exact-match signals.

    Preserved as-is without normalisation — BM25 needs exact token matches.
    Codes placed first: highest precision signals for industrial retrieval.

    For SPIQA: "BLEU" "WMT2014" "d_k=64" "h=8" "28.4"
    For Scania: "F3-447" "E404" "45 Nm" "SDP3" "D13K" "v2.40"
    """
    parts = []
    if codes:
        parts.extend(codes)
    if entities:
        parts.extend(entities[:10])
    if keywords:
        parts.extend(keywords[:10])
    if annots:
        parts.append(annots)
    return cap_words(" ".join(parts), max_words=_BM25_MAX_WORDS)


def build_title_text(
    doc_title:       str,
    section_title:   str,
    semantic_anchor: str,
    tier1:           str = "",
    tier2:           str = "",
) -> str:
    """
    Build document-routing text for Layer 1 structural retrieval.

    Contains ONLY structural signals — no chunk content. Creates a pure
    document/section embedding for routing queries before chunk search.

    For Scania: routes "engine oil pressure fault" to the correct manual
    section without being confused by other sections mentioning oil.
    For SPIQA: routes "multi-head attention mechanism" to section 3.2.2.
    """
    return cap_words(join_nonempty([
        doc_title,
        tier1,
        tier2,
        section_title,
        semantic_anchor,
    ]), max_words=_TITLE_MAX_WORDS)


# ── Text chunk views ───────────────────────────────────────────────────────────

def build_text_views(
    chunk:           Dict[str, Any],
    doc_meta:        Dict[str, Any],
    unit:            Dict[str, Any],
    section_context: str = "",
) -> Dict[str, str]:
    raw   = chunk.get("text_original_content") or chunk.get("transcript_text") or ""
    clean = clean_text(raw)
    summary = clean_text(chunk.get("contextual_summary")) or summarize(clean)
    local_context  = clean_text(chunk.get("local_context", ""))
    keywords       = extract_keywords(join_nonempty([clean, summary], sep=" "))
    entities       = chunk.get("entities") or extract_entities(clean)
    detected_codes = chunk.get("detected_codes") or []

    entity_text = join_nonempty(list(entities) + keywords + detected_codes, sep=", ")

    bm25_text = build_bm25_text(
        codes    = detected_codes,
        entities = list(entities)[:10],
        keywords = keywords[:10],
        annots   = "",
    )

    section_title = (
        chunk.get("section_title") or
        unit.get("title", "") or
        unit.get("structure_unit_title", "")
    )

    title_text = build_title_text(
        doc_title       = doc_meta.get("doc_title", ""),
        section_title   = section_title,
        semantic_anchor = unit.get("semantic_anchor", ""),
        tier1           = doc_meta.get("tier1", ""),
        tier2           = doc_meta.get("tier2", ""),
    )

    # PRIMARY dense embedding — highest-signal fields first to survive truncation
    retrieval_text = cap_words(join_nonempty([
        doc_meta.get("doc_title", ""),
        section_title,
        unit.get("semantic_anchor", ""),
        summary,
        clean,
        local_context,
        cap_words(section_context, max_words=60),
    ]), max_words=_RETRIEVAL_MAX_WORDS)

    # Reranker: includes evidence_role for role-aware ranking
    evidence_role = chunk.get("evidence_role", "")
    rerank_text = cap_words(join_nonempty([
        section_title,
        f"[{evidence_role}]" if evidence_role else "",
        summary,
        clean,
        local_context,
    ], sep=" "), max_words=_RERANK_MAX_WORDS)

    fused_retrieval_text = cap_words(
        join_nonempty([summary, clean, cap_words(section_context, max_words=80)], sep=" "),
        max_words=_RETRIEVAL_MAX_WORDS
    )

    return {
        "raw_text":              raw,
        "clean_text":            clean,
        "retrieval_text":        retrieval_text,
        "title_text":            title_text,
        "fused_retrieval_text":  fused_retrieval_text,
        "summary_text":          cap_words(summary, max_words=_SUMMARY_MAX_WORDS),
        "entity_text":           entity_text,
        "bm25_text":             bm25_text,
        "rerank_text":           rerank_text,
        "local_context_text":    local_context,
        "section_context_text":  cap_words(section_context, max_words=_SECTION_CTX_WORDS),
        "transcript_text":       chunk.get("transcript_text", ""),
        "segment_summary_text":  summary if chunk.get("transcript_text") else "",
        "image_caption_text":    "",
        "ocr_text":              "",
        "visual_summary_text":   "",
        "table_summary_text":    "",
        "table_purpose_text":    "",
        "schema_text":           "",
        "row_group_text":        "",
        "frame_caption_text":    "",
    }


# ── Image / video frame views ──────────────────────────────────────────────────

def build_image_views(
    chunk:           Dict[str, Any],
    doc_meta:        Dict[str, Any],
    unit:            Dict[str, Any],
    section_context: str = "",
) -> Dict[str, str]:
    caption    = clean_text(chunk.get("image_caption", ""))
    ocr        = clean_text(chunk.get("ocr_text", ""))
    summary    = clean_text(chunk.get("contextual_summary", "")) or summarize(
        join_nonempty([caption, ocr], sep=" ")
    )
    component  = clean_text(chunk.get("depicted_component", ""))
    annots     = clean_text(chunk.get("visible_annotations", ""))
    frame_role = clean_text(chunk.get("frame_role", ""))
    image_type = clean_text(chunk.get("image_type", ""))
    section_title = unit.get("title", "") or unit.get("structure_unit_title", "")

    title_text = build_title_text(
        doc_title       = doc_meta.get("doc_title", ""),
        section_title   = section_title,
        semantic_anchor = unit.get("semantic_anchor", ""),
        tier1           = doc_meta.get("tier1", ""),
        tier2           = doc_meta.get("tier2", ""),
    )

    # bm25_text: exact labels visible in figures — "h=8" "d_k=64" "45 Nm"
    bm25_text = build_bm25_text(
        codes    = chunk.get("detected_codes") or [],
        entities = extract_entities(join_nonempty([component, annots], sep=" "))[:10],
        keywords = extract_keywords(join_nonempty([component, annots, caption], sep=" "))[:10],
        annots   = annots,
    )

    entity_text = join_nonempty(
        extract_entities(join_nonempty([component, caption, ocr, summary, annots], sep=" ")) +
        extract_keywords(join_nonempty([component, caption, ocr, summary, annots], sep=" ")),
        sep=", "
    )

    # component first — highest precision for visual queries
    retrieval_text = cap_words(join_nonempty([
        doc_meta.get("doc_title", ""),
        section_title,
        unit.get("semantic_anchor", ""),
        image_type,
        frame_role,
        component,
        caption,
        ocr,
        summary,
        annots,
        cap_words(section_context, max_words=40),
    ]), max_words=_RETRIEVAL_MAX_WORDS)

    rerank_text = cap_words(join_nonempty([
        section_title,
        component,
        caption,
        ocr,
        summary,
        annots,
        section_context,
    ], sep=" "), max_words=_RERANK_MAX_WORDS)

    return {
        "raw_text":              caption,
        "clean_text":            caption,
        "retrieval_text":        retrieval_text,
        "title_text":            title_text,
        "fused_retrieval_text":  cap_words(
            join_nonempty([summary, component, caption, ocr, section_context], sep=" "),
            max_words=_RETRIEVAL_MAX_WORDS
        ),
        "summary_text":          cap_words(summary, max_words=_SUMMARY_MAX_WORDS),
        "entity_text":           entity_text,
        "bm25_text":             bm25_text,
        "rerank_text":           rerank_text,
        "local_context_text":    "",
        "section_context_text":  cap_words(section_context, max_words=_SECTION_CTX_WORDS),
        "image_caption_text":    caption,
        "ocr_text":              ocr,
        "visual_summary_text":   summary,
        "table_summary_text":    "",
        "table_purpose_text":    "",
        "schema_text":           "",
        "row_group_text":        "",
        "transcript_text":       "",
        "segment_summary_text":  "",
        "frame_caption_text":    caption,
    }


# ── Table views ────────────────────────────────────────────────────────────────

def build_table_views(
    chunk:           Dict[str, Any],
    doc_meta:        Dict[str, Any],
    unit:            Dict[str, Any],
    section_context: str = "",
) -> Dict[str, str]:
    summary   = clean_text(chunk.get("table_summary", ""))
    purpose   = clean_text(chunk.get("table_purpose", ""))
    markdown  = clean_text(chunk.get("markdown", "") or chunk.get("table_markdown", ""))
    caption   = clean_text(chunk.get("table_caption", ""))

    schema_cols = chunk.get("column_names") or []
    if schema_cols:
        schema_text = "Columns: " + ", ".join(schema_cols)
    else:
        schema_text = f"rows={chunk.get('row_count', 0)} cols={chunk.get('col_count', 0)}"

    if not summary:
        summary = summarize(join_nonempty([purpose, caption, markdown, schema_text], sep=" "))

    section_title = unit.get("title", "") or unit.get("structure_unit_title", "")

    title_text = build_title_text(
        doc_title       = doc_meta.get("doc_title", ""),
        section_title   = section_title,
        semantic_anchor = unit.get("semantic_anchor", ""),
        tier1           = doc_meta.get("tier1", ""),
        tier2           = doc_meta.get("tier2", ""),
    )

    detected_codes = chunk.get("detected_codes") or []
    table_keywords = extract_keywords(join_nonempty([summary, purpose, schema_text, markdown], sep=" "))
    table_entities = extract_entities(join_nonempty([summary, caption, schema_text, markdown], sep=" "))

    # bm25: column names + codes in cells — critical for Scania data tables
    bm25_text = build_bm25_text(
        codes    = detected_codes,
        entities = table_entities[:10],
        keywords = table_keywords[:10],
        annots   = schema_text,
    )

    entity_text = join_nonempty(table_entities + table_keywords + detected_codes, sep=", ")

    retrieval_text = cap_words(join_nonempty([
        doc_meta.get("doc_title", ""),
        section_title,
        unit.get("semantic_anchor", ""),
        "table",
        caption,
        summary,
        purpose,
        schema_text,
        markdown[:400],
        cap_words(section_context, max_words=40),
    ]), max_words=_RETRIEVAL_MAX_WORDS)

    rerank_text = cap_words(join_nonempty([
        section_title,
        summary,
        purpose,
        schema_text,
        markdown[:1000],
        section_context,
    ], sep=" "), max_words=_RERANK_MAX_WORDS)

    return {
        "raw_text":              markdown or summary,
        "clean_text":            summary,
        "retrieval_text":        retrieval_text,
        "title_text":            title_text,
        "fused_retrieval_text":  cap_words(
            join_nonempty([summary, purpose, schema_text, section_context], sep=" "),
            max_words=_RETRIEVAL_MAX_WORDS
        ),
        "summary_text":          cap_words(summary, max_words=_SUMMARY_MAX_WORDS),
        "entity_text":           entity_text,
        "bm25_text":             bm25_text,
        "rerank_text":           rerank_text,
        "local_context_text":    "",
        "section_context_text":  cap_words(section_context, max_words=_SECTION_CTX_WORDS),
        "image_caption_text":    "",
        "ocr_text":              "",
        "visual_summary_text":   "",
        "table_summary_text":    summary,
        "table_purpose_text":    purpose,
        "schema_text":           schema_text,
        "row_group_text":        markdown,
        "transcript_text":       "",
        "segment_summary_text":  "",
        "frame_caption_text":    "",
    }