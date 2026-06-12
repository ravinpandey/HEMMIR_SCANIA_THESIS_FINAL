"""
encoding_layer/models.py

Data structures for encoded chunk records.

Research design:

EncodingViews — multiple specialised text representations of one chunk.
Each view targets a different retrieval mechanism:

  retrieval_text      → PRIMARY dense embedding (all semantic signals)
  fused_retrieval_text → SECONDARY dense retrieval (summary + content + context)
  entity_text         → KEYWORD/SPARSE embedding (entities + codes + terms)
  bm25_text           → BM25 SPARSE INDEX (exact-match: codes, part numbers,
                         fault codes, version strings, technical identifiers)
                         Critical for Scania industrial queries where users search
                         for exact codes like "F3-447", "E404", "45 Nm".
                         Dense embeddings miss exact codes; BM25 catches them.
  title_text          → DOCUMENT ROUTING embedding (doc + section + anchor)
                         Used in Layer 1 structural retrieval to route a query
                         to the correct document and section before chunk search.
                         Keeps doc_title + section_title + semantic_anchor without
                         chunk content noise — pure structural signal.
  rerank_text         → CROSS-ENCODER RERANKER input (section + summary + content
                         + evidence_role context, no doc_title noise, 700 words)
  summary_text        → COMPACT embedding for quick similarity checks
  section_context_text → STRUCTURAL PRIOR (summaries of sibling chunks in section)

LinkageViews — graph structure written to ChromaDB metadata.
Enables the agent retriever to expand results without re-querying:
  related_figures     → expand text chunk to its referenced figures
  related_tables      → expand text chunk to its referenced tables
  related_section_ids → follow "see Section 3.2" cross-references
  sibling_chunk_ids   → all chunks in same structure unit (section context)
  neighbor_chunk_ids  → ±1 sequential neighbours (reading flow)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


@dataclass
class EncodingViews:
    # ── Raw content ────────────────────────────────────────────────────
    raw_text:             str = ""
    clean_text:           str = ""

    # ── Dense embedding views ──────────────────────────────────────────
    retrieval_text:       str = ""   # PRIMARY: all signals concatenated
    fused_retrieval_text: str = ""   # SECONDARY: summary + content + context
    summary_text:         str = ""   # COMPACT: contextual_summary only
    title_text:           str = ""   # ROUTING: doc + section + anchor (no content)

    # ── Sparse / keyword views ─────────────────────────────────────────
    entity_text:          str = ""   # Named entities + keywords (dense + sparse)
    bm25_text:            str = ""   # BM25 exact-match: codes, IDs, values

    # ── Reranker view ─────────────────────────────────────────────────
    rerank_text:          str = ""   # Cross-encoder input (700 words, no title)

    # ── Context views ─────────────────────────────────────────────────
    local_context_text:   str = ""   # ±2 neighbour chunks
    section_context_text: str = ""   # All chunk summaries in same section

    # ── Modality-specific views ────────────────────────────────────────
    image_caption_text:   str = ""
    ocr_text:             str = ""
    visual_summary_text:  str = ""
    table_summary_text:   str = ""
    table_purpose_text:   str = ""
    schema_text:          str = ""
    row_group_text:       str = ""
    transcript_text:      str = ""
    segment_summary_text: str = ""
    frame_caption_text:   str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LinkageViews:
    # ── Document provenance ────────────────────────────────────────────
    source_id:              str = ""
    parent_id:              str = ""

    # ── Structure navigation ───────────────────────────────────────────
    structure_unit_id:      str = ""
    structure_unit_type:    str = ""
    structure_unit_title:   str = ""
    semantic_anchor:        str = ""
    section_title:          str = ""
    section_id:             str = ""

    # ── Cross-modal expansion ──────────────────────────────────────────
    related_figures:        List[str] = field(default_factory=list)
    related_tables:         List[str] = field(default_factory=list)
    related_frames:         List[str] = field(default_factory=list)

    # ── Cross-section navigation ───────────────────────────────────────
    # Resolved from "see Section 3.2" references by cross_referencer.
    # Enables agent retriever to follow explicit cross-references between
    # sections without relying on embedding similarity.
    related_section_ids:    List[str] = field(default_factory=list)

    # ── Sequential navigation ──────────────────────────────────────────
    related_chunk_ids:      List[str] = field(default_factory=list)
    neighbor_chunk_ids:     List[str] = field(default_factory=list)
    sibling_chunk_ids:      List[str] = field(default_factory=list)

    # ── Positional signals ─────────────────────────────────────────────
    page_number:            int = 0
    slide_index:            int = 0
    sheet_name:             str = ""
    start_time_s:           Optional[float] = None
    end_time_s:             Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EncodedRecord:
    chunk_id:           str
    doc_id:             str
    source_id:          str
    structure_unit_id:  str
    source_modality:    str
    file_format:        str
    encoding_views:     EncodingViews
    linkage:            LinkageViews
    promoted_fields:    Dict[str, Any] = field(default_factory=dict)
    metadata:           Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["encoding_views"] = self.encoding_views.to_dict()
        d["linkage"]        = self.linkage.to_dict()
        return d