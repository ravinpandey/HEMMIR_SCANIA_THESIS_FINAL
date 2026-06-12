"""
shared/models/pipeline_models.py

Pydantic models for the full HEMMIR pipeline.

New vs V4:
  - QueryAnalysis: adds query_intent_type, uncertainty_prior, requires_temporal,
                   sub_questions, section_hint, use_hyde, clip_query
  - ScoreBreakdown: adds salience_boost, cross_encoder_score, evidence_role_boost
  - RetrievalTrace: NEW — full audit trail of every retrieval decision
  - AttributedClaim: NEW — per-claim attribution with evidence role + location
  - UncertaintyReport: NEW — composite multi-signal uncertainty score
  - ExplainableAnswer: NEW — replaces FinalAnswer, carries full explainability output
  - VideoSegmentMetadata / VideoFrameMetadata: NEW — for video modality plugins
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from shared.models.metadata_models import (
    DocumentMetadata,
    ImageChunkMetadata,
    TableChunkMetadata,
    TextChunkMetadata,
    VideoSegmentChunkMetadata,
    VideoFrameChunkMetadata,
)


# ── Chunk metadata union (discriminated by source_modality) ───────────────────

ChunkMetadata = Annotated[
    Union[
        TextChunkMetadata,
        ImageChunkMetadata,
        TableChunkMetadata,
        VideoSegmentChunkMetadata,
        VideoFrameChunkMetadata,
    ],
    Field(discriminator="source_modality"),
]


# ── Query analysis ─────────────────────────────────────────────────────────────

class Filters(BaseModel):
    document_type: Optional[str] = None
    tier1:         Optional[str] = None
    tier2:         Optional[str] = None
    project_id:    Optional[str] = None
    doc_title:     Optional[str] = None
    doc_id:        Optional[str] = None  # direct Chroma item-ID bypass (RQ1 use)


class BoostSignals(BaseModel):
    exact_code_match:         List[str]      = Field(default_factory=list)
    document_type_preference: Optional[str]  = None
    modality_preference:      Optional[str]  = None


class QueryAnalysis(BaseModel):
    """
    Full structured analysis of a user query.
    Produced by query_analyser.py before any retrieval begins.
    """
    # Routing
    path:              Literal["rag", "agent"] = "rag"
    complexity_reason: str                     = ""

    # Intent — maps directly to evidence_role filtering
    intent:            Literal[
        "factual", "comparative", "procedural", "exploratory",
        "diagnostic", "visual", "temporal"
    ]                                          = "factual"
    query_intent_type: str                     = "definition"

    # Modalities to activate (ordered by expected relevance)
    modalities:        List[str]               = Field(
        default_factory=lambda: ["text_to_text"]
    )

    # Agent-specific
    sub_questions:     List[Dict[str, Any]]    = Field(default_factory=list)
    section_hint:      Optional[str]           = None
    synthesis_instruction: str                 = ""
    requires_cross_document: bool              = False

    # Retrieval signals
    entities:          List[str]               = Field(default_factory=list)
    filters:           Filters                 = Field(default_factory=Filters)
    boost_signals:     BoostSignals            = Field(default_factory=BoostSignals)
    rewritten_query:   str                     = ""

    # HyDE / CLIP
    use_hyde:          bool                    = False
    clip_query:        Optional[str]           = None
    requires_temporal: bool                    = False

    # Uncertainty prior — set before retrieval, updated after
    uncertainty_prior: float                   = 0.7


class QueryContext(BaseModel):
    raw_query:            str
    rewritten_query:      str
    query_text:           str
    query_image_b64:      Optional[str]  = None
    query_table_text:     Optional[str]  = None
    requested_modalities: List[str]      = Field(default_factory=list)
    retrieval_mode:       Literal["rag", "agent"] = "rag"


# ── Score breakdown — extended with enrichment signals ────────────────────────

class ScoreBreakdown(BaseModel):
    """
    Complete score breakdown for one retrieved chunk.
    All components are individually inspectable for explainability.
    """
    vector_score:       float = 0.0   # cosine similarity from ChromaDB
    cross_encoder_score: float = 0.0  # cross-encoder re-ranking score
    code_boost:         float = 0.0   # exact code/ID match boost
    type_boost:         float = 0.0   # document type preference boost
    modality_boost:     float = 0.0   # modality preference boost
    salience_boost:     float = 0.0   # from enrichment: salience_score × weight
    evidence_role_boost: float = 0.0  # role match with query_intent_type
    noise_penalty:      float = 0.0   # boilerplate / noise penalty
    final_score:        float = 0.0   # weighted composite


# ── Retrieved chunk ────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    plugin_name:     str
    retrieval_mode:  str
    collection_name: str
    metadata:        ChunkMetadata
    score_breakdown: ScoreBreakdown
    content:         str             = ""
    extra_payload:   Dict[str, Any]  = Field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.metadata.chunk_id

    @property
    def doc_id(self) -> str:
        return self.metadata.doc_id

    @property
    def source_modality(self) -> str:
        return self.metadata.source_modality


# ── Document candidate ────────────────────────────────────────────────────────

class DocCandidate(BaseModel):
    doc_id:   str
    score:    float
    metadata: Optional[DocumentMetadata] = None


# ── Retrieval trace — full audit trail ────────────────────────────────────────

class SubQuestionResult(BaseModel):
    """Result of one sub-question in the Agent path."""
    question:            str
    evidence_role:       str
    preferred_modality:  str
    section_hint:        str                 = ""
    sections_navigated:  List[str]           = Field(default_factory=list)
    navigation_confidence: float             = 0.0
    sufficiency:         Literal["full", "partial", "insufficient"] = "partial"
    sufficiency_score:   float               = 0.0
    missing_aspects:     List[str]           = Field(default_factory=list)
    role_mismatch:       bool                = False
    iterations:          int                 = 1
    chunks_retrieved:    int                 = 0


class RetrievalTrace(BaseModel):
    """
    Complete audit trail of the retrieval process.
    The primary instrument for E1 (retrieval trace) explainability.
    """
    query:                  str
    path:                   Literal["rag", "agent"]
    complexity_reason:      str                      = ""
    query_intent_type:      str                      = ""
    uncertainty_prior:      float                    = 0.7

    # RAG-specific
    plugins_fired:          List[str]                = Field(default_factory=list)
    hyde_used:              bool                     = False
    multi_query_used:       bool                     = False
    docs_retrieved:         int                      = 0
    total_chunks_candidate: int                      = 0
    total_chunks_selected:  int                      = 0

    # Agent-specific
    sub_question_results:   List[SubQuestionResult]  = Field(default_factory=list)
    requires_cross_document: bool                    = False
    synthesis_instruction:  str                      = ""

    # Overall
    overall_sufficiency:    Literal["full", "partial", "insufficient"] = "partial"
    retrieval_duration_ms:  float                    = 0.0


# ── Evidence pack ─────────────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    chunk_id:        str
    doc_id:          str
    chunk_index:     int
    source_modality: str
    score:           float
    score_breakdown: ScoreBreakdown
    metadata:        ChunkMetadata
    content:         str            = ""
    retrieval_plugin: str           = ""
    retrieval_mode:  str            = ""
    collection_name: str            = ""
    extra_payload:   Dict[str, Any] = Field(default_factory=dict)


class EvidencePack(BaseModel):
    items:         List[EvidenceItem]
    total_items:   int
    was_truncated: bool = False


# ── Generation intermediaries ─────────────────────────────────────────────────

class DraftAnswer(BaseModel):
    text:             str
    cited_chunk_ids:  List[str] = Field(default_factory=list)
    abstained:        bool      = False
    raw_confidence:   float     = 0.5   # LLM self-reported confidence
    raw_missing:      str       = ""    # what LLM said was missing


class SupportedClaim(BaseModel):
    claim:               str
    supporting_chunk_id: str
    evidence_role:       str   = ""
    is_direct:           bool  = True
    citation_correct:    bool  = True


class UnsupportedClaim(BaseModel):
    claim:            str
    reason:           str  = ""
    suggested_search: str  = ""


class RoleMismatch(BaseModel):
    claim:         str
    expected_role: str
    actual_role:   str
    impact:        Literal["minor", "significant"] = "minor"


class FaithfulnessReport(BaseModel):
    supported_claims:    List[SupportedClaim]   = Field(default_factory=list)
    unsupported_claims:  List[UnsupportedClaim] = Field(default_factory=list)
    role_mismatches:     List[RoleMismatch]     = Field(default_factory=list)
    missing_evidence:    List[str]              = Field(default_factory=list)
    overall_faithfulness: float                 = 0.5
    contains_hallucination: bool                = False


# ── Explainability outputs ────────────────────────────────────────────────────

class AttributedClaim(BaseModel):
    """
    E2 — Per-claim attribution.
    Every factual claim in the answer traced to its source.
    """
    claim:           str
    chunk_id:        str
    doc_id:          str
    section_id:      str    = ""
    section_title:   str    = ""
    evidence_role:   str    = ""    # procedure/measurement/definition/context
    salience_score:  float  = 0.0
    modality:        str    = ""    # text/image/table/video_segment/video_frame
    location:        str    = ""    # "p.12" / "4:32s" / "Slide 7" / "rows 120-180"
    confidence:      float  = 0.5   # contextual_summary_confidence
    is_direct:       bool   = True  # evidence_role matches query_intent_type


class UncertaintyReport(BaseModel):
    """
    E3 — Composite multi-signal uncertainty score.
    The primary research contribution of the uncertainty layer.
    """
    # Composite score (0=uncertain, 1=certain)
    score:              float
    level:              Literal["HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE"]

    # Component signals (all individually inspectable)
    mean_salience:      float  = 0.0
    mean_cross_enc:     float  = 0.0
    mean_conf:          float  = 0.0
    sufficiency_value:  float  = 0.0   # 1.0=full, 0.5=partial, 0.0=insufficient
    faithfulness:       float  = 0.0
    uncertainty_prior:  float  = 0.7   # set at query analysis time

    # Penalty signals
    unsupported_count:  int    = 0
    missing_gap_count:  int    = 0
    role_mismatch_count: int   = 0
    hallucination_flag: bool   = False

    # Modality-specific signals
    dominant_modality:  str    = "text"
    asr_confidence:     Optional[float] = None   # video only
    caption_confidence: Optional[float] = None   # image only

    # Human-readable output
    level_explanation:  str    = ""
    recommendation:     str    = ""

    # Missing evidence gaps (from sufficiency checker + verifier)
    missing_evidence:   List[str] = Field(default_factory=list)
    role_mismatches:    List[RoleMismatch] = Field(default_factory=list)


class ExplainableAnswer(BaseModel):
    """
    The complete output of the HEMMIR pipeline.
    Replaces FinalAnswer — carries full explainability + uncertainty outputs.

    E1: retrieval_trace
    E2: attributed_claims
    E3: uncertainty
    E4: modality_provenance (inside evidence items)
    """
    # Core answer
    answer:              str
    follow_up_questions: List[str]          = Field(default_factory=list)

    # Evidence
    evidence:            List[EvidenceItem] = Field(default_factory=list)
    evidence_pack:       Optional[EvidencePack] = None

    # E1 — Retrieval trace
    retrieval_trace:     RetrievalTrace

    # E2 — Per-claim attribution
    attributed_claims:   List[AttributedClaim] = Field(default_factory=list)
    unsupported_claims:  List[UnsupportedClaim] = Field(default_factory=list)

    # E3 — Uncertainty
    uncertainty:         UncertaintyReport

    # Faithfulness
    faithfulness_report: FaithfulnessReport

    # Meta
    retrieval_path:      Literal["rag", "agent"]
    total_llm_calls:     int   = 0
    total_duration_ms:   float = 0.0
    argrag_claims:         List[str] = Field(default_factory=list)
    argrag_contested:      List[str] = Field(default_factory=list)
    argrag_claim_strength: float     = 0.0
    contested_note:        str       = ""
