"""
retrieval_layer/retrieval_pipeline.py

HEMMIR Retrieval Pipeline — two paths, one retrieval core.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE — where Simple RAG ends and ArgRAG begins
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SHARED CORE (retrieve()):
    Both paths go through this. Always runs first.
    ┌─────────────────────────────────────────────────────────────┐
    │  Query understanding → Stage 1 docs → Stage 2 sections      │
    │  → Stage 3 chunks (8 plugins) → cross-modal pairing         │
    │  → BM25 boost → section ref expansion → reranking           │
    │  → evidence_score E                                          │
    └────────────────────────────┬────────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
         C >= threshold                    C < threshold
         (generation layer decides)        (generation layer decides)
                 │                               │
                 ▼                               ▼
        ┌─────────────────┐          ┌──────────────────────┐
        │  SIMPLE RAG     │          │  ArgRAG PATH         │
        │  (fast path)    │          │  retrieve_for_agent() │
        └────────┬────────┘          └──────────┬───────────┘
                 │                               │
                 ▼                               ▼
    evidence_pack + E               sub-question decomposition
    → answer generation             → iterative retrieval
    C = wf·F + we·E                 → merged evidence_pack + E
      + wc·COV + ws·CONS            → sub-claims → support/attack
                                    → S(Ci) = αS - βA + γC
                                    C = wf·F + we·S(Ci)
                                      + wc·COV + ws·CONS

KEY POINT:
  The retrieval layer does NOT decide the path.
  retrieve() always runs the shared core and returns evidence_score E.
  The generation layer generates a Simple RAG answer, computes C.
  If C < threshold → generation layer calls retrieve_for_agent().

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESEARCH CONTRIBUTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Composite reranking score (6 signals including enrichment)
  2. Evidence score E = mean(salience × role_conf × cross_enc)
  3. BM25 hybrid boost (sparse+dense)
  4. 3-stage hierarchical retrieval with semantic_anchor
  5. retrieve_for_agent(): sub-question targeted retrieval for ArgRAG
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from generation_layer.evidence import build_evidence_pack, build_evidence_blocks
from generation_layer.fuse import fuse_chunks
from retrieval_layer.Agent.query_decomposer import QueryDecomposer
from retrieval_layer.Agent.section_navigator import SectionNavigator
from retrieval_layer.Agent.iterative_retriever import IterativeRetriever
from retrieval_layer.RAG.retrieve_chunks import retrieve_chunks
from retrieval_layer.RAG.retrieve_docs import (
    retrieve_documents, extract_doc_ids, retrieve_sections
)
from retrieval_layer.modality_plugin.registry import build_plugin_registry
from retrieval_layer.reranker.cross_encoder_reranker import CrossEncoderReranker
from retrieval_layer.utils.query_analyser import analyse_query
from retrieval_layer.utils.hyde import HyDE, MultiQueryExpander
from shared.models.pipeline_models import (
    EvidencePack,
    QueryAnalysis,
    QueryContext,
    RetrievalTrace,
    RetrievedChunk,
    SubQuestionResult,
)

# When ArgRAG re-retrieval runs, only accept improved evidence if E increases
# by at least this margin. Prevents infinite re-retrieval loops.
# Tunable hyperparameter — report in thesis.
MIN_EVIDENCE_IMPROVEMENT = 0.05


class RetrievalPipeline:
    """
    HEMMIR retrieval pipeline.

    retrieve()            — SHARED CORE: called first for both paths
    retrieve_for_agent()  — ArgRAG PATH: called by generation on escalation
    """

    def __init__(
        self,
        store,
        text_embedder,
        image_embedder,
        llm_client,
        top_k_docs:     int = 5,
        top_k_chunks:   int = 10,
        top_k_sections: int = 8,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.store          = store
        self.text_embedder  = text_embedder
        self.image_embedder = image_embedder
        self.llm_client     = llm_client
        self.top_k_docs     = top_k_docs
        self.top_k_chunks   = top_k_chunks
        self.top_k_sections = top_k_sections

        self.registry = build_plugin_registry(
            text_embedder  = text_embedder,
            image_embedder = image_embedder,
            llm_client     = llm_client,
            store          = store,
        )

        # ArgRAG-only components — used only in retrieve_for_agent()
        self.decomposer = QueryDecomposer(llm_client) if llm_client else None
        self.navigator  = SectionNavigator(llm_client, text_embedder) \
            if (llm_client and text_embedder) else None

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        self.reranker = CrossEncoderReranker(model_name=reranker_model, device=device)

        # HyDE + MultiQuery — used in BOTH paths inside shared core
        self._domain = "scania"
        self.hyde = None
        if llm_client and text_embedder:
            self.hyde           = HyDE(llm_client, text_embedder, domain=self._domain)
            self.multi_expander = MultiQueryExpander(llm_client, domain=self._domain)
        else:
            self.multi_expander = None

    def set_domain(self, domain: str) -> None:
        """
        Set domain for HyDE/MultiQuery prompts.
        Call pipeline.set_domain("academic") before SPIQA evaluation.
        Call pipeline.set_domain("scania") before Scania evaluation.
        """
        from retrieval_layer.utils.hyde import get_domain_prompts
        self._domain = domain
        if self.hyde:
            self.hyde.domain   = domain
            self.hyde._prompts = get_domain_prompts(domain)
        if self.multi_expander:
            self.multi_expander.domain   = domain
            self.multi_expander._prompts = get_domain_prompts(domain)

    # ══════════════════════════════════════════════════════════════
    # ┌─────────────────────────────────────────────────────────────┐
    # │  SHARED RETRIEVAL CORE                                      │
    # │  Called FIRST for BOTH Simple RAG and ArgRAG paths          │
    # └─────────────────────────────────────────────────────────────┘
    # ══════════════════════════════════════════════════════════════

    def retrieve(
        self,
        query:            str,
        filters:          Optional[Dict]      = None,
        query_image_b64:  Optional[str]       = None,
        query_table_text: Optional[str]       = None,
        force_modalities: Optional[List[str]] = None,
        rq1_config=       None,   # Optional[RQ1Config] — controls ablation knobs
    ) -> Dict[str, Any]:
        """
        SHARED RETRIEVAL CORE — runs for both Simple RAG and ArgRAG.

        Always runs first. Returns evidence_pack + evidence_score E.
        The generation layer decides path based on confidence C:
          C >= threshold → Simple RAG: use this evidence directly
          C <  threshold → ArgRAG:     call retrieve_for_agent()

        Returns dict:
            analysis        — QueryAnalysis (intent, modalities, flags)
            context         — QueryContext
            doc_candidates  — List[DocCandidate]
            doc_ids         — List[str]
            section_ids     — List[str]
            chunks          — List[RetrievedChunk] (reranked)
            evidence_pack   — EvidencePack
            evidence_score  — float E ∈ [0,1]  ← research output, feeds C
            retrieval_trace — RetrievalTrace
        """
        t0 = time.time()

        # ──────────────────────────────────────────────────────────────────────
        # Step 1: Query understanding
        # SHARED — used by both paths
        # Detects intent_type (procedural/measurement/definition/comparative/visual)
        # Sets HyDE flag, modality list, uncertainty prior
        # query_intent_type feeds reranker role-matching + claim generation
        # ──────────────────────────────────────────────────────────────────────
        analysis, context = analyse_query(
            query             = query,
            llm_client        = self.llm_client,
            filters           = filters,
            query_image_b64   = query_image_b64,
            query_table_text  = query_table_text,
        )

        analysis.modalities = ["text_to_text", "text_to_image", "text_to_table"]
        if force_modalities is not None:
            analysis.modalities = force_modalities

        # RQ1 ablation: override HyDE and MultiQuery flags
        _rq1 = rq1_config
        if _rq1 is not None:
            if not _rq1.use_hyde:
                analysis.use_hyde = False
            # use_multi_query handled when passing multi_expander below

        # Always start RAG — agent escalation is generation layer's decision
        analysis.path = "rag"

        logger.info(
            f"\n{'='*60}\n"
            f"  [SHARED] Query: {query[:80]}\n"
            f"  Intent: {analysis.query_intent_type} | "
            f"Uncertainty: {analysis.uncertainty_prior}\n"
            f"{'='*60}"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Step 2: Stage 1 — Document retrieval
        # SHARED — scopes all downstream retrieval to relevant documents
        # documents_collection (doc_embedding 1536-dim, 8901 chars)
        # Min score 0.20 — filters unrelated documents
        # ──────────────────────────────────────────────────────────────────────
        doc_candidates = retrieve_documents(
            self.store, self.text_embedder, analysis, self.top_k_docs
        )
        doc_ids = extract_doc_ids(doc_candidates)

        # RQ1 override: when caller passes {"doc_id": <hash>}, use it directly.
        # The Filters/analyse_query path doesn't reliably forward doc_id, so we
        # intercept it here on the raw filters dict before Stage 2.
        if _rq1 is not None and isinstance(filters, dict) and filters.get("doc_id"):
            doc_ids = [filters["doc_id"]]
            logger.info(f"  [RQ1] doc_id override: {doc_ids}")
        else:
            logger.info(f"  [SHARED] Stage 1: {len(doc_ids)} documents")

        # ──────────────────────────────────────────────────────────────────────
        # Step 3: Stage 2 — Section retrieval
        # SHARED — further scopes retrieval to relevant sections
        # sections_collection embedded using semantic_anchor + summary + keywords
        # semantic_anchor is enrichment output — richer than raw headings
        # Min score 0.30 — filters irrelevant sections before chunk search
        # ──────────────────────────────────────────────────────────────────────
        section_ids = retrieve_sections(
            store         = self.store,
            text_embedder = self.text_embedder,
            query         = analysis.rewritten_query,
            doc_ids       = doc_ids,
            top_k         = self.top_k_sections,
        )
        logger.info(f"  [SHARED] Stage 2: {len(section_ids)} sections")

        # ──────────────────────────────────────────────────────────────────────
        # Step 4: Stage 3 — Chunk retrieval (8 modality plugins)
        # SHARED — runs all modality plugins in parallel
        # Each plugin filters by doc_id + structure_unit_id from stages 1+2
        # HyDE embeds hypothetical passage if use_hyde=True
        # MultiQuery generates 3 variants, RRF fusion
        # ──────────────────────────────────────────────────────────────────────
        for plugin in self.registry.values():
            if hasattr(plugin, "store"):
                plugin.store = self.store

        _use_multi = True if _rq1 is None else _rq1.use_multi_query
        # RQ1 ablation: disable section_id filter so doc_id-only filtering is used.
        # Section IDs from Stage 2 (structure_unit_id) can mismatch text/image chunk
        # metadata, returning 0 chunks even when chunks exist for the document.
        # Tables often bypass this; text and image chunks are silently zeroed out.
        _rq1_section_ids = None if _rq1 is not None else (section_ids if section_ids else None)
        chunks = retrieve_chunks(
            store          = self.store,
            registry       = self.registry,
            analysis       = analysis,
            context        = context,
            doc_ids        = doc_ids,
            top_k          = self.top_k_chunks,
            hyde           = self.hyde if analysis.use_hyde else None,
            multi_expander = self.multi_expander if _use_multi else None,
            section_ids    = _rq1_section_ids,
        )
        logger.info(
            f"  [SHARED] Stage 3: {len(chunks)} chunks from "
            f"{len(analysis.modalities)} plugins"
        )

        trace = RetrievalTrace(
            query                  = context.raw_query,
            path                   = "rag",
            complexity_reason      = analysis.complexity_reason,
            query_intent_type      = analysis.query_intent_type,
            uncertainty_prior      = analysis.uncertainty_prior,
            plugins_fired          = list(analysis.modalities),
            hyde_used              = analysis.use_hyde and self.hyde is not None,
            multi_query_used       = self.multi_expander is not None,
            docs_retrieved         = len(doc_ids),
            total_chunks_candidate = len(chunks),
            overall_sufficiency    = "partial",
        )

        # ──────────────────────────────────────────────────────────────────────
        # Step 5: Fuse + cross-modal pairing
        # SHARED
        # Merges enrichment boost signals into chunk scores
        # Bidirectional text↔image↔table linking using cross-reference graph
        #   image → related text (for SPIQA: figure → explaining text)
        #   text  → related figure (for ArgRAG: text claim → visual evidence)
        #   table semantic → HTML content sibling (exact cell values)
        # ──────────────────────────────────────────────────────────────────────
        fused = fuse_chunks(
            chunks            = chunks,
            boost_signals     = analysis.boost_signals,
            query_intent_type = analysis.query_intent_type,
        )
        # RQ1 ablation: cross-modal pairing controlled by use_cross_modal
        if _rq1 is None or _rq1.use_cross_modal:
            fused = self._pair_table_siblings(fused)
            fused = self._pair_image_siblings(fused)
            fused = self._pair_figure_siblings(fused)

        # RQ1 ablation: BM25 boost controlled by use_bm25
        if _rq1 is None or _rq1.use_bm25:
            fused = self._boost_bm25_matches(fused, query)

        # RQ1 ablation: section expansion controlled by use_section_expansion
        if _rq1 is None or _rq1.use_section_expansion:
            fused = self._expand_section_references(fused)

        # ──────────────────────────────────────────────────────────────────────
        # Step 8: Composite reranking   [RESEARCH — core contribution]
        # SHARED
        # final = 0.40·dense + 0.30·cross_encoder + 0.12·salience
        #       + 0.08·role_boost × role_confidence + 0.05·summary_conf
        #       + 0.05·role_confidence
        # Enrichment-grounded scoring — what distinguishes HEMMIR from RAG
        # role_boost dampened by role_confidence — uncertain LLM tags don't
        # inflate scores (novel relative to standard RAG-Fusion literature)
        # ──────────────────────────────────────────────────────────────────────
        # Score all candidates, then apply soft-minimum modality selection.
        # Passing top_k=len(fused) ensures every candidate gets a score before
        # modality_aware_select picks the final 12 with diversity guarantees.
        _scored_all = self.reranker.rerank(
            query             = query,
            chunks            = fused,
            query_intent_type = analysis.query_intent_type,
            top_k             = len(fused) if fused else self.top_k_chunks,
        )
        reranked = modality_aware_select(
            _scored_all,
            top_k     = 12,
            min_image = 2,
            min_table = 3,
            min_text  = 6,
        )
        logger.info(
            f"  [SHARED] Reranked: top-{len(reranked)} chunks "
            f"(img={sum(1 for c in reranked if _chunk_modality(c)=='image')} "
            f"tbl={sum(1 for c in reranked if _chunk_modality(c)=='table')} "
            f"txt={sum(1 for c in reranked if _chunk_modality(c)=='text')})"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Step 9: Evidence score E   [RESEARCH — feeds Final Confidence C]
        # SHARED
        # E = mean( salience × role_confidence × cross_encoder ) over top-5
        #
        # Used in Final Confidence formula:
        #   Simple RAG: C = wf·F + we·E         + wc·COV + ws·CONS
        #   ArgRAG:     C = wf·F + we·S(Ci)     + wc·COV + ws·CONS
        #     ArgRAG replaces E with ClaimStrength S(Ci) which is richer
        #     because it integrates support/attack relations over E.
        #
        # E here is also used as the threshold for re-retrieval guard
        # in retrieve_for_agent() — only accept new evidence if new_E > E + 0.05
        # ──────────────────────────────────────────────────────────────────────
        evidence_score = compute_evidence_score(reranked, query)
        logger.info(f"  [SHARED] Evidence score E = {evidence_score:.4f}")

        # ──────────────────────────────────────────────────────────────────────
        # Step 10: Build evidence pack
        # SHARED
        # Packages reranked chunks into EvidencePack for generation layer
        # ──────────────────────────────────────────────────────────────────────
        evidence_pack = build_evidence_pack(
            reranked, self.registry, len(reranked)
        )

        trace.total_chunks_selected = len(evidence_pack.items)
        trace.retrieval_duration_ms = round((time.time() - t0) * 1000, 1)

        logger.info(
            f"  [SHARED] Done: {len(evidence_pack.items)} items | "
            f"E={evidence_score:.3f} | {trace.retrieval_duration_ms}ms\n"
            f"  → generation layer decides: Simple RAG or ArgRAG"
        )

        return {
            "analysis":        analysis,
            "context":         context,
            "doc_candidates":  doc_candidates,
            "doc_ids":         doc_ids,
            "section_ids":     section_ids,
            "chunks":          reranked,
            "evidence_pack":   evidence_pack,
            "evidence_score":  evidence_score,   # E → feeds C in both paths
            "retrieval_trace": trace,
        }

    # ══════════════════════════════════════════════════════════════
    # ┌─────────────────────────────────────────────────────────────┐
    # │  ArgRAG PATH                                                │
    # │  Called by generation layer ONLY when C < threshold         │
    # │  NOT called automatically — generation layer decides        │
    # └─────────────────────────────────────────────────────────────┘
    # ══════════════════════════════════════════════════════════════

    def retrieve_for_agent(
        self,
        query:                str,
        analysis:             QueryAnalysis,
        context:              QueryContext,
        doc_ids:              List[str],
        existing_evidence:    Optional[EvidencePack] = None,
        prior_evidence_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        ArgRAG retrieval — called by generation layer when C < threshold.

        Difference from retrieve():
          - Decomposes query into focused sub-questions (QueryDecomposer)
          - Retrieves per sub-question (IterativeRetriever)
          - Each sub-question has a target evidence_role that seeds
            ArgRAG claim generation (Step 4 in the system diagram)
          - Evidence guard: if new_E <= prior_E + 0.05, returns existing evidence

        Returns dict:
            analysis        — updated QueryAnalysis
            evidence_pack   — merged EvidencePack for ArgRAG steps 4-8
            evidence_score  — new E (or prior E if no improvement)
            retrieval_trace — RetrievalTrace with path="argrag"
        """
        t0 = time.time()
        logger.info(
            f"\n{'='*60}\n"
            f"  [ArgRAG] Retrieval — prior_E={prior_evidence_score:.3f}\n"
            f"  Escalated from generation layer (C < threshold)\n"
            f"{'='*60}"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Sub-question decomposition
        # ArgRAG ONLY
        # Breaks complex query into sub-questions each with target evidence_role
        # These sub-questions seed claim generation (Step 4 of system diagram)
        # decomp_conf: how confident is decomposition — used in synthesis
        # ──────────────────────────────────────────────────────────────────────
        sub_questions, synthesis_instr, decomp_conf = self.decomposer.decompose(
            query           = context.raw_query,
            intent_type     = analysis.query_intent_type,
            section_hint    = analysis.section_hint,
            existing_sub_qs = analysis.sub_questions,
        )
        logger.info(
            f"  [ArgRAG] {len(sub_questions)} sub-questions | "
            f"decomp_conf={decomp_conf:.2f}"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Iterative retrieval per sub-question
        # ArgRAG ONLY
        # Each sub-question gets a targeted retrieval pass within doc_ids scope
        # SectionNavigator resolves section hints in sub-question text
        # ──────────────────────────────────────────────────────────────────────
        iterative = IterativeRetriever(
            llm_client    = self.llm_client,
            text_embedder = self.text_embedder,
            store         = self.store,
            navigator     = self.navigator,
            registry      = self.registry,
            top_k_chunks  = self.top_k_chunks,
        )

        agent_chunks, sq_results = iterative.retrieve(
            sub_questions = sub_questions,
            doc_ids       = doc_ids,
            query_context = context,
        )
        logger.info(
            f"  [ArgRAG] {len(agent_chunks)} chunks from "
            f"{len(sub_questions)} sub-questions"
        )

        suf_values  = [r.sufficiency for r in sq_results]
        overall_suf = (
            "full"         if all(s == "full"         for s in suf_values) else
            "insufficient" if all(s == "insufficient" for s in suf_values) else
            "partial"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Merge with existing RAG evidence
        # ArgRAG ONLY
        # Agent evidence takes priority (purpose-targeted per sub-question)
        # Existing RAG evidence fills gaps for uncovered aspects
        # ──────────────────────────────────────────────────────────────────────
        all_chunks = list(agent_chunks)

        # Same post-processing as shared core
        fused    = fuse_chunks(all_chunks, analysis.boost_signals, analysis.query_intent_type)
        fused    = self._pair_table_siblings(fused)
        fused    = self._pair_image_siblings(fused)
        fused    = self._pair_figure_siblings(fused)
        fused    = self._boost_bm25_matches(fused, query)
        fused    = self._expand_section_references(fused)
        reranked = self.reranker.rerank(
            query             = query,
            chunks            = fused,
            query_intent_type = analysis.query_intent_type,
            top_k             = self.top_k_chunks,
        )

        # ──────────────────────────────────────────────────────────────────────
        # Evidence score guard
        # ArgRAG ONLY
        # Only accept new evidence if materially better than prior
        # Prevents HyDE re-retrieval loop (Step 11) from running indefinitely
        # MIN_EVIDENCE_IMPROVEMENT = 0.05 — tunable, report in thesis
        # ──────────────────────────────────────────────────────────────────────
        new_E = compute_evidence_score(reranked, query)
        logger.info(
            f"  [ArgRAG] new_E={new_E:.4f} prior_E={prior_evidence_score:.4f} "
            f"improvement={new_E - prior_evidence_score:+.4f}"
        )

        if (
            existing_evidence is not None
            and new_E <= prior_evidence_score + MIN_EVIDENCE_IMPROVEMENT
        ):
            logger.info(
                "  [ArgRAG] No material improvement — returning prior evidence"
            )
            return {
                "analysis":        analysis,
                "evidence_pack":   existing_evidence,
                "evidence_score":  prior_evidence_score,
                "retrieval_trace": RetrievalTrace(
                    query                 = context.raw_query,
                    path                  = "argrag_no_improvement",
                    query_intent_type     = analysis.query_intent_type,
                    docs_retrieved        = len(doc_ids),
                    total_chunks_selected = len(existing_evidence.items),
                    overall_sufficiency   = "partial",
                    retrieval_duration_ms = round((time.time() - t0) * 1000, 1),
                ),
            }

        evidence_pack = build_evidence_pack(
            reranked, self.registry, self.top_k_chunks
        )

        trace = RetrievalTrace(
            query                   = context.raw_query,
            path                    = "argrag",
            complexity_reason       = "confidence-based escalation",
            query_intent_type       = analysis.query_intent_type,
            uncertainty_prior       = analysis.uncertainty_prior,
            plugins_fired           = list(analysis.modalities),
            docs_retrieved          = len(doc_ids),
            total_chunks_candidate  = len(all_chunks),
            total_chunks_selected   = len(evidence_pack.items),
            sub_question_results    = sq_results,
            requires_cross_document = analysis.requires_cross_document,
            synthesis_instruction   = synthesis_instr,
            overall_sufficiency     = overall_suf,
            retrieval_duration_ms   = round((time.time() - t0) * 1000, 1),
        )

        logger.info(
            f"  [ArgRAG] Done: {len(evidence_pack.items)} items | "
            f"E={new_E:.3f} | {trace.retrieval_duration_ms}ms\n"
            f"  → generation layer: sub-claims → support/attack → S(Ci) → C"
        )

        return {
            "analysis":        analysis,
            "evidence_pack":   evidence_pack,
            "evidence_score":  new_E,
            "retrieval_trace": trace,
        }

    def get_evidence_blocks(self, evidence_pack: EvidencePack) -> str:
        return build_evidence_blocks(evidence_pack, self.registry)

    # ══════════════════════════════════════════════════════════════
    # Internal helpers — used by both retrieve() and retrieve_for_agent()
    # ══════════════════════════════════════════════════════════════

    def _pair_table_siblings(self, chunks):
        """Fetch HTML content sibling for each semantic table chunk."""
        collection = self.store.collections.get("tables")
        if not collection:
            return chunks
        existing_ids = {c.chunk_id for c in chunks}
        additions = []
        for chunk in chunks:
            if chunk.source_modality != "table" or chunk.chunk_id.endswith("_html"):
                continue
            sibling_id = chunk.chunk_id + "_html"
            if sibling_id in existing_ids:
                continue
            try:
                result = collection.get(ids=[sibling_id], include=["documents", "metadatas"])
                if not result.get("ids"):
                    continue
                sibling = chunk.model_copy(update={
                    "chunk_id":      sibling_id,
                    "content":       (result["documents"] or [""])[0] or "",
                    "extra_payload": (result["metadatas"]  or [{}])[0] or {},
                })
                if hasattr(sibling.metadata, "chunk_id"):
                    object.__setattr__(sibling.metadata, "chunk_id", sibling_id)
                additions.append(sibling)
                existing_ids.add(sibling_id)
            except Exception as e:
                logger.debug(f"  _pair_table_siblings: {e}")
        if additions:
            logger.info(f"  Table siblings: +{len(additions)}")
        return chunks + additions

    def _pair_image_siblings(self, chunks):
        """Image → text: fetch related text chunks using related_sections."""
        import json
        text_collection = self.store.collections.get("text")
        if not text_collection:
            return chunks
        existing_ids = {c.chunk_id for c in chunks}
        additions = []
        for chunk in chunks:
            if chunk.source_modality != "image":
                continue
            related_sections = []
            if hasattr(chunk.metadata, "related_sections"):
                related_sections = chunk.metadata.related_sections or []
            elif "related_sections" in chunk.extra_payload:
                raw = chunk.extra_payload.get("related_sections", "[]")
                try:
                    related_sections = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    related_sections = []
            for related_id in related_sections[:2]:
                if related_id in existing_ids:
                    continue
                try:
                    result = text_collection.get(ids=[related_id], include=["documents", "metadatas"])
                    if not result.get("ids"):
                        continue
                    from shared.models.pipeline_models import ScoreBreakdown
                    meta = (result["metadatas"] or [{}])[0] or {}
                    additions.append(chunk.model_copy(update={
                        "chunk_id":        related_id,
                        "source_modality": "text",
                        "content":         (result["documents"] or [""])[0] or "",
                        "extra_payload":   {**meta, "linked_from_image": chunk.chunk_id},
                        "score_breakdown": ScoreBreakdown(
                            vector_score = chunk.score_breakdown.vector_score * 0.85,
                            final_score  = chunk.score_breakdown.final_score  * 0.85,
                        ),
                    }))
                    existing_ids.add(related_id)
                except Exception as e:
                    logger.debug(f"  _pair_image_siblings: {e}")
        if additions:
            logger.info(f"  Image siblings: +{len(additions)}")
        return chunks + additions

    def _pair_figure_siblings(self, chunks):
        """Text → image: fetch related figure chunks using related_figures."""
        import json
        image_collection = self.store.collections.get("images_text")
        if not image_collection:
            return chunks
        existing_ids = {c.chunk_id for c in chunks}
        additions = []
        for chunk in chunks:
            if chunk.source_modality != "text":
                continue
            raw = chunk.extra_payload.get("related_figures", "[]")
            try:
                related_figures = json.loads(raw) if isinstance(raw, str) else (raw or [])
            except Exception:
                related_figures = []
            for fig_id in related_figures[:2]:
                if fig_id in existing_ids:
                    continue
                try:
                    result = image_collection.get(ids=[fig_id], include=["documents", "metadatas"])
                    if not result.get("ids"):
                        continue
                    from shared.models.pipeline_models import ScoreBreakdown
                    meta = (result["metadatas"] or [{}])[0] or {}
                    additions.append(chunk.model_copy(update={
                        "chunk_id":        fig_id,
                        "source_modality": "image",
                        "content":         (result["documents"] or [""])[0] or "",
                        "extra_payload":   {**meta, "linked_from_text": chunk.chunk_id},
                        "score_breakdown": ScoreBreakdown(
                            vector_score = chunk.score_breakdown.vector_score * 0.80,
                            final_score  = chunk.score_breakdown.final_score  * 0.80,
                        ),
                    }))
                    existing_ids.add(fig_id)
                except Exception as e:
                    logger.debug(f"  _pair_figure_siblings: {e}")
        if additions:
            logger.info(f"  Figure siblings: +{len(additions)}")
        return chunks + additions

    def _boost_bm25_matches(self, chunks, query):
        """
        BM25 hybrid boost — sparse exact-match signal.  [RESEARCH]
        Reads bm25_text from ChromaDB metadata. +0.08 per term, cap 0.20.
        """
        import re
        query_terms = set(re.findall(r'[A-Za-z0-9][A-Za-z0-9\-_.]{1,}', query.lower()))
        if not query_terms:
            return chunks
        BM25_BOOST = 0.08
        boosted = []
        for chunk in chunks:
            bm25_text = (chunk.extra_payload.get("bm25_text") or "").lower()
            if bm25_text:
                matches = sum(1 for t in query_terms if t in bm25_text)
                if matches > 0:
                    boost = min(BM25_BOOST * matches, 0.20)
                    new_score = min(1.0, chunk.score_breakdown.final_score + boost)
                    from shared.models.pipeline_models import ScoreBreakdown
                    chunk = chunk.model_copy(update={"score_breakdown": ScoreBreakdown(
                        vector_score        = chunk.score_breakdown.vector_score,
                        cross_encoder_score = chunk.score_breakdown.cross_encoder_score,
                        code_boost          = round(boost, 4),
                        salience_boost      = chunk.score_breakdown.salience_boost,
                        evidence_role_boost = chunk.score_breakdown.evidence_role_boost,
                        noise_penalty       = chunk.score_breakdown.noise_penalty,
                        final_score         = round(new_score, 4),
                    )})
            boosted.append(chunk)
        return boosted

    def _expand_section_references(self, chunks):
        """
        Follow related_section_ids cross-references in chunk metadata.
        Fetches top chunk from each referenced section at score × 0.75.
        """
        import json
        text_collection = self.store.collections.get("text")
        if not text_collection:
            return chunks
        existing_ids = {c.chunk_id for c in chunks}
        additions = []
        for chunk in chunks:
            if chunk.source_modality != "text":
                continue
            raw = chunk.extra_payload.get("related_section_ids", "[]")
            try:
                section_ids = json.loads(raw) if isinstance(raw, str) else (raw or [])
            except Exception:
                section_ids = []
            for su_id in section_ids[:2]:
                try:
                    result = text_collection.get(
                        where={"structure_unit_id": {"$eq": su_id}},
                        limit=1, include=["documents", "metadatas"]
                    )
                    if not result.get("ids"):
                        continue
                    ref_id = result["ids"][0]
                    if ref_id in existing_ids:
                        continue
                    from shared.models.pipeline_models import ScoreBreakdown
                    meta = (result["metadatas"] or [{}])[0] or {}
                    additions.append(chunk.model_copy(update={
                        "chunk_id":        ref_id,
                        "source_modality": "text",
                        "content":         (result["documents"] or [""])[0] or "",
                        "extra_payload":   {**meta, "linked_from_section_ref": chunk.chunk_id},
                        "score_breakdown": ScoreBreakdown(
                            vector_score = chunk.score_breakdown.vector_score * 0.75,
                            final_score  = chunk.score_breakdown.final_score  * 0.75,
                        ),
                    }))
                    existing_ids.add(ref_id)
                except Exception as e:
                    logger.debug(f"  _expand_section_references: {e}")
        if additions:
            logger.info(f"  Section refs: +{len(additions)}")
        return chunks + additions


# ══════════════════════════════════════════════════════════════════════════════
# Modality-aware soft-minimum selection
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_modality(chunk: "RetrievedChunk") -> str:
    ct = (
        chunk.extra_payload.get("source_modality")
        or chunk.extra_payload.get("chunk_type")
        or chunk.extra_payload.get("content_type")
        or getattr(chunk.metadata, "source_modality", "")
        or "text"
    )
    if ct in ("image", "figure"):
        return "image"
    if ct == "table":
        return "table"
    return "text"


def modality_aware_select(
    chunks:     "List[RetrievedChunk]",
    top_k:      int = 12,
    min_image:  int = 2,
    min_table:  int = 3,
    min_text:   int = 6,
) -> "List[RetrievedChunk]":
    """
    Soft-minimum modality-aware selection from a score-sorted candidate pool.

    Phase 1 — guarantee up to min_X chunks per modality (if available).
    Phase 2 — fill remaining slots with best-scored chunks from any modality.
    Phase 3 — re-sort selected chunks by final_score descending.

    Degrades gracefully: if a modality has no chunks its guaranteed slots
    fall through to the free pool, so text-only docs are unaffected.
    """
    selected:  "List[RetrievedChunk]" = []
    used_ids: set = set()

    # Build per-modality lists (chunks already sorted by score)
    buckets: dict = {"image": [], "table": [], "text": []}
    for c in chunks:
        buckets.setdefault(_chunk_modality(c), []).append(c)

    # Phase 1: guaranteed minimum slots
    for mod, min_n in [("image", min_image), ("table", min_table), ("text", min_text)]:
        count = 0
        for c in buckets.get(mod, []):
            if count >= min_n or len(selected) >= top_k:
                break
            if c.chunk_id not in used_ids:
                selected.append(c)
                used_ids.add(c.chunk_id)
                count += 1

    # Phase 2: fill remaining with best-scored regardless of modality
    for c in chunks:
        if len(selected) >= top_k:
            break
        if c.chunk_id not in used_ids:
            selected.append(c)
            used_ids.add(c.chunk_id)

    # Phase 3: re-sort by final score
    selected.sort(key=lambda c: c.score_breakdown.final_score, reverse=True)
    return selected[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# Evidence score E — standalone function used by both paths
# ══════════════════════════════════════════════════════════════════════════════

def compute_evidence_score(
    chunks: List[RetrievedChunk],
    query:  str,
) -> float:
    """
    Evidence quality score E ∈ [0,1].  [RESEARCH]

    E = mean( salience × role_confidence × cross_encoder ) over top-5 chunks

    Feeds Final Confidence C in both paths:
        Simple RAG:  C = wf·F + we·E         + wc·COV + ws·CONS
        ArgRAG:      C = wf·F + we·ClaimStrength + wc·COV + ws·CONS

    In ArgRAG, ClaimStrength S(Ci) = αS - βA + γC replaces E.
    S(Ci) is richer because it integrates support/attack argumentation
    over the raw evidence — but E is still computed here as the
    pre-reasoning baseline and as the re-retrieval guard threshold.

    Design: all three factors must be high for strong E.
        salience high + low role_conf = LLM unsure about chunk role
        salience high + low CE score  = salient but off-topic for query
        CE high + low salience        = relevant but peripheral to document
    """
    if not chunks:
        return 0.0
    scores = []
    for chunk in chunks[:5]:
        sb  = chunk.score_breakdown
        ep  = chunk.extra_payload
        ct  = ep.get("chunk_type") or ep.get("content_type") or ""

        if ct == "table":
            # Use enrichment confidences as salience proxy for table chunks
            # (salience_score is not set for non-text chunks)
            confs = [
                ep.get("table_summary_confidence"),
                ep.get("table_purpose_confidence"),
            ]
            valid = [float(c) for c in confs if c is not None]
            salience = sum(valid) / len(valid) if valid else 0.5
        elif ct in ("image", "figure"):
            confs = [
                ep.get("image_caption_confidence"),
                ep.get("depicted_component_confidence"),
                ep.get("contextual_summary_confidence"),
                ep.get("visible_annotations_confidence"),
            ]
            valid = [float(c) for c in confs if c is not None]
            salience = sum(valid) / len(valid) if valid else 0.5
        else:
            salience = float(
                ep.get("salience_score")
                or getattr(chunk.metadata, "salience_score", 0.5)
                or 0.5
            )

        role_conf = float(
            ep.get("evidence_role_confidence")
            or getattr(chunk.metadata, "evidence_role_confidence", 0.5)
            or 0.5
        )
        ce_score = float(sb.cross_encoder_score or sb.vector_score or 0.5)
        scores.append(salience * role_conf * ce_score)
    return round(sum(scores) / len(scores), 4) if scores else 0.0