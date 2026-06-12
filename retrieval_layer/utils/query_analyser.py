"""
retrieval_layer/utils/query_analyser.py

Research-grade query analysis for HEMMIR.

Produces QueryAnalysis with:
  - path: rag | agent
  - query_intent_type: maps to evidence_role filtering
  - sub_questions: agent decomposition (structured with evidence_role per sub-q)
  - uncertainty_prior: seeds uncertainty layer before retrieval
  - modalities: ordered plugin activation list
  - use_hyde: whether to use Hypothetical Document Embedding
  - requires_temporal: activates video_segment + video_frame plugins

Three-tier priority:
  1. Hard regex overrides (certain simple/complex signals)
  2. LLM full analysis (preferred — uses research-grade prompts)
  3. Regex heuristic fallback (when LLM unavailable)

The LLM call is structured to produce both routing AND decomposition in
one call for efficiency — avoids a separate decomposer call for agent queries.

Changelog:
  - Added _COMPARATIVE_TABLE_SIGNALS regex to detect queries that need
    text_to_table even when no explicit table keyword is present
  - _regex_analyse: comparative/measurement queries now include text_to_table
  - _build_complex_analysis: always includes text_to_table for comparative
  - ANALYSIS_PROMPT: explicit rule to include text_to_table for comparative
  - _apply_external_filters: ensures text_to_table added when comparative intent
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models.pipeline_models import (
    BoostSignals,
    Filters,
    QueryAnalysis,
    QueryContext,
)

# ── Domain-aware regex signals ────────────────────────────────────────────────

_TABLE_SIGNALS   = re.compile(
    r'\b(table|list|values?|how many|port|rule|config|setting|parameter|'
    r'row|column|statistics?|specification|threshold|limit|range|fault code|'
    r'fault log|dataset|sheet|csv|xlsx)\b', re.I
)
_IMAGE_SIGNALS   = re.compile(
    r'\b(diagram|figure|chart|image|picture|show|illustrat|architecture|'
    r'layout|visual|draw|photo|graph|plot|visualization|schematic|thumbnail)\b', re.I
)
_VIDEO_SIGNALS   = re.compile(
    r'\b(video|in the video|at what point|timestamp|minute|scene|'
    r'instructor|demonstration|show me when|training video|clip)\b', re.I
)
_CODE_PATTERN    = re.compile(
    r'\b([A-Z]{2,}-\d+|\d{4,}|v\d+\.\d+|[A-Z]{2,}_[A-Z0-9_]{2,}|'
    r'\d+\s*Nm|\d+\s*bar|\d+\s*kPa)\b'
)
_HARD_SIMPLE     = re.compile(
    r'\b(what is the|define|definition of|find the value|what port|'
    r'show me the diagram|find figure|what version|what is|who is)\b', re.I
)
_HARD_COMPLEX    = re.compile(
    r'\b(compare|comparison between|difference between|across all|'
    r'summarize|summarise|explain (why|how) .{10,}|'
    r'relationship between|impact of|trade.?off|pros and cons|'
    r'contrast|evaluate|assess)\b', re.I
)
_PROCEDURAL      = re.compile(
    r'\b(how (do|to|can|should)|procedure|step(s)?|process|'
    r'install|replace|maintain|service|repair|calibrate|configure)\b', re.I
)
_MEASUREMENT     = re.compile(
    r'\b(torque|pressure|temperature|voltage|resistance|rpm|nm|kpa|bar|'
    r'specification|tolerance|limit|range|value|threshold)\b', re.I
)
_TEMPORAL        = re.compile(
    r'\b(timestamp|at what (point|time|minute)|when does|'
    r'in the video|scene|clip|at \d+:\d+)\b', re.I
)

# ── NEW: comparative/metric signals that strongly indicate table evidence ──────
# These are queries where the decisive answer lives in a table even when
# the user does not use the word "table". Examples:
#   "How does self-attention compare to recurrent in complexity?"
#   "What BLEU score did the Transformer achieve?"
#   "Which model performs best on EN-DE translation?"
_COMPARATIVE_TABLE_SIGNALS = re.compile(
    r'\b(compar|versus|vs\.?|differ|complex|complexit|'
    r'perform|performanc|metric|benchmark|ablat|evaluat|'
    r'BLEU|accuracy|precision|recall|F1|score|result|'
    r'faster|slower|better|worse|efficient|'
    r'per.layer|sequential|path.length|'
    r'O\s*\(|Big.?O|trade.?off|overhead)\b', re.I
)


# ── Research-grade prompts ────────────────────────────────────────────────────

ANALYSIS_SYSTEM = """\
You are a retrieval routing expert for HEMMIR — an Explainable Multimodal \
Information Retrieval system for industrial technical documents.

The corpus contains: PDF manuals, DOCX reports, PPTX presentations, \
CSV/XLSX datasets, and MP4 training videos from a heavy vehicle manufacturer.

ROUTING RULES:
  RAG path  → single-fact lookup, definition, image retrieval, code/ID search,
               direct table query, answerable from one chunk with high confidence
  Agent path → multi-hop reasoning, cross-section synthesis, comparative analysis,
               summarisation of a section/scene, cross-document queries,
               causal explanations spanning multiple concepts

BIAS: When uncertain, choose Agent.
  Cost of wrong RAG = silent incomplete answer (unacceptable for industrial safety)
  Cost of wrong Agent = 3-5s extra latency (acceptable)

QUERY INTENT TYPES (maps to evidence_role in the index):
  procedural   → how-to, step-by-step, maintenance procedure
  measurement  → specific values, tolerances, specifications
  definition   → what is X, explain X, define X
  comparative  → difference between, compare, which is better
  diagnostic   → fault codes, error conditions, troubleshooting
  exploratory  → overview, summary, what does section X cover
  visual       → find diagram, show image, what does X look like
  temporal     → when in the video, at what timestamp\
"""

ANALYSIS_PROMPT = """\
Analyse this query for industrial document retrieval and produce routing + decomposition.

Query: "{query}"

Think step by step:
1. How many distinct pieces of information does answering require?
2. Does answering require navigating document structure (sections/scenes/slides)?
3. What type of evidence chunk answers this best?
4. Which modalities are likely relevant?
5. Should HyDE be used? (yes when query vocabulary differs from document vocabulary)

Respond in EXACTLY this JSON (no markdown):
{{
  "path": "rag" or "agent",
  "complexity_reason": "<one sentence>",
  "query_intent_type": "procedural|measurement|definition|comparative|diagnostic|exploratory|visual|temporal",
  "modalities": ["text_to_text","text_to_image","text_to_table","video_segment","video_frame"],
  "use_hyde": true or false,
  "requires_temporal": true or false,
  "clip_query": "<visual description for CLIP or null>",
  "section_hint": "<keyword to guide section navigation or null>",
  "exact_codes": ["<part numbers, fault codes, torque specs found in query>"],
  "filters": {{"tier1": null, "tier2": null, "document_type": null, "project_id": null}},
  "uncertainty_prior": 0.0,
  "sub_questions": [
    {{
      "question": "<specific sub-question>",
      "evidence_role": "procedural|measurement|definition|comparative|diagnostic",
      "preferred_modality": "text|image|table|video_segment|video_frame",
      "section_hint": "<keyword>",
      "is_prerequisite_for": null
    }}
  ],
  "synthesis_instruction": "<how to combine sub-answers, or empty string for rag>",
  "requires_cross_document": false
}}

Rules:
- sub_questions: populated only when path=agent, empty list for rag
- modalities: ordered by expected relevance, 1-3 items
- uncertainty_prior: 0.9=direct fact, 0.7=moderate, 0.5=multi-hop, 0.3=exploratory
- use_hyde: true when user might phrase differently from document language
- IMPORTANT: always include "text_to_table" in modalities when the query involves
  any of: comparison, performance metrics, complexity, BLEU scores, ablation
  results, numerical benchmarks, vs/versus, trade-offs, efficiency analysis,
  or any question where the answer is likely in a results/comparison table\
"""


# ── Main analysis function ────────────────────────────────────────────────────

def analyse_query(
    query:          str,
    llm_client=     None,
    filters:        Optional[Dict] = None,
    query_image_b64: Optional[str] = None,
    query_table_text: Optional[str] = None,
) -> tuple[QueryAnalysis, QueryContext]:
    """
    Full query analysis producing QueryAnalysis + QueryContext.

    Priority:
      1. Hard overrides (regex) — definite simple/complex signals
      2. LLM full analysis (preferred)
      3. Regex heuristic fallback
    """
    # Handle both dict and Filters object
    if hasattr(filters, "__dict__") and not isinstance(filters, dict):
        raw_filters = {k: v for k, v in filters.__dict__.items() if v is not None}
    else:
        raw_filters = {k: v for k, v in vars(filters).items() if v is not None} if hasattr(filters, "__fields__") else (filters or {})

    # ── Hard simple override ───────────────────────────────────────────
    has_image  = query_image_b64 is not None
    has_table  = query_table_text is not None
    hard_s     = _HARD_SIMPLE.search(query)
    hard_c     = _HARD_COMPLEX.search(query)

    if hard_s and not hard_c and not has_image and not has_table:
        img_score = len(_IMAGE_SIGNALS.findall(query))
        tbl_score = len(_TABLE_SIGNALS.findall(query))
        if img_score == 0 and tbl_score <= 1 and not _VIDEO_SIGNALS.search(query):
            analysis = _build_simple_analysis(query, raw_filters)
            analysis.complexity_reason = "Hard simple signal — single fact lookup"
            context  = _build_context(query, analysis, query_image_b64, query_table_text)
            logger.debug(f"Hard simple override: {query[:60]}")
            return analysis, context

    # ── Hard complex override ──────────────────────────────────────────
    if hard_c:
        if llm_client:
            try:
                analysis = _llm_analyse(query, llm_client)
                analysis.path = "agent"
                analysis.complexity_reason = (
                    "Hard complex signal — " + analysis.complexity_reason
                )
                # Ensure text_to_table for comparative queries
                _ensure_table_modality(analysis, query)
                _apply_external_filters(analysis, raw_filters, has_image, has_table)
                context = _build_context(query, analysis, query_image_b64, query_table_text)
                return analysis, context
            except Exception as e:
                logger.warning(f"LLM analysis failed on hard complex: {e}")
        analysis = _build_complex_analysis(query, raw_filters)
        analysis.complexity_reason = "Hard complex signal — comparison/synthesis"
        context  = _build_context(query, analysis, query_image_b64, query_table_text)
        return analysis, context

    # ── LLM full analysis (preferred path) ────────────────────────────
    if llm_client:
        try:
            analysis = _llm_analyse(query, llm_client)
            # Ensure text_to_table for comparative queries even if LLM missed it
            _ensure_table_modality(analysis, query)
            _apply_external_filters(analysis, raw_filters, has_image, has_table)
            context  = _build_context(query, analysis, query_image_b64, query_table_text)
            logger.debug(
                f"LLM analysis: path={analysis.path} "
                f"intent={analysis.query_intent_type} "
                f"modalities={analysis.modalities} "
                f"prior={analysis.uncertainty_prior}"
            )
            return analysis, context
        except Exception as e:
            logger.warning(f"LLM analysis failed — using regex fallback: {e}")

    # ── Regex fallback ────────────────────────────────────────────────
    analysis = _regex_analyse(query, raw_filters, has_image, has_table)
    context  = _build_context(query, analysis, query_image_b64, query_table_text)
    return analysis, context


# ── LLM analysis ─────────────────────────────────────────────────────────────

def _llm_analyse(query: str, llm_client) -> QueryAnalysis:
    prompt   = ANALYSIS_PROMPT.format(query=query)
    response = llm_client.invoke(
        system=ANALYSIS_SYSTEM, prompt=prompt, max_tokens=800
    )
    data = _extract_json(response)
    if not data:
        raise ValueError("No valid JSON in LLM analysis response")

    exact_codes = data.get("exact_codes", []) or list(
        set(_CODE_PATTERN.findall(query))
    )
    filters_raw = data.get("filters", {}) or {}

    return QueryAnalysis(
        path               = data.get("path", "rag"),
        complexity_reason  = data.get("complexity_reason", ""),
        intent             = _map_intent(data.get("query_intent_type", "factual")),
        query_intent_type  = data.get("query_intent_type", "definition"),
        modalities         = (["video_segment", "video_frame", "text_to_text", "text_to_image", "text_to_table"]
                               if data.get("requires_temporal") else
                               ["text_to_text", "text_to_image", "text_to_table"]),
        use_hyde           = bool(data.get("use_hyde", False)),
        requires_temporal  = bool(data.get("requires_temporal", False)),
        clip_query         = data.get("clip_query"),
        section_hint       = data.get("section_hint"),
        sub_questions      = data.get("sub_questions", []) or [],
        synthesis_instruction = data.get("synthesis_instruction", ""),
        requires_cross_document = bool(data.get("requires_cross_document", False)),
        entities           = _extract_entities(query),
        filters            = Filters(
            document_type = filters_raw.get("document_type") if isinstance(filters_raw.get("document_type"), str) else None,
            tier1         = filters_raw.get("tier1") if isinstance(filters_raw.get("tier1"), str) else None,
            tier2         = filters_raw.get("tier2") if isinstance(filters_raw.get("tier2"), str) else None,
            project_id    = filters_raw.get("project_id") if isinstance(filters_raw.get("project_id"), str) else None,
        ),
        boost_signals      = BoostSignals(
            exact_code_match = exact_codes,
        ),
        rewritten_query    = query.strip(),
        uncertainty_prior  = float(data.get("uncertainty_prior", 0.7)),
    )


# ── Regex fallback ────────────────────────────────────────────────────────────

def _regex_analyse(
    query: str, raw_filters: Dict, has_image: bool, has_table: bool
) -> QueryAnalysis:
    q_lower     = query.lower()
    word_count  = len(query.split())

    # Complexity scoring
    complexity  = 0
    if re.search(r'\b(and|also|as well as|furthermore)\b', q_lower): complexity += 1
    if re.search(r'\b(compar|versus|differ|between|contrast)\b', q_lower): complexity += 2
    if re.search(r'\b(why|how does|explain|reason|cause|impact)\b', q_lower): complexity += 1
    if re.search(r'\b(section|chapter|overview|summarize|summarise)\b', q_lower): complexity += 2
    if re.search(r'\b(all documents?|across|multiple documents?)\b', q_lower): complexity += 3
    if word_count > 25: complexity += 1

    is_complex      = complexity >= 2
    is_temporal     = bool(_VIDEO_SIGNALS.search(query))
    is_visual       = bool(_IMAGE_SIGNALS.search(query))
    is_table        = bool(_TABLE_SIGNALS.search(query))
    is_proc         = bool(_PROCEDURAL.search(query))
    is_meas         = bool(_MEASUREMENT.search(query))
    # NEW: detect comparative/metric queries that need table retrieval
    is_comparative  = bool(_COMPARATIVE_TABLE_SIGNALS.search(query))

    # Intent type
    if is_proc:            intent_type = "procedural"
    elif is_meas:          intent_type = "measurement"
    elif is_temporal:      intent_type = "temporal"
    elif is_visual:        intent_type = "visual"
    elif is_comparative:   intent_type = "comparative"   # NEW
    elif is_table:         intent_type = "definition"
    else:                  intent_type = "definition"

    # ── Modalities — always retrieve all document modalities ──────────
    # Text + Image + Table retrieved simultaneously for every query.
    # Fusion Score handles ranking — irrelevant chunks score low.
    # Video is a separate Stage 2 escalation triggered by low confidence.
    if is_temporal:
        modalities = ["video_segment", "video_frame", "text_to_text",
                      "text_to_image", "text_to_table"]
    elif has_image:
        modalities = ["image_to_image", "image_to_text",
                      "text_to_text", "text_to_table"]
    else:
        modalities = ["text_to_text", "text_to_image", "text_to_table"]

    prior = 0.5 if is_complex else (0.6 if is_proc or is_meas else 0.8)

    exact_codes = list(set(_CODE_PATTERN.findall(query)))

    return QueryAnalysis(
        path              = "agent" if is_complex else "rag",
        complexity_reason = f"Regex fallback: complexity_score={complexity}",
        intent            = _map_intent(intent_type),
        query_intent_type = intent_type,
        modalities        = modalities,
        use_hyde          = word_count > 12,
        requires_temporal = is_temporal,
        clip_query        = query if is_visual else None,
        entities          = _extract_entities(query),
        filters           = Filters(**{k: v for k, v in raw_filters.items() if v}),
        boost_signals     = BoostSignals(exact_code_match=exact_codes),
        rewritten_query   = query.strip(),
        uncertainty_prior = prior,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_table_modality(analysis: QueryAnalysis, query: str) -> None:
    """
    Post-LLM safety check: if the query matches comparative/metric signals
    but the LLM did not include text_to_table in modalities, add it.

    This fixes the case where the LLM correctly classifies intent as
    'comparative' but forgets to add text_to_table to the modality list.
    Called after every LLM analysis before returning to the pipeline.
    """
    if "text_to_table" in analysis.modalities:
        return  # already set — nothing to do

    intent_needs_table = analysis.query_intent_type in (
        "comparative", "measurement"
    )
    regex_needs_table = bool(_COMPARATIVE_TABLE_SIGNALS.search(query))

    if intent_needs_table or regex_needs_table:
        # text_to_table already in default modalities — skip if present
        if "text_to_table" not in analysis.modalities:
            if "text_to_text" in analysis.modalities:
                idx = analysis.modalities.index("text_to_text")
                analysis.modalities.insert(idx + 1, "text_to_table")
            else:
                analysis.modalities.append("text_to_table")
        logger.debug(
            f"  _ensure_table_modality: added text_to_table for "
            f"intent={analysis.query_intent_type} query={query[:60]}"
        )


def _build_simple_analysis(query: str, raw_filters: Dict) -> QueryAnalysis:
    exact_codes = list(set(_CODE_PATTERN.findall(query)))
    is_visual   = bool(_IMAGE_SIGNALS.search(query))
    is_table    = bool(_TABLE_SIGNALS.search(query))
    modalities  = ["text_to_text", "text_to_image", "text_to_table"]
    return QueryAnalysis(
        path              = "rag",
        query_intent_type = "definition",
        modalities        = modalities,
        entities          = _extract_entities(query),
        filters           = Filters(**{k: v for k, v in raw_filters.items() if v}),
        boost_signals     = BoostSignals(exact_code_match=exact_codes),
        rewritten_query   = query.strip(),
        uncertainty_prior = 0.85,
    )


def _build_complex_analysis(query: str, raw_filters: Dict) -> QueryAnalysis:
    """
    Regex fallback for complex queries when LLM is unavailable.
    Always includes text_to_table for comparative queries because
    comparison evidence almost always lives in tables.
    """
    exact_codes   = list(set(_CODE_PATTERN.findall(query)))
    is_comparative = bool(_COMPARATIVE_TABLE_SIGNALS.search(query))

    # Include text_to_table if comparative signals found
    modalities = ["text_to_text", "text_to_image", "text_to_table"]

    return QueryAnalysis(
        path              = "agent",
        query_intent_type = "comparative",
        modalities        = modalities,
        entities          = _extract_entities(query),
        filters           = Filters(**{k: v for k, v in raw_filters.items() if v}),
        boost_signals     = BoostSignals(exact_code_match=exact_codes),
        rewritten_query   = query.strip(),
        uncertainty_prior = 0.5,
    )


def _apply_external_filters(
    analysis: QueryAnalysis,
    raw_filters: Dict,
    has_image: bool,
    has_table: bool,
) -> None:
    """Override LLM filters with explicitly passed filters. Mutates analysis."""
    if raw_filters.get("tier1") and not analysis.filters.tier1:
        analysis.filters.tier1 = raw_filters["tier1"]
    if raw_filters.get("tier2") and not analysis.filters.tier2:
        analysis.filters.tier2 = raw_filters["tier2"]
    if raw_filters.get("document_type") and not analysis.filters.document_type:
        analysis.filters.document_type = raw_filters["document_type"]
    if raw_filters.get("project_id") and not analysis.filters.project_id:
        analysis.filters.project_id = raw_filters["project_id"]

    # Ensure image/table modalities when caller provides input
    if has_image and "image_to_image" not in analysis.modalities:
        analysis.modalities.insert(0, "image_to_image")
    if has_table and "text_to_table" not in analysis.modalities:
        analysis.modalities.insert(0, "text_to_table")


def _build_context(
    query:           str,
    analysis:        QueryAnalysis,
    query_image_b64: Optional[str],
    query_table_text: Optional[str],
) -> QueryContext:
    return QueryContext(
        raw_query            = query,
        rewritten_query      = analysis.rewritten_query,
        query_text           = analysis.rewritten_query or query,
        query_image_b64      = query_image_b64,
        query_table_text     = query_table_text,
        requested_modalities = analysis.modalities,
        retrieval_mode       = analysis.path,
    )


def _extract_entities(query: str) -> List[str]:
    stop = {
        "what","is","are","the","a","an","in","of","for","how","show",
        "me","find","tell","does","do","this","that","which","where","when"
    }
    words = re.findall(r'\b\w+\b', query.lower())
    return list(dict.fromkeys(w for w in words if w not in stop))[:12]


def _map_intent(intent_type: str) -> str:
    mapping = {
        "procedural": "procedural",
        "measurement": "factual",
        "definition": "factual",
        "comparative": "comparative",
        "diagnostic": "factual",
        "exploratory": "exploratory",
        "visual": "factual",
        "temporal": "factual",
    }
    return mapping.get(intent_type, "factual")


def _extract_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    candidate = fenced.group(1).strip() if fenced else text
    start = candidate.find("{")
    if start == -1:
        return None
    depth = 0
    end   = -1
    for i, ch in enumerate(candidate[start:], start=start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    try:
        result = json.loads(candidate[start:end+1])
        return result if isinstance(result, dict) and result else None
    except json.JSONDecodeError:
        return None