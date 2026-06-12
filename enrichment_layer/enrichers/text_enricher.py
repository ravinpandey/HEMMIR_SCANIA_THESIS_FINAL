"""
enrichment_layer/enrichers/text_enricher.py

Anthropic-style contextual chunking for text chunks (PDF / DOCX / PPTX).

Core algorithm per chunk:
  1. Sort all chunks by chunk_index (document order)
  2. Build local_window: text of ±2 neighbours (400 chars each)
  3. Build user prompt including:
       - section heading + semantic_anchor (from doc_meta_enricher)
       - heading_breadcrumb (DOCX only)
       - speaker_notes (PPTX only)
       - window text
       - chunk text
  4. System prompt = full document text (cached via invoke_with_cache)
  5. LLM returns situated_context prefix + metadata fields
  6. PREPEND situated_context to text_original_content
  7. Write contextual_summary, local_context, detected_codes,
     evidence_role, salience_score, contextual_summary_confidence

Writes to chunk dict (plain dict — Pydantic validation at disk boundary):
  chunk["text_original_content"]          ← situated_context + \\n\\n + original
  chunk["contextual_summary"]             ← 2-3 sentence LLM prefix
  chunk["local_context"]                  ← ±2 window text
  chunk["detected_codes"]                 ← list of part numbers / fault codes
  chunk["evidence_role"]                  ← definition/procedure/measurement/reference/context
  chunk["salience_score"]                 ← 0.0-1.0
  chunk["contextual_summary_confidence"]  ← 0.0-1.0
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from enrichment_layer.enrichers._parse_utils import (
    extract_field,
    extract_float,
    extract_list,
    invoke_with_retry,
)
from enrichment_layer.utils.llm_client import LLMClient


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """\
You are a technical document analyst for a multimodal industrial retrieval system.
Your task is to situate document chunks so they are self-sufficient for search.

<full_document>
{document_text}
</full_document>\
"""

CHUNK_PROMPT = """\
Situate the chunk below within the document for search retrieval.

Section: {section_title}
Section anchor: {semantic_anchor}
{breadcrumb_line}\
{notes_line}\
Local context (surrounding chunks):
{window_text}

<chunk>
{chunk_text}
</chunk>

Write a SHORT, DENSE SITUATED_CONTEXT (1-2 sentences maximum) that:
  - States the document and section this chunk comes from
  - Captures the SPECIFIC technical content: key terms, values,
    mechanisms, formulas, procedures, or findings in this chunk

CRITICAL RULES:
  - Be concise and information-dense — every word must carry retrieval signal
  - DO NOT write "A reader seeking..." or "A user would find..." phrases
  - DO NOT use vague phrases like "this chunk discusses" or "this section covers"
  - DO use specific technical terms, numbers, model names, algorithm names
  - Make the summary self-sufficient: someone reading only the summary should
    understand what technical content is in this chunk

GOOD: "From 'Attention Is All You Need' sec 3.2.2: multi-head attention projects
       queries/keys/values h=8 times to d_k=d_v=64 dimensions in parallel."
BAD:  "This chunk appears in Section 3.2.2. A reader seeking multi-head attention
       would find this chunk."

Then extract technical codes, part numbers, fault codes, values, or identifiers.

Respond in EXACTLY this format (all fields required):
SITUATED_CONTEXT: <1-2 dense sentences>
CONFIDENCE: <0.0-1.0>
CODES: <comma-separated codes/values, or "none">
CODES_CONFIDENCE: <0.0-1.0 — how certain are the extracted codes/values>
EVIDENCE_ROLE: <one of: definition / procedure / measurement / reference / context>
EVIDENCE_ROLE_CONFIDENCE: <0.0-1.0 — how certain are you about the role classification>
SALIENCE: <0.0-1.0 — how central is this chunk to the document's main purpose>\
"""

RETRY_PROMPT = """\
Your previous response was incomplete. Missing fields: {missing}

Previous response:
{prev_response}

Respond again in EXACTLY this format:
SITUATED_CONTEXT: <2-3 sentences>
CONFIDENCE: <0.0-1.0>
CODES: <comma-separated codes, or "none">
CODES_CONFIDENCE: <0.0-1.0>
EVIDENCE_ROLE: <one of: definition / procedure / measurement / reference / context>
EVIDENCE_ROLE_CONFIDENCE: <0.0-1.0>
SALIENCE: <0.0-1.0>\
"""

_REQUIRED = ["SITUATED_CONTEXT", "CONFIDENCE"]
_WINDOW_CHARS = 400   # chars per neighbour in the ±2 window
_CHUNK_CHARS  = 2000  # max chars from chunk text sent to LLM
_DOC_CHARS    = 12000 # max chars of full doc sent in system prompt




def _extract_named_entities(text: str, max_terms: int = 12) -> list:
    """
    Extract named entities from text — capitalised technical terms,
    model names, dataset names, algorithm names, standard identifiers.

    Filters out sentence starters and common words to focus on
    genuinely named technical entities that aid retrieval.

    Examples of what gets extracted:
      "Transformer", "BERT", "MNIST", "ImageNet", "BLEU", "ReLU",
      "ResNet", "Adam", "LSTM", "GPT", "ANOVA", "IEEE", "RFC"
    """
    import re
    # Match capitalised tokens: start with uppercase, contain letters
    # Exclude pure stopwords and sentence starters
    _SKIP = {
        "The", "This", "These", "That", "Those", "In", "On", "At",
        "For", "To", "Of", "And", "Or", "But", "A", "An", "It",
        "As", "By", "We", "Our", "Their", "Is", "Are", "Was", "Were",
        "Has", "Have", "Had", "Be", "Been", "Being", "Not", "No",
        "Also", "With", "From", "About", "When", "Where", "Which",
        "Who", "How", "What", "Why", "Can", "Will", "May", "Should",
    }
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9\-_]{2,}\b", text)
    seen = []
    for t in tokens:
        if t in _SKIP:
            continue
        if t not in seen:
            seen.append(t)
        if len(seen) >= max_terms:
            break
    return seen

class TextEnricher:

    def __init__(self, llm: LLMClient, delay: float = 0.2):
        self.llm   = llm
        self.delay = delay

    def enrich(
        self,
        text_chunks:     List[Dict[str, Any]],
        structure_units: List[Dict[str, Any]],
        document_text:   str,
    ) -> List[Dict[str, Any]]:
        """
        Enrich all text chunks with contextual summaries.
        Idempotent: skips chunks that already have contextual_summary set.
        Returns the mutated list.
        """
        # Sort by chunk_index for window building
        chunks = sorted(text_chunks, key=lambda c: c.get("chunk_index", 0))

        # Build structure unit lookup for section metadata
        su_lookup: Dict[str, Dict] = {
            su["structure_unit_id"]: su
            for su in structure_units
            if su.get("structure_unit_id")
        }

        to_enrich = [c for c in chunks if not c.get("contextual_summary") or c.get("evidence_role_confidence") is None]
        if not to_enrich:
            logger.info("  TextEnricher: all chunks already enriched")
            return text_chunks

        logger.info(f"  TextEnricher: enriching {len(to_enrich)}/{len(chunks)} chunks")
        system = SYSTEM_TEMPLATE.format(document_text=document_text[:_DOC_CHARS])

        # Build chunk index → position map for window access
        ordered_ids = [c["chunk_id"] for c in chunks]

        for i, chunk in enumerate(chunks):
            if chunk.get("contextual_summary") and chunk.get("evidence_role_confidence") is not None:
                continue

            chunk_id  = chunk.get("chunk_id", f"idx_{i}")
            su_id     = chunk.get("structure_unit_id", "")
            su        = su_lookup.get(su_id, {})

            # ── Build local window ─────────────────────────────────────
            window_parts = []
            for delta in (-2, -1, 1, 2):
                j = i + delta
                if 0 <= j < len(chunks):
                    neighbour_text = (
                        chunks[j].get("text_original_content") or ""
                    )[:_WINDOW_CHARS]
                    label = "before" if delta < 0 else "after"
                    window_parts.append(f"[{label}] {neighbour_text}")
            window_text = "\n".join(window_parts) or "(no surrounding chunks)"

            # Patch local_context on chunk (prev only for PDF; add next here)
            chunk["local_context"] = window_text

            # ── Build prompt ───────────────────────────────────────────
            section_title   = (
                chunk.get("section_title")
                or chunk.get("slide_title")
                or su.get("title", "")
                or ""
            )
            semantic_anchor = su.get("semantic_anchor") or su.get("title", "")
            breadcrumb      = chunk.get("heading_breadcrumb", "")
            speaker_notes   = (chunk.get("speaker_notes") or "")[:300]
            chunk_text      = (chunk.get("text_original_content") or "")[:_CHUNK_CHARS]

            breadcrumb_line = (
                f"Breadcrumb: {breadcrumb}\n" if breadcrumb else ""
            )
            notes_line = (
                f"Speaker notes: {speaker_notes}\n" if speaker_notes else ""
            )

            user_prompt = CHUNK_PROMPT.format(
                section_title   = section_title,
                semantic_anchor = semantic_anchor,
                breadcrumb_line = breadcrumb_line,
                notes_line      = notes_line,
                window_text     = window_text,
                chunk_text      = chunk_text,
            )

            # ── LLM call with retry ────────────────────────────────────
            response = invoke_with_retry(
                llm_client             = self.llm,
                first_prompt           = user_prompt,
                retry_prompt_template  = RETRY_PROMPT,
                required_keys          = _REQUIRED,
                use_cache              = True,
                system_doc             = system,
            )

            if response is None:
                logger.warning(f"  TextEnricher: {chunk_id} failed after retries")
                continue

            situated_context = extract_field(response, "SITUATED_CONTEXT") or ""
            confidence       = extract_float(response, "CONFIDENCE", default=0.5)
            codes               = extract_list(response, "CODES")
            codes_conf          = extract_float(response, "CODES_CONFIDENCE",          default=0.5)
            evidence_role       = extract_field(response, "EVIDENCE_ROLE") or "context"
            evidence_role_conf  = extract_float(response, "EVIDENCE_ROLE_CONFIDENCE",  default=0.5)
            salience            = extract_float(response, "SALIENCE", default=0.5)

            if situated_context:
                original_text = chunk.get("text_original_content") or ""
                # Prepend situated context only if not already prepended
                if situated_context not in original_text:
                    chunk["text_original_content"] = (
                        situated_context + "\n\n" + original_text
                    )
                chunk["contextual_summary"]                = situated_context
                chunk["contextual_summary_confidence"]     = confidence
                chunk["detected_codes"]                    = codes
                chunk["detected_codes_confidence"]         = codes_conf
                chunk["evidence_role"]                     = evidence_role.lower().strip()
                chunk["evidence_role_confidence"]          = evidence_role_conf
                chunk["salience_score"]                    = salience

                # Fix 2: Extract named entities from the situated context
                # + original text. The encoding layer uses chunk["entities"]
                # for entity_text — previously always [] because text_enricher
                # never populated it. Now we extract capitalised named terms
                # (algorithms, model names, dataset names, standards) which
                # the encoding layer cannot reliably infer from raw text alone.
                # Combined text: situated context names the section/document
                # so entity extraction finds document-level named entities too.
                combined_for_entities = (situated_context + " " + original_text)[:2000]
                extracted_entities = _extract_named_entities(combined_for_entities)
                if extracted_entities:
                    chunk["entities"] = extracted_entities

            logger.debug(
                f"  TextEnricher: {chunk_id} | role={evidence_role} | "
                f"salience={salience} | codes={len(codes)}"
            )
            time.sleep(self.delay)

        logger.success(
            f"  TextEnricher: enriched "
            f"{sum(1 for c in chunks if c.get('contextual_summary'))}/{len(chunks)}"
        )
        return chunks