"""
enrichment_layer/enrichers/table_enricher.py

LLM enrichment for table chunks across all formats.

Three modes (detected from chunk fields):

Mode 1 — PDF / DOCX / PPTX tables (table_html present, is_schema_chunk absent/False)
    Sends table_html + document context
    Writes: table_caption, table_summary, table_purpose + confidence scores

Mode 2 — CSV / XLSX schema chunks (is_schema_chunk = True)
    Sends only column names
    Writes: table_summary (dataset description + column semantics),
            table_purpose, column_semantics dict

Mode 3 — CSV / XLSX row group chunks (is_schema_chunk = False, sheet-based)
    Sends schema summary (from already-enriched schema chunk) + row HTML
    Writes: table_summary (what pattern this row group shows), table_purpose
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from enrichment_layer.enrichers._parse_utils import (
    extract_field,
    extract_float,
    missing_keys,
    invoke_with_retry,
)
from enrichment_layer.utils.llm_client import LLMClient

_REQUIRED = ["CAPTION", "SUMMARY"]
_SCHEMA_REQUIRED = ["DATASET_SUMMARY", "PURPOSE"]


# ── Prompts ───────────────────────────────────────────────────────────────────

TABLE_SYSTEM = """\
You are a technical analyst for an industrial document retrieval system.

<document_context>
{document_text}
</document_context>\
"""

TABLE_PROMPT = """\
Analyse this HTML table from a technical document.

Document context:
{doc_context}

Table HTML:
{table_html}

Respond in EXACTLY this format:
CAPTION: <1-sentence caption for this table>
SUMMARY: <2-3 sentences: what data the table contains, key columns/rows, units>
PURPOSE: <1-2 sentences: what decision or action this table supports>
QUESTIONS: <2-3 natural language questions that this table directly answers, separated by | >
CAPTION_CONFIDENCE: <0.0-1.0>
SUMMARY_CONFIDENCE: <0.0-1.0>
PURPOSE_CONFIDENCE: <0.0-1.0>\
"""

TABLE_RETRY = """\
Your previous response was incomplete. Missing fields: {missing}

Previous response:
{prev_response}

Respond again in EXACTLY this format:
CAPTION: <1-sentence caption>
SUMMARY: <2-3 sentences>
PURPOSE: <1-2 sentences>
QUESTIONS: <2-3 questions this table answers, separated by | >
CAPTION_CONFIDENCE: <0.0-1.0>
SUMMARY_CONFIDENCE: <0.0-1.0>
PURPOSE_CONFIDENCE: <0.0-1.0>\
"""

SCHEMA_PROMPT = """\
This is the header schema of a spreadsheet dataset.

Sheet name: {sheet_name}
Column names: {column_names}

Analyse what this dataset contains and respond in EXACTLY this format:
DATASET_SUMMARY: <3-4 sentences: domain, what each column means in plain language, \
what kind of records are stored, what time period or scope if inferable>
PURPOSE: <1-2 sentences: what business or engineering questions this dataset answers>
COLUMN_SEMANTICS: <JSON dict: {{"col_name": "plain english description", ...}}>
CONFIDENCE: <0.0-1.0>\
"""

SCHEMA_RETRY = """\
Your previous response was incomplete. Missing fields: {missing}

Previous response:
{prev_response}

Respond in EXACTLY this format:
DATASET_SUMMARY: <3-4 sentences>
PURPOSE: <1-2 sentences>
COLUMN_SEMANTICS: <JSON dict>
CONFIDENCE: <0.0-1.0>\
"""

ROWGROUP_PROMPT = """\
This is a batch of rows from a spreadsheet dataset.

Dataset description:
{schema_summary}

Row data (HTML):
{row_html}

Respond in EXACTLY this format:
CAPTION: <1-sentence: what rows are in this batch — row range, key filters>
SUMMARY: <2-3 sentences: what pattern, range, or cluster of data this batch represents>
PURPOSE: <1 sentence: what specific data does this batch contain?>
SUMMARY_CONFIDENCE: <0.0-1.0>\
"""

ROWGROUP_RETRY = """\
Your previous response was incomplete. Missing fields: {missing}

Previous response:
{prev_response}

Respond in EXACTLY this format:
CAPTION: <1-sentence>
SUMMARY: <2-3 sentences>
PURPOSE: <1 sentence>
SUMMARY_CONFIDENCE: <0.0-1.0>\
"""


class TableEnricher:

    def __init__(self, llm: LLMClient, delay: float = 0.5):
        self.llm   = llm
        self.delay = delay

    def enrich(
        self,
        table_chunks:  List[Dict[str, Any]],
        document_text: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Enrich all table chunks.
        Detects mode per chunk (PDF table / schema / row group).
        Idempotent: skips chunks that already have table_summary.
        """
        to_enrich = [c for c in table_chunks if not c.get("table_summary")]
        if not to_enrich:
            logger.info("  TableEnricher: all table chunks already enriched")
            return table_chunks

        logger.info(
            f"  TableEnricher: enriching {len(to_enrich)}/{len(table_chunks)} tables"
        )

        system     = TABLE_SYSTEM.format(document_text=document_text[:6000])
        doc_ctx    = document_text[:500]

        # Build schema_summary lookup per sheet (for row group mode)
        schema_summary_by_su: Dict[str, str] = {}
        for c in table_chunks:
            if c.get("is_schema_chunk") and c.get("table_summary"):
                su_id = c.get("structure_unit_id", "")
                schema_summary_by_su[su_id] = c["table_summary"]

        for chunk in to_enrich:
            chunk_id = chunk.get("chunk_id", "?")
            try:
                if chunk.get("is_schema_chunk"):
                    self._enrich_schema(chunk, system)
                    # After enriching, register for row group lookup
                    su_id = chunk.get("structure_unit_id", "")
                    if chunk.get("table_summary") and su_id:
                        schema_summary_by_su[su_id] = chunk["table_summary"]
                elif chunk.get("sheet_name") or chunk.get("sheet_index") is not None:
                    # Row group from spreadsheet
                    su_id  = chunk.get("structure_unit_id", "")
                    schema = schema_summary_by_su.get(su_id, "")
                    self._enrich_rowgroup(chunk, schema, system)
                else:
                    # PDF / DOCX / PPTX table
                    self._enrich_table(chunk, doc_ctx, system)
            except Exception as e:
                logger.error(f"  TableEnricher: {chunk_id} failed: {e}")
            time.sleep(self.delay)

        enriched = sum(1 for c in table_chunks if c.get("table_summary"))
        logger.success(f"  TableEnricher: {enriched}/{len(table_chunks)} enriched")
        return table_chunks

    # ── Mode 1: PDF / DOCX / PPTX table ──────────────────────────────

    def _enrich_table(
        self, chunk: Dict, doc_ctx: str, system: str
    ) -> None:
        table_html = chunk.get("table_html", "")
        if not table_html:
            logger.warning(f"  TableEnricher: {chunk.get('chunk_id')} — no table_html, skipping")
            return

        response = invoke_with_retry(
            llm_client            = self.llm,
            first_prompt          = TABLE_PROMPT.format(
                doc_context=doc_ctx,
                table_html=table_html[:3000],
            ),
            retry_prompt_template = TABLE_RETRY,
            required_keys         = _REQUIRED,
            use_cache             = True,
            system_doc            = system,
        )
        if not response:
            return

        chunk["table_caption"]            = extract_field(response, "CAPTION")
        chunk["table_summary"]            = extract_field(response, "SUMMARY")
        chunk["table_purpose"]            = extract_field(response, "PURPOSE")
        chunk["table_questions"]          = extract_field(response, "QUESTIONS")
        chunk["table_caption_confidence"] = extract_float(response, "CAPTION_CONFIDENCE", 0.5)
        chunk["table_summary_confidence"] = extract_float(response, "SUMMARY_CONFIDENCE", 0.5)
        chunk["table_purpose_confidence"] = extract_float(response, "PURPOSE_CONFIDENCE", 0.5)

    # ── Mode 2: Schema chunk (CSV/XLSX header) ─────────────────────────

    def _enrich_schema(self, chunk: Dict, system: str) -> None:
        import json, re

        col_names  = chunk.get("column_names", [])
        sheet_name = chunk.get("sheet_name", "")

        response = invoke_with_retry(
            llm_client            = self.llm,
            first_prompt          = SCHEMA_PROMPT.format(
                sheet_name=sheet_name,
                column_names=", ".join(col_names),
            ),
            retry_prompt_template = SCHEMA_RETRY,
            required_keys         = _SCHEMA_REQUIRED,
            use_cache             = True,
            system_doc            = system,
        )
        if not response:
            return

        chunk["table_summary"]            = extract_field(response, "DATASET_SUMMARY")
        chunk["table_purpose"]            = extract_field(response, "PURPOSE")
        chunk["table_summary_confidence"] = extract_float(response, "CONFIDENCE", 0.5)
        chunk["table_caption"]            = f"Schema for sheet: {sheet_name}"

        # Parse COLUMN_SEMANTICS JSON
        m = re.search(r"COLUMN_SEMANTICS:\s*(\{[\s\S]+?\})", response)
        if m:
            try:
                chunk["column_semantics"] = json.loads(m.group(1))
            except json.JSONDecodeError:
                chunk["column_semantics"] = {}

    # ── Mode 3: Row group chunk (CSV/XLSX data rows) ───────────────────

    def _enrich_rowgroup(
        self, chunk: Dict, schema_summary: str, system: str
    ) -> None:
        table_html = chunk.get("table_html", "")
        if not table_html and not schema_summary:
            return

        response = invoke_with_retry(
            llm_client            = self.llm,
            first_prompt          = ROWGROUP_PROMPT.format(
                schema_summary=schema_summary[:600],
                row_html=(table_html or "")[:2000],
            ),
            retry_prompt_template = ROWGROUP_RETRY,
            required_keys         = ["CAPTION", "SUMMARY"],
            use_cache             = True,
            system_doc            = system,
        )
        if not response:
            return

        chunk["table_caption"]            = extract_field(response, "CAPTION")
        chunk["table_summary"]            = extract_field(response, "SUMMARY")
        chunk["table_purpose"]            = extract_field(response, "PURPOSE")
        chunk["table_summary_confidence"] = extract_float(response, "SUMMARY_CONFIDENCE", 0.5)
