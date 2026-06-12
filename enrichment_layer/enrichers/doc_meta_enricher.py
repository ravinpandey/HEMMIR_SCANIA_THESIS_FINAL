"""
enrichment_layer/enrichers/doc_meta_enricher.py

Enriches doc_metadata.json and structure_units.json for every format.

Runs FIRST — before any chunk enricher — because:
  - section summaries are needed by text_enricher to build contextual prompts
  - semantic_anchor upgrade on structure units is read by encoding layer

One LLM call per document. Full document text cached in system prompt.

Outputs written:
  doc_metadata["doc_summary"]
  doc_metadata["tier1"], ["tier2"], ["tier_confidence"]
  doc_metadata["section_map"][heading]["summary"]
  doc_metadata["section_map"][heading]["keywords"]
  doc_metadata["section_map"][heading]["entities"]
  doc_metadata["section_map"][heading]["subsections"]
  structure_units[i]["semantic_anchor"]    ← upgraded from heading text
  structure_units[i]["section_summary"]
  structure_units[i]["keywords"]
  structure_units[i]["entities"]
  structure_units[i]["subsections"]

Format-specific behaviour:
  PDF / DOCX  → section heading + chunk text samples → retrieval-focused summaries
  PPTX        → slide titles + body + speaker_notes  → deck narrative summary
  CSV / XLSX  → sheet names + column names           → dataset purpose description
  Video       → full transcript text                 → video topic and key points
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from enrichment_layer.enrichers._parse_utils import extract_field, extract_float, extract_list
from enrichment_layer.utils.llm_client import LLMClient

VIDEO_FORMATS = {"mp4", "avi", "mkv", "mov", "webm", "m4v", "video"}
SHEET_FORMATS = {"csv", "xlsx", "xls", "ods", "tsv"}


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """\
You are a technical document analyst.
You provide precise, factual analysis grounded strictly in document content.

Confidence scale:
1.0 = Directly and explicitly stated
0.8 = Strongly implied or clearly inferable
0.6 = Reasonable inference with supporting evidence
0.4 = Limited context, educated guess

<full_document_text>
{document_text}
</full_document_text>\
"""

DOC_PROMPT = """\
Analyse the full document above and return ONLY valid JSON — no markdown, no preamble.

Use this EXACT schema:
{{
  "doc_summary": "<4-6 sentence summary: purpose, major topics, key ideas, contribution>",
  "tier1": "<one of: Software / Hardware / Operations / Research / Business / Legal / Healthcare / General>",
  "tier2": "<max 4 words, specific topic, e.g. Vehicle Maintenance>",
  "tier_confidence": <0.0-1.0>,
  "section_summaries": {{
    "<exact_section_heading>": {{
      "summary":     "<2-3 sentence RETRIEVAL-FOCUSED summary — what specific questions does this section answer?>",
      "semantic_anchor": "<1 sentence that upgrades the heading into a meaningful retrieval anchor>",
      "semantic_anchor_confidence": <0.0-1.0>,
      "summary_confidence": <0.0-1.0 — how well does the summary capture the section>,
      "keywords":    ["<key technical term>", ...],
      "keywords_confidence": <0.0-1.0 — how relevant and complete are the keywords>,
      "entities":    ["<named component/code/standard>", ...],
      "subsections": ["<child heading if numbered hierarchy detected>", ...]
    }}
  }}
}}

Rules:
- section_summaries keys MUST be the EXACT headings from the input — do not rename them
- semantic_anchor must name specific concepts, NOT just repeat the heading
- keywords: technical terms a user would search for (max 12 per section)
- entities: named things — part numbers, standards, model codes, error codes (max 12)
- subsections: child headings only if 3.1/3.2 numbering is visible; otherwise []

BAD  summary: "Describes the methodology used in this study."
GOOD summary: "Details the ANOVA and regression methods, sample size N=240, \
inclusion criteria, and data collection procedure used across three test sites."

section_contents (heading → first 1200 chars of section text):
{section_contents}

Document metadata:
title: {title}
format: {file_format}
pages: {total_pages}\
"""

DOC_PROMPT = """\
Analyse the full document above and return ONLY valid JSON — no markdown, no preamble.

Use this EXACT schema:
{{
  "doc_summary": "<4-6 sentence summary: purpose, major topics, key ideas, contribution>",
  "doc_summary_confidence": <0.0-1.0>,
  "document_type": "<one of: Research Paper / Technical Report / User Manual / Policy Document / Standard Operating Procedure / Tutorial / Presentation / Other>",
  "tier1": "<one of: Software / Hardware / Operations / Research / Business / Legal / Healthcare / General>",
  "tier2": "<max 4 words, specific topic, e.g. Vehicle Maintenance>",
  "tier_confidence": <0.0-1.0>,
  "section_summaries": {{
    "<exact_section_heading>": {{
      "summary":     "<2-3 sentence RETRIEVAL-FOCUSED summary — what specific questions does this section answer?>",
      "semantic_anchor": "<1 sentence that upgrades the heading into a meaningful retrieval anchor>",
      "semantic_anchor_confidence": <0.0-1.0>,
      "summary_confidence": <0.0-1.0 — how well does the summary capture the section>,
      "keywords":    ["<key technical term>", ...],
      "keywords_confidence": <0.0-1.0 — how relevant and complete are the keywords>,
      "entities":    ["<named component/code/standard>", ...],
      "subsections": ["<child heading if numbered hierarchy detected>", ...]
    }}
  }}
}}

Rules:
- ALL 6 top-level fields are REQUIRED — do not omit any
- doc_summary_confidence: how confident are you in the summary (1.0 = very confident)
- document_type: classify based on content and structure
- section_summaries keys MUST be the EXACT headings from the input\
"""


SECTION_BATCH_PROMPT = """Analyse these document sections and return ONLY valid JSON — no markdown, no preamble.

For EACH section heading provided, return a summary entry.
Use this EXACT schema:
{{
  "section_summaries": {{
    "<exact_section_heading>": {{
      "summary":         "<2-3 sentence RETRIEVAL-FOCUSED summary>",
      "semantic_anchor": "<1 sentence retrieval anchor>",
      "keywords":        ["<key technical term>", ...],
      "entities":        ["<named component/algorithm/dataset>", ...],
      "subsections":     []
    }}
  }}
}}

Rules:
- Return ALL {n_sections} sections — do not skip any
- Keys MUST be EXACT headings from input
- keywords: max 8 per section
- entities: max 8 per section

Section contents:
{section_contents}
"""

DECK_PROMPT = """Analyse this presentation deck and return ONLY valid JSON — no markdown, no preamble.

Use this EXACT schema:
{{
  "doc_summary": "<4-6 sentences: deck purpose, key message, audience, main topics covered>",
  "doc_summary_confidence": <0.0-1.0>,
  "document_type": "Presentation",
  "tier1": "<one of: Software / Hardware / Operations / Research / Business / Legal / Healthcare / General>",
  "tier2": "<max 4 words, specific topic, e.g. Product Launch Strategy>",
  "tier_confidence": <0.0-1.0>,
  "section_summaries": {{
    "<exact_slide_title_or_heading>": {{
      "summary":     "<2-3 sentences: what does this slide communicate, what data or argument does it present?>",
      "semantic_anchor": "<1 sentence: the core claim or topic of this slide>",
      "semantic_anchor_confidence": <0.0-1.0>,
      "summary_confidence": <0.0-1.0>,
      "keywords":    ["<key term visible on slide>", ...],
      "keywords_confidence": <0.0-1.0>,
      "entities":    ["<named product, company, metric, or component>", ...],
      "subsections": []
    }}
  }}
}}

Rules:
- ALL 6 top-level fields are REQUIRED
- section_summaries keys MUST be the EXACT slide titles from the input
- semantic_anchor must capture the slide's core argument or finding
- keywords: terms visible on the slide (max 8 per slide)
- entities: named things — product names, companies, KPIs, standards (max 8)

Deck slide contents:
{section_contents}

Deck title: {title}"""

DATASET_PROMPT = """\
This is a spreadsheet dataset. Analyse the schema and return ONLY valid JSON.

{{
  "doc_summary": "<4-6 sentences: what dataset, domain, what each sheet covers, \
what questions it answers, data volume>",
  "tier1": "<one of: Software / Hardware / Operations / Research / Business / Legal / Healthcare / General>",
  "tier2": "<max 4 words, e.g. Fault Code Log>",
  "tier_confidence": <0.0-1.0>,
  "section_summaries": {{
    "<sheet_name>": {{
      "summary":      "<what data does this sheet contain? what queries does it answer?>",
      "semantic_anchor": "<1 sentence: purpose of this sheet>",
      "keywords":    ["<domain term>", ...],
      "entities":    ["<column group or key identifier>", ...],
      "subsections": []
    }}
  }}
}}

Sheets and column names:
{section_contents}

Dataset title: {title}\
"""

VIDEO_PROMPT = """\
This is a video transcript. Analyse the content and return ONLY valid JSON.

{{
  "doc_summary": "<4-6 sentences: video topic, key points covered, speaker focus areas, \
overall structure>",
  "tier1": "<one of: Software / Hardware / Operations / Research / Business / Legal / Healthcare / General>",
  "tier2": "<max 4 words, e.g. Maintenance Training>",
  "tier_confidence": <0.0-1.0>,
  "section_summaries": {{
    "<scene_title>": {{
      "summary":      "<what is discussed in this scene segment?>",
      "semantic_anchor": "<1 sentence capturing scene topic>",
      "keywords":    ["<term>", ...],
      "entities":    ["<named item>", ...],
      "subsections": []
    }}
  }}
}}

Scenes (title → transcript excerpt):
{section_contents}

Video title: {title}\
"""


class DocMetaEnricher:

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def enrich(
        self,
        doc_metadata:     Dict[str, Any],
        structure_units:  List[Dict[str, Any]],
        text_chunks:      List[Dict[str, Any]],
        doc_output_dir:   Path,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Enrich doc_metadata and structure_units in-place.
        Returns updated (doc_metadata, structure_units).
        """
        if doc_metadata.get("doc_summary"):
            logger.info("  doc_meta already enriched — skipping")
            return doc_metadata, structure_units

        fmt    = (doc_metadata.get("file_format") or "").lower()
        title  = doc_metadata.get("doc_title", "")
        doc_id = doc_metadata.get("doc_id", "")
        logger.info(f"  DocMetaEnricher: {doc_id} [{fmt}]")

        # ── Build document text for system prompt ──────────────────────
        if fmt in VIDEO_FORMATS:
            doc_text = self._load_video_transcript(doc_output_dir, doc_metadata)
        else:
            doc_text = self._build_doc_text(text_chunks)

        # ── Build section_contents for user prompt ─────────────────────
        if fmt in VIDEO_FORMATS:
            section_contents = self._build_video_section_contents(structure_units, text_chunks)
            user_prompt = VIDEO_PROMPT.format(
                section_contents=section_contents[:6000],
                title=title,
            )
        elif fmt in SHEET_FORMATS:
            section_contents = self._build_sheet_section_contents(structure_units, text_chunks)
            user_prompt = DATASET_PROMPT.format(
                section_contents=section_contents[:4000],
                title=title,
            )
        elif fmt == "pptx":
            section_contents = self._build_pptx_section_contents(structure_units, text_chunks)
            user_prompt = DECK_PROMPT.format(
                section_contents=section_contents[:6000],
                title=title,
            )
        else:
            section_contents = self._build_doc_section_contents(
                doc_metadata, structure_units, text_chunks
            )
            user_prompt = DOC_PROMPT.format(
                section_contents=section_contents[:6000],
                title=title,
                file_format=fmt,
                total_pages=doc_metadata.get("total_pages", 0),
            )

        system = SYSTEM_TEMPLATE.format(document_text=doc_text[:8000])

        # ── LLM call with retry ────────────────────────────────────────
        enriched = self._call_and_parse(system, user_prompt)
        if not enriched:
            logger.warning(f"  DocMetaEnricher: parse failed for {doc_id}")
            return doc_metadata, structure_units

        # ── Write doc-level fields ─────────────────────────────────────
        '''doc_metadata["doc_summary"]    = enriched.get("doc_summary") or doc_metadata.get("doc_summary")
        doc_metadata["doc_summary_confidence"]  = enriched.get("doc_summary_confidence", 0.0)  # ← ADD THIS
        doc_metadata["tier1"]          = enriched.get("tier1") or doc_metadata.get("tier1")
        doc_metadata["tier2"]          = enriched.get("tier2") or doc_metadata.get("tier2")
        doc_metadata["tier_confidence"] = enriched.get("tier_confidence", 0.0)'''
        doc_metadata["doc_summary"]            = enriched.get("doc_summary") or doc_metadata.get("doc_summary")
        doc_metadata["doc_summary_confidence"] = float(enriched.get("doc_summary_confidence") or 0.85)
        doc_metadata["document_type"]          = enriched.get("document_type") or doc_metadata.get("document_type") or ""
        doc_metadata["tier1"]                  = enriched.get("tier1") or doc_metadata.get("tier1")
        doc_metadata["tier2"]                  = enriched.get("tier2") or doc_metadata.get("tier2")
        doc_metadata["tier_confidence"]        = float(enriched.get("tier_confidence") or 0.85)

        # ── Write section-level fields ─────────────────────────────────
        section_summaries = enriched.get("section_summaries", {}) or {}
        section_map       = doc_metadata.get("section_map", {})
        unit_lookup       = {su["title"]: su for su in structure_units if su.get("title")}

        matched = 0
        for llm_heading, data in section_summaries.items():
            if not isinstance(data, dict):
                continue

            # Resolve LLM heading to actual key (exact → case-insensitive → fuzzy)
            resolved = self._match_heading(llm_heading, section_map, unit_lookup)
            if not resolved:
                continue

            # Update section_map entry
            if resolved in section_map and isinstance(section_map[resolved], dict):
                section_map[resolved]["summary"]     = data.get("summary", "")
                section_map[resolved]["keywords"]    = data.get("keywords", [])
                section_map[resolved]["entities"]    = data.get("entities", [])
                section_map[resolved]["subsections"] = data.get("subsections", [])
                matched += 1

            # Update matching structure unit
            if resolved in unit_lookup:
                su = unit_lookup[resolved]
                su["semantic_anchor"]            = (
                    data.get("semantic_anchor")
                    or su.get("semantic_anchor", resolved)
                )
                su["semantic_anchor_confidence"] = float(data.get("semantic_anchor_confidence") or 0.5)
                su["section_summary"]            = data.get("summary", "")
                su["section_summary_confidence"] = float(data.get("summary_confidence") or 0.5)
                su["keywords"]                   = data.get("keywords", [])
                su["keywords_confidence"]        = float(data.get("keywords_confidence") or 0.5)
                su["entities"]                   = data.get("entities", [])
                su["subsections"]                = data.get("subsections", [])

        doc_metadata["section_map"] = section_map

        logger.success(
            f"  DocMetaEnricher: tier={doc_metadata.get('tier1')}/{doc_metadata.get('tier2')} "
            f"| sections_enriched={matched}"
        )
        # Batch enrich remaining sections without summaries
        remaining = [su for su in structure_units if su.get('title') and not su.get('section_summary')]
        if remaining:
            section_map = doc_metadata.get('section_map', {})
            batch_matched = self._enrich_sections_batch(remaining, text_chunks, section_map, system)
            doc_metadata['section_map'] = section_map
            logger.info(f'  DocMetaEnricher: batch enriched {batch_matched} additional sections')
        return doc_metadata, structure_units

    # ── LLM call ──────────────────────────────────────────────────────


    def _enrich_sections_batch(self, structure_units, text_chunks, section_map, system, batch_size=5):
        """Enrich section summaries in batches — fixes 6000-char truncation bug."""
        chunk_by_id  = {c["chunk_id"]: c for c in text_chunks if c.get("chunk_id")}
        unit_lookup  = {su["title"]: su for su in structure_units if su.get("title")}
        total_matched = 0
        batches = [structure_units[i:i+batch_size] for i in range(0, len(structure_units), batch_size)]
        for batch_idx, batch in enumerate(batches):
            lines = []
            for su in batch:
                title   = su.get("title", "")
                cids    = su.get("chunk_ids", [])
                content = " ".join(
                    (chunk_by_id.get(cid, {}).get("text_original_content") or "")
                    for cid in cids
                ).strip()[:800]
                lines.append(f"### {title}\n{content}")
            section_contents = "\n\n".join(lines)
            prompt = SECTION_BATCH_PROMPT.format(
                section_contents=section_contents,
                n_sections=len(batch),
            )
            try:
                enriched = self._call_and_parse(system, prompt)
                if not enriched:
                    continue
                section_summaries = enriched.get("section_summaries", {}) or {}
                for llm_heading, data in section_summaries.items():
                    if not isinstance(data, dict):
                        continue
                    resolved = self._match_heading(llm_heading, section_map, unit_lookup)
                    if not resolved:
                        continue
                    if resolved in section_map and isinstance(section_map[resolved], dict):
                        section_map[resolved]["summary"]  = data.get("summary", "")
                        section_map[resolved]["keywords"] = data.get("keywords", [])
                        section_map[resolved]["entities"] = data.get("entities", [])
                    if resolved in unit_lookup:
                        su = unit_lookup[resolved]
                        su["section_summary"]            = data.get("summary", "")
                        su["section_summary_confidence"] = float(data.get("summary_confidence") or 0.5)
                        su["semantic_anchor"]            = data.get("semantic_anchor", resolved)
                        su["semantic_anchor_confidence"] = float(data.get("semantic_anchor_confidence") or 0.5)
                        su["keywords"]                   = data.get("keywords", [])
                        su["keywords_confidence"]        = float(data.get("keywords_confidence") or 0.5)
                        su["entities"]                   = data.get("entities", [])
                        total_matched += 1
                logger.info(f"  SectionBatch {batch_idx+1}/{len(batches)}: {len(section_summaries)}/{len(batch)} enriched")
            except Exception as e:
                logger.warning(f"  SectionBatch {batch_idx+1} error: {e}")
        return total_matched

    def _call_and_parse(self, system: str, prompt: str) -> Optional[Dict]:
        """
        Call LLM and parse JSON response robustly.

        Strategy:
          Attempt 1 — parse response directly.
          Attempt 2 — if JSON is malformed, send a repair prompt that
                      includes the broken response and asks the model to
                      fix only the JSON syntax.
          Gives up after 2 total attempts.
        """
        response = None
        for attempt in range(2):
            try:
                if attempt == 0:
                    response = self.llm.invoke(
                        system=system, prompt=prompt, max_tokens=4000
                    )
                else:
                    # Repair prompt — send back the broken JSON and ask for a fix
                    repair_prompt = (
                        "The JSON you returned was malformed. "
                        "Return ONLY the corrected JSON with no other text, "
                        "no markdown fences, no explanation.\n\n"
                        f"Broken response:\n{response}"
                    )
                    response = self.llm.invoke(
                        system=system, prompt=repair_prompt, max_tokens=4000
                    )

                logger.debug(
                    f"  DocMetaEnricher attempt {attempt+1} "
                    f"response[:200]: {response[:200]!r}"
                )

                parsed = self._extract_json(response)
                if parsed:
                        print("LLM returned keys:", list(parsed.keys()))
                        print("doc_summary_confidence:", parsed.get("doc_summary_confidence"))
                        print("document_type:", parsed.get("document_type"))
                        return parsed

                # Parsed returned None — JSON not found or empty object
                logger.warning(
                    f"  DocMetaEnricher attempt {attempt+1}: "
                    "no valid JSON found in response"
                )

            except Exception as e:
                logger.warning(f"  DocMetaEnricher attempt {attempt+1} error: {e}")

        logger.error("  DocMetaEnricher: failed to parse LLM response after 2 attempts")
        return None

    def _extract_json(self, text: str) -> Optional[Dict]:
        """
        Extract the first complete JSON object from LLM response text.

        Handles:
          - Responses wrapped in ```json ... ``` fences
          - Leading prose before the JSON object
          - Trailing text after the closing brace
          - json.JSONDecodeError on malformed JSON (returns None, triggering retry)
        """
        if not text:
            return None

        # Strip markdown code fences if present
        fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        candidate = fenced.group(1).strip() if fenced else text

        # Find the outermost { ... } using brace counting
        # (more reliable than greedy regex when JSON contains nested objects)
        start = candidate.find("{")
        if start == -1:
            return None

        depth = 0
        end   = -1
        for i, ch in enumerate(candidate[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end == -1:
            return None

        json_str = candidate[start : end + 1]
        try:
            result = json.loads(json_str)
            # Must be a non-empty dict to be valid
            if isinstance(result, dict) and result:
                return result
            return None
        except json.JSONDecodeError as e:
            logger.debug(f"  _extract_json JSONDecodeError: {e}")
            return None

    # ── Section matching ───────────────────────────────────────────────

    def _match_heading(
        self,
        llm_heading: str,
        section_map: Dict[str, Any],
        unit_lookup:  Dict[str, Any],
        threshold:    float = 0.82,
    ) -> Optional[str]:
        """
        Find the best matching key in section_map / unit_lookup for a heading
        returned by the LLM.

        Strategy:
          1. Exact match (fastest, most common case)
          2. Case-insensitive exact match
          3. difflib.SequenceMatcher fuzzy match above threshold

        Returns the matched key string, or None if no match found.
        Logs fuzzy matches at DEBUG level so they are auditable.
        """
        # 1. Exact match
        if llm_heading in section_map or llm_heading in unit_lookup:
            return llm_heading

        # 2. Case-insensitive
        llm_lower = llm_heading.lower().strip()
        for key in set(list(section_map.keys()) + list(unit_lookup.keys())):
            if key.lower().strip() == llm_lower:
                logger.debug(
                    f"  DocMetaEnricher: case-insensitive match "
                    f"{llm_heading!r} → {key!r}"
                )
                return key

        # 3. Fuzzy match
        import difflib
        all_keys = sorted(set(list(section_map.keys()) + list(unit_lookup.keys())))
        best_key   = None
        best_ratio = 0.0
        for key in all_keys:
            ratio = difflib.SequenceMatcher(
                None, llm_lower, key.lower().strip()
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_key   = key

        if best_key and best_ratio >= threshold:
            logger.debug(
                f"  DocMetaEnricher: fuzzy match {llm_heading!r} → "
                f"{best_key!r} (ratio={best_ratio:.2f})"
            )
            return best_key

        logger.warning(
            f"  DocMetaEnricher: no match for LLM heading {llm_heading!r} "
            f"(best ratio={best_ratio:.2f} < {threshold})"
        )
        return None

    # ── Section content builders ───────────────────────────────────────

    def _build_doc_text(self, text_chunks: List[Dict]) -> str:
        return " ".join(
            (c.get("text_original_content") or "")
            for c in sorted(text_chunks, key=lambda x: x.get("chunk_index", 0))
        )

    def _build_doc_section_contents(
        self,
        doc_metadata:    Dict,
        structure_units: List[Dict],
        text_chunks:     List[Dict],
    ) -> str:
        chunk_by_id = {c["chunk_id"]: c for c in text_chunks if c.get("chunk_id")}
        lines = []
        for su in structure_units:
            title    = su.get("title", "")
            cids     = su.get("chunk_ids", [])
            content  = " ".join(
                (chunk_by_id.get(cid, {}).get("text_original_content") or "")
                for cid in cids
            ).strip()[:1200]
            lines.append(f"### {title}\n{content}")
        return "\n\n".join(lines)

    def _build_pptx_section_contents(
        self, structure_units: List[Dict], text_chunks: List[Dict]
    ) -> str:
        chunk_by_su = {}
        for c in text_chunks:
            su_id = c.get("structure_unit_id", "")
            chunk_by_su.setdefault(su_id, []).append(c)

        lines = []
        for su in structure_units:
            title    = su.get("title", "")
            su_id    = su.get("structure_unit_id", "")
            chunks   = chunk_by_su.get(su_id, [])
            body     = " ".join(c.get("text_original_content", "") for c in chunks)[:600]
            notes    = " ".join(c.get("speaker_notes", "") or "" for c in chunks)[:300]
            content  = body
            if notes.strip():
                content += f"\n[Speaker notes]: {notes}"
            lines.append(f"### {title}\n{content}")
        return "\n\n".join(lines)

    def _build_sheet_section_contents(
        self, structure_units: List[Dict], table_chunks: List[Dict]
    ) -> str:
        schema_by_su: Dict[str, List[str]] = {}
        for tc in table_chunks:
            if tc.get("is_schema_chunk"):
                su_id = tc.get("structure_unit_id", "")
                cols  = tc.get("column_names", [])
                schema_by_su.setdefault(su_id, []).extend(cols)

        lines = []
        for su in structure_units:
            title = su.get("title", su.get("sheet_name", ""))
            su_id = su.get("structure_unit_id", "")
            cols  = schema_by_su.get(su_id, [])
            lines.append(f"### {title}\nColumns: {', '.join(cols)}")
        return "\n\n".join(lines)

    def _build_video_section_contents(
        self, structure_units: List[Dict], video_segments: List[Dict]
    ) -> str:
        seg_by_su: Dict[str, List[str]] = {}
        for seg in video_segments:
            su_id = seg.get("structure_unit_id", "")
            seg_by_su.setdefault(su_id, []).append(
                seg.get("transcript_text", "")[:300]
            )

        lines = []
        for su in structure_units:
            title = su.get("title", "")
            su_id = su.get("structure_unit_id", "")
            texts = seg_by_su.get(su_id, [])
            excerpt = " ".join(texts)[:600]
            lines.append(f"### {title}\n{excerpt}")
        return "\n\n".join(lines)

    def _load_video_transcript(self, doc_output_dir: Path, doc_meta: Dict) -> str:
        stem = doc_meta.get("doc_title", "")
        trans_dir = doc_output_dir / "transcripts"
        txt_path  = trans_dir / f"{stem}_full_transcript.txt"
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8")[:8000]
        # Fallback: join segment transcripts
        return ""