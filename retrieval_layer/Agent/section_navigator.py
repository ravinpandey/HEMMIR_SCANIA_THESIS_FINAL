"""
retrieval_layer/Agent/section_navigator.py

Navigates the unified sections_collection to find which structural units
(sections, scenes, slides, sheets) are most relevant for each sub-question.

The sections_collection is format-agnostic — it contains entries for:
  - PDF/DOCX sections (heading text + LLM summary + keywords + entities)
  - PPTX slides (slide title + body + speaker notes summary)
  - CSV/XLSX sheets (sheet name + column semantics from enrichment)
  - Video scenes (scene title + transcript summary)

Each entry has the same structure: embedded text, section_summary,
keywords_str, entities_str, structure_unit_id, section_order.

The navigator:
  1. Embeds the sub-question
  2. Searches sections_collection with ChromaDB vector search
  3. Optionally asks LLM to select the best sections from candidates
  4. Returns selected structure_unit_ids + metadata for chunk retrieval
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


NAVIGATOR_SYSTEM = """\
You are selecting which document sections to retrieve evidence from \
for an industrial document retrieval system.

Sections come from multiple document types:
  - PDF/DOCX manual sections (technical procedures, specifications)
  - PPTX presentation slides (summaries, overviews, safety requirements)
  - CSV/XLSX spreadsheet sheets (fault logs, maintenance records)
  - Video scenes (training demonstrations, walkthroughs)

You select sections based on relevance to the sub-question AND whether \
the section is likely to contain the right TYPE of evidence.\
"""

NAVIGATOR_PROMPT = """\
Select the best sections to retrieve evidence for this sub-question.

Sub-question: "{sub_question}"
Evidence role needed: {evidence_role}
Section hint keyword: {section_hint}

Section candidates (from vector search, ordered by similarity):
{section_candidates}

Format per candidate:
  [ID] title | summary | keywords | modality | similarity_score

Select 1-3 sections that best answer the sub-question.
Prefer sections where:
  1. The section summary directly addresses the sub-question
  2. The evidence role matches (procedure section for procedural sub-question)
  3. Multiple modalities together give a complete answer

Respond in EXACTLY this JSON (no markdown):
{{
  "selected_sections": [
    {{
      "structure_unit_id": "<exact ID from candidates>",
      "section_title": "<title>",
      "relevance_reason": "<one sentence>",
      "expected_evidence_role_match": true,
      "modality": "text|image|table|video|spreadsheet"
    }}
  ],
  "navigation_confidence": 0.0,
  "expected_answer_completeness": "full|partial|insufficient"
}}\
"""


class SectionNavigator:
    """
    Finds relevant structural units for each sub-question using
    the sections_collection — the unified structural index of the corpus.

    This is the core innovation of the Agent path: instead of searching
    all chunks simultaneously (RAG), we first find the right sections,
    then retrieve chunks only from those sections.
    """

    def __init__(self, llm_client, text_embedder, top_k_sections: int = 5):
        self.llm             = llm_client
        self.text_embedder   = text_embedder
        self.top_k_sections  = top_k_sections

    def navigate(
        self,
        sub_question:  Dict[str, Any],
        store,
        doc_ids:       Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], float, str]:
        """
        Find relevant sections for one sub-question.

        Args:
            sub_question: Dict with question, evidence_role, section_hint
            store:        ChromaStore instance
            doc_ids:      Limit search to these doc_ids (from doc retrieval)

        Returns:
            (selected_sections, navigation_confidence, expected_completeness)
        """
        question    = sub_question.get("question", "")
        role        = sub_question.get("evidence_role", "context")
        hint        = sub_question.get("section_hint", "")

        # Build search text — question + hint for better section matching
        search_text = question
        if hint:
            search_text = f"{question} {hint}"

        # Embed and search sections_collection
        candidates  = self._search_sections(search_text, store, doc_ids)

        if not candidates:
            logger.warning(f"  Navigator: no section candidates for: {question[:60]}")
            return [], 0.1, "insufficient"

        # If only 1-2 candidates, select automatically without LLM
        if len(candidates) <= 2:
            selected = [
                {
                    "structure_unit_id": c["id"],
                    "section_title":     c["meta"].get("section_name", ""),
                    "relevance_reason":  "Top vector match",
                    "expected_evidence_role_match": True,
                    "modality":          _infer_modality(c["meta"]),
                }
                for c in candidates
            ]
            conf = candidates[0]["score"] if candidates else 0.3
            return selected, conf, "partial"

        # LLM selection for richer candidate sets
        try:
            return self._llm_select(question, role, hint, candidates)
        except Exception as e:
            logger.warning(f"  Navigator LLM selection failed: {e} — using top-2")
            selected = [
                {
                    "structure_unit_id": c["id"],
                    "section_title":     c["meta"].get("section_name", ""),
                    "relevance_reason":  "Top vector match (LLM unavailable)",
                    "expected_evidence_role_match": True,
                    "modality":          _infer_modality(c["meta"]),
                }
                for c in candidates[:2]
            ]
            return selected, 0.5, "partial"

    def _search_sections(
        self,
        search_text: str,
        store,
        doc_ids:     Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Search sections_collection and return formatted candidates."""
        query_vec = self.text_embedder.embed_query(search_text)
        if not query_vec:
            return []

        collection = store.collections.get("sections")
        if not collection:
            logger.warning("  Navigator: sections collection not found")
            return []

        count = max(0, int(collection.count()))
        if count == 0:
            return []

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vec],
            "n_results":        min(self.top_k_sections, count),
            "include":          ["documents", "metadatas", "distances"],
        }
        if doc_ids and len(doc_ids) == 1:
            kwargs["where"] = {"doc_id": {"$eq": doc_ids[0]}}
        elif doc_ids and len(doc_ids) > 1:
            kwargs["where"] = {"doc_id": {"$in": doc_ids}}

        try:
            results = collection.query(**kwargs)
        except Exception as e:
            logger.warning(f"  Navigator: sections query failed: {e}")
            return []

        ids   = results.get("ids",       [[]])[0]
        docs  = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        candidates = []
        for sec_id, doc, meta, dist in zip(ids, docs, metas, dists):
            score = round(max(0.0, 1.0 - float(dist)), 4)
            candidates.append({
                "id":    sec_id,
                "doc":   doc or "",
                "meta":  meta or {},
                "score": score,
            })

        return sorted(candidates, key=lambda x: x["score"], reverse=True)

    def _llm_select(
        self,
        question:   str,
        role:       str,
        hint:       str,
        candidates: List[Dict[str, Any]],
    ) -> Tuple[List[Dict], float, str]:
        """Use LLM to select the best sections from candidates."""
        # Format candidates for prompt
        formatted = []
        for c in candidates:
            meta     = c["meta"]
            title    = meta.get("section_name", "")[:60]
            summary  = meta.get("section_summary", "")[:120]
            keywords = meta.get("keywords_str", "")[:80]
            modality = _infer_modality(meta)
            score    = c["score"]
            formatted.append(
                f"  [{c['id']}] {title} | {summary} | "
                f"kw: {keywords} | {modality} | score={score:.3f}"
            )

        prompt = NAVIGATOR_PROMPT.format(
            sub_question       = question,
            evidence_role      = role,
            section_hint       = hint or "none",
            section_candidates = "\n".join(formatted),
        )
        response = self.llm.invoke(
            system     = NAVIGATOR_SYSTEM,
            prompt     = prompt,
            max_tokens = 800,
        )
        data = _extract_json(response)
        selected = (data or {}).get("selected_sections") or []
        if not selected:
            logger.debug("  Navigator: no selected_sections in LLM response — using top-2")
            return [
                {
                    "structure_unit_id": c["id"],
                    "section_title":     c["meta"].get("section_name", ""),
                    "relevance_reason":  "Top vector match",
                    "expected_evidence_role_match": True,
                    "modality":          _infer_modality(c["meta"]),
                }
                for c in candidates[:2]
            ], 0.5, "partial"
        confidence  = float(data.get("navigation_confidence", 0.5))
        completeness = data.get("expected_answer_completeness", "partial")

        logger.info(
            f"  Navigator: selected {len(selected)} section(s) "
            f"| confidence={confidence:.2f} | completeness={completeness}"
        )
        return selected, confidence, completeness


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_modality(meta: Dict) -> str:
    """Infer document modality from section metadata."""
    doc_id = meta.get("doc_id", "")
    name   = meta.get("section_name", "").lower()
    if "scene" in name or "seg" in doc_id.lower():
        return "video"
    if "slide" in name:
        return "presentation"
    if "sheet" in name or "row" in name:
        return "spreadsheet"
    return "text"


def _extract_json(text: str):
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
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None
