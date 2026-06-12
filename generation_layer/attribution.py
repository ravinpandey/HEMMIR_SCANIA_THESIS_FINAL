"""
generation_layer/attribution.py

Per-claim attribution builder — E2 explainability component.

Enriches each supported claim from the Verifier with:
  - section_id and section_title (from chunk metadata)
  - evidence_role (from enrichment layer: procedure/measurement/definition/context)
  - salience_score (from enrichment layer: chunk centrality 0-1)
  - modality (text/image/table/video_segment/video_frame)
  - location string: format-specific human-readable position
      PDF:   "p.12"
      PPTX:  "Slide 7"
      Video: "4:32"
      CSV:   "rows 120-180"
  - is_direct: True if evidence_role matches query_intent_type

The is_direct flag is the key explainability signal:
  True  → answer is directly grounded in the right type of evidence
  False → answer draws from indirect/contextual evidence

This distinction matters for industrial safety contexts:
  A maintenance procedure answered from "context" chunks rather than
  "procedure" chunks should surface a warning to the engineer.

Changelog:
  - is_table_direct: table chunks with _tbl_ or _html in chunk_id are marked
    direct for comparative/measurement queries regardless of evidence_role,
    because table chunks do not go through TextEnricher and have no role set.
  - salience fallback: when salience_score is None (table chunks), falls back
    to table_summary_confidence from extra_payload (0.88-0.98 range).
  - inferred_modality: chunk_id pattern used to detect table/image chunks
    when source_modality is incorrectly serialized as "text".
  - DIRECT_ROLE_MAP comparative: added comparison/comparative/quantitative
    to handle verifier role labels for table evidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models.pipeline_models import (
    AttributedClaim,
    EvidenceItem,
    EvidencePack,
    FaithfulnessReport,
)

# Maps query_intent_type → evidence_roles that count as "direct"
DIRECT_ROLE_MAP: Dict[str, List[str]] = {
    "procedural":  ["procedure", "procedural"],
    "measurement": ["measurement"],
    "definition":  ["definition"],
    "comparative": ["procedure", "measurement", "definition", "comparison", "comparative", "quantitative"],
    "diagnostic":  ["measurement", "procedure"],
    "exploratory": ["definition", "context", "procedure"],
    "visual":      ["context", "definition"],
    "temporal":    ["context", "procedure"],
}


def build_attribution(
    faithfulness_report: FaithfulnessReport,
    evidence_pack:       EvidencePack,
    query_intent_type:   str = "definition",
) -> List[AttributedClaim]:
    """
    Build per-claim attribution for all supported claims.

    Args:
        faithfulness_report: From Verifier — contains supported_claims
        evidence_pack:       Final evidence pack — for metadata lookup
        query_intent_type:   From QueryAnalysis — for is_direct determination

    Returns:
        List[AttributedClaim] — one per supported claim
    """
    # Build chunk_id → EvidenceItem lookup
    # Also build a secondary lookup stripping _html suffix so verifier
    # citations to semantic chunks match HTML siblings and vice versa
    item_lookup: Dict[str, EvidenceItem] = {}
    for item in evidence_pack.items:
        item_lookup[item.chunk_id] = item
        # Also index by base_id (without _html) so citations match either view
        base_id = item.chunk_id.replace("_html", "")
        if base_id not in item_lookup:
            item_lookup[base_id] = item

    direct_roles = DIRECT_ROLE_MAP.get(query_intent_type, ["context"])
    attributed:  List[AttributedClaim] = []

    for claim in faithfulness_report.supported_claims:
        chunk_id = claim.supporting_chunk_id
        item     = item_lookup.get(chunk_id)
        # LLM sometimes prefixes chunk_id with "chunk_" — strip and retry
        if item is None and chunk_id.startswith("chunk_"):
            item = item_lookup.get(chunk_id[6:])

        if item is None:
            # Claim cited a chunk_id not in evidence pack — still record it
            attributed.append(AttributedClaim(
                claim        = claim.claim,
                chunk_id     = chunk_id,
                doc_id       = "",
                evidence_role = claim.evidence_role or "unknown",
                modality     = "unknown",
                location     = "unknown",
                confidence   = 0.0,
                is_direct    = False,
            ))
            continue

        meta         = item.metadata
        modality     = item.source_modality
        evidence_role = (
            claim.evidence_role
            or getattr(meta, "evidence_role", None)
            or "context"
        )

        # ── Salience score ─────────────────────────────────────────────
        # Text chunks have salience_score set by TextEnricher (0.0–1.0).
        # Table chunks never go through TextEnricher so salience_score is
        # None. Fall back to table_summary_confidence from extra_payload.
        # Also check the HTML sibling in item_lookup if base item lacks tsc.
        salience = _safe_float(getattr(meta, "salience_score", None), None)
        if salience is None or salience == 0.0:
            # Try current item's extra_payload first
            _ep = item.extra_payload if isinstance(item.extra_payload, dict) else {}
            _tsc = _ep.get("table_summary_confidence")
            # Try HTML sibling if not found
            if _tsc is None:
                html_item = item_lookup.get(item.chunk_id + "_html")
                if html_item:
                    _ep2 = html_item.extra_payload if isinstance(html_item.extra_payload, dict) else {}
                    _tsc = _ep2.get("table_summary_confidence")
            salience = _safe_float(_tsc, 0.5) if _tsc is not None else 0.5

        confidence = _safe_float(
            getattr(meta, "contextual_summary_confidence", None), 0.5
        )
        section_id    = _get_str(meta, "section_id")
        section_title = _get_str(meta, "section_title") or _get_str(meta, "slide_title")
        location      = _build_location(meta, modality)

        # ── is_direct determination ────────────────────────────────────
        # Primary: evidence_role matches expected roles for this intent
        # Secondary: table chunks are direct for comparative/measurement
        #   even without an enriched evidence_role, because table chunks
        #   contain the decisive quantitative evidence for these intents.
        #   Use chunk_id pattern to detect tables when source_modality
        #   is incorrectly serialized as "text".
        inferred_modality = modality
        if inferred_modality == "text" or not inferred_modality:
            cid = (item.chunk_id or "").lower()
            if "_tbl_" in cid or cid.endswith("_html"):
                inferred_modality = "table"
            elif "_img_" in cid:
                inferred_modality = "image"

        is_table_direct = (
            inferred_modality == "table"
            and query_intent_type in ("comparative", "measurement")
        )
        is_image_direct = (
            inferred_modality == "image"
            and query_intent_type in ("visual", "exploratory")
        )
        is_direct = (
            evidence_role.lower() in [r.lower() for r in direct_roles]
            or is_table_direct
            or is_image_direct
        )

        attributed.append(AttributedClaim(
            claim          = claim.claim,
            chunk_id       = chunk_id,
            doc_id         = item.doc_id,
            section_id     = section_id,
            section_title  = section_title,
            evidence_role  = evidence_role,
            salience_score = salience,
            modality       = modality,
            location       = location,
            confidence     = confidence,
            is_direct      = is_direct,
        ))

        logger.debug(
            f"  Attribution: claim[{len(attributed)}] "
            f"role={evidence_role} direct={is_direct} "
            f"loc={location} sal={salience:.2f}"
        )

    logger.info(
        f"  Attribution: {len(attributed)} claims attributed | "
        f"direct={sum(1 for a in attributed if a.is_direct)} | "
        f"indirect={sum(1 for a in attributed if not a.is_direct)}"
    )
    return attributed


def _build_location(meta: Any, modality: str) -> str:
    """Build human-readable location string for each modality."""
    if modality in ("video_segment",):
        start = getattr(meta, "start_time_s", None)
        end   = getattr(meta, "end_time_s", None)
        if start is not None and end is not None:
            return f"{_fmt_time(float(start))}–{_fmt_time(float(end))}"
        return "video"

    if modality in ("video_frame",):
        ts = getattr(meta, "timestamp_s", None)
        if ts is not None:
            return _fmt_time(float(ts))
        return "video frame"

    if modality == "image":
        page  = getattr(meta, "page_number", None)
        slide = getattr(meta, "slide_index", None)
        if slide is not None and int(slide) > 0:
            return f"Slide {slide}"
        if page is not None and int(page) > 0:
            return f"p.{page}"
        return "figure"

    if modality == "table":
        row_start = getattr(meta, "row_start", None)
        row_end   = getattr(meta, "row_end", None)
        sheet     = getattr(meta, "sheet_name", None)
        if row_start is not None and row_end is not None and int(row_end) > 0:
            prefix = f"{sheet} " if sheet else ""
            return f"{prefix}rows {row_start}–{row_end}"
        page = getattr(meta, "page_number", None)
        if page is not None and int(page) > 0:
            return f"table p.{page}"
        return "table"

    # Text chunk (PDF / DOCX / PPTX)
    slide = getattr(meta, "slide_index", None)
    if slide is not None and int(slide) > 0:
        title = getattr(meta, "slide_title", None) or ""
        return f"Slide {slide}" + (f": {title}" if title else "")

    page = getattr(meta, "page_number", None)
    if page is not None and int(page) > 0:
        return f"p.{page}"

    return "document"


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _safe_float(v: Any, default) -> Optional[float]:
    if v is None:
        return default
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def _get_str(meta: Any, attr: str) -> str:
    val = getattr(meta, attr, None)
    if val is None:
        return ""
    return str(val).strip()