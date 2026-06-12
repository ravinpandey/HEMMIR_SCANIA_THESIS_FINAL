"""
enrichment_layer/enrichers/image_enricher.py

Vision LLM enrichment for image chunks (PDF / DOCX / PPTX).

For each image chunk:
  1. Resolve image_path relative to doc_output_dir
  2. Load image, encode as base64
  3. Build RICH context from section summary + enriched page text
  4. Call vision LLM with image + context prompt
  5. Write: image_caption, contextual_summary, depicted_component,
            visible_annotations, image_type + confidence scores

PPTX thumbnails get a different prompt framing.

Writes to chunk dict:
  chunk["image_caption"]
  chunk["image_caption_confidence"]
  chunk["contextual_summary"]
  chunk["contextual_summary_confidence"]
  chunk["depicted_component"]
  chunk["visible_annotations"]
  chunk["image_type"]

Changes from previous version — Fix 1: Rich image context:
  OLD: build lookup by structure_unit_id → always empty at enrichment time
       because cross-referencer runs AFTER enrichment. Context always fell
       back to first 400 chars of full document — completely wrong section.

  NEW: Build context using TWO sources both available at enrichment time:
    1. section_map section_summary (filled by DocMetaEnricher which runs
       BEFORE ImageEnricher) — gives the LLM the semantic meaning of the
       section the image appears in.
    2. contextual_summary of text chunks on same page (filled by
       TextEnricher which runs BEFORE ImageEnricher) — gives the LLM the
       enriched, situated version of the surrounding text.

  This combination gives the vision LLM:
    "Section 3.1: Describes encoder-decoder structure using multi-head
     self-attention with 6 stacked layers...
     Relevant text on this page: This chunk describes the encoder stack
     which processes input tokens through 6 identical layers each containing
     a multi-head self-attention sublayer..."

  Instead of the first 400 chars of the abstract — completely irrelevant
  for an image on page 3 of a 15-page paper.

  Research impact: Image embeddings now carry correct section semantics.
  A query about "encoder architecture diagram" will correctly match the
  Transformer architecture figure because the LLM described it using the
  actual section context, not the abstract.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from enrichment_layer.enrichers._parse_utils import (
    extract_field,
    extract_float,
    missing_keys,
)
from enrichment_layer.utils.llm_client import LLMClient

ALLOWED_IMAGE_TYPES = {
    "flowchart", "layout_diagram", "architecture_diagram", "bar_chart",
    "line_chart", "scatter_plot", "pie_chart", "table", "schematic",
    "photo", "slide_thumbnail", "diagram", "other",
}
_REQUIRED = ["CAPTION", "DESCRIPTION"]

IMAGE_PROMPT = """\
You are analysing a technical image for a Retrieval-Augmented Generation (RAG) system. \
Your description is the ONLY representation of this image the retrieval system can search — \
make it exhaustive and information-dense. A future QA system must be able to answer ANY \
question about this image using only your description.

Document context:
{context}

Analyse the image with expert precision. Then respond in EXACTLY this format \
(all fields required, no length limit on DESCRIPTION):

CAPTION: <1-2 sentences: what this image is and its primary subject>

DESCRIPTION: <Comprehensive technical description that covers ALL of the following \
that are present in the image:
- ALL visible text, labels, numbers, units, and annotations quoted exactly as they appear
- Spatial layout and arrangement: what is positioned left / right / top / bottom / centre, \
how elements are arranged relative to each other, whether the layout is linear / U-shaped / \
circular / hierarchical / grid-based
- For flowcharts and process diagrams: exact sequence of steps in order, direction of every \
arrow and flow, all decision points, branch conditions, start and end points, parallel paths
- For layout and floor-plan diagrams: named positions of all elements, paths between areas, \
distances or scale if shown, entry and exit points
- For charts and graphs: chart type, all axis labels with units, all data series names, \
all explicitly shown values and ranges, overall trend or pattern, key comparisons between \
data points or groups
- For architecture and system diagrams: all named components, connection types, direction of \
data or signal flow, hierarchical relationships, interfaces
- For tables shown as images: all column headers, all row labels, key cell values
- For photos: all visible objects, people, equipment, environment, any text in the scene
- What the image communicates in the context of the document: its purpose and key insight>

COMPONENT: <The primary technical system, process, or subject depicted — be specific, \
not generic>
ANNOTATIONS: <Every piece of text visible in the image, listed exactly as it appears, \
comma-separated — include axis labels, legend entries, callouts, station names, values>
IMAGE_TYPE: <one of: flowchart / layout_diagram / architecture_diagram / bar_chart / \
line_chart / scatter_plot / pie_chart / table / schematic / photo / other>
CAPTION_CONFIDENCE: <0.0-1.0>
SUMMARY_CONFIDENCE: <0.0-1.0>
COMPONENT_CONFIDENCE: <0.0-1.0>
ANNOTATIONS_CONFIDENCE: <0.0-1.0>\
"""

SLIDE_PROMPT = """\
You are analysing a presentation slide for a RAG system. Your description is the ONLY \
representation of this slide the retrieval system can search — make it complete enough \
to answer any question about its content without seeing the slide.

Slide {slide_index}: "{slide_title}"
Deck context: {context}

Respond in EXACTLY this format (no length limit on DESCRIPTION):

CAPTION: <1-2 sentences: the slide's subject and its central claim or finding>

DESCRIPTION: <Exhaustive slide description covering ALL of:
- The slide's main argument, claim, or message stated explicitly
- ALL text visible on the slide quoted exactly: title, subheadings, every bullet point, \
body text, labels, captions, footnotes, legends, axis values
- Every visual element described fully:
  * Diagrams: spatial layout (linear/U-shaped/circular/etc.), all component names, \
    direction of every arrow and flow, connections and relationships between elements, \
    start and end points, any branching or parallel paths
  * Charts: chart type, all axis labels and units, all data series, all explicitly shown \
    values, overall trend or pattern, key comparisons
  * Tables: all headers, all visible cell values
  * Photos or illustrations: all subjects, equipment, environment, visible text
- Any highlighted, coloured, bolded, or visually emphasised elements and what they signify
- How this slide's content relates to and advances the overall presentation narrative>

COMPONENT: <Main visual element type: process_diagram / layout_map / data_chart / \
architecture_diagram / comparison_table / text_slide / mixed>
ANNOTATIONS: <Every piece of text visible on the slide, quoted exactly, comma-separated>
IMAGE_TYPE: slide_thumbnail
CAPTION_CONFIDENCE: <0.0-1.0>
SUMMARY_CONFIDENCE: <0.0-1.0>
COMPONENT_CONFIDENCE: <0.0-1.0>
ANNOTATIONS_CONFIDENCE: <0.0-1.0>\
"""

RETRY_PROMPT = """\
Your previous response was incomplete. Missing required fields: {missing}

Previous response:
{prev_response}

Respond again in EXACTLY this format — DESCRIPTION is required and must be comprehensive:
CAPTION: <1-2 sentence caption>
DESCRIPTION: <comprehensive description covering all visible content, spatial layout, \
text, values, and what the image communicates>
COMPONENT: <specific component or system depicted>
ANNOTATIONS: <all visible text quoted exactly, comma-separated>
IMAGE_TYPE: <type>
CAPTION_CONFIDENCE: <0.0-1.0>
SUMMARY_CONFIDENCE: <0.0-1.0>
COMPONENT_CONFIDENCE: <0.0-1.0>
ANNOTATIONS_CONFIDENCE: <0.0-1.0>\
"""


class ImageEnricher:

    def __init__(self, llm: LLMClient, delay: float = 1.0):
        self.llm   = llm
        self.delay = delay

    def enrich(
        self,
        image_chunks:    List[Dict[str, Any]],
        text_chunks:     List[Dict[str, Any]],
        doc_output_dir:  Path,
        document_text:   str = "",
        doc_metadata:    Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Enrich image chunks with vision LLM.
        Idempotent: skips chunks that already have contextual_summary.
        Returns mutated list.
        """
        to_enrich = [c for c in image_chunks if not c.get("contextual_summary")]
        if not to_enrich:
            logger.info("  ImageEnricher: all image chunks already enriched")
            return image_chunks

        logger.info(
            f"  ImageEnricher: enriching {len(to_enrich)}/{len(image_chunks)} images"
        )

        # ── Build page → enriched text context ────────────────────────
        # TextEnricher runs BEFORE ImageEnricher so contextual_summaries
        # are already available. Use them as context — far better than raw text.
        page_to_context = self._build_page_context(text_chunks)

        # ── Build page → section summary context ──────────────────────
        # DocMetaEnricher runs BEFORE ImageEnricher so section summaries
        # are already filled. This gives the precise section meaning.
        section_map = (doc_metadata or {}).get("section_map", {})
        page_to_section = self._build_page_to_section(section_map)

        for chunk in to_enrich:
            chunk_id = chunk.get("chunk_id", "?")
            try:
                self._enrich_one(
                    chunk, page_to_context, page_to_section,
                    doc_output_dir, document_text
                )
            except Exception as e:
                logger.error(f"  ImageEnricher: {chunk_id} failed: {e}")
            time.sleep(self.delay)

        enriched = sum(1 for c in image_chunks if c.get("contextual_summary"))
        logger.success(f"  ImageEnricher: {enriched}/{len(image_chunks)} enriched")
        return image_chunks

    # ── Context builders ───────────────────────────────────────────────

    def _build_page_context(self, text_chunks: List[Dict]) -> Dict[int, str]:
        """
        Build page_number → enriched context from text chunks.

        Uses contextual_summary (enriched) when available, raw text otherwise.
        TextEnricher runs before ImageEnricher so summaries are ready.
        """
        page_to_texts: Dict[int, List[str]] = {}
        for tc in text_chunks:
            page = int(tc.get("page_number") or 0)
            text = (
                tc.get("contextual_summary")
                or tc.get("text_original_content")
                or ""
            )[:300]
            if text:
                page_to_texts.setdefault(page, []).append(text)

        return {
            page: " ".join(texts[:3])
            for page, texts in page_to_texts.items()
        }

    def _build_page_to_section(
        self, section_map: Dict[str, Any]
    ) -> Dict[int, str]:
        """
        Build page_number → section summary string.

        DocMetaEnricher fills section summaries before ImageEnricher runs.
        Each page maps to the section summary of the section it falls in.
        """
        page_to_section: Dict[int, str] = {}
        for heading, entry in section_map.items():
            if not isinstance(entry, dict):
                continue
            summary = entry.get("summary") or ""
            if not summary:
                continue
            start = int(entry.get("start_page") or 0)
            end   = int(entry.get("end_page") or start)
            for page in range(start, end + 1):
                if page not in page_to_section:
                    page_to_section[page] = (
                        f"Section: {heading}\n{summary}"
                    )
        return page_to_section

    def _build_context(
        self,
        chunk:           Dict[str, Any],
        page_to_context: Dict[int, str],
        page_to_section: Dict[int, str],
        document_text:   str,
    ) -> str:
        """
        Build the richest possible context for a given image chunk.

        Priority:
          1. Section summary + enriched page text (best case)
          2. Section summary alone
          3. Enriched page text alone
          4. First 400 chars of document (last resort — signals context missing)
        """
        page = int(chunk.get("page_number") or 0)

        section_ctx = page_to_section.get(page, "")
        page_ctx    = page_to_context.get(page, "")

        if section_ctx and page_ctx:
            return (
                f"{section_ctx}\n\n"
                f"Relevant text on this page:\n{page_ctx[:400]}"
            )
        elif section_ctx:
            return section_ctx
        elif page_ctx:
            return page_ctx
        else:
            logger.debug(
                f"  ImageEnricher: no page/section context for page {page} "
                f"— falling back to document text"
            )
            return document_text[:400]

    def _enrich_one(
        self,
        chunk:           Dict[str, Any],
        page_to_context: Dict[int, str],
        page_to_section: Dict[int, str],
        doc_output_dir:  Path,
        document_text:   str,
    ) -> None:
        chunk_id = chunk.get("chunk_id", "?")

        # ── Load image ─────────────────────────────────────────────────
        img_b64, media_type = self._load_image(chunk, doc_output_dir)
        if not img_b64:
            logger.warning(f"  ImageEnricher: {chunk_id} — image not found, skipping")
            return

        # ── Build rich context ─────────────────────────────────────────
        context = self._build_context(
            chunk, page_to_context, page_to_section, document_text
        )

        # ── Build prompt ───────────────────────────────────────────────
        is_slide    = (chunk.get("image_type") == "slide_thumbnail" or
                       chunk.get("slide_index") is not None)
        slide_idx   = chunk.get("slide_index", "")
        slide_title = chunk.get("slide_title", "")

        if is_slide:
            prompt = SLIDE_PROMPT.format(
                slide_index=slide_idx,
                slide_title=slide_title or f"Slide {slide_idx}",
                context=context[:500],
            )
        else:
            prompt = IMAGE_PROMPT.format(context=context[:700])

        # ── LLM call with retry ────────────────────────────────────────
        response = self._invoke_with_retry(img_b64, media_type, prompt)
        if not response:
            logger.warning(f"  ImageEnricher: {chunk_id} — enrichment failed after retries")
            return

        caption    = extract_field(response, "CAPTION") or ""
        # DESCRIPTION replaces the old 2-3 sentence SUMMARY — no length limit,
        # covers spatial layout, flow direction, quantitative data, all visible text.
        description = extract_field(response, "DESCRIPTION") or ""
        component  = extract_field(response, "COMPONENT") or ""
        annots     = extract_field(response, "ANNOTATIONS") or ""
        img_type   = (extract_field(response, "IMAGE_TYPE") or "other").lower().strip()
        cap_conf   = extract_float(response, "CAPTION_CONFIDENCE",     default=0.5)
        sum_conf   = extract_float(response, "SUMMARY_CONFIDENCE",     default=0.5)
        comp_conf  = extract_float(response, "COMPONENT_CONFIDENCE",   default=0.5)
        annot_conf = extract_float(response, "ANNOTATIONS_CONFIDENCE", default=0.5)

        # Normalise image type
        if img_type not in ALLOWED_IMAGE_TYPES:
            img_type = "other"

        if caption:
            chunk["image_caption"]                 = caption
            chunk["image_caption_confidence"]      = cap_conf
        if description:
            # Store full description as contextual_summary — downstream retrieval
            # uses this field. No truncation here; chroma_store caps at index time.
            chunk["contextual_summary"]            = description
            chunk["contextual_summary_confidence"] = sum_conf
        if component and component.lower() not in ("n/a", "none", ""):
            chunk["depicted_component"]            = component
            chunk["depicted_component_confidence"] = comp_conf
        if annots and annots.lower() not in ("none", "n/a", ""):
            chunk["visible_annotations"]           = annots
            chunk["visible_annotations_confidence"] = annot_conf
        chunk["image_type"] = img_type

        logger.debug(
            f"  ImageEnricher: {chunk_id} | type={img_type} | "
            f"desc_len={len(description)} | cap_conf={cap_conf} | "
            f"context_len={len(context)}"
        )

    def _invoke_with_retry(
        self, img_b64: str, media_type: str, prompt: str, max_retries: int = 2
    ) -> Optional[str]:
        response = None
        for attempt in range(1, max_retries + 2):
            try:
                response = self.llm.invoke_with_image(
                    prompt=prompt,
                    image_b64=img_b64,
                    media_type=media_type,
                    max_tokens=1200,
                )
            except Exception as e:
                logger.warning(f"  ImageEnricher vision attempt {attempt}: {e}")
                if attempt <= max_retries:
                    time.sleep(1.5 * attempt)
                continue

            if not missing_keys(response, _REQUIRED):
                return response

            if attempt <= max_retries:
                prompt = RETRY_PROMPT.format(
                    missing=", ".join(missing_keys(response, _REQUIRED)),
                    prev_response=response[:500],
                )
                time.sleep(1.5)

        return response

    def _load_image(
        self, chunk: Dict, doc_output_dir: Path
    ) -> tuple[Optional[str], str]:
        rel = chunk.get("image_path", "")
        if not rel:
            return None, ""
        path = doc_output_dir / rel
        if not path.exists():
            return None, ""
        try:
            raw = path.read_bytes()
            # Detect actual format from magic bytes — extension is unreliable
            # (PPT extractor saves all images as .png regardless of actual format)
            if raw[:3] == b'\xff\xd8\xff':
                media_type = "image/jpeg"
            elif raw[:8] == b'\x89PNG\r\n\x1a\n':
                media_type = "image/png"
            elif raw[:6] in (b'GIF87a', b'GIF89a'):
                media_type = "image/gif"
            elif raw[:4] == b'RIFF' and raw[8:12] == b'WEBP':
                media_type = "image/webp"
            else:
                # Fall back to extension
                ext = path.suffix.lower()
                media_type = (
                    "image/png"  if ext == ".png"  else
                    "image/gif"  if ext == ".gif"  else
                    "image/webp" if ext == ".webp" else
                    "image/jpeg"
                )
            data = base64.b64encode(raw).decode()
            return data, media_type
        except Exception as e:
            logger.warning(f"  Cannot load image {path}: {e}")
            return None, ""