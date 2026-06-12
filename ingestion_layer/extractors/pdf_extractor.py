"""
ingestion_layer/extractors/pdf_extractor.py

Updated PDF Extractor with three chunking strategies:
    - paragraph: one chunk per Docling paragraph (default)
    - title:     one chunk per section/heading group
    - size:      fixed token-size with overlap (for experiments)

Also adds:
    - page_number on every chunk
    - section_title on text chunks (title strategy)
    - token_count on every text chunk
    - row_count, col_count on table chunks
    - total_pages on document metadata
    - chunk_strategy stored in doc_metadata for reproducibility

Changes from previous version:
    - Bug fix: row_count / col_count initialized to 0 before try block
      so UnboundLocalError cannot occur when DataFrame export fails on
      malformed/scanned tables (was a hard crash on the entire document).
    - Bug fix: _get_title() now uses a multi-strategy approach to return
      the actual paper title instead of the arxiv filename (e.g.
      "Attention Is All You Need" not "1706.03762v7"). Strategy order:
        1. Docling PDF metadata title field
        2. First Docling title-labeled element on page 1
        3. First section_header element on page 1
        4. File stem fallback (original behavior — last resort only)
    - doc_embedding_input_text now uses doc_title (resolved above) as the
      title prefix, not the raw file stem — fixes the embedding quality gap
      identified in metadata comparison.
"""

import sys
import io
from pathlib import Path
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from PIL import Image as PILImage
from loguru import logger

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.models.metadata_models import (
    DocumentMetadata, TextChunkMetadata,
    ImageChunkMetadata, TableChunkMetadata,
)
from ingestion_layer.utils.chunking_utils import apply_chunking_strategy

logger.remove()
logger.add(sys.stdout, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")


def generate_doc_id(file_path: Path) -> str:
    import hashlib
    return hashlib.md5(file_path.name.encode()).hexdigest()[:12]

def generate_text_chunk_id(doc_id, idx):   return f"{doc_id}_chunk_{idx:03d}"
def generate_image_chunk_id(doc_id, p, i): return f"{doc_id}_page_{p:03d}_img_{i:03d}"
def generate_table_chunk_id(doc_id, p, i): return f"{doc_id}_page_{p:03d}_tbl_{i:03d}"


class PDFExtractor:
    """
    Extracts text, images, and tables from PDFs using Docling.
    Supports three chunking strategies: paragraph, title, size.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.converter  = self._build_converter()

    def _build_converter(self) -> DocumentConverter:
        opts = PdfPipelineOptions()
        opts.images_scale           = 2.0
        opts.generate_page_images   = False
        opts.generate_picture_images = True
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )

    # ──────────────────────────────────────────────────────────────────
    # Main Entry Point
    # ──────────────────────────────────────────────────────────────────

    def extract(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """
        Extract all content from a PDF into structured metadata.

        kwargs:
            project_id, tier1, tier2, document_type, language
            chunk_strategy: "paragraph" | "title" | "size"
            chunk_size:     int (for size strategy)
            chunk_overlap:  int (for size strategy)
        """
        chunk_strategy = kwargs.get("chunk_strategy", "paragraph")
        chunk_size     = int(kwargs.get("chunk_size", 512))
        chunk_overlap  = int(kwargs.get("chunk_overlap", 50))

        logger.info(f"Extracting: {file_path.name} | strategy={chunk_strategy} | size={chunk_size}")

        pdf_stem = file_path.stem
        folders  = self._create_folders(pdf_stem)
        doc_id   = generate_doc_id(file_path)

        # Run Docling
        result = self.converter.convert(str(file_path))
        doc    = result.document

        # Extract total pages
        total_pages = self._get_total_pages(doc)

        # ── Compute section spans from raw Docling elements (BEFORE chunking) ──
        # Must run before chunking — the size strategy destroys element boundaries.
        # Reads Docling layout labels (section_header, title) directly from doc.
        # Falls back to page-based synthetic sections if no headings found.
        ingestion_section_spans = self._compute_section_spans_from_raw(doc, total_pages)

        # Extract all modalities
        text_chunks = self._extract_text_chunks(
            doc, doc_id, chunk_strategy, chunk_size, chunk_overlap
        )
        image_chunks, figure_index_map = self._extract_image_chunks(
            doc, doc_id, folders["images"], pdf_stem, result
        )
        table_chunks, table_index_map  = self._extract_table_chunks(
            doc, doc_id, folders["tables"], pdf_stem
        )

        # ── Build section_map skeleton with page ranges + empty lists ─
        # chunk_ids / figure_ids / table_ids will be filled by encoding layer.
        # summaries / subsections will be filled by enrichment layer.
        # synthetic=True marks page-based fallback sections (no real headings found).
        code_section_map: Dict[str, Any] = {}
        for span in ingestion_section_spans:
            entry = {
                "summary":           None,
                "keywords":          [],
                "entities":          [],
                "subsections":       [],
                "start_page":        span["start_page"],
                "end_page":          span["end_page"],
                "start_element_idx": span.get("start_element_idx"),
                "end_element_idx":   span.get("end_element_idx"),
                "chunk_ids":         [],
                "figure_ids":        [],
                "table_ids":         [],
            }
            if span.get("synthetic"):
                entry["synthetic"] = True
            code_section_map[span["heading"]] = entry

        # ── Resolve title, author, project_id ─────────────────────────
        # _get_title() uses multi-strategy extraction to return the real
        # paper/document title rather than the raw file stem. This directly
        # fixes the doc_embedding_input_text quality gap where arxiv filenames
        # like "1706.03762v7" were used instead of "Attention Is All You Need".
        doc_title  = self._get_title(doc, file_path)
        raw_author = self._get_author(doc)
        # project_id: prefer caller-supplied; fall back to doc_title
        resolved_project_id = kwargs.get("project_id") or doc_title

        logger.info(f"  Section spans found (code): {len(ingestion_section_spans)}")
        for span in ingestion_section_spans:
            logger.info(f"    {span['heading']!r:40s} p{span['start_page']}–{span['end_page']}")
        logger.info(f"  Title  : {doc_title}")
        logger.info(f"  Author : {raw_author}")
        logger.info(f"  Project: {resolved_project_id}")

        # Build document metadata
        doc_metadata = DocumentMetadata(
            doc_id           = doc_id,
            doc_title        = doc_title,
            project_id       = resolved_project_id,
            author           = raw_author,
            chunk_count      = len(text_chunks) + len(image_chunks) + len(table_chunks),
            related_figures  = [c.figure_id for c in image_chunks],
            related_tables   = [c.chunk_id  for c in table_chunks],
            last_modified    = datetime.utcfromtimestamp(file_path.stat().st_mtime).isoformat(),
            document_type    = kwargs.get("document_type"),
            language         = kwargs.get("language", "English"),
            total_pages      = total_pages,
            chunk_strategy   = chunk_strategy,
            chunk_size       = chunk_size if chunk_strategy == "size" else None,
            chunk_overlap    = chunk_overlap if chunk_strategy == "size" else None,
            figure_index_map = figure_index_map,
            table_index_map  = table_index_map,
            section_map      = code_section_map,
            tier1            = None,
            tier2            = None,
            doc_summary      = None,
            doc_summary_confidence = None,
            tier_confidence  = None,
            outline_summary  = None,
            doc_embedding    = None,
            outline_summary_confidence = None,
        )

        # Save all metadata
        import json
        def save_json(data, path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved: {path.name}")

        # Attach ingestion_section_spans to the dict before saving
        # (not a Pydantic field — stored as extra metadata for downstream layers)
        doc_meta_dict = doc_metadata.to_dict()
        doc_meta_dict["ingestion_section_spans"] = ingestion_section_spans

        # Modality count signals — inferrable downstream but stored explicitly
        # here for fast doc-level filtering without list traversal.
        doc_meta_dict["figure_count"]  = len(image_chunks)
        doc_meta_dict["table_count"]   = len(table_chunks)
        doc_meta_dict["has_figures"]   = len(image_chunks) > 0
        doc_meta_dict["has_tables"]    = len(table_chunks) > 0
        doc_meta_dict["section_titles"] = [
            span["heading"] for span in ingestion_section_spans
            if not span.get("synthetic")
        ]

        save_json(doc_meta_dict,                              folders["metadata"] / "doc_metadata.json")
        save_json([c.to_dict() for c in text_chunks], folders["metadata"] / "text_chunks.json")
        save_json([c.to_dict() for c in image_chunks],folders["metadata"] / "image_chunks.json")
        save_json([c.to_dict() for c in table_chunks],folders["metadata"] / "table_chunks.json")

        logger.success(
            f"Done: {file_path.name} → "
            f"{len(text_chunks)} text | {len(image_chunks)} images | {len(table_chunks)} tables"
        )

        return {
            "doc_id":       doc_id,
            "text_count":   len(text_chunks),
            "image_count":  len(image_chunks),
            "table_count":  len(table_chunks),
        }

    # ──────────────────────────────────────────────────────────────────
    # Text Extraction + Chunking Strategy
    # ──────────────────────────────────────────────────────────────────

    def _extract_text_chunks(
        self, doc, doc_id: str,
        strategy: str, chunk_size: int, chunk_overlap: int
    ) -> List[TextChunkMetadata]:
        """
        Extract text using the selected chunking strategy.
        Returns list of TextChunkMetadata.
        section_map is now built deterministically in extract() via _compute_section_spans().
        """
        # Collect raw items from Docling — (text, page, elem_type, elem_idx) tuples
        # CRITICAL: elem_count must increment for EVERY element including tables,
        # pictures, and empty elements — to stay in sync with the counter in
        # _compute_section_spans_from_raw which also counts every element.
        # If we skip non-text elements here, the counters diverge and chunks
        # get assigned to wrong sections.
        raw_items  = []
        elem_count = 0
        for element, _level in doc.iterate_items():
            elem_count += 1   # always increment — every element counts
            if isinstance(element, (TableItem, PictureItem)):
                continue
            text = getattr(element, "text", None)
            if not text or not text.strip():
                continue
            page_no   = self._get_page(element)
            elem_type = getattr(element, "label", "text")
            raw_items.append((text.strip(), page_no, str(elem_type), elem_count))

        # Apply chunking strategy — all strategies receive triples
        raw_chunks = apply_chunking_strategy(
            raw_items     = raw_items,
            strategy      = strategy,
            chunk_size    = chunk_size,
            chunk_overlap = chunk_overlap,
        )

        # Convert to TextChunkMetadata
        chunks = []
        for i, raw in enumerate(raw_chunks):
            chunk_id  = generate_text_chunk_id(doc_id, i + 1)
            prev      = raw_chunks[i-1].text if i > 0 else ""
            nxt       = raw_chunks[i+1].text if i < len(raw_chunks)-1 else ""
            local_ctx = f"{prev} {nxt}".strip() or None

            chunk = TextChunkMetadata(
                chunk_id              = chunk_id,
                doc_id                = doc_id,
                chunk_index           = i + 1,
                text_original_content = raw.text,
                local_context         = local_ctx,
                page_number           = raw.page_number,
                section_title         = raw.section_title,
                chunk_strategy        = strategy,
                token_count           = raw.token_count,
                element_index         = raw.element_index,
                end_element_index     = raw.end_element_index,
            )
            chunks.append(chunk)

        return chunks

    # ──────────────────────────────────────────────────────────────────
    # Image Extraction
    # ──────────────────────────────────────────────────────────────────

    def _extract_image_chunks(
        self, doc, doc_id, images_dir, pdf_stem, result
    ) -> Tuple[List[ImageChunkMetadata], Dict[int, str]]:
        chunks           = []
        page_img_counter = {}
        global_fig_num   = 0
        figure_index_map = {}

        for element, _level in doc.iterate_items():
            if not isinstance(element, PictureItem):
                continue

            global_fig_num += 1
            page_no         = self._get_page(element)
            page_img_counter[page_no] = page_img_counter.get(page_no, 0) + 1
            img_idx         = page_img_counter[page_no]
            chunk_id        = generate_image_chunk_id(doc_id, page_no, img_idx)
            figure_index_map[str(global_fig_num)] = chunk_id

            # Save PNG
            img_filename = f"{pdf_stem}_page_{page_no:03d}_img_{img_idx:03d}.png"
            img_path     = images_dir / img_filename
            try:
                img = element.get_image(result.document)
                if img:
                    img.save(str(img_path))
            except Exception as e:
                logger.warning(f"Could not save image {global_fig_num}: {e}")

            caption = self._get_caption(element)

            chunk = ImageChunkMetadata(
                chunk_id    = chunk_id,
                doc_id      = doc_id,
                chunk_index = global_fig_num,
                figure_id   = chunk_id,
                image_path  = str(Path("images") / img_filename),
                page_number = page_no,
                image_caption = caption,
            )
            chunks.append(chunk)

        return chunks, figure_index_map

    # ──────────────────────────────────────────────────────────────────
    # Table Extraction
    # ──────────────────────────────────────────────────────────────────

    def _extract_table_chunks(
        self, doc, doc_id, tables_dir, pdf_stem
    ) -> Tuple[List[TableChunkMetadata], Dict[int, str]]:
        chunks           = []
        page_tbl_counter = {}
        global_tbl_num   = 0
        table_index_map  = {}

        for element, _level in doc.iterate_items():
            if not isinstance(element, TableItem):
                continue

            global_tbl_num += 1
            page_no         = self._get_page(element)
            page_tbl_counter[page_no] = page_tbl_counter.get(page_no, 0) + 1
            tbl_idx         = page_tbl_counter[page_no]
            chunk_id        = generate_table_chunk_id(doc_id, page_no, tbl_idx)
            table_index_map[str(global_tbl_num)] = chunk_id

            csv_filename = f"{pdf_stem}_page_{page_no:03d}_tbl_{tbl_idx:03d}.csv"
            csv_path     = tables_dir / csv_filename
            html_filename = f"{pdf_stem}_page_{page_no:03d}_tbl_{tbl_idx:03d}.html"
            html_path     = tables_dir / html_filename

            # Save CSV
            # Initialize with safe defaults BEFORE the try block.
            # If export_to_dataframe() raises (scanned tables, merged-cell
            # layouts, image-only tables), the variables must still be defined
            # when TableChunkMetadata is constructed below — otherwise Python
            # raises UnboundLocalError and aborts extraction for the entire doc.
            row_count = 0
            col_count = 0
            try:
                df = element.export_to_dataframe()
                df.to_csv(csv_path, index=False)
                row_count = len(df)
                col_count = len(df.columns)
            except Exception as e:
                logger.warning(
                    f"Could not export table {global_tbl_num} to CSV: {e} "
                    f"— chunk stored with row_count=0, col_count=0"
                )

            # Get HTML — try Docling first, fallback to pandas-generated HTML
            table_html = None
            try:
                table_html = element.export_to_html()
                if table_html:
                    logger.debug(f"Table {global_tbl_num}: Docling HTML ({len(table_html)} chars)")
            except Exception as e:
                logger.warning(f"Docling HTML failed for table {global_tbl_num}: {e}")

            # Fallback: generate HTML from DataFrame (always works)
            if not table_html:
                try:
                    df_html = element.export_to_dataframe()
                    table_html = df_html.to_html(index=False, border=1, classes="table")
                    logger.info(f"Table {global_tbl_num}: generated HTML from DataFrame ({len(table_html)} chars)")
                except Exception as e2:
                    logger.warning(f"DataFrame HTML fallback failed for table {global_tbl_num}: {e2}")

            # Save HTML file
            if table_html:
                try:
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(table_html)
                    logger.info(f"Saved HTML: {html_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to save HTML file for table {global_tbl_num}: {e}")

            caption = self._get_caption(element)

            chunk = TableChunkMetadata(
                chunk_id      = chunk_id,
                doc_id        = doc_id,
                chunk_index   = global_tbl_num,
                table_html    = table_html,
                table_csv_path = str(Path("tables") / csv_filename),
                html_file_path = str(Path("tables") / html_filename),
                page_number   = page_no,
                row_count     = row_count,
                col_count     = col_count,
                table_caption = caption,
            )
            chunks.append(chunk)

        return chunks, table_index_map

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _create_folders(self, pdf_stem: str) -> Dict[str, Path]:
        root = self.output_dir / pdf_stem
        folders = {
            "root":     root,
            "images":   root / "images",
            "tables":   root / "tables",
            "metadata": root / "metadata",
        }
        for f in folders.values():
            f.mkdir(parents=True, exist_ok=True)
        return folders

    # ──────────────────────────────────────────────────────────────────
    # Code-driven Section Span Extraction
    # ──────────────────────────────────────────────────────────────────

    # ── Noise patterns for heading detection ──────────────────────────
    _HEADING_NOISE = re.compile(
        r'^(figure|fig\.|table|tbl\.|equation|eq\.|appendix)\s*\d',
        re.IGNORECASE
    )
    _IS_PURE_DIGIT   = re.compile(r'^\d+$')
    _MAX_HEADING_LEN = 200
    _MIN_HEADING_LEN = 3

    def _compute_section_spans_from_raw(
        self, doc, total_pages: int
    ) -> List[Dict[str, Any]]:
        """
        Detect section headings from raw Docling elements BEFORE chunking.

        Reads Docling layout labels (section_header, title) directly from
        doc.iterate_items(). These labels are set by Docling's neural layout
        model which reads font size, weight, whitespace, and page position —
        far more reliable than post-chunking regex heuristics.

        Must run before any chunking strategy is applied, because the size
        strategy destroys element boundaries by merging all text into a flat
        word stream, losing the section_header labels.

        Falls back to page-based synthetic sections when no headings found —
        guarantees section_map is never empty, so encoding always has page
        ranges to link chunks against.

        Returns:
            [{"heading": str, "start_page": int, "end_page": int,
              "synthetic"?: True}, ...]
        """
        import re as _re

        _HEADING_LABELS = {"section_header", "title"}
        seen:  Dict[str, int] = {}   # base_heading → count (for dedup)
        order: List[tuple]    = []   # [(heading, page, elem_idx), ...] in document order
        pages_with_text:      set    = set()
        elem_counter: int     = 0    # global element index across doc.iterate_items()

        for element, _level in doc.iterate_items():
            elem_counter += 1
            label    = str(getattr(element, "label", "")).lower()
            page     = self._get_page(element)
            elem_idx = elem_counter

            # Track all pages that have any text content (for fallback)
            raw_text = (getattr(element, "text", "") or "").strip()
            if raw_text:
                pages_with_text.add(page)

            if label not in _HEADING_LABELS:
                continue

            heading = raw_text
            if not heading:
                continue

            # Filter noise — captions, labels, page numbers, single chars
            if len(heading) > self._MAX_HEADING_LEN:
                continue
            if len(heading) < self._MIN_HEADING_LEN:
                continue
            if self._IS_PURE_DIGIT.match(heading):
                continue
            if self._HEADING_NOISE.match(heading):
                continue

            # Deduplicate repeated headings — keep both with suffix
            base = heading
            if base in seen:
                seen[base] += 1
                heading = f"{base} ({seen[base]})"
            else:
                seen[base] = 1

            order.append((heading, page, elem_idx))

        # ── Compute page spans + element index spans ───────────────────
        if order:
            spans = []
            for i, (heading, start_page, start_elem) in enumerate(order):
                if i + 1 < len(order):
                    next_page = order[i + 1][1]
                    next_elem = order[i + 1][2]
                    end_page  = max(start_page, next_page - 1)
                    end_elem  = next_elem - 1
                else:
                    end_page  = max(start_page, int(total_pages or start_page))
                    end_elem  = elem_counter  # last element in doc
                spans.append({
                    "heading":           heading,
                    "start_page":        start_page,
                    "end_page":          end_page,
                    "start_element_idx": start_elem,
                    "end_element_idx":   end_elem,
                })
            logger.info(f"  Section spans detected from raw layout: {len(spans)}")
            return spans

        # ── Fallback: page-based synthetic sections ────────────────────
        logger.warning(
            "  No section headings found in raw layout — "
            "creating page-based synthetic sections"
        )
        spans = [
            {
                "heading":    f"Page {p}",
                "start_page": p,
                "end_page":   p,
                "synthetic":  True,
            }
            for p in sorted(pages_with_text)
        ]
        if not spans:
            logger.warning("  No text content found — creating full-document section")
            spans = [{
                "heading":    "Document",
                "start_page": 1,
                "end_page":   int(total_pages or 1),
                "synthetic":  True,
            }]
        logger.info(f"  Synthetic page sections created: {len(spans)}")
        return spans

    def _get_page(self, element) -> int:
        try:
            if element.prov and element.prov[0]:
                return element.prov[0].page_no
        except Exception:
            pass
        return 1

    def _get_total_pages(self, doc) -> int:
        try:
            return doc.num_pages() if hasattr(doc, "num_pages") else 0
        except Exception:
            return 0

    def _get_caption(self, element) -> Optional[str]:
        try:
            if hasattr(element, "captions") and element.captions:
                text = " ".join(
                    c.text for c in element.captions
                    if hasattr(c, "text") and c.text
                ).strip()
                return text or None
        except Exception:
            pass
        return None

    def _get_title(self, doc, file_path: Path) -> str:
        """
        Extract the real document title using a multi-strategy approach.

        This replaces the original single-strategy method that returned the
        file stem (e.g. "1706.03762v7") whenever PDF metadata was absent,
        which caused doc_embedding_input_text to use the arxiv filename
        instead of the paper title — weakening document-level retrieval.

        Strategy order (stops at first non-empty result):
          1. Docling PDF metadata title field — most reliable when present.
          2. First element with label="title" on page 1 — paper title line
             as detected by Docling's neural layout model.
          3. First element with label="section_header" on page 1 — fallback
             when the title is not labeled as "title" but as a top-level heading.
          4. File stem — last resort only. Logged as a warning so it is visible
             in the pipeline run log.
        """
        # Strategy 1: Docling PDF metadata
        try:
            if doc.metadata and doc.metadata.title:
                title = doc.metadata.title.strip()
                if title:
                    logger.debug(f"  Title via PDF metadata: {title!r}")
                    return title
        except Exception:
            pass

        # Strategy 2 + 3: scan first-page Docling elements for title/heading labels
        try:
            for label_target in ("title", "section_header"):
                for element, _level in doc.iterate_items():
                    page = self._get_page(element)
                    if page > 2:
                        break
                    label = str(getattr(element, "label", "")).lower()
                    if label != label_target:
                        continue
                    text = (getattr(element, "text", "") or "").strip()
                    if not text:
                        continue
                    # Reject strings that look like arxiv IDs, emails, URLs
                    if re.match(r'^\d{4}\.\d{4,5}', text):
                        continue
                    if re.search(r'@|http|www\.', text):
                        continue
                    # Reject very short or all-digit strings
                    if len(text) < 5 or text.isdigit():
                        continue
                    logger.debug(f"  Title via Docling label={label_target!r}: {text!r}")
                    return text
        except Exception as e:
            logger.debug(f"  Title scan failed: {e}")

        # Strategy 4: file stem fallback
        logger.warning(
            f"  Could not extract real title from PDF — "
            f"falling back to file stem: {file_path.stem!r}. "
            f"Check that Docling parsed the first page correctly."
        )
        return file_path.stem

    def _get_author(self, doc) -> Optional[str]:
        """
        Extract author from PDF metadata with heuristic fallback.
        Strategy 1: Docling PDF metadata authors field.
        Strategy 2: Pattern-match author-like lines on first 2 pages.
        """
        import re
        # Strategy 1: Docling metadata
        try:
            if doc.metadata and doc.metadata.authors:
                names = [a.name for a in doc.metadata.authors
                         if getattr(a, "name", None)]
                if names:
                    return ", ".join(names)
        except Exception:
            pass

        # Strategy 2: Heuristic scan of first 2 pages
        try:
            first_page_texts = []
            for element, _level in doc.iterate_items():
                if isinstance(element, (TableItem, PictureItem)):
                    continue
                page_no = self._get_page(element)
                if page_no > 2:
                    break
                text      = getattr(element, "text", "").strip()
                elem_type = str(getattr(element, "label", ""))
                if text:
                    first_page_texts.append((text, elem_type))

            author_re = re.compile(
                r'^[A-Z][a-z]+([\s\-][A-Z][a-z]+)+'   # John Smith / John Von Neumann
                r'|^[A-Z]\.\s*[A-Z][a-z]+'              # J. Smith
                r'|^[A-Z][a-z]+,\s*[A-Z]\.'             # Smith, J.
            )
            candidates = []
            for text, elem_type in first_page_texts:
                if elem_type in ("section_header", "title"):
                    continue
                # Skip affiliation / institution lines
                if re.search(r'\d{4}|@|http|www|university|institute|department|abstract',
                             text, re.IGNORECASE):
                    continue
                if len(text) > 150:
                    continue
                parts = [p.strip() for p in text.split(",")]
                if all(author_re.match(p) for p in parts if p):
                    candidates.append(text)

            if candidates:
                return "; ".join(candidates[:3])
        except Exception as e:
            logger.debug(f"Author heuristic failed: {e}")

        return None
