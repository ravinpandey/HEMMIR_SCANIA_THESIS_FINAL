"""
ingestion_layer/utils/chunking_utils.py

Three chunking strategies for text extraction from Docling documents.

STRATEGY 1 — paragraph (default, recommended for structured docs):
    One chunk per paragraph/text block as identified by Docling.
    Preserves document structure.
    Best for: technical documents with clear paragraph breaks.
    Typical chunk size: 50-300 tokens.

STRATEGY 2 — title:
    Groups all text under a section heading into ONE chunk.
    Heading itself becomes the section_title field.
    Best for: documents with clear headings (e.g., "3.2 Firewall Rules").
    Typical chunk size: 200-1500 tokens (whole sections).

STRATEGY 3 — size:
    Fixed token count with overlap, regardless of document structure.
    Classic RAG approach. Use --chunk-size and --chunk-overlap to tune.
    Best for: experiments to find optimal chunk size for retrieval.
    Typical chunk size: 128, 256, 512, 1024 tokens.

EXPERIMENT GUIDE:
    Run pipeline 3 times and compare retrieval metrics:
    python run_pipeline.py --chunk-strategy paragraph
    python run_pipeline.py --chunk-strategy title
    python run_pipeline.py --chunk-strategy size --chunk-size 256
    python run_pipeline.py --chunk-strategy size --chunk-size 512
    python run_pipeline.py --chunk-strategy size --chunk-size 1024

Changes from previous version:
    - Bug fix in chunk_by_size: end_word now uses words_per_chunk (word units)
      instead of chunk_size (token units). Using chunk_size caused end_word to
      point ~25% past the chunk's actual last word, placing end_element_index
      in the next chunk's territory and producing wrong section assignment in
      the cross-reference layer for all --chunk-strategy size documents.
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RawChunk:
    """Raw text chunk before metadata assignment."""
    text:          str
    section_title: Optional[str] = None    # Only filled for title strategy
    page_number:   Optional[int] = None
    token_count:   int = 0
    element_index:     Optional[int] = None  # start position in doc.iterate_items()
    end_element_index: Optional[int] = None  # end position (inclusive)
                                              # Both used by encoding for overlap-based
                                              # section assignment. Match the counter in
                                              # _compute_section_spans_from_raw().


def estimate_tokens(text: str) -> int:
    """
    Estimate token count. Approximation: 1 token ≈ 4 characters.
    Good enough for chunking decisions without loading a tokenizer.
    """
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Paragraph Chunking (Docling native)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_by_paragraph(raw_texts: List[Tuple]) -> List[RawChunk]:
    """
    One chunk per paragraph/text block from Docling.
    This is the most structure-aware approach — Docling already
    understands where paragraphs begin and end.

    Args:
        raw_texts: List of (text, page_number, elem_type, elem_index) tuples.

    Returns:
        List of RawChunk objects, one per paragraph.
    """
    chunks = []
    for item in raw_texts:
        text     = item[0].strip()
        page_no  = item[1]
        elem_idx = item[3] if len(item) > 3 else None
        if not text:
            continue
        chunks.append(RawChunk(
            text              = text,
            page_number       = page_no,
            token_count       = estimate_tokens(text),
            element_index     = elem_idx,
            end_element_index = elem_idx,  # paragraph = single element
        ))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Title-Based Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_by_title(
    raw_texts: List[Tuple],  # (text, page_no, element_type, elem_index)
) -> List[RawChunk]:
    """
    Groups all text under each section heading into one chunk.
    The heading text becomes the section_title field.
    element_index of the first element in each group is stored on the chunk.

    Args:
        raw_texts: List of (text, page_number, element_type, elem_index) tuples.

    Returns:
        List of RawChunk objects, one per section.
    """
    chunks         = []
    current_title    = None
    current_texts    = []
    current_page     = None
    current_elem     = None
    current_end_elem = None

    for item in raw_texts:
        text      = item[0].strip()
        page_no   = item[1]
        elem_type = item[2]
        elem_idx  = item[3] if len(item) > 3 else None
        if not text:
            continue

        is_heading = (
            elem_type in ("section_header", "title", "heading") or
            _looks_like_heading(text)
        )

        if is_heading:
            if current_texts:
                combined = "\n".join(current_texts)
                chunks.append(RawChunk(
                    text          = combined,
                    section_title = current_title,
                    page_number   = current_page,
                    token_count   = estimate_tokens(combined),
                    element_index     = current_elem,
                    end_element_index = current_end_elem,
                ))
            current_title     = text
            current_texts     = []
            current_page      = page_no
            current_elem      = elem_idx
            current_end_elem  = elem_idx
        else:
            if current_page is None:
                current_page = page_no
            if current_elem is None:
                current_elem = elem_idx
            current_end_elem = elem_idx  # track last element in this section group
            current_texts.append(text)

    if current_texts:
        combined = "\n".join(current_texts)
        chunks.append(RawChunk(
            text              = combined,
            section_title     = current_title,
            page_number       = current_page,
            token_count       = estimate_tokens(combined),
            element_index     = current_elem,
            end_element_index = current_end_elem,
        ))

    return chunks


def _looks_like_heading(text: str) -> bool:
    """
    Heuristic to detect headings when Docling element type is ambiguous.
    Checks: short text, starts with number pattern, all caps, etc.
    """
    text = text.strip()
    if len(text) > 120:
        return False
    if re.match(r'^\d+(\.\d+)*[\.\s]', text):   # "3.2 Firewall Rules"
        return True
    if text.isupper() and len(text) < 80:        # "FIREWALL CONFIGURATION"
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Fixed Size Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_by_size(
    raw_texts:    List[Tuple],
    chunk_size:   int = 512,
    chunk_overlap: int = 50,
) -> List[RawChunk]:
    """
    Fixed token-size chunks with overlap.
    Classic RAG approach — useful for experiments to find optimal size.

    Strategy:
        1. Concatenate all paragraphs into one big text stream
        2. Split into chunks of approximately chunk_size tokens
        3. Add chunk_overlap tokens from previous chunk at the start

    This ignores document structure intentionally — useful for comparison:
    Does structure-aware chunking (paragraph/title) outperform fixed-size?

    Args:
        raw_texts:     List of (text, page_number) tuples.
        chunk_size:    Target tokens per chunk.
        chunk_overlap: Overlap tokens between consecutive chunks.

    Returns:
        List of RawChunk objects with approximately chunk_size tokens each.
    """
    # Flatten all text into words, tracking page and element boundaries
    all_words  = []
    page_at    = {}   # word_index → page_number
    elem_at    = {}   # word_index → element_index

    for item in raw_texts:
        text     = item[0]
        page_no  = item[1]
        elem_idx = item[3] if len(item) > 3 else None
        words = text.strip().split()
        for w in words:
            page_at[len(all_words)] = page_no
            elem_at[len(all_words)] = elem_idx
            all_words.append(w)

    if not all_words:
        return []

    # Convert token count to word count (approx: 1 token ≈ 0.75 words)
    words_per_chunk   = max(10, int(chunk_size * 0.75))
    words_overlap     = max(0,  int(chunk_overlap * 0.75))

    chunks = []
    start  = 0

    while start < len(all_words):
        end        = min(start + words_per_chunk, len(all_words))
        chunk_words = all_words[start:end]
        chunk_text  = " ".join(chunk_words)
        page_no     = page_at.get(start)

        # end_word must use words_per_chunk (word units), NOT chunk_size (token
        # units). chunk_size is tokens; words_per_chunk = int(chunk_size * 0.75)
        # converts to words. Using chunk_size here made end_word point ~25%
        # beyond the chunk's last actual word, placing end_element_index in the
        # next chunk's territory and causing wrong section assignment downstream.
        end_word = min(start + words_per_chunk - 1, len(all_words) - 1)
        chunks.append(RawChunk(
            text              = chunk_text,
            page_number       = page_no,
            token_count       = estimate_tokens(chunk_text),
            element_index     = elem_at.get(start),
            end_element_index = elem_at.get(end_word),
        ))

        # Move forward by chunk_size minus overlap
        step   = words_per_chunk - words_overlap
        start += max(1, step)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher — called by PDFExtractor
# ─────────────────────────────────────────────────────────────────────────────

def apply_chunking_strategy(
    raw_items:      list,       # Items from Docling doc.iterate_items()
    strategy:       str,        # "paragraph" | "title" | "size"
    chunk_size:     int = 512,
    chunk_overlap:  int = 50,
) -> List[RawChunk]:
    """
    Main dispatcher — applies the requested chunking strategy to Docling items.

    Args:
        raw_items:    List of (text, page_no) or (text, page_no, type) tuples
        strategy:     Chunking strategy name
        chunk_size:   Token target for size strategy
        chunk_overlap: Overlap tokens for size strategy

    Returns:
        List of RawChunk objects ready for metadata assignment.
    """
    strategy = strategy.lower().strip()

    if strategy == "paragraph":
        return chunk_by_paragraph(raw_items)

    elif strategy == "title":
        return chunk_by_title(raw_items)

    elif strategy == "size":
        return chunk_by_size(raw_items, chunk_size, chunk_overlap)

    else:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            f"Choose from: paragraph, title, size"
        )
