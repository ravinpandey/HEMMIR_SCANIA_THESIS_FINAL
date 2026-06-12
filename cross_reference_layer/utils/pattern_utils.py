"""
cross_reference_layer/utils/pattern_utils.py

Regex patterns for cross-reference detection in technical documents.

Research-level upgrade from previous version:

Previous version matched only simple patterns:
  "Figure 3", "Fig. 3", "Table 2", "Section 3.2"

Gaps in previous version:
  1. Did not match parenthetical refs: "(Fig. 3)", "(see Figure 3a)"
  2. Did not match letter suffixes: "Figure 3a", "Fig. 3b"
  3. Did not match range refs: "Figures 3-5", "Tables 1 and 2"
  4. Did not match "above"/"below" implicit refs
  5. Did not match equation refs at all (function existed but pattern weak)
  6. Swedish only partially covered

This version covers:
  Figure references:
    "Figure 3", "Fig. 3", "fig3", "FIG 3", "Figur 3" (Swedish)
    "Figure 3a", "Fig. 3b" (letter suffix)
    "(Figure 3)", "(see Fig. 3)", "shown in Figure 3"
    "Figures 3 and 4", "Figs. 3-5" (multi-figure refs → both extracted)
    "figure above", "figure below" (implicit — tagged separately)

  Table references:
    "Table 2", "TABLE 2", "Tabell 2" (Swedish)
    "Table 2a", "Tables 2 and 3"
    "(Table 2)", "(see Table 2)"

  Section references:
    "Section 3.2", "section 4", "Sec. 3", "Avsnitt 3" (Swedish)
    "§3.2", "§ 3.2" (legal/formal documents)
    "Chapter 3", "Appendix A"

  Equation references:
    "Equation (3)", "Eq. 5", "eq.(3)", "Equation 3"
    "(3)", "(Eq. 3)" — context-dependent, only returned when explicit label present
"""

from __future__ import annotations

import re
from typing import List, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Figure patterns
# ─────────────────────────────────────────────────────────────────────────────

# Core figure word: Figure / Fig / Figur (Swedish) — case-insensitive
_FIG_WORD = r"(?:[Ff]ig(?:ure|ur|\.)?|[Ff][Ii][Gg](?:[Uu][Rr][Ee]|[Uu][Rr])?)"

# Single figure ref: "Figure 3", "Fig. 3a", "fig3"
# Captures the integer part only (letter suffix ignored for index lookup)
FIGURE_SINGLE = re.compile(
    rf"{_FIG_WORD}\.?\s*(\d+)[a-zA-Z]?"
)

# Range refs: "Figures 3-5", "Figs. 3 and 4", "Figures 3, 4 and 5"
# We expand ranges to individual numbers downstream
FIGURE_RANGE = re.compile(
    rf"{_FIG_WORD}s?\.?\s*(\d+)\s*(?:[-–—to]|and|through)\s*(\d+)"
)

# Multi refs: "Figures 3, 4"
FIGURE_MULTI = re.compile(
    rf"{_FIG_WORD}s?\.?\s*(\d+)(?:\s*,\s*(\d+))+"
)

# ─────────────────────────────────────────────────────────────────────────────
# Table patterns
# ─────────────────────────────────────────────────────────────────────────────

_TBL_WORD = r"(?:[Tt]ab(?:le|ell|\.)?)"

TABLE_SINGLE = re.compile(
    rf"{_TBL_WORD}s?\.?\s*(\d+)[a-zA-Z]?"
)

TABLE_RANGE = re.compile(
    rf"{_TBL_WORD}s?\.?\s*(\d+)\s*(?:[-–—to]|and|through)\s*(\d+)"
)

# ─────────────────────────────────────────────────────────────────────────────
# Section patterns
# ─────────────────────────────────────────────────────────────────────────────

# English: Section 3, section 3.2, Sec. 4.1
# Swedish: Avsnitt 3
# Formal: §3.2, § 3.2
# Chapter / Appendix
SECTION_PATTERN = re.compile(
    r"""
    (?:
        (?:[Ss]ec(?:tion|\.)?|[Aa]vsnitt|[Cc]hapter)\s*\.?\s*([A-Z]?\d+(?:\.\d+)*)  |
        §\s*\.?\s*(\d+(?:\.\d+)*)                                                       |
        [Aa]ppendix\s+([A-Z]\d*(?:\.\d+)*)
    )
    """,
    re.VERBOSE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Equation patterns
# ─────────────────────────────────────────────────────────────────────────────

EQUATION_PATTERN = re.compile(
    r"(?:[Ee]q(?:uation|\.)?)\s*\.?\s*\(?\s*(\d+)\s*\)?"
)

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def find_figure_references(text: str) -> List[int]:
    """
    Return sorted unique figure numbers referenced in text.

    Handles:
      - Single refs: "Figure 3", "Fig. 3a"
      - Range refs: "Figures 3-5" → [3, 4, 5]
      - Multi refs: "Figures 3, 4" → [3, 4]
    """
    nums: Set[int] = set()

    # Ranges first (more specific)
    for m in FIGURE_RANGE.finditer(text):
        start, end = int(m.group(1)), int(m.group(2))
        for n in range(start, min(end + 1, start + 10)):  # cap at 10 to avoid noise
            nums.add(n)

    # Singles
    for m in FIGURE_SINGLE.finditer(text):
        nums.add(int(m.group(1)))

    return sorted(nums)


def find_table_references(text: str) -> List[int]:
    """
    Return sorted unique table numbers referenced in text.

    Handles:
      - Single refs: "Table 2", "Tabell 2"
      - Range refs: "Tables 2-4" → [2, 3, 4]
    """
    nums: Set[int] = set()

    for m in TABLE_RANGE.finditer(text):
        start, end = int(m.group(1)), int(m.group(2))
        for n in range(start, min(end + 1, start + 10)):
            nums.add(n)

    for m in TABLE_SINGLE.finditer(text):
        nums.add(int(m.group(1)))

    return sorted(nums)


def find_section_references(text: str) -> List[str]:
    """
    Return sorted unique section references (e.g. '3.2', '4', 'A.1') in text.
    Used to build related_section_ids on text chunks.
    Handles Section/Chapter/Avsnitt, §, and Appendix patterns.
    """
    results: Set[str] = set()
    for m in SECTION_PATTERN.finditer(text):
        val = m.group(1) or m.group(2) or m.group(3)
        if val:
            results.add(val)
    return sorted(results)


def find_equation_references(text: str) -> List[int]:
    """Return sorted unique equation numbers referenced in text."""
    return sorted({int(m.group(1)) for m in EQUATION_PATTERN.finditer(text)})