"""
rq3_extended/router.py

Component-aware repair router for HEMMIR self-correction.

Reads the per-component C score breakdown and decides which
repair action to trigger, or None if the answer is already good.

Routing priority (ordered):
  1. Full escalation   — all three components simultaneously very low
  2. Faith repair      — faithfulness < THR_FAITH_LOW (answer not grounded)
  3. Evidence repair   — evidence_strength < THR_EVSTR_LOW (weak retrieval)
  4. Gap repair        — completeness < THR_COMP_LOW (key aspects missing)
  5. None              — C score above all thresholds, no repair needed
"""

from __future__ import annotations

from typing import Optional

from .config import (
    THR_FAITH_LOW, THR_FAITH_VLOW,
    THR_EVSTR_LOW, THR_EVSTR_VLOW,
    THR_COMP_LOW,  THR_COMP_VLOW,
)

REPAIR_TYPES = ("full_escalation", "faith_repair", "evidence_repair", "gap_repair")


def route(
    faithfulness:      float,
    evidence_strength: float,
    completeness:      float,
) -> Optional[str]:
    """
    Returns the repair type to trigger, or None if no repair needed.

    Routing is worst-signal-first: the component furthest below its threshold
    is repaired first, not the first one that crosses a fixed priority order.

    full_escalation : all three below very-low threshold
    evidence_repair : evidence pool doesn't cover question needs (worst bottleneck
                      when ev_support < THR_EVSTR_LOW — re-retrieval unlocks all others)
    faith_repair    : answer contains unsupported claims → tighten ArgRAG threshold
    gap_repair      : key aspects missing from answer → targeted gap retrieval

    Why evidence_repair before faith_repair:
      faith_repair re-synthesizes from the SAME evidence — if ev_support is
      already the binding constraint, tightening claim threshold only removes
      claims without adding new information, hitting a C score ceiling.
      Evidence_repair expands the evidence pool first, unlocking both
      faithfulness and completeness improvements in one step.
    """
    if (
        faithfulness      < THR_FAITH_VLOW and
        evidence_strength < THR_EVSTR_VLOW and
        completeness      < THR_COMP_VLOW
    ):
        return "full_escalation"

    # Compute normalised deficits (how far below threshold, as fraction of threshold)
    faith_deficit  = max(0.0, THR_FAITH_LOW  - faithfulness)      / THR_FAITH_LOW
    evstr_deficit  = max(0.0, THR_EVSTR_LOW  - evidence_strength) / THR_EVSTR_LOW
    comp_deficit   = max(0.0, THR_COMP_LOW   - completeness)      / THR_COMP_LOW

    # Route to the worst failing signal
    if evstr_deficit > 0 and evstr_deficit >= faith_deficit and evstr_deficit >= comp_deficit:
        return "evidence_repair"

    if faith_deficit > 0:
        return "faith_repair"

    if comp_deficit > 0:
        return "gap_repair"

    return None


def ceiling_diagnosis(
    faithfulness:      float,
    evidence_strength: float,
    completeness:      float,
    repair_history:    list,
) -> Optional[str]:
    """
    Detects repair stalling patterns and explains why.
    Returns a diagnostic message, or None if no stall detected.
    """
    if not repair_history:
        return None

    faith_repairs    = [h for h in repair_history if h.get("repair_type") == "faith_repair"]
    evidence_repairs = [h for h in repair_history if h.get("repair_type") == "evidence_repair"]

    # Pattern 1: faith_repair hit ceiling because ev_support is the real bottleneck
    if faith_repairs and evidence_strength < THR_EVSTR_LOW:
        total_faith_gain = sum(h.get("delta_c", 0) for h in faith_repairs)
        if total_faith_gain < 0.10:
            return (
                f"⚠️ **faith_repair ceiling** — ev_support={evidence_strength:.2f} < {THR_EVSTR_LOW}: "
                f"only {round(evidence_strength*100):.0f}% of question requirements are in the evidence. "
                f"faith_repair cannot add missing information — try **evidence_repair**."
            )

    # Pattern 2: evidence_repair made things worse → document likely doesn't contain the answer
    evidence_degraded = [h for h in evidence_repairs if h.get("delta_c", 0) < 0]
    if evidence_degraded and evidence_strength < THR_EVSTR_LOW:
        return (
            f"🚫 **Information not available** — evidence_repair made the score worse "
            f"(ΔC={evidence_degraded[-1]['delta_c']:+.3f}). "
            f"The document likely does not contain information about this aspect of the question. "
            f"Consider: (1) rephrasing the question to what the document CAN answer, "
            f"(2) checking if a different document covers this topic, "
            f"or (3) accepting the current answer as the best achievable from this source."
        )

    # Pattern 3: multiple repairs all failed — systemic ceiling
    all_failed = [h for h in repair_history
                  if not h.get("improved") and h.get("repair_type") != "faith_repair"]
    if len(all_failed) >= 2:
        return (
            f"🚫 **Repair ceiling reached** — {len(repair_history)} repairs attempted, "
            f"none improved the score meaningfully. "
            f"The current answer (C={max(h.get('after_c', 0) for h in repair_history):.3f}) "
            f"is the best achievable from the available evidence."
        )

    return None


def describe_route(repair_type: Optional[str]) -> str:
    descriptions = {
        "full_escalation": "All signals very low → MultiQuery + expanded K + full R2-G",
        "faith_repair":    "Low faithfulness → re-synthesize with stricter claim threshold",
        "evidence_repair": "Low evidence strength → MultiQuery re-retrieval",
        "gap_repair":      "Low completeness → targeted gap-query retrieval",
        None:              "No repair needed — C score above all thresholds",
    }
    return descriptions.get(repair_type, "Unknown")
