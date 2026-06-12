"""
enrichment_layer/enrichers/video_segment_enricher.py

Contextual chunking for video ASR segments.

Key difference from text_enricher:
  Window = entire SCENE (all segments in the same structure_unit_id, up to 10 segs)
  NOT ±2 chunks. Video context is temporal — the scene is the natural unit.

System prompt: full transcript from transcripts/{stem}_full_transcript.txt (cached).

Per segment:
  1. Collect all segments in the same scene (structure_unit_id)
  2. Build scene_context from their transcript_text (trimmed)
  3. Build user prompt: scene_context + current segment
  4. LLM returns situated_context (2-3 sentences)
  5. Set contextual_summary (do NOT touch transcript_text)
  6. Upgrade local_context to full scene context

Writes to segment dict:
  segment["contextual_summary"]
  segment["contextual_summary_confidence"]
  segment["local_context"]              ← upgraded from naive prev+next to scene context
  segment["detected_codes"]
  segment["evidence_role"]
  segment["salience_score"]

Does NOT modify: transcript_text, text_original_content, word_timestamps, asr_confidence
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from enrichment_layer.enrichers._parse_utils import (
    extract_field,
    extract_float,
    extract_list,
    invoke_with_retry,
)
from enrichment_layer.utils.llm_client import LLMClient

_REQUIRED      = ["SITUATED_CONTEXT", "CONFIDENCE"]
_SCENE_CHARS   = 800   # chars per scene context
_SEG_CHARS     = 600   # chars from current segment transcript
_FULL_TRANS_CHARS = 10000


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """\
You are analysing a video transcript for an industrial retrieval system.
Your task is to situate each segment so it is self-sufficient for search.

<full_transcript>
{full_transcript}
</full_transcript>\
"""

SEGMENT_PROMPT = """\
Situate this video segment in context.

Video title: {video_title}
Scene: {scene_title} (segment {seg_index} of {total_segments})
Timestamp: {start_time:.1f}s – {end_time:.1f}s

Scene context (other segments in this scene):
{scene_context}

<current_segment>
{transcript}
</current_segment>

Write 2-3 sentences (SITUATED_CONTEXT) that:
  - State what topic is being discussed at this point in the video
  - Describe the specific content of this segment
  - Note what question a user would ask to find this segment

Then extract any technical terms, procedure names, part numbers, or specifications mentioned.

Respond in EXACTLY this format:
SITUATED_CONTEXT: <2-3 sentences>
CONFIDENCE: <0.0-1.0>
CODES: <comma-separated technical terms, or "none">
EVIDENCE_ROLE: <one of: explanation / demonstration / specification / context>
EVIDENCE_ROLE_CONFIDENCE: <0.0-1.0 — how certain are you about the role classification>
SALIENCE: <0.0-1.0>\
"""

RETRY_PROMPT = """\
Your previous response was incomplete. Missing fields: {missing}

Previous response:
{prev_response}

Respond in EXACTLY this format:
SITUATED_CONTEXT: <2-3 sentences>
CONFIDENCE: <0.0-1.0>
CODES: <comma-separated terms or "none">
EVIDENCE_ROLE: <explanation / demonstration / specification / context>
SALIENCE: <0.0-1.0>\
"""


class VideoSegmentEnricher:

    def __init__(self, llm: LLMClient, delay: float = 0.2):
        self.llm   = llm
        self.delay = delay

    def enrich(
        self,
        video_segments:  List[Dict[str, Any]],
        structure_units: List[Dict[str, Any]],
        doc_metadata:    Dict[str, Any],
        doc_output_dir:  Path,
    ) -> List[Dict[str, Any]]:
        """
        Enrich all video segments with contextual summaries.
        Idempotent: skips segments that already have contextual_summary.
        """
        to_enrich = [s for s in video_segments if not s.get("contextual_summary")]
        if not to_enrich:
            logger.info("  VideoSegmentEnricher: all segments already enriched")
            return video_segments

        logger.info(
            f"  VideoSegmentEnricher: enriching {len(to_enrich)}/{len(video_segments)} segments"
        )

        # ── Load full transcript for system prompt ─────────────────────
        full_transcript = self._load_transcript(doc_output_dir, doc_metadata)

        # ── Build structure unit lookup ────────────────────────────────
        su_lookup: Dict[str, Dict] = {
            su["structure_unit_id"]: su
            for su in structure_units
            if su.get("structure_unit_id")
        }

        # ── Group segments by scene (structure_unit_id) ────────────────
        scene_groups: Dict[str, List[Dict]] = defaultdict(list)
        for seg in sorted(video_segments, key=lambda s: s.get("segment_index", 0)):
            su_id = seg.get("structure_unit_id", "")
            scene_groups[su_id].append(seg)

        system = SYSTEM_TEMPLATE.format(
            full_transcript=full_transcript[:_FULL_TRANS_CHARS]
        )
        video_title = doc_metadata.get("doc_title", "")
        total_segs  = len(video_segments)

        for seg in video_segments:
            if seg.get("contextual_summary"):
                continue

            seg_id  = seg.get("chunk_id", "?")
            su_id   = seg.get("structure_unit_id", "")
            su      = su_lookup.get(su_id, {})
            scene   = scene_groups.get(su_id, [])

            # ── Build scene context ────────────────────────────────────
            scene_context_parts = []
            for other in scene:
                if other.get("chunk_id") == seg.get("chunk_id"):
                    continue
                text = (other.get("transcript_text") or "")[:200]
                ts   = other.get("start_time_s", 0)
                scene_context_parts.append(f"[{ts:.1f}s] {text}")
            scene_context = "\n".join(scene_context_parts[:6]) or "(only segment in scene)"

            # Update local_context to scene context (better than prev+next)
            seg["local_context"] = scene_context[:_SCENE_CHARS]

            # ── Build prompt ───────────────────────────────────────────
            prompt = SEGMENT_PROMPT.format(
                video_title   = video_title,
                scene_title   = su.get("title", su_id),
                seg_index     = seg.get("segment_index", 0) + 1,
                total_segments= total_segs,
                start_time    = seg.get("start_time_s", 0.0),
                end_time      = seg.get("end_time_s", 0.0),
                scene_context = scene_context[:_SCENE_CHARS],
                transcript    = (seg.get("transcript_text") or "")[:_SEG_CHARS],
            )

            # ── LLM call with retry ────────────────────────────────────
            response = invoke_with_retry(
                llm_client            = self.llm,
                first_prompt          = prompt,
                retry_prompt_template = RETRY_PROMPT,
                required_keys         = _REQUIRED,
                use_cache             = True,
                system_doc            = system,
            )

            if not response:
                logger.warning(f"  VideoSegmentEnricher: {seg_id} failed after retries")
                continue

            situated = extract_field(response, "SITUATED_CONTEXT") or ""
            if situated:
                seg["contextual_summary"]                = situated
                seg["contextual_summary_confidence"]     = extract_float(response, "CONFIDENCE", 0.5)
                seg["detected_codes"]                    = extract_list(response, "CODES")
                seg["evidence_role"]                     = (
                    extract_field(response, "EVIDENCE_ROLE") or "context"
                ).lower().strip()
                seg["evidence_role_confidence"]          = extract_float(response, "EVIDENCE_ROLE_CONFIDENCE", 0.5)
                seg["salience_score"]                    = extract_float(response, "SALIENCE", 0.5)

            logger.debug(f"  VideoSegmentEnricher: {seg_id} enriched")
            time.sleep(self.delay)

        enriched = sum(1 for s in video_segments if s.get("contextual_summary"))
        logger.success(
            f"  VideoSegmentEnricher: {enriched}/{len(video_segments)} enriched"
        )
        return video_segments

    def _load_transcript(self, doc_output_dir: Path, doc_meta: Dict) -> str:
        stem = doc_meta.get("doc_title", "")
        path = doc_output_dir / "transcripts" / f"{stem}_full_transcript.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        # Fallback: join segment transcripts
        return ""