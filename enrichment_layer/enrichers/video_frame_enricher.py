"""
enrichment_layer/enrichers/video_frame_enricher.py

Vision LLM enrichment for video keyframes.

Each keyframe knows its parent_id → VideoSegmentChunk.
The enrichment prompt sends:
  - The frame image
  - The parent segment's transcript_text as context
  - Frame position within the segment (1–4)
  - Timestamp

This cross-modal grounding — visual frame + spoken audio — is what makes
video frame enrichment unique vs regular image enrichment.

Writes to frame dict:
  frame["image_caption"]
  frame["image_caption_confidence"]
  frame["contextual_summary"]
  frame["contextual_summary_confidence"]
  frame["frame_role"]   ← read by build_image_views() in encoding layer
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

ALLOWED_FRAME_ROLES = {
    "title_slide", "diagram", "demo", "talking_head",
    "data_view", "whiteboard", "equipment", "other",
}

_REQUIRED = ["CAPTION", "SUMMARY"]

FRAME_PROMPT = """\
This is keyframe {frame_position} of {total_frames} extracted from a video segment.

Timestamp: {timestamp_s:.1f}s
Video: {video_title}
Scene: {scene_title}

The speaker is saying at this moment:
"{transcript}"

Describe what is visible in this frame and how it relates to the spoken content.

Respond in EXACTLY this format:
CAPTION: <1-sentence: what is visible in the frame>
SUMMARY: <2-3 sentences: how this visual relates to the spoken content and the video's topic>
FRAME_ROLE: <one of: title_slide / diagram / demo / talking_head / data_view / whiteboard / equipment / other>
CAPTION_CONFIDENCE: <0.0-1.0>
SUMMARY_CONFIDENCE: <0.0-1.0>
FRAME_ROLE_CONFIDENCE: <0.0-1.0 — how certain are you about the frame role classification>\
"""

RETRY_PROMPT = """\
Your previous response was incomplete. Missing fields: {missing}

Previous response:
{prev_response}

Respond in EXACTLY this format:
CAPTION: <1-sentence>
SUMMARY: <2-3 sentences>
FRAME_ROLE: <frame role>
CAPTION_CONFIDENCE: <0.0-1.0>
SUMMARY_CONFIDENCE: <0.0-1.0>
FRAME_ROLE_CONFIDENCE: <0.0-1.0>\
"""


class VideoFrameEnricher:

    def __init__(self, llm: LLMClient, delay: float = 1.0):
        self.llm   = llm
        self.delay = delay

    def enrich(
        self,
        video_frames:    List[Dict[str, Any]],
        video_segments:  List[Dict[str, Any]],
        structure_units: List[Dict[str, Any]],
        doc_metadata:    Dict[str, Any],
        doc_output_dir:  Path,
    ) -> List[Dict[str, Any]]:
        """
        Enrich video frames using vision LLM + parent segment transcript.
        Idempotent: skips frames that already have contextual_summary.
        """
        to_enrich = [f for f in video_frames if not f.get("contextual_summary")]
        if not to_enrich:
            logger.info("  VideoFrameEnricher: all frames already enriched")
            return video_frames

        logger.info(
            f"  VideoFrameEnricher: enriching {len(to_enrich)}/{len(video_frames)} frames"
        )

        # Build segment lookup: chunk_id → segment dict
        seg_lookup: Dict[str, Dict] = {
            s["chunk_id"]: s for s in video_segments if s.get("chunk_id")
        }
        su_lookup: Dict[str, Dict] = {
            su["structure_unit_id"]: su
            for su in structure_units
            if su.get("structure_unit_id")
        }

        video_title   = doc_metadata.get("doc_title", "")
        total_frames  = 4  # uniform keyframes per segment (from extractor config)

        for frame in to_enrich:
            frame_id = frame.get("chunk_id", "?")
            try:
                self._enrich_one(
                    frame, seg_lookup, su_lookup, doc_output_dir,
                    video_title, total_frames,
                )
            except Exception as e:
                logger.error(f"  VideoFrameEnricher: {frame_id} failed: {e}")
            time.sleep(self.delay)

        enriched = sum(1 for f in video_frames if f.get("contextual_summary"))
        logger.success(
            f"  VideoFrameEnricher: {enriched}/{len(video_frames)} enriched"
        )
        return video_frames

    def _enrich_one(
        self,
        frame:        Dict[str, Any],
        seg_lookup:   Dict[str, Dict],
        su_lookup:    Dict[str, Dict],
        doc_output_dir: Path,
        video_title:  str,
        total_frames: int,
    ) -> None:
        frame_id = frame.get("chunk_id", "?")

        # ── Load image ─────────────────────────────────────────────────
        img_b64, media_type = self._load_image(frame, doc_output_dir)
        if not img_b64:
            logger.warning(f"  VideoFrameEnricher: {frame_id} — image not found")
            return

        # ── Look up parent segment ─────────────────────────────────────
        parent_id = frame.get("parent_id", "")
        parent    = seg_lookup.get(parent_id, {})
        transcript = (parent.get("transcript_text") or "")[:400]

        su_id      = frame.get("structure_unit_id", "")
        su         = su_lookup.get(su_id, {})
        scene_title = su.get("title", su_id)

        # ── Build prompt ───────────────────────────────────────────────
        prompt = FRAME_PROMPT.format(
            frame_position = frame.get("frame_position", 1),
            total_frames   = total_frames,
            timestamp_s    = frame.get("timestamp_s", 0.0),
            video_title    = video_title,
            scene_title    = scene_title,
            transcript     = transcript or "(no transcript available)",
        )

        # ── LLM vision call with retry ─────────────────────────────────
        response = self._invoke_with_retry(img_b64, media_type, prompt)
        if not response:
            logger.warning(f"  VideoFrameEnricher: {frame_id} — enrichment failed")
            return

        caption    = extract_field(response, "CAPTION") or ""
        summary    = extract_field(response, "SUMMARY") or ""
        frame_role      = (extract_field(response, "FRAME_ROLE") or "other").lower().strip()
        cap_conf        = extract_float(response, "CAPTION_CONFIDENCE",    0.5)
        sum_conf        = extract_float(response, "SUMMARY_CONFIDENCE",    0.5)
        role_conf       = extract_float(response, "FRAME_ROLE_CONFIDENCE", 0.5)

        if frame_role not in ALLOWED_FRAME_ROLES:
            frame_role = "other"

        if caption:
            frame["image_caption"]                 = caption
            frame["image_caption_confidence"]      = cap_conf
        if summary:
            frame["contextual_summary"]            = summary
            frame["contextual_summary_confidence"] = sum_conf
        frame["frame_role"]            = frame_role
        frame["frame_role_confidence"] = role_conf

        logger.debug(
            f"  VideoFrameEnricher: {frame_id} | role={frame_role} | "
            f"ts={frame.get('timestamp_s',0):.1f}s"
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
                    max_tokens=500,
                )
            except Exception as e:
                logger.warning(f"  VideoFrameEnricher vision attempt {attempt}: {e}")
                if attempt <= max_retries:
                    time.sleep(1.5 * attempt)
                continue

            if not missing_keys(response, _REQUIRED):
                return response

            if attempt <= max_retries:
                prompt = RETRY_PROMPT.format(
                    missing=", ".join(missing_keys(response, _REQUIRED)),
                    prev_response=response[:400],
                )
                time.sleep(1.5)

        return response

    def _load_image(
        self, frame: Dict, doc_output_dir: Path
    ) -> tuple[Optional[str], str]:
        rel = frame.get("image_path", "")
        if not rel:
            return None, ""
        path = doc_output_dir / rel
        if not path.exists():
            return None, ""
        try:
            ext        = path.suffix.lower()
            media_type = "image/png" if ext == ".png" else "image/jpeg"
            data       = base64.b64encode(path.read_bytes()).decode()
            return data, media_type
        except Exception as e:
            logger.warning(f"  Cannot load frame {path}: {e}")
            return None, ""