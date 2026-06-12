"""
cross_reference_layer/utils/video_linker.py

Video-specific cross-reference utilities.

Three functions:

1. build_scene_sibling_map
   Groups segment chunk_ids by structure_unit_id (scene).
   Returns {chunk_id: [sibling_chunk_ids]} for every segment.
   Written to segment["scene_sibling_ids"].

2. validate_frame_parent_links
   Checks that every frame's parent_id points to a real segment.
   Logs warnings for orphaned frames.
   Returns clean {frame_chunk_id: segment_dict} lookup.

3. build_segment_frame_map
   Confirms keyframe_ids on each segment match actual frames
   that have that segment as parent_id.
   Repairs mismatches in-place.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from loguru import logger


def build_scene_sibling_map(
    video_segments: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Group segments by scene (structure_unit_id).
    Returns {chunk_id: [sibling_chunk_ids excluding self]}.
    """
    scene_groups: Dict[str, List[str]] = defaultdict(list)
    for seg in video_segments:
        su_id    = seg.get("structure_unit_id", "")
        chunk_id = seg.get("chunk_id", "")
        if su_id and chunk_id:
            scene_groups[su_id].append(chunk_id)

    sibling_map: Dict[str, List[str]] = {}
    for su_id, ids in scene_groups.items():
        for cid in ids:
            sibling_map[cid] = [other for other in ids if other != cid]

    return sibling_map


def validate_frame_parent_links(
    video_segments: List[Dict[str, Any]],
    video_frames:   List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict], List[str]]:
    """
    Validate that every frame's parent_id points to a real segment.

    Returns:
        (parent_lookup: {frame_chunk_id: segment_dict},
         orphaned_frame_ids: list of frame_ids with no valid parent)
    """
    seg_lookup  = {s["chunk_id"]: s for s in video_segments if s.get("chunk_id")}
    parent_lookup: Dict[str, Dict] = {}
    orphaned: List[str] = []

    for frame in video_frames:
        frame_id  = frame.get("chunk_id", "?")
        parent_id = frame.get("parent_id", "")
        if parent_id and parent_id in seg_lookup:
            parent_lookup[frame_id] = seg_lookup[parent_id]
        else:
            orphaned.append(frame_id)
            logger.warning(
                f"  VideoLinker: frame {frame_id} has no valid parent "
                f"(parent_id={parent_id!r})"
            )

    return parent_lookup, orphaned


def build_segment_frame_map(
    video_segments: List[Dict[str, Any]],
    video_frames:   List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Confirm and repair keyframe_ids on each segment.

    The ingestion layer sets keyframe_ids at extraction time.
    This function verifies they match the frames that claim this segment as parent.
    Repairs any mismatches in-place.

    Returns the mutated video_segments list.
    """
    # Build {segment_chunk_id: [frame_chunk_ids]} from frames side
    frames_by_parent: Dict[str, List[str]] = defaultdict(list)
    for frame in video_frames:
        parent_id = frame.get("parent_id", "")
        frame_id  = frame.get("chunk_id", "")
        if parent_id and frame_id:
            frames_by_parent[parent_id].append(frame_id)

    repaired = 0
    for seg in video_segments:
        seg_id      = seg.get("chunk_id", "")
        declared    = set(seg.get("keyframe_ids", []))
        actual      = set(frames_by_parent.get(seg_id, []))

        if declared != actual:
            logger.debug(
                f"  VideoLinker: {seg_id} keyframe_ids mismatch "
                f"declared={len(declared)} actual={len(actual)} — repairing"
            )
            seg["keyframe_ids"] = sorted(actual)
            repaired += 1

    if repaired:
        logger.info(f"  VideoLinker: repaired keyframe_ids on {repaired} segments")

    return video_segments


def link_schema_to_rowgroups(
    table_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For CSV/XLSX: link each row group chunk to its sheet's schema chunk.
    Writes schema_chunk_id on every row group.
    Returns mutated list.
    """
    schema_by_su: Dict[str, str] = {}
    for c in table_chunks:
        if c.get("is_schema_chunk"):
            su_id = c.get("structure_unit_id", "")
            if su_id:
                schema_by_su[su_id] = c["chunk_id"]

    for c in table_chunks:
        if not c.get("is_schema_chunk"):
            su_id = c.get("structure_unit_id", "")
            schema_id = schema_by_su.get(su_id, "")
            if schema_id:
                c["schema_chunk_id"] = schema_id

    return table_chunks
