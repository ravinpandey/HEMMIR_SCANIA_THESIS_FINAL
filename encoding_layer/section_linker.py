from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Any, List


def build_neighbors(chunk_ids: List[str]) -> Dict[str, List[str]]:
    out = {}
    for i, cid in enumerate(chunk_ids):
        n = []
        if i - 1 >= 0:
            n.append(chunk_ids[i - 1])
        if i + 1 < len(chunk_ids):
            n.append(chunk_ids[i + 1])
        out[cid] = n
    return out


def link_chunks_to_structure_units(
    structure_units: List[Dict[str, Any]],
    text_dicts: List[Dict[str, Any]],
    image_dicts: List[Dict[str, Any]],
    table_dicts: List[Dict[str, Any]],
    video_seg_dicts: List[Dict[str, Any]] | None = None,
    video_frame_dicts: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    unit_map = {u["structure_unit_id"]: u for u in structure_units if u.get("structure_unit_id")}
    for su in unit_map.values():
        su["chunk_ids"] = []
        su["figure_ids"] = []
        su["table_ids"] = []
        su["frame_ids"] = []

    def add(rec: Dict[str, Any], field: str):
        uid = rec.get("structure_unit_id")
        cid = rec.get("chunk_id")
        if uid in unit_map and cid:
            if cid not in unit_map[uid][field]:
                unit_map[uid][field].append(cid)

    for r in text_dicts or []:
        add(r, "chunk_ids")
    for r in image_dicts or []:
        add(r, "figure_ids")
    for r in table_dicts or []:
        add(r, "table_ids")
    for r in video_seg_dicts or []:
        add(r, "chunk_ids")
    for r in video_frame_dicts or []:
        add(r, "frame_ids")

    return structure_units


def build_group_context(
    structure_units: List[Dict[str, Any]],
    text_chunks: List[Dict[str, Any]],
    image_chunks: List[Dict[str, Any]],
    table_chunks: List[Dict[str, Any]],
    video_segments: List[Dict[str, Any]],
    video_frames: List[Dict[str, Any]],
) -> Dict[str, Any]:
    text_lookup = {r["chunk_id"]: r for r in (text_chunks or []) + (video_segments or [])}
    chunks_by_structure_unit = defaultdict(list)
    images_by_structure_unit = defaultdict(list)
    tables_by_structure_unit = defaultdict(list)

    for rec in (text_chunks or []) + (video_segments or []):
        structure_unit_id = (
            rec.get("structure_unit_id")
            or rec.get("section_id")
            or rec.get("_structure_unit_id")
        )
        if structure_unit_id:
            chunks_by_structure_unit[structure_unit_id].append(rec)
        else:
            logging.warning(f"Chunk missing structure_unit_id/section_id: {rec.get('chunk_id', rec)}")
            continue
    for rec in (image_chunks or []) + (video_frames or []):
        structure_unit_id = (
            rec.get("structure_unit_id")
            or rec.get("section_id")
            or rec.get("_structure_unit_id")
        )
        if structure_unit_id:
            images_by_structure_unit[structure_unit_id].append(rec)
        else:
            logging.warning(f"Image chunk missing structure_unit_id/section_id: {rec.get('chunk_id', rec)}")
            continue
    for rec in (table_chunks or []):
        structure_unit_id = (
            rec.get("structure_unit_id")
            or rec.get("section_id")
            or rec.get("_structure_unit_id")
        )
        if structure_unit_id:
            tables_by_structure_unit[structure_unit_id].append(rec)
        else:
            logging.warning(f"Table chunk missing structure_unit_id/section_id: {rec.get('chunk_id', rec)}")
            continue

    neighbor_chunks = {}
    for su_id, items in chunks_by_structure_unit.items():
        ids = [x["chunk_id"] for x in sorted(items, key=lambda x: x.get("chunk_index", 0))]
        neighbor_chunks.update(build_neighbors(ids))

    image_to_text_chunks = defaultdict(list)
    table_to_text_chunks = defaultdict(list)
    chunk_relations = defaultdict(list)

    for rec in text_chunks or []:
        for fig in rec.get("related_figures", []) or []:
            image_to_text_chunks[fig].append(rec["chunk_id"])
            chunk_relations[rec["chunk_id"]].append(fig)
        for tbl in rec.get("related_tables", []) or []:
            table_to_text_chunks[tbl].append(rec["chunk_id"])
            chunk_relations[rec["chunk_id"]].append(tbl)

    for rec in video_segments or []:
        for frame in rec.get("keyframe_ids", []) or []:
            image_to_text_chunks[frame].append(rec["chunk_id"])
            chunk_relations[rec["chunk_id"]].append(frame)

    for su_id, imgs in images_by_structure_unit.items():
        su_chunk_ids = [x["chunk_id"] for x in chunks_by_structure_unit.get(su_id, [])]
        for img in imgs:
            image_to_text_chunks[img["chunk_id"]] = list(dict.fromkeys(image_to_text_chunks.get(img["chunk_id"], []) + su_chunk_ids))

    for su_id, tbls in tables_by_structure_unit.items():
        su_chunk_ids = [x["chunk_id"] for x in chunks_by_structure_unit.get(su_id, [])]
        for tbl in tbls:
            table_to_text_chunks[tbl["chunk_id"]] = list(dict.fromkeys(table_to_text_chunks.get(tbl["chunk_id"], []) + su_chunk_ids))

    return {
        "text_lookup": text_lookup,
        "chunks_by_structure_unit": chunks_by_structure_unit,
        "images_by_structure_unit": images_by_structure_unit,
        "tables_by_structure_unit": tables_by_structure_unit,
        "neighbor_chunks": neighbor_chunks,
        "image_to_text_chunks": image_to_text_chunks,
        "table_to_text_chunks": table_to_text_chunks,
        "chunk_relations": chunk_relations,
    }
