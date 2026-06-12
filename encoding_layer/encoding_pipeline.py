from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from .models import EncodedRecord, EncodingViews, LinkageViews
from .retrieval_text_builders import build_text_views, build_image_views, build_table_views
from .promoted_fields_builder import build_promoted_fields
from .section_linker import build_group_context, link_chunks_to_structure_units


def _load(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class EncodingPipeline:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def process_document(self, doc_output_dir: Path) -> Dict[str, Any]:
        metadata_dir = doc_output_dir / "metadata"
        if not metadata_dir.exists():
            return {}

        doc_meta = _load(metadata_dir / "doc_metadata.json") or {}
        if not isinstance(doc_meta, dict) or not doc_meta.get("doc_id"):
            raise ValueError(f"Invalid doc_metadata in {metadata_dir}")

        structure_units = _load(metadata_dir / "structure_units.json") or []
        text_chunks = _load(metadata_dir / "text_chunks.json") or []
        image_chunks = _load(metadata_dir / "image_chunks.json") or []
        table_chunks = _load(metadata_dir / "table_chunks.json") or []
        video_segments = _load(metadata_dir / "video_segments.json") or []
        video_frames = _load(metadata_dir / "video_frames.json") or []

        structure_lookup = {u["structure_unit_id"]: u for u in structure_units if u.get("structure_unit_id")}
        group_context = build_group_context(structure_units, text_chunks, image_chunks, table_chunks, video_segments, video_frames)

        encoded_text = [self._encode_text(r, doc_meta, structure_lookup, group_context).to_dict() for r in text_chunks]
        encoded_images = [self._encode_image(r, doc_meta, structure_lookup, group_context).to_dict() for r in image_chunks]
        encoded_tables = [self._encode_table(r, doc_meta, structure_lookup, group_context).to_dict() for r in table_chunks]
        encoded_video_segments = [self._encode_text(r, doc_meta, structure_lookup, group_context).to_dict() for r in video_segments]
        encoded_video_frames = [self._encode_image(r, doc_meta, structure_lookup, group_context).to_dict() for r in video_frames]

        # Merge enrichment fields stripped by Pydantic back onto encoded dicts
        _ENRICH = {
            "evidence_role_confidence", "contextual_summary_confidence",
            "detected_codes_confidence", "salience_score", "evidence_role",
            "contextual_summary", "detected_codes", "entities",
            "image_caption_confidence", "depicted_component_confidence",
            "visible_annotations_confidence", "frame_role_confidence",
            "table_summary_confidence", "table_purpose_confidence",
        }
        for raw_list, enc_list in [
            (text_chunks, encoded_text),
            (image_chunks, encoded_images),
            (table_chunks, encoded_tables),
        ]:
            raw_by_id = {
                r.get("chunk_id"): r
                for r in raw_list
                if isinstance(r, dict) and r.get("chunk_id")
            }
            for d in enc_list:
                raw = raw_by_id.get(d.get("chunk_id", ""))
                if raw:
                    for field in _ENRICH:
                        if field in raw and raw[field] is not None and (d.get(field) is None or d.get(field) == 0.0):
                            d[field] = raw[field]

        structure_units = link_chunks_to_structure_units(
            structure_units,
            text_chunks,
            image_chunks,
            table_chunks,
            video_segments,
            video_frames,
        )

        # Preserve enrichment outline_summary if it already exists (structured list
        # from DocMetaEnricher). Only build a fallback string version if missing.
        if not doc_meta.get("outline_summary"):
            doc_meta["outline_summary"] = self._build_outline_summary(structure_units)
        doc_meta["section_map_text"] = self._build_section_map_text(structure_units)
        doc_meta["doc_summary"] = doc_meta.get("doc_summary") or self._infer_doc_summary(doc_meta, structure_units)
        doc_meta["doc_embedding_input_text"] = self._build_doc_embedding_input_text(doc_meta)
        doc_meta["encoding_summary"] = {
            "text_encoded": len(encoded_text),
            "image_encoded": len(encoded_images),
            "table_encoded": len(encoded_tables),
            "video_segments_encoded": len(encoded_video_segments),
            "video_frames_encoded": len(encoded_video_frames),
        }

        if not self.dry_run:
            _save(doc_meta, metadata_dir / "doc_metadata.json")
            _save(structure_units, metadata_dir / "structure_units.json")
            _save(encoded_text, metadata_dir / "encoded_text_chunks.json")
            _save(encoded_images, metadata_dir / "encoded_image_chunks.json")
            _save(encoded_tables, metadata_dir / "encoded_table_chunks.json")
            _save(encoded_video_segments, metadata_dir / "encoded_video_segments.json")
            _save(encoded_video_frames, metadata_dir / "encoded_video_frames.json")

        return {
            "doc_name": doc_output_dir.name,
            "format": doc_meta.get("file_format", ""),
            "text_encoded": len(encoded_text) + len(encoded_video_segments),
            "img_encoded": len(encoded_images) + len(encoded_video_frames),
            "tbl_encoded": len(encoded_tables),
        }

    def _encode_text(self, rec: Dict[str, Any], doc_meta: Dict[str, Any], structure_lookup: Dict[str, Dict[str, Any]], group_context: Dict[str, Any]) -> EncodedRecord:
        su = structure_lookup.get(rec.get("structure_unit_id", ""), {})
        section_chunks = group_context["chunks_by_structure_unit"].get(rec.get("structure_unit_id", ""), [])
        # Use contextual_summary (enriched, signal-dense) not raw text_original_content
        # which has the summary prepended making it redundant. Contextual summaries
        # are ~50 tokens each vs 300 tokens raw — fits more sections in context window.
        section_context = " ".join([
            x.get("contextual_summary", "") or
            x.get("text_original_content", "") or
            x.get("transcript_text", "")
            for x in section_chunks[:8]
        ])
        views_dict = build_text_views(rec, doc_meta, su, section_context)
        # EncodingViews now includes title_text and bm25_text
        views = EncodingViews(**{k: v for k, v in views_dict.items() if hasattr(EncodingViews, k)})
        sibling_ids = [x["chunk_id"] for x in section_chunks if x.get("chunk_id") != rec.get("chunk_id")][:30]
        linkage = LinkageViews(
            source_id=rec.get("source_id", ""),
            parent_id=rec.get("parent_id", ""),
            structure_unit_id=rec.get("structure_unit_id", ""),
            structure_unit_type=su.get("structure_unit_type", ""),
            structure_unit_title=su.get("title", "") or su.get("structure_unit_title", ""),
            semantic_anchor=su.get("semantic_anchor", ""),
            section_title=rec.get("section_title", "") or su.get("title", "") or su.get("structure_unit_title", ""),
            section_id=rec.get("section_id", "") or rec.get("structure_unit_id", ""),
            related_figures=list(rec.get("related_figures", []) or rec.get("keyframe_ids", []) or []),
            related_tables=list(rec.get("related_tables", []) or []),
            related_frames=list(rec.get("keyframe_ids", []) or []),
            related_section_ids=list(rec.get("related_section_ids", []) or []),
            related_chunk_ids=group_context["chunk_relations"].get(rec.get("chunk_id", ""), []),
            neighbor_chunk_ids=group_context["neighbor_chunks"].get(rec.get("chunk_id", ""), []),
            sibling_chunk_ids=sibling_ids,
            page_number=int(rec.get("page_number", 0) or 0),
            slide_index=int(rec.get("slide_index", 0) or 0),
            sheet_name=rec.get("sheet_name", "") or "",
            start_time_s=rec.get("start_time_s"),
            end_time_s=rec.get("end_time_s"),
        )
        out_dict = dict(rec)
        out_dict.update(views.to_dict())
        out_dict["promoted_fields"] = build_promoted_fields(out_dict, doc_meta)
        default_modality = "video" if rec.get("transcript_text") else "text"
        return EncodedRecord(
            chunk_id=rec["chunk_id"],
            doc_id=rec["doc_id"],
            source_id=rec.get("source_id", ""),
            structure_unit_id=rec.get("structure_unit_id", ""),
            source_modality=rec.get("source_modality") or default_modality,
            file_format=rec.get("file_format", doc_meta.get("file_format", "")),
            encoding_views=views,
            linkage=linkage,
            promoted_fields=out_dict["promoted_fields"],
            metadata=rec,
        )

    def _encode_image(self, rec: Dict[str, Any], doc_meta: Dict[str, Any], structure_lookup: Dict[str, Dict[str, Any]], group_context: Dict[str, Any]) -> EncodedRecord:
        su = structure_lookup.get(rec.get("structure_unit_id", ""), {})
        related_chunk_ids = group_context["image_to_text_chunks"].get(rec.get("chunk_id", ""), [])
        # Primary: use text chunks that explicitly reference this image
        # Fallback: use contextual_summaries from the same structure unit
        # (handles unreferenced images that got section_id via proximity linking)
        if related_chunk_ids:
            section_context = " ".join([
                group_context["text_lookup"][cid].get("contextual_summary", "") or
                group_context["text_lookup"][cid].get("text_original_content", "") or
                group_context["text_lookup"][cid].get("transcript_text", "")
                for cid in related_chunk_ids if cid in group_context["text_lookup"]
            ])
        else:
            su_chunks = group_context["chunks_by_structure_unit"].get(rec.get("structure_unit_id", ""), [])
            section_context = " ".join([
                x.get("contextual_summary", "") or x.get("text_original_content", "")
                for x in su_chunks[:4]
            ])
        views_dict = build_image_views(rec, doc_meta, su, section_context)
        views = EncodingViews(**views_dict)
        sibling_ids = [x["chunk_id"] for x in group_context["images_by_structure_unit"].get(rec.get("structure_unit_id", ""), []) if x.get("chunk_id") != rec.get("chunk_id")][:30]
        image_type = rec.get("image_type") or ""
        is_video_frame = image_type.startswith("video_") or rec.get("file_format", "") in {"mp4","avi","mkv","mov","webm","m4v"}
        linkage = LinkageViews(
            source_id=rec.get("source_id", ""),
            parent_id=rec.get("parent_id", ""),
            structure_unit_id=rec.get("structure_unit_id", ""),
            structure_unit_type=su.get("structure_unit_type", ""),
            structure_unit_title=su.get("title", "") or su.get("structure_unit_title", ""),
            semantic_anchor=su.get("semantic_anchor", ""),
            section_title=su.get("title", "") or su.get("structure_unit_title", ""),
            section_id=rec.get("structure_unit_id", ""),
            related_figures=[] if is_video_frame else related_chunk_ids,
            related_tables=[],
            related_frames=[rec.get("chunk_id", "")] if is_video_frame else [],
            related_chunk_ids=related_chunk_ids,
            neighbor_chunk_ids=[],
            sibling_chunk_ids=sibling_ids,
            page_number=int(rec.get("page_number", 0) or 0),
            slide_index=int(rec.get("slide_index", 0) or 0),
            sheet_name=rec.get("sheet_name", "") or "",
            start_time_s=rec.get("timestamp_s", rec.get("start_time_s")),
            end_time_s=rec.get("timestamp_s", rec.get("end_time_s")),
        )
        out_dict = dict(rec)
        out_dict.update(views.to_dict())
        out_dict["promoted_fields"] = build_promoted_fields(out_dict, doc_meta)
        return EncodedRecord(
            chunk_id=rec["chunk_id"],
            doc_id=rec["doc_id"],
            source_id=rec.get("source_id", ""),
            structure_unit_id=rec.get("structure_unit_id", ""),
            source_modality=rec.get("source_modality", "image"),
            file_format=rec.get("file_format", doc_meta.get("file_format", "")),
            encoding_views=views,
            linkage=linkage,
            promoted_fields=out_dict["promoted_fields"],
            metadata=rec,
        )

    def _encode_table(self, rec: Dict[str, Any], doc_meta: Dict[str, Any], structure_lookup: Dict[str, Dict[str, Any]], group_context: Dict[str, Any]) -> EncodedRecord:
        su = structure_lookup.get(rec.get("structure_unit_id", ""), {})
        related_chunk_ids = group_context["table_to_text_chunks"].get(rec.get("chunk_id", ""), [])
        if related_chunk_ids:
            section_context = " ".join([
                group_context["text_lookup"][cid].get("contextual_summary", "") or
                group_context["text_lookup"][cid].get("text_original_content", "") or
                group_context["text_lookup"][cid].get("transcript_text", "")
                for cid in related_chunk_ids if cid in group_context["text_lookup"]
            ])
        else:
            su_chunks = group_context["chunks_by_structure_unit"].get(rec.get("structure_unit_id", ""), [])
            section_context = " ".join([
                x.get("contextual_summary", "") or x.get("text_original_content", "")
                for x in su_chunks[:4]
            ])
        views_dict = build_table_views(rec, doc_meta, su, section_context)
        views = EncodingViews(**views_dict)
        sibling_ids = [x["chunk_id"] for x in group_context["tables_by_structure_unit"].get(rec.get("structure_unit_id", ""), []) if x.get("chunk_id") != rec.get("chunk_id")][:30]
        linkage = LinkageViews(
            source_id=rec.get("source_id", ""),
            parent_id=rec.get("parent_id", ""),
            structure_unit_id=rec.get("structure_unit_id", ""),
            structure_unit_type=su.get("structure_unit_type", ""),
            structure_unit_title=su.get("title", "") or su.get("structure_unit_title", ""),
            semantic_anchor=su.get("semantic_anchor", ""),
            section_title=su.get("title", "") or su.get("structure_unit_title", ""),
            section_id=rec.get("structure_unit_id", ""),
            related_figures=[],
            related_tables=[rec.get("chunk_id", "")],
            related_frames=[],
            related_chunk_ids=related_chunk_ids,
            neighbor_chunk_ids=[],
            sibling_chunk_ids=sibling_ids,
            page_number=int(rec.get("page_number", 0) or 0),
            slide_index=int(rec.get("slide_index", 0) or 0),
            sheet_name=rec.get("sheet_name", "") or "",
            start_time_s=rec.get("start_time_s"),
            end_time_s=rec.get("end_time_s"),
        )
        out_dict = dict(rec)
        out_dict.update(views.to_dict())
        out_dict["promoted_fields"] = build_promoted_fields(out_dict, doc_meta)
        return EncodedRecord(
            chunk_id=rec["chunk_id"],
            doc_id=rec["doc_id"],
            source_id=rec.get("source_id", ""),
            structure_unit_id=rec.get("structure_unit_id", ""),
            source_modality=rec.get("source_modality", "table"),
            file_format=rec.get("file_format", doc_meta.get("file_format", "")),
            encoding_views=views,
            linkage=linkage,
            promoted_fields=out_dict["promoted_fields"],
            metadata=rec,
        )

    def _build_outline_summary(self, structure_units: List[Dict[str, Any]]) -> str:
        parts = []
        for su in structure_units[:20]:
            title = su.get("title", "") or su.get("structure_unit_title", "")
            anchor = su.get("semantic_anchor", "")
            if title or anchor:
                parts.append(f"{title}: {anchor}".strip(": "))
        return " | ".join(parts)

    def _build_section_map_text(self, structure_units: List[Dict[str, Any]]) -> str:
        lines = []
        for su in structure_units:
            title = su.get("title", "") or su.get("structure_unit_title", "")
            anchor = su.get("semantic_anchor", "")
            n_chunks = len(su.get("chunk_ids", []))
            n_figs = len(su.get("figure_ids", []))
            n_tbls = len(su.get("table_ids", []))
            n_frames = len(su.get("frame_ids", []))
            lines.append(f"{title} | anchor={anchor} | chunks={n_chunks} figs={n_figs} tbls={n_tbls} frames={n_frames}")
        return "\n".join(lines)

    def _infer_doc_summary(self, doc_meta: Dict[str, Any], structure_units: List[Dict[str, Any]]) -> str:
        title = doc_meta.get("doc_title", "")
        summary_bits = []
        if structure_units:
            summary_bits.append(self._build_outline_summary(structure_units[:5]))
        if doc_meta.get("file_format") in {"mp4","avi","mkv","mov","webm","m4v"}:
            summary_bits.append(f"Video with {doc_meta.get('total_segments', 0)} segments and {doc_meta.get('total_frames', 0)} frames.")
        return " ".join([x for x in [title] + summary_bits if x]).strip()

    def _build_doc_embedding_input_text(self, doc_meta: Dict[str, Any]) -> str:
        parts = []

        # doc_title and doc_summary
        for key in ("doc_title", "doc_summary"):
            val = doc_meta.get(key)
            if val and isinstance(val, str) and val.strip():
                parts.append(val.strip())

        # outline_summary: DocMetaEnricher stores as list of {order, section, summary}
        # Encoding pipeline may also have converted to string. Handle both.
        outline = doc_meta.get("outline_summary")
        if outline:
            if isinstance(outline, list):
                # Convert structured list to readable text for embedding
                lines = []
                for entry in outline[:20]:
                    if isinstance(entry, dict):
                        sec = entry.get("section", "")
                        summ = entry.get("summary", "")
                        if sec or summ:
                            lines.append(f"{sec}: {summ}".strip(": "))
                    elif isinstance(entry, str):
                        lines.append(entry)
                outline_text = " | ".join(lines)
                if outline_text:
                    parts.append(outline_text)
            elif isinstance(outline, str) and outline.strip():
                parts.append(outline.strip())

        # section_map_text
        smt = doc_meta.get("section_map_text")
        if smt and isinstance(smt, str) and smt.strip():
            parts.append(smt.strip())

        return "\n\n".join(parts).strip()