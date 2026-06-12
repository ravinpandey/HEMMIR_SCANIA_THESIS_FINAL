"""
cross_reference_layer/cross_referencer.py

Links text ↔ images ↔ tables across documents.
Assigns section_id AND structure_unit_id to every chunk.
Updates promoted_fields after all linking is complete.
Handles all formats: PDF, DOCX, PPTX, CSV, XLSX, Video.

Requires enrichment to have run first:
  section_map["heading"]["start_page"] and ["end_page"] are set at ingestion.
  section_map["heading"]["summary"] is set by enrichment.

What gets written:

PDF / DOCX / PPTX:
  text_chunk["related_figures"]    ← resolved image chunk_ids (explicit + proximity)
  text_chunk["related_tables"]     ← resolved table chunk_ids
  text_chunk["section_id"]         ← section heading from page-range lookup
  text_chunk["structure_unit_id"]  ← structure unit ID from structure_units.json
  text_chunk["related_section_ids"] ← NEW: structure unit IDs from "see Section X" refs
  text_chunk["related_figures_str"] ← updated denormalized string for ChromaDB
  text_chunk["related_tables_str"]  ← updated denormalized string for ChromaDB
  image_chunk["related_sections"]  ← text chunk_ids that reference this image
  image_chunk["section_id"]        ← assigned to ALL images (not just cited ones)
  image_chunk["structure_unit_id"] ← assigned to ALL images via page-range lookup
  table_chunk["section_id"]        ← section heading from page-range lookup
  table_chunk["structure_unit_id"] ← structure unit ID from structure_units.json
  ALL promoted_fields updated      ← NEW: ChromaDB sees correct structure_unit_id

CSV / XLSX:
  table_chunk["schema_chunk_id"]   ← pointing to schema chunk for the same sheet
  table_chunk["structure_unit_id"] ← from structure_units (sheet-level unit)

Video:
  segment["scene_sibling_ids"]     ← other segments in same scene
  segment["keyframe_ids"]          ← verified/repaired from frames side
  frame["parent_segment_id"]       ← confirmed parent segment chunk_id
  frame["parent_id"]               ← repaired from related_sections fallback

Research-level upgrades in this version:

  1. ALL images get section_id + structure_unit_id via page-range lookup,
     not just images cited as "Figure N" in text. In research papers, slides,
     and manuals many figures are placed contextually without explicit citation.
     Previously unreferenced images had empty structure_unit_id — they could
     not be filtered or ranked by section in retrieval.

  2. Proximity linking: text chunks on the same page as an image are linked
     as related_sections even without explicit "Figure N" citation. This is
     the correct design for contextual figure placement in documents.

  3. promoted_fields rebuilt after all linking — structure_unit_id, related_figures,
     related_tables are written to chunk dicts but promoted_fields (what ChromaDB
     indexes) was stale from the normalizer. Now promoted_fields is rebuilt after
     cross-referencing so ChromaDB filters on structure_unit_id actually work.
     This directly enables Layer 1 (structural prior) and Layer 3 (cross-modal
     expansion) of the retrieval spec.

  4. Section reference linking: "see Section 3.2" patterns in text are resolved
     to structure_unit_ids and written as related_section_ids on the chunk.
     This creates explicit navigational links between text chunks and sections
     they discuss — enabling the agent retriever to follow cross-references.

  5. related_figures_str and related_tables_str (denormalized ChromaDB strings)
     are updated after linking so ChromaDB has the correct lists.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from cross_reference_layer.utils.pattern_utils import (
    find_figure_references,
    find_table_references,
    find_section_references,
)
from cross_reference_layer.utils.video_linker import (
    build_scene_sibling_map,
    validate_frame_parent_links,
    build_segment_frame_map,
    link_schema_to_rowgroups,
)
from shared.models.metadata_models import (
    validate_doc_metadata,
    validate_structure_units,
    validate_text_chunks,
    validate_image_chunks,
    validate_table_chunks,
    validate_video_segments,
    validate_video_frames,
)

VIDEO_FORMATS  = {"mp4", "avi", "mkv", "mov", "webm", "m4v", "video"}
SHEET_FORMATS  = {"csv", "xlsx", "xls", "ods", "tsv"}


def _load(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _find_section_by_page(page_number: int, section_map: Dict) -> str:
    """
    Find which section a chunk belongs to using page-range lookup.
    Returns the section name with the highest start_page ≤ page_number.
    Returns "" if no match.
    """
    if not section_map or not page_number:
        return ""
    best_section = ""
    best_start   = -1
    for section_name, entry in section_map.items():
        if not isinstance(entry, dict):
            continue
        start = int(entry.get("start_page") or 0)
        end   = int(entry.get("end_page") or 999999)
        if start <= page_number <= end and start > best_start:
            best_section = section_name
            best_start   = start
    return best_section


def _build_su_id_lookup(structure_units: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Build reverse lookup: section_title → structure_unit_id.
    Used to write structure_unit_id onto chunks after section_id is resolved.
    """
    lookup: Dict[str, str] = {}
    for su in structure_units:
        title = su.get("title", "")
        su_id = su.get("structure_unit_id", "")
        if title and su_id:
            lookup[title] = su_id
    return lookup


def _build_section_number_lookup(
    structure_units: List[Dict[str, Any]]
) -> Dict[str, str]:
    """
    Build lookup: section_number_string → structure_unit_id.

    Extracts the leading number from section titles:
      "3.2 Attention" → "3.2" → su_id
      "4 Why Self-Attention" → "4" → su_id

    Used to resolve "see Section 3.2" references in text to structure_unit_ids.
    This enables direct navigational links from text chunks to the sections
    they discuss — critical for the agent retriever's cross-reference traversal.
    """
    import re
    lookup: Dict[str, str] = {}
    number_re = re.compile(r"^(\d+(?:\.\d+)*)\s")
    for su in structure_units:
        title = su.get("title", "")
        su_id = su.get("structure_unit_id", "")
        if not (title and su_id):
            continue
        m = number_re.match(title)
        if m:
            lookup[m.group(1)] = su_id
    return lookup


def _rebuild_promoted_fields(
    chunk: Dict[str, Any],
    doc_meta: Dict[str, Any],
) -> None:
    """
    Rebuild promoted_fields on a chunk after cross-referencing updates its fields.

    Cross-referencing writes structure_unit_id, related_figures, related_tables,
    section_id onto chunk dicts — but promoted_fields (what ChromaDB indexes)
    was built by the normalizer BEFORE cross-referencing ran. It is stale.

    This function updates only the fields that cross-referencing changes:
      structure_unit_id    ← newly assigned
      section_id (as str)  ← newly assigned
      evidence_role        ← already from enrichment, keep
      salience_score       ← already from enrichment, keep

    We do NOT do a full rebuild (which requires ExtendedPromotedFields Pydantic)
    to avoid import coupling in this layer. Instead we patch in-place.
    This is intentional — the encoding layer does a full promoted_fields rebuild
    using the correct Pydantic model with all embedding signals.
    """
    pf = chunk.get("promoted_fields")
    if not isinstance(pf, dict):
        return

    # Update structure navigation fields
    pf["structure_unit_id"] = chunk.get("structure_unit_id", "")

    # Update section_id (stored as string in promoted_fields)
    section_id = chunk.get("section_id", "") or chunk.get("structure_unit_id", "")
    if section_id:
        pf["section_id"] = section_id

    # Propagate tier1/tier2 from doc_meta if not already set
    if not pf.get("tier1") and doc_meta.get("tier1"):
        pf["tier1"] = doc_meta["tier1"]
    if not pf.get("tier2") and doc_meta.get("tier2"):
        pf["tier2"] = doc_meta["tier2"]

    # Copy enrichment quality signals from chunk dict into promoted_fields.
    # These are set by TextEnricher/ImageEnricher/TableEnricher but the
    # normalizer built promoted_fields BEFORE enrichment ran — so they are
    # None in promoted_fields even though they exist on the chunk dict.
    # ChromaDB uses promoted_fields for filtering and reranking, so these
    # must be present there.
    if chunk.get("evidence_role"):
        pf["evidence_role"] = chunk["evidence_role"]
    if chunk.get("evidence_role_confidence") is not None:
        pf["evidence_role_confidence"] = float(chunk["evidence_role_confidence"] or 0.0)
    if chunk.get("salience_score") is not None:
        pf["salience_score"] = float(chunk["salience_score"] or 0.0)
    if chunk.get("contextual_summary_confidence") is not None:
        pf["contextual_summary_confidence"] = float(chunk["contextual_summary_confidence"] or 0.0)
    if chunk.get("detected_codes_confidence") is not None:
        pf["detected_codes_confidence"] = float(chunk["detected_codes_confidence"] or 0.0)
    # Image-specific
    if chunk.get("image_caption_confidence") is not None:
        pf["image_caption_confidence"] = float(chunk["image_caption_confidence"] or 0.0)
    if chunk.get("depicted_component_confidence") is not None:
        pf["depicted_component_confidence"] = float(chunk["depicted_component_confidence"] or 0.0)
    if chunk.get("visible_annotations_confidence") is not None:
        pf["visible_annotations_confidence"] = float(chunk["visible_annotations_confidence"] or 0.0)
    # Table-specific
    if chunk.get("table_summary_confidence") is not None:
        pf["table_summary_confidence"] = float(chunk["table_summary_confidence"] or 0.0)


class CrossReferencer:

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def process_document(self, doc_output_dir: Path) -> Dict[str, Any]:
        doc_name     = doc_output_dir.name
        metadata_dir = doc_output_dir / "metadata"

        if not metadata_dir.exists():
            logger.warning(f"  No metadata/ in {doc_output_dir}")
            return {}

        logger.info(f"\n{'='*55}")
        logger.info(f"  CrossRef: {doc_name}")
        logger.info(f"{'='*55}")

        # ── Load raw JSON ──────────────────────────────────────────────
        raw_doc_meta  = _load(metadata_dir / "doc_metadata.json")    or {}
        raw_su        = _load(metadata_dir / "structure_units.json") or []
        raw_text      = _load(metadata_dir / "text_chunks.json")     or []
        raw_images    = _load(metadata_dir / "image_chunks.json")    or []
        raw_tables    = _load(metadata_dir / "table_chunks.json")    or []
        raw_vid_segs  = _load(metadata_dir / "video_segments.json")  or []
        raw_vid_frms  = _load(metadata_dir / "video_frames.json")    or []

        # ── Pydantic boundary validation ───────────────────────────────
        try:
            doc_meta_model = validate_doc_metadata(raw_doc_meta)
        except Exception as e:
            logger.error(f"  CrossRef doc_metadata invalid: {e}")
            return {}

        su_models   = validate_structure_units(raw_su)
        text_models = validate_text_chunks(raw_text)
        img_models  = validate_image_chunks(raw_images)
        tbl_models  = validate_table_chunks(raw_tables)
        seg_models  = validate_video_segments(raw_vid_segs)
        frm_models  = validate_video_frames(raw_vid_frms)

        # Work on plain dicts
        doc_meta      = doc_meta_model.to_dict()
        struct_units  = [m.to_dict() for m in su_models]
        text_chunks   = [m.to_dict() for m in text_models]
        image_chunks  = [m.to_dict() for m in img_models]
        table_chunks  = [m.to_dict() for m in tbl_models]
        vid_segments  = [m.to_dict() for m in seg_models]
        vid_frames    = [m.to_dict() for m in frm_models]


        # Merge enrichment fields stripped by Pydantic during validation
        _ENRICH = {
            "evidence_role_confidence", "contextual_summary_confidence",
            "detected_codes_confidence", "salience_score", "evidence_role",
            "contextual_summary", "detected_codes", "entities",
            "image_caption_confidence", "depicted_component_confidence",
            "visible_annotations_confidence", "frame_role_confidence",
            "table_summary_confidence", "table_purpose_confidence",
        }
        for raw_list, dict_list in [
            (raw_text,   text_chunks),
            (raw_images, image_chunks),
            (raw_tables, table_chunks),
        ]:
            raw_by_id = {
                r.get("chunk_id"): r
                for r in raw_list
                if isinstance(r, dict) and r.get("chunk_id")
            }
            for d in dict_list:
                raw = raw_by_id.get(d.get("chunk_id", ""))
                if raw:
                    for field in _ENRICH:
                        if field in raw and raw[field] is not None and (d.get(field) is None or d.get(field) == 0.0):
                            d[field] = raw[field]

        fmt = (doc_meta.get("file_format") or "").lower().strip(".")

        # ── Route by format ────────────────────────────────────────────
        stats: Dict[str, int] = {}

        if fmt in VIDEO_FORMATS:
            stats = self._process_video(vid_segments, vid_frames)
        elif fmt in SHEET_FORMATS:
            stats = self._process_spreadsheet(table_chunks, struct_units)
        else:
            stats = self._process_document(
                doc_meta, text_chunks, image_chunks, table_chunks, struct_units
            )

        # ── Save ───────────────────────────────────────────────────────
        if not self.dry_run:
            _save(text_chunks,  metadata_dir / "text_chunks.json")

            # image_chunks: skip for CSV/XLSX (no images)
            if fmt not in SHEET_FORMATS:
                _save(image_chunks, metadata_dir / "image_chunks.json")

            # table_chunks: skip for video (no tables)
            if fmt not in VIDEO_FORMATS:
                _save(table_chunks, metadata_dir / "table_chunks.json")

            # video files: only for video format
            if fmt in VIDEO_FORMATS:
                _save(vid_segments, metadata_dir / "video_segments.json")
                _save(vid_frames,   metadata_dir / "video_frames.json")

            logger.success(f"  CrossRef saved: {doc_name} | {stats}")
        else:
            logger.info(f"[DRY RUN] {doc_name}: {stats}")

        return {"doc_name": doc_name, **stats}

    # ── Document processing (PDF / DOCX / PPTX) ───────────────────────

    def _process_document(
        self,
        doc_meta:     Dict[str, Any],
        text_chunks:  List[Dict[str, Any]],
        image_chunks: List[Dict[str, Any]],
        table_chunks: List[Dict[str, Any]],
        struct_units: List[Dict[str, Any]],
    ) -> Dict[str, int]:

        section_map      = doc_meta.get("section_map", {})
        figure_index_map = {
            int(k): v
            for k, v in doc_meta.get("figure_index_map", {}).items()
        }
        table_index_map = {
            int(k): v
            for k, v in doc_meta.get("table_index_map", {}).items()
        }

        has_page_ranges = any(
            isinstance(v, dict) and v.get("start_page")
            for v in section_map.values()
        )
        if not has_page_ranges:
            logger.warning(
                "  CrossRef: section_map has no page ranges — "
                "section_id and structure_unit_id will be empty"
            )

        # ── Lookups ────────────────────────────────────────────────────
        # title → structure_unit_id (for page-range resolution)
        su_id_by_title = _build_su_id_lookup(struct_units)

        # section_number → structure_unit_id (for "see Section 3.2" refs)
        su_id_by_number = _build_section_number_lookup(struct_units)

        # image chunk_id → index (for fast update)
        img_lookup: Dict[str, int] = {
            c["chunk_id"]: i for i, c in enumerate(image_chunks)
        }

        # page_number → list of image chunk_ids on that page
        # Used for proximity linking (Upgrade 2)
        page_to_images: Dict[int, List[str]] = defaultdict(list)
        for ic in image_chunks:
            page = int(ic.get("page_number") or 0)
            if page > 0:
                page_to_images[page].append(ic["chunk_id"])

        # page_number → list of text chunk_ids on that page
        # Used for reverse proximity linking back to images
        page_to_texts: Dict[int, List[str]] = defaultdict(list)
        for tc in text_chunks:
            page = int(tc.get("page_number") or 0)
            if page > 0:
                page_to_texts[page].append(tc["chunk_id"])

        # Explicit figure citation: image chunk_id → set of text chunk_ids
        img_to_text: Dict[str, Set[str]] = defaultdict(set)

        total_fig_links    = 0
        total_tbl_links    = 0
        section_ids_set    = 0
        su_ids_set         = 0
        section_ref_links  = 0
        proximity_links    = 0

        # ── Pass 1: Text chunks ────────────────────────────────────────
        for chunk in text_chunks:
            cid         = chunk.get("chunk_id", "")
            content     = chunk.get("text_original_content", "")
            page_number = int(chunk.get("page_number") or 0)

            # ── Explicit figure references ("Figure 3", "Fig. 2") ──────
            fig_nums      = find_figure_references(content)
            resolved_figs = []
            for n in fig_nums:
                if n in figure_index_map:
                    img_id = figure_index_map[n]
                    resolved_figs.append(img_id)
                    img_to_text[img_id].add(cid)
            chunk["related_figures"] = list(dict.fromkeys(resolved_figs))
            total_fig_links += len(resolved_figs)

            # ── Proximity linking: images on same page ─────────────────
            # Upgrade 2: if a text chunk shares a page with images that
            # are NOT yet cited by this chunk, add them as proximity links.
            # This handles contextual figure placement without "Figure N".
            # We mark these separately from explicit links.
            if page_number > 0:
                same_page_imgs = page_to_images.get(page_number, [])
                for img_id in same_page_imgs:
                    if img_id not in resolved_figs:
                        # Add to chunk's related_figures
                        if img_id not in chunk["related_figures"]:
                            chunk["related_figures"].append(img_id)
                            proximity_links += 1
                        # Add reverse link (weaker than explicit — tagged)
                        img_to_text[img_id].add(cid)

            # ── Explicit table references ("Table 2") ──────────────────
            tbl_nums      = find_table_references(content)
            resolved_tbls = [
                table_index_map[n]
                for n in tbl_nums
                if n in table_index_map
            ]
            chunk["related_tables"] = list(dict.fromkeys(resolved_tbls))
            total_tbl_links += len(resolved_tbls)

            # ── Section ID + structure_unit_id assignment ──────────────
            if has_page_ranges and page_number > 0:
                section_id = _find_section_by_page(page_number, section_map)
                if section_id:
                    chunk["section_id"] = section_id
                    section_ids_set    += 1

                    su_id = su_id_by_title.get(section_id, "")
                    if su_id:
                        chunk["structure_unit_id"] = su_id
                        su_ids_set += 1

            # ── Section reference linking ("see Section 3.2") ──────────
            # Upgrade 4: resolve "Section N.M" citations to structure_unit_ids.
            # Writes related_section_ids — a list of su_ids this chunk discusses.
            # The agent retriever uses this to follow cross-references between
            # sections without relying on embedding similarity alone.
            section_refs = find_section_references(content)
            if section_refs:
                related_su_ids = []
                for ref in section_refs:
                    ref_su_id = su_id_by_number.get(ref, "")
                    if ref_su_id and ref_su_id != chunk.get("structure_unit_id"):
                        related_su_ids.append(ref_su_id)
                if related_su_ids:
                    chunk["related_section_ids"] = list(dict.fromkeys(related_su_ids))
                    section_ref_links += len(related_su_ids)

            # ── Update denormalized strings for ChromaDB ───────────────
            # Upgrade 5: keep *_str fields in sync after linking.
            chunk["related_figures_str"] = json.dumps(
                chunk.get("related_figures") or []
            )
            chunk["related_tables_str"] = json.dumps(
                chunk.get("related_tables") or []
            )

            # ── Rebuild promoted_fields ────────────────────────────────
            # Upgrade 3: promoted_fields was built before cross-referencing.
            # Update structure_unit_id, tier1/tier2 so ChromaDB sees correct values.
            _rebuild_promoted_fields(chunk, doc_meta)

        # ── Pass 2: Image chunks — assign section to ALL images ────────
        # Upgrade 1: ALL images get section_id + structure_unit_id, not
        # just those cited as "Figure N". Uses page-range lookup.
        # Also builds back-links from img_to_text (explicit + proximity).
        img_section_ids_set = 0
        for ic in image_chunks:
            img_id = ic.get("chunk_id", "")

            # Back-links from text chunks that reference or share page
            text_ids = img_to_text.get(img_id, set())
            if text_ids:
                existing = set(ic.get("related_sections", []))
                ic["related_sections"] = sorted(existing | text_ids)
            elif not ic.get("related_sections"):
                # No text reference at all — link to text chunks on same page
                page = int(ic.get("page_number") or 0)
                same_page_texts = page_to_texts.get(page, [])
                if same_page_texts:
                    ic["related_sections"] = same_page_texts[:3]

            # Section + structure_unit_id assignment for ALL images
            if has_page_ranges:
                page_number = int(ic.get("page_number") or 0)
                if page_number > 0:
                    section_id = _find_section_by_page(page_number, section_map)
                    if section_id:
                        ic["section_id"] = section_id
                        su_id = su_id_by_title.get(section_id, "")
                        if su_id:
                            ic["structure_unit_id"] = su_id
                            img_section_ids_set    += 1

            # Update denormalized string and promoted_fields
            ic["related_sections_str"] = json.dumps(
                ic.get("related_sections") or []
            )
            _rebuild_promoted_fields(ic, doc_meta)

        # ── Pass 3: Table chunks ───────────────────────────────────────
        tbl_su_set = 0
        if has_page_ranges:
            for tbl_chunk in table_chunks:
                page_number = int(tbl_chunk.get("page_number") or 0)
                if page_number > 0:
                    section_id = _find_section_by_page(page_number, section_map)
                    if section_id:
                        tbl_chunk["section_id"] = section_id
                        su_id = su_id_by_title.get(section_id, "")
                        if su_id:
                            tbl_chunk["structure_unit_id"] = su_id
                            tbl_su_set += 1
                _rebuild_promoted_fields(tbl_chunk, doc_meta)

        # ── Hierarchical propagation: push chunk_ids to parent sections ─
        # Sections like "3.2 Attention" and "Abstract" span pages that are
        # dominated by subsections in page-range lookup — so they get 0 chunks.
        # Fix: for any structure unit with 0 chunk_ids, inherit chunk_ids from
        # structure units whose section_id begins with this unit's title number.
        # e.g. "3.2" inherits from "3.2.1", "3.2.2", "3.2.3".
        su_by_id = {su["structure_unit_id"]: su for su in struct_units}
        propagated = 0
        for su in struct_units:
            if su.get("chunk_ids"):
                continue
            su_title = su.get("title", "")
            # Find child units: structure units whose title starts with our title
            # e.g. "3.2" matches "3.2.1", "3.2.2", "3.2.3 Applications..."
            inherited_chunks = []
            inherited_figs   = []
            inherited_tables = []
            for other_su in struct_units:
                if other_su.get("structure_unit_id") == su.get("structure_unit_id"):
                    continue
                other_title = other_su.get("title", "")
                # Child if other starts with our title followed by . or space
                import re as _re
                if _re.match(r"^" + _re.escape(su_title) + r"[\s\.]", other_title):
                    inherited_chunks.extend(other_su.get("chunk_ids", []))
                    inherited_figs.extend(other_su.get("figure_ids", []))
                    inherited_tables.extend(other_su.get("table_ids", []))
            if inherited_chunks:
                su["chunk_ids"]  = list(dict.fromkeys(inherited_chunks))
                su["figure_ids"] = list(dict.fromkeys(su.get("figure_ids", []) + inherited_figs))
                su["table_ids"]  = list(dict.fromkeys(su.get("table_ids", []) + inherited_tables))
                propagated += 1

        if propagated:
            logger.info(f"  CrossRef: propagated chunk_ids to {propagated} parent sections")

        logger.info(
            f"  CrossRef document: "
            f"fig_explicit={total_fig_links} fig_proximity={proximity_links} "
            f"tbl={total_tbl_links} section_refs={section_ref_links} "
            f"text_su_ids={su_ids_set} img_su_ids={img_section_ids_set} "
            f"tbl_su_ids={tbl_su_set} parent_propagated={propagated}"
        )

        return {
            "figure_links":        total_fig_links,
            "proximity_links":     proximity_links,
            "table_links":         total_tbl_links,
            "section_ref_links":   section_ref_links,
            "section_ids":         section_ids_set,
            "structure_unit_ids":  su_ids_set + img_section_ids_set + tbl_su_set,
            "images_referenced":   len(img_to_text),
            "tables_sectioned":    tbl_su_set,
        }

    # ── Spreadsheet processing (CSV / XLSX) ────────────────────────────

    def _process_spreadsheet(
        self,
        table_chunks: List[Dict[str, Any]],
        struct_units: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """
        Links schema chunks to row-group chunks within each sheet.
        Assigns structure_unit_id from sheet-level structure units.
        """
        linked = link_schema_to_rowgroups(table_chunks)

        su_id_by_sheet: Dict[str, str] = {}
        for su in struct_units:
            sheet = su.get("sheet_name") or su.get("title", "")
            su_id = su.get("structure_unit_id", "")
            if sheet and su_id:
                su_id_by_sheet[sheet] = su_id

        su_ids_set = 0
        for chunk in linked:
            if not chunk.get("structure_unit_id"):
                sheet = (
                    chunk.get("sheet_name")
                    or (chunk.get("format_specific") or {}).get("sheet_name", "")
                )
                su_id = su_id_by_sheet.get(sheet, "")
                if su_id:
                    chunk["structure_unit_id"] = su_id
                    su_ids_set += 1

        schema_links = sum(1 for c in linked if c.get("schema_chunk_id"))
        logger.info(
            f"  CrossRef spreadsheet: schema_links={schema_links} "
            f"structure_unit_ids={su_ids_set}"
        )
        return {"schema_links": schema_links, "structure_unit_ids": su_ids_set}

    # ── Video processing ───────────────────────────────────────────────

    def _process_video(
        self,
        vid_segments: List[Dict[str, Any]],
        vid_frames:   List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """
        Links video segments to their scene siblings and validates
        frame-to-segment parent links.
        """
        # ── Scene sibling links ────────────────────────────────────────
        sibling_map = build_scene_sibling_map(vid_segments)
        for seg in vid_segments:
            seg_id = seg.get("chunk_id", "")
            seg["scene_sibling_ids"] = sibling_map.get(seg_id, [])

        # ── Repair parent_id on frames ────────────────────────────────
        for frame in vid_frames:
            if not frame.get("parent_id"):
                related = frame.get("related_sections") or []
                fallback = related[0] if related else ""
                if not fallback:
                    fallback = (
                        frame.get("format_specific") or {}
                    ).get("segment_id", "")
                if fallback:
                    frame["parent_id"] = fallback
                    logger.debug(
                        f"  VideoLinker: repaired parent_id on "
                        f"{frame.get('chunk_id')} → {fallback}"
                    )

        # ── Frame parent validation ────────────────────────────────────
        parent_lookup, orphaned = validate_frame_parent_links(
            vid_segments, vid_frames
        )
        for frame in vid_frames:
            frame_id  = frame.get("chunk_id", "")
            parent_id = frame.get("parent_id", "")
            if frame_id in parent_lookup:
                frame["parent_segment_id"] = parent_id

        if orphaned:
            logger.warning(
                f"  CrossRef video: {len(orphaned)} orphaned frames"
            )

        # ── Segment keyframe verification ─────────────────────────────
        vid_segments = build_segment_frame_map(vid_segments, vid_frames)

        scene_count = sum(1 for v in sibling_map.values() if v)
        logger.info(
            f"  CrossRef video: scene_links={scene_count} "
            f"frame_links={len(parent_lookup)} orphaned={len(orphaned)}"
        )

        return {
            "scene_links":     scene_count,
            "frame_links":     len(parent_lookup),
            "orphaned_frames": len(orphaned),
        }