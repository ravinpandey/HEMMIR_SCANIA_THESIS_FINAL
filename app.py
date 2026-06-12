"""
HEMMIR Interactive Demo — app.py

Streamlit front-end for the full HEMMIR 7-layer pipeline:
  Upload PDF → Enrich & Index (Docling + LLM) → Retrieve (R1-A)
  → Generate (R2-G ArgRAG) → Confidence Score (C) → Self-Correct (RQ3 routing)

Run:
    cd HEMMIR_updated
    streamlit run app.py --server.port 8501
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import logging

import streamlit as st

# BM25 — installed via: pip install rank-bm25
try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

# Per-chroma-dir BM25 index cache  {chroma_dir: {"ids": [...], "bm25": BM25Okapi}}
_BM25_INDEX_CACHE: Dict[str, dict] = {}

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="HEMMIR",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stMetric { background: #1e1e2e; border-radius: 8px; padding: 10px; }
  .stExpanderHeader { font-weight: 600; }
  .chunk-card { border-left: 3px solid #6366f1; padding: 8px 12px; margin: 4px 0; }
  code { font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_CHROMA_DIR   = str(ROOT / "chroma_app")
DEFAULT_TOP_K        = 10
# SPIQA-calibrated defaults (empirical: S4 AUROC=0.752, S3=0.733, S2=0.460 near-chance)
DEFAULT_W_FAITH  = 0.15   # S2 faith_new  — ceiling effect on SPIQA, low discriminability
DEFAULT_W_EVSTR  = 0.35   # S4 ev_support — strongest single signal (AUROC 0.752***)
DEFAULT_W_COMP   = 0.35   # S3 comp_new   — second strongest (AUROC 0.733***)
DEFAULT_W_DIRECT = 0.15   # Direct bonus  — ev_support_ratio >= threshold
HAL_CAP          = 0.35
COVERAGE_DIRECT_THR  = 0.80
MIN_TEXT_CHARS       = 80
MAX_TEXT_CHARS       = 1200

# ── faith_new / comp_new judge (reference-free, validated on SPIQA + Vectara) ─
_JUDGE_NEW_SYSTEM = """You are an expert reliability evaluation system for Retrieval-Augmented Generation (RAG).

Your task is to evaluate a generated answer using TWO separate reference-free reliability dimensions:

1. Faithfulness
   → Is the generated answer grounded in the retrieved evidence?
   → Detect hallucinations and unsupported claims.

2. Completeness
   → Does the generated answer sufficiently cover the important aspects required by the question?
   → Detect omission and missing information.

IMPORTANT:
- Do NOT use external knowledge.
- Do NOT assume the generated answer is correct.
- Do NOT use any gold/reference answer.
- Use ONLY: (a) the question, (b) the retrieved evidence, (c) the generated answer.

========================================================
PART 1 — FAITHFULNESS EVALUATION
========================================================
STEP 1A — Decompose the generated answer into atomic factual claims.
Each claim must contain exactly ONE factual proposition. Keep claims concise.
Return: [{"id": "C1", "text": "..."}, ...]

STEP 1B — For EACH claim determine:
SUPPORTED (evidence explicitly confirms) → 1.0
PARTIALLY_SUPPORTED (evidence supports only part) → 0.5
UNSUPPORTED (evidence does not justify) → 0.0
CONTRADICTED (evidence conflicts) → 0.0
Return: [{"claim_id": "...", "label": "...", "confidence": 0.0-1.0, "reason": "..."}]

STEP 1C — Faithfulness = sum(claim scores) / number of claims

========================================================
PART 2 — COMPLETENESS EVALUATION
========================================================
STEP 2A — From the QUESTION ALONE generate the atomic semantic requirements
needed to fully answer it. Do NOT use the evidence or generated answer.
Return: [{"id": "R1", "text": "..."}, ...]

STEP 2B — For EACH requirement determine whether retrieved evidence contains
enough information: SUPPORTED (true/false).
Return: [{"req_id": "...", "supported": true/false, "confidence": 0.0-1.0, "reason": "..."}]

STEP 2C — ONLY for evidence-supported requirements, determine whether the
generated answer covers them: FULL → 1.0 / PARTIAL → 0.5 / MISSING → 0.0
Return: [{"unit_id": "...", "coverage": "FULL|PARTIAL|MISSING", "confidence": 0.0-1.0, "reason": "..."}]

STEP 2D — Completeness = sum(coverage scores) / number of evidence-supported requirements

========================================================
FINAL OUTPUT FORMAT (STRICT JSON — no markdown, no code fences)
========================================================
{
  "faithfulness": {
    "claims": [...],
    "claim_verification": [...],
    "score": 0.0,
    "summary": "..."
  },
  "completeness": {
    "requirements": [...],
    "requirement_support": [...],
    "coverage": [...],
    "score": 0.0,
    "summary": "..."
  },
  "final_assessment": {
    "overall_reliability": "HIGH | MODERATE | LOW",
    "reason": "..."
  }
}"""

_JUDGE_NEW_USER = """QUESTION:
{question}

RETRIEVED EVIDENCE:
{evidence}

GENERATED ANSWER:
{answer}

Perform the full evaluation following all steps. Return ONLY the JSON object."""


def _run_judge_new(llm, question: str, evidence: str, answer: str) -> dict:
    """
    Single LLM call that returns both faith_new and comp_new.
    Uses the validated prompt from RAVI_THESIS/rag-reliability/src/run_judge_new.py.
    """
    import re as _re
    if not answer or not answer.strip():
        return {"faith_new": 0.0, "comp_new": 0.0,
                "evidence_support_ratio": 0.0,
                "n_claims": 0, "n_unsupported_from_judge": 0,
                "n_requirements": 0, "n_supported_reqs": 0,
                "missing_requirements": [],
                "faith_summary": "", "comp_summary": "", "reliability": "LOW"}

    # Abstained answers make no claims → faithful by definition, completeness = 0
    if answer.strip().lower().startswith("insufficient evidence"):
        return {"faith_new": 1.0, "comp_new": 0.0,
                "evidence_support_ratio": 0.0,
                "n_claims": 0, "n_unsupported_from_judge": 0,
                "n_requirements": 0, "n_supported_reqs": 0,
                "missing_requirements": [],
                "faith_summary": "abstained — no claims to verify",
                "comp_summary": "abstained — no coverage",
                "reliability": "LOW"}

    prompt = _JUDGE_NEW_USER.format(
        question=question,
        evidence=evidence[:4000],
        answer=answer[:1500],
    )
    try:
        raw = llm.invoke(system=_JUDGE_NEW_SYSTEM, prompt=prompt,
                         max_tokens=4096, temperature=0)
        # Strip markdown fences if present
        clean = _re.sub(r"```(?:json)?", "", raw).strip()
        s = clean.find("{"); e = clean.rfind("}") + 1
        result = json.loads(clean[s:e]) if s >= 0 and e > s else {}
    except Exception as exc:
        return {"faith_new": 0.5, "comp_new": 0.5,
                "evidence_support_ratio": 0.5,
                "n_claims": 0, "n_unsupported_from_judge": 0,
                "n_requirements": 0, "n_supported_reqs": 0,
                "missing_requirements": [],
                "faith_summary": f"judge error: {exc}",
                "comp_summary": "", "reliability": "LOW"}

    # Extract faithfulness
    faith_data   = result.get("faithfulness", {})
    faith_score  = float(faith_data.get("score", 0.0))
    claims       = faith_data.get("claims", []) or []
    n_claims     = len(claims)
    claim_verif  = faith_data.get("claim_verification", []) or []
    n_unsupported_from_judge = sum(
        1 for cv in claim_verif
        if cv.get("label", "").upper() in ("UNSUPPORTED", "CONTRADICTED")
    )

    # Extract completeness
    comp_data        = result.get("completeness", {})
    comp_score       = float(comp_data.get("score", 0.0))
    requirements     = comp_data.get("requirements", []) or []
    req_support      = comp_data.get("requirement_support", []) or []
    n_requirements   = len(requirements)
    # Robust boolean coercion: LLM sometimes returns "true"/"false" (string) instead
    # of true/false (JSON bool). Strict `is True` would count all as 0, tanking S4.
    def _is_supported(val) -> bool:
        if isinstance(val, bool): return val
        if isinstance(val, str):  return val.strip().lower() in ("true", "yes", "1")
        return bool(val)
    n_supported_reqs = sum(1 for r in req_support if _is_supported(r.get("supported")))
    ev_support_ratio = n_supported_reqs / max(n_requirements, 1)

    # Fallback: independently recompute comp_score from structured coverage array.
    # LLM sometimes omits STEP 2D (score field = 0.0). Recompute from coverage labels
    # so comp is never silently zeroed when the answer is actually good.
    coverage_list_pre = comp_data.get("coverage", []) or []
    if comp_score == 0.0 and n_supported_reqs > 0 and coverage_list_pre:
        _cov_map = {"FULL": 1.0, "PARTIAL": 0.5, "MISSING": 0.0}
        _sup_ids = {r.get("req_id", "") for r in req_support if _is_supported(r.get("supported"))}
        _cov_vals = [
            _cov_map.get(c.get("coverage", "").upper(), 0.0)
            for c in coverage_list_pre
            if c.get("unit_id", "") in _sup_ids
        ]
        if _cov_vals:
            comp_score = sum(_cov_vals) / len(_cov_vals)

    # Extract missing/partial requirements — used as gap_repair query seeds
    req_id_to_text = {r.get("id", ""): r.get("text", "") for r in requirements}
    supported_req_ids = {r.get("req_id", "") for r in req_support if _is_supported(r.get("supported"))}
    coverage_list = comp_data.get("coverage", []) or []
    missing_requirements: list = []
    for cov in coverage_list:
        if cov.get("coverage") in ("MISSING", "PARTIAL"):
            rid = cov.get("unit_id", "")
            txt = req_id_to_text.get(rid, "")
            if txt:
                missing_requirements.append(txt)

    # Build full requirements detail for UI display
    coverage_by_rid = {c.get("unit_id", ""): c for c in coverage_list}
    support_by_rid  = {r.get("req_id", ""): r for r in req_support}
    requirements_detail: list = []
    for req in requirements:
        rid = req.get("id", "")
        cov = coverage_by_rid.get(rid, {})
        sup = support_by_rid.get(rid, {})
        raw_cov = cov.get("coverage", "")
        if not raw_cov:
            # LLM omitted this req from coverage array — infer from requirement_support
            sup_val = sup.get("supported")
            raw_cov = "FULL" if _is_supported(sup_val) else (
                      "MISSING" if sup_val is not None else "UNKNOWN")
        requirements_detail.append({
            "id":         rid,
            "text":       req.get("text", ""),
            "coverage":   raw_cov,
            "cov_reason": cov.get("reason", sup.get("reason", "")),
            "supported":  _is_supported(sup.get("supported", False)),
            "sup_reason": sup.get("reason", ""),
        })

    final = result.get("final_assessment", {})
    return {
        "faith_new":                  round(max(0.0, min(1.0, faith_score)), 4),
        "comp_new":                   round(max(0.0, min(1.0, comp_score)), 4),
        "evidence_support_ratio":     round(ev_support_ratio, 4),
        "n_claims":                   n_claims,
        "n_unsupported_from_judge":   n_unsupported_from_judge,
        "n_requirements":             n_requirements,
        "n_supported_reqs":           n_supported_reqs,
        "missing_requirements":       missing_requirements,   # gap_repair seeds
        "requirements_detail":        requirements_detail,    # full detail for UI
        "faith_summary":              str(faith_data.get("summary", "")),
        "comp_summary":               str(comp_data.get("summary", "")),
        "reliability":                str(final.get("overall_reliability", "")),
    }


# Keywords that flag a question as asking about document metadata
_META_KEYWORDS = frozenset([
    "author", "authors", "who wrote", "written by", "researcher", "researchers",
    "title", "paper title", "what is the title", "paper called", "paper named",
    "publication year", "when was published", "when was it published",
    "journal", "venue", "workshop", "proceedings",
    "abstract", "what is this paper about", "what does this paper",
    "institution", "university", "affiliation", "department",
    "which organization published", "what organization", "organization behind",
    "doi", "arxiv id", "arxiv number",
    # Presentation / PPTX metadata
    "who presented", "who gave", "presenter", "presented by",
    "on what date", "what date was", "when was this presented", "presentation date",
    "slide author", "who created this", "who made this presentation",
])

MODALITY_ICON = {"text": "📝", "table": "📊", "image": "🖼️"}
ROLE_COLOR    = {
    "methodology": "#6366f1",
    "finding":     "#10b981",
    "measurement": "#f59e0b",
    "background":  "#6b7280",
    "reference":   "#8b5cf6",
}

# ── LLM enrichment prompts ────────────────────────────────────────────────────
_ENRICH_SYS = "You are a scientific text analyst. Respond ONLY with valid JSON."

_TEXT_ENRICH = """\
Analyze this chunk from a scientific paper. Return JSON with exactly these keys:

text: {text}

{{
  "semantic_anchor":    "<one sentence capturing the core claim or finding>",
  "evidence_role":      "<one of: methodology|finding|measurement|background|reference>",
  "salience_score":     <0.0-1.0, importance to the paper's main contribution>,
  "contextual_summary": "<2-3 sentences providing context>"
}}"""

_TABLE_ENRICH = """\
Analyze this table from a scientific paper. Return JSON:

table: {text}

{{
  "semantic_anchor": "<one sentence: what this table shows>",
  "evidence_role":   "<one of: measurement|finding|methodology|background|reference>",
  "salience_score":  <0.0-1.0>,
  "contextual_summary": "<what question does this table answer?>"
}}"""

_IMAGE_ENRICH = """\
Analyze this figure description from a scientific paper. Return JSON:

figure: {text}

{{
  "semantic_anchor": "<one sentence: what this figure shows>",
  "evidence_role":   "<one of: finding|methodology|measurement|background|reference>",
  "salience_score":  <0.0-1.0>,
  "contextual_summary": "<what insight does this figure provide?>"
}}"""


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _doc_id(name: str, ext: str = ".pdf") -> str:
    """MD5(name<ext>)[:12] — matches ingestion_layer extractor convention."""
    return hashlib.md5(f"{name}{ext}".encode()).hexdigest()[:12]

def _chunk_id(doc_id_hash: str, modality: str, idx: int) -> str:
    return f"{doc_id_hash}_{modality}_{idx:04d}"

def _get_weights() -> tuple[float, float, float, float]:
    """Return (w_faith, w_evstr, w_comp, w_direct) from session state or defaults."""
    return (
        st.session_state.get("w_faith",  DEFAULT_W_FAITH),
        st.session_state.get("w_evstr",  DEFAULT_W_EVSTR),
        st.session_state.get("w_comp",   DEFAULT_W_COMP),
        st.session_state.get("w_direct", DEFAULT_W_DIRECT),
    )


def _parse_json(text: str) -> dict:
    import re
    clean = re.sub(r"```(?:json)?", "", text).strip()
    s = clean.find("{"); e = clean.rfind("}") + 1
    if s >= 0 and e > s:
        try:
            return json.loads(clean[s:e])
        except Exception:
            pass
    return {}

def _c_score(faith: float, evstr: float, comp: float, coverage: float,
             abstained: bool, unsupported: int, total_claims: int) -> float:
    w_faith, w_evstr, w_comp, w_direct = _get_weights()
    direct = w_direct if coverage >= COVERAGE_DIRECT_THR else 0.0
    raw    = w_faith * faith + w_evstr * evstr + w_comp * comp + direct
    if abstained or (total_claims > 0 and unsupported / total_claims >= 0.5):
        raw = min(raw, HAL_CAP)
    return round(max(0.0, min(1.0, raw)), 4)

def _evstr_cosine(dense_scores: List[float]) -> float:
    """Mean cosine similarity of retrieved chunks to query — real evidence relevance."""
    if not dense_scores:
        return 0.30
    return round(max(0.30, min(0.95, sum(dense_scores) / len(dense_scores))), 4)

def _is_metadata_question(question: str) -> bool:
    """True when the question asks about document-level facts (authors, title, year…)."""
    q = question.lower()
    return any(kw in q for kw in _META_KEYWORDS)

def _score_color(v: float) -> str:
    if v >= 0.70: return "#10b981"
    if v >= 0.50: return "#f59e0b"
    return "#ef4444"


# ── Cached pipeline resources ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading text embedder…")
def get_embedder():
    from embedding_layer.embedders.text_embedder import TextEmbedder
    return TextEmbedder()


@st.cache_resource(show_spinner="Connecting to LLM…")
def get_llm(provider: str, api_key: str):
    from enrichment_layer.utils.llm_client import build_llm_client
    kwargs = {}
    if provider == "anthropic" and api_key:
        kwargs["api_key"] = api_key
    return build_llm_client(provider=provider, **kwargs)


@st.cache_resource(show_spinner="Opening ChromaDB…")
def get_db(chroma_dir: str):
    import chromadb
    client = chromadb.PersistentClient(path=chroma_dir)
    # Six collections created by the full ingestion pipeline + doc_meta (app-managed)
    colls = {
        "documents":   client.get_or_create_collection("documents_collection"),
        "sections":    client.get_or_create_collection("sections_collection"),
        "text":        client.get_or_create_collection("text_chunks"),
        "tables":      client.get_or_create_collection("table_chunks"),
        "images_text": client.get_or_create_collection("image_chunks_text"),
        "images_clip": client.get_or_create_collection("image_chunks_clip"),
        "doc_meta":    client.get_or_create_collection("doc_meta"),
    }
    return client, colls


# ── Ingestion — full 5-layer pipeline ────────────────────────────────────────

WORKSPACE_DIR = ROOT / "app_workspace"
_PYTHON       = "/usr/bin/python3.12"   # the interpreter that has all pipeline deps

# ── Answer History (persistent JSONL log) ─────────────────────────────────────

HISTORY_FILE = WORKSPACE_DIR / "answer_history.jsonl"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


def _build_history_record(
    record_id: str,
    question: str,
    gen: dict,
    score: dict,
    hits: list,
    repair_history: list,
    was_repaired: bool,
) -> dict:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    # Chunks actually used in generation (may differ from all retrieved hits after repair)
    chunks_used = [
        {
            "chunk_id":    cid,
            "modality":    mod,
            "text_preview": txt[:250],
        }
        for cid, mod, txt in zip(
            gen.get("chunk_ids", []),
            gen.get("modalities", []),
            gen.get("texts", []),
        )
    ]

    # All retrieved hits — summary for provenance
    retrieved_summary = [
        {
            "chunk_id":    h["chunk_id"],
            "modality":    h["modality"],
            "score":       round(h.get("score", 0.0), 4),
            "doc":         h.get("meta", {}).get("arxiv_id",
                           h.get("meta", {}).get("source_id",
                           h.get("meta", {}).get("doc_id", "—"))),
            "page":        h.get("meta", {}).get("page_number", "?"),
            "text_preview": h["text"][:250],
        }
        for h in hits
    ]

    return {
        "record_id":        record_id,
        "timestamp":        now,
        "question":         question,
        "answer":           gen.get("answer", ""),
        "abstained":        bool(gen.get("abstained", False)),
        "generation_path":  gen.get("generation_path", "r2g"),
        "c_score":          score.get("c_score", 0.0),
        "faithfulness":     score.get("faithfulness", 0.0),
        "evidence_strength": score.get("evidence_strength", 0.0),
        "completeness":     score.get("completeness", 0.0),
        "direct_bonus":     score.get("direct_bonus", 0.0),
        "n_claims":         score.get("n_claims", 0),
        "n_requirements":   score.get("n_requirements", 0),
        "n_supported_reqs": score.get("n_supported_reqs", 0),
        "was_repaired":     was_repaired,
        "repair_history":   repair_history,
        "chunks_used":      chunks_used,
        "retrieved_chunks": retrieved_summary,
    }


def save_history_record(record: dict) -> None:
    """Append a new record to the JSONL history file."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_history_record(record_id: str, updated: dict) -> None:
    """Replace the record matching record_id in place (rewrite the file)."""
    records = load_history()
    found = False
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        for r in records:
            if r.get("record_id") == record_id:
                f.write(json.dumps(updated, ensure_ascii=False) + "\n")
                found = True
            else:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not found:
        save_history_record(updated)


def load_history() -> list:
    """Return all records from the JSONL history file, newest first."""
    if not HISTORY_FILE.exists():
        return []
    records = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return list(reversed(records))  # newest first


_EXT_TO_FORMAT = {".pdf": "pdf", ".pptx": "pptx", ".ppt": "pptx"}


def ingest_document(file_bytes: bytes, doc_name: str, file_ext: str, provider: str,
                    chroma_dir: str, log_q: queue.Queue) -> None:
    """
    Run the full 5-layer HEMMIR pipeline as subprocesses for any supported format:
      pdf, pptx, ppt
      1. ingestion_layer  — Docling extract → raw chunks
      2. enrichment_layer — LLM semantic anchor / evidence role / salience
      3. encoding_layer   — retrieval views, cross-reference linkage
      4. embedding_layer  — dense text + CLIP visual vectors
      5. indexing_layer   — upsert into ChromaDB (6 collections)

    Streams each layer's stdout to log_q.
    Sends __DONE__<doc_id> on success or __DONE__error on failure.
    """
    ext    = file_ext.lower() if file_ext.startswith(".") else f".{file_ext.lower()}"
    fmt    = _EXT_TO_FORMAT.get(ext, "pdf")

    workspace  = WORKSPACE_DIR / doc_name
    input_dir  = workspace / "input"
    output_dir = workspace / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = input_dir / f"{doc_name}{ext}"
    file_path.write_bytes(file_bytes)
    doc_id = _doc_id(doc_name, ext)

    log_q.put(f"📄 {doc_name}{ext}  (doc_id={doc_id})")

    steps = [
        {
            "name": "1/5 Ingestion — Docling extract",
            "cmd":  [_PYTHON, str(ROOT / "ingestion_layer" / "main_multiformat.py"),
                     "--data-dir",   str(input_dir),
                     "--source-id",  doc_name,
                     "--output-dir", str(output_dir),
                     "--formats",    fmt],
        },
        {
            "name": "2/5 Enrichment — LLM metadata",
            "cmd":  [_PYTHON, str(ROOT / "enrichment_layer" / "main.py"),
                     "--output-dir", str(output_dir),
                     "--doc-name",   doc_name,
                     "--provider",   provider],
        },
        {
            "name": "3/5 Encoding — retrieval views",
            "cmd":  [_PYTHON, str(ROOT / "encoding_layer" / "main.py"),
                     "--output-dir", str(output_dir),
                     "--doc-name",   doc_name],
        },
        {
            "name": "4/5 Embedding — dense vectors",
            "cmd":  [_PYTHON, str(ROOT / "embedding_layer" / "main.py"),
                     "--output-dir", str(output_dir),
                     "--doc-name",   doc_name],
        },
        {
            "name": "5/5 Indexing — ChromaDB upsert",
            "cmd":  [_PYTHON, str(ROOT / "indexing_layer" / "main.py"),
                     "--mode",       "index",
                     "--output-dir", str(output_dir),
                     "--chroma-dir", chroma_dir,
                     "--doc-name",   doc_name],
        },
    ]

    import subprocess
    for step in steps:
        log_q.put(f"\n⏳ {step['name']}")
        proc = subprocess.Popen(
            step["cmd"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT),
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log_q.put(f"    {line}")
        proc.wait()
        if proc.returncode != 0:
            log_q.put(f"  ❌ {step['name']} failed (exit {proc.returncode})")
            log_q.put("__DONE__error")
            return
        log_q.put(f"  ✅ {step['name']} done")

    log_q.put(f"\n  ✅ {doc_name} fully indexed into ChromaDB (doc_id={doc_id})")
    log_q.put(f"__DONE__{doc_id}")


# ── Retrieval (R1-A hierarchical) ─────────────────────────────────────────────

def _make_where(field: str, ids: List[str]) -> Optional[dict]:
    """Build a ChromaDB where filter for one or many IDs."""
    if not ids:
        return None
    if len(ids) == 1:
        return {field: {"$eq": ids[0]}}
    return {field: {"$in": ids}}


def _expand_doc_filter(doc_id: str, colls: dict) -> List[str]:
    """
    Return doc_id plus all other doc_ids that share the same source_id.

    When the same source file was ingested as both a PDF (Docling) and a PPTX
    (ingest_pptx_research), they get different doc_ids but the same source_id.
    Expanding the filter ensures both versions are searched together, so PPTX
    context chunks (which often contain richer slide-level descriptions, presenter
    names, dates, and annotated measurements) are not silently excluded.
    """
    dc = colls.get("documents")
    if dc is None:
        return [doc_id]
    try:
        r = dc.get(ids=[doc_id], include=["metadatas"])
        if not r or not r.get("metadatas"):
            return [doc_id]
        source_id = r["metadatas"][0].get("source_id", "")
        if not source_id:
            return [doc_id]
        all_docs = dc.get(include=["metadatas"])
        ids_out = [doc_id]
        for i, meta in enumerate(all_docs["metadatas"]):
            alt_id = all_docs["ids"][i]
            if meta.get("source_id") == source_id and alt_id != doc_id:
                ids_out.append(alt_id)
        if len(ids_out) > 1:
            logger.debug(
                "  _expand_doc_filter: %s → %d docs (source_id=%s)",
                doc_id, len(ids_out), source_id,
            )
        return ids_out
    except Exception:
        return [doc_id]


def _add_doc_cards(
    doc_ids: List[str],
    colls: dict,
    emb: List[float],
    results: List[Dict],
    seen_ids: set,
) -> None:
    """Inject documents_collection cards into results (score=1.0, always first)."""
    dc = colls.get("documents")
    if dc is None:
        return
    card_found = False
    try:
        r = dc.get(ids=doc_ids, include=["documents", "metadatas"])
        for cid, text, meta in zip(r.get("ids", []), r.get("documents", []), r.get("metadatas", [])):
            card_id = f"{cid}_doccard"
            if card_id in seen_ids:
                continue
            results.append({
                "chunk_id":    card_id,
                "text":        text or "",
                "modality":    "text",
                "dense_score": 1.0,
                "score":       1.0,
                "meta": {
                    **meta,
                    "doc_id":          cid,
                    "is_doc_card":     1,
                    "semantic_anchor": meta.get("doc_title", "Document metadata card"),
                },
            })
            seen_ids.add(card_id)
            card_found = True
    except Exception:
        pass

    # Fallback: documents_collection has no record → pull early text_chunks (page ≤ 2)
    if not card_found and colls.get("text"):
        try:
            tc    = colls["text"]
            where = _make_where("doc_id", doc_ids)
            all_r = tc.get(limit=200, where=where, include=["metadatas", "documents"])
            pairs = list(zip(all_r.get("metadatas", []), all_r.get("documents", [])))
            early = sorted(
                [(m, d) for m, d in pairs if int(m.get("page_number") or 99) <= 2],
                key=lambda x: int(x[0].get("chunk_index") or 99),
            )[:4] or sorted(pairs, key=lambda x: int(x[0].get("chunk_index") or 99))[:4]
            for i, (meta, doc) in enumerate(early):
                sid = f"{meta.get('doc_id', 'unknown')}_autocard_{i}"
                if sid in seen_ids:
                    continue
                results.append({
                    "chunk_id":    sid,
                    "text":        doc or "",
                    "modality":    "text",
                    "dense_score": 0.80,
                    "score":       1.0,
                    "meta": {**meta, "is_doc_card": 1,
                             "semantic_anchor": "Auto-card: early page text"},
                })
                seen_ids.add(sid)
        except Exception:
            pass


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split — used for BM25."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


# Common English stop words removed from BM25 queries so scoring focuses on
# content words (nouns, measurements, technical terms) rather than function words.
_BM25_STOPWORDS = frozenset([
    "what", "which", "where", "when", "who", "why", "how",
    "is", "are", "was", "were", "be", "been", "being",
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "it", "its", "this", "that", "these", "those",
    "do", "does", "did", "can", "could", "should", "would", "will",
    "have", "has", "had", "with", "by", "from", "as", "into", "about",
    "between", "during", "before", "after", "above", "below",
    "there", "their", "they", "them", "then", "than",
])


def _bm25_query_tokens(text: str) -> List[str]:
    """Tokenise query and drop stop words so BM25 matches on content terms only."""
    tokens = _tokenize(text)
    content = [t for t in tokens if t not in _BM25_STOPWORDS and len(t) > 1]
    return content if content else tokens  # fallback: never return empty list


def _enrich_table_text(doc_text: str, meta: dict, chunk_id: str = "") -> str:
    """
    Build a readable evidence string for table chunks that includes actual cell
    data. ChromaDB stores only caption+summary as the document string, and the
    metadata table_html is truncated to 500 chars (header only). The full HTML
    (all rows) is written to disk during ingestion — reconstruct the path from
    source_id + page/table numbers encoded in chunk_id and read it directly.
    Falls back to the truncated metadata HTML if the disk file is not found.
    """
    caption   = (meta.get("table_caption") or "").strip()
    summary   = (meta.get("table_summary") or "").strip()
    source_id = (meta.get("source_id")     or "").strip()
    html      = ""

    # ── Try to read the full HTML from disk ───────────────────────────────────
    if chunk_id and source_id:
        m = re.search(r"_page_(\d+)_tbl_(\d+)$", chunk_id)
        if m:
            page_str = m.group(1)   # e.g. '005'
            tbl_str  = m.group(2)   # e.g. '001'
            html_path = (WORKSPACE_DIR / source_id / "output" / source_id
                         / "tables"
                         / f"{source_id}_page_{page_str}_tbl_{tbl_str}.html")
            if html_path.exists():
                try:
                    html = html_path.read_text(encoding="utf-8")
                except Exception:
                    pass

    # ── Fall back to truncated metadata HTML ──────────────────────────────────
    if not html:
        html = (meta.get("table_html") or "").strip()

    # Strip HTML tags — LLM gets clean tabular text, not raw markup
    html_text = re.sub(r"<[^>]+>", " ", html)
    html_text = re.sub(r"\s+", " ", html_text).strip()

    parts = []
    if caption:
        parts.append(f"Table: {caption}")
    if summary:
        parts.append(summary)
    if html_text:
        parts.append(html_text[:2500])

    enriched = "\n".join(parts).strip()
    return enriched if enriched else (doc_text or "")


def _bm25_score_candidates(query_text: str, chunks: List[Dict]) -> List[float]:
    """
    Compute BM25 scores for `query_text` against a small candidate set.
    Returns normalised scores [0, 1] aligned with `chunks`.

    Corpus text = bm25_text metadata (enriched keywords/annotations) concatenated
    with the chunk's document text (vision description for images, body for text/tables).
    This gives richer signal than document text alone, especially for image chunks
    where bm25_text captures visible annotations not always present in the description.

    Falls back to zeros if rank_bm25 is not installed or corpus is empty.
    """
    if not _BM25_AVAILABLE or not chunks:
        return [0.0] * len(chunks)
    # Combine enriched metadata keywords with document text for a richer BM25 corpus
    texts = [
        (c.get("meta", {}).get("bm25_text", "") + " " + c.get("text", "")).strip()
        for c in chunks
    ]
    tokenized = [_tokenize(t) for t in texts]
    if not any(tokenized):
        return [0.0] * len(chunks)
    try:
        bm25   = _BM25Okapi(tokenized)
        scores = bm25.get_scores(_bm25_query_tokens(query_text))
        max_s  = float(max(scores)) if max(scores) > 0 else 1.0
        return [float(s / max_s) for s in scores]
    except Exception:
        return [0.0] * len(chunks)


# ── Scania-aware query rewriter ───────────────────────────────────────────────

_SCANIA_REWRITE_SYSTEM = """\
You are a precision search-query optimizer for Scania vehicle technical documentation.
Scania documents include: service manuals, workshop guides, technical bulletins (TBSA/TBIS),
parts catalogs, fault code databases, maintenance schedules, and engine/transmission specs.
"""

_SCANIA_REWRITE_PROMPT = """\
Rewrite the question below into a dense, search-optimized technical query that maximizes
retrieval of the most relevant passages from Scania documentation.

RULES — follow every rule strictly:

1. PRESERVE EXACTLY (never alter):
   - Part/component numbers: e.g. 1759869, 2177536, R730
   - Fault codes: SPN xxx FMI x, DTC Pxxxx, SA x
   - Model/engine designations: D13K500, GRS905R, OC9G290, DC13, DC09, GR875
   - Torque values, measurements, and any numeric specs

2. EXPAND abbreviations to full Scania technical terms:
   EGR → Exhaust Gas Recirculation EGR valve
   DEF / AdBlue → Diesel Exhaust Fluid SCR dosing
   SCR → Selective Catalytic Reduction aftertreatment
   DPF → Diesel Particulate Filter DPF regeneration
   VGT → Variable Geometry Turbocharger
   EBS → Electronic Braking System
   ABS → Anti-lock Braking System
   PTO → Power Take-Off
   ECU / EMS → Engine Control Unit Engine Management System
   CPC → Coordinator Processor Communication
   OBD → On-Board Diagnostics fault code
   ACM → Aftertreatment Control Module
   NOx → Nitrogen Oxide sensor emission

3. MAP user vocabulary to Scania manual vocabulary:
   "oil leak" → "lubricant seepage oil leak gasket seal failure"
   "won't start / no start" → "engine cranking no start starting failure"
   "check / inspect" → "inspection verification functional check procedure"
   "replace / swap" → "replacement installation removal procedure"
   "torque" → "torque specification tightening torque Nm"
   "fault / error / warning" → "fault code diagnostic trouble code DTC SPN FMI"
   "reset" → "reset calibration parameter reset after repair"
   "bleed" → "bleeding procedure air purge pressure relief"
   "filter" → "filter replacement service interval"
   "coolant" → "coolant temperature cooling system antifreeze"

4. CONVERT question form to noun phrases (remove how/what/why/where):
   "How do I replace X?" → "X replacement removal installation procedure steps"
   "What causes X?" → "X cause root cause diagnosis fault troubleshooting"
   "What is the torque for X?" → "X torque specification tightening value Nm"
   "Where is X located?" → "X location position assembly diagram"

5. ADD modality hints when content type is implied:
   Numeric spec question → append "specification table"
   Procedure question → append "procedure step-by-step"
   Location/diagram question → append "diagram exploded view"
   Fault/DTC question → append "fault code cause corrective action"

OUTPUT: Return ONLY the rewritten search query — 1 to 2 lines of technical terms.
No explanation, no markdown, no bullet points.

Question: {question}
Rewritten search query:"""


# Signals that indicate the question is genuinely about Scania service-manual content.
# Only apply the Scania vocabulary rewriter when these appear — prevents it from
# corrupting queries about PPTX/presentation content (spatial layouts, diagrams, etc.)
_SCANIA_SERVICE_SIGNALS = frozenset([
    "egr", "dpf", "def", "adblue", "scr", "vgt", "ebs", "pto",
    "ecu", "ems", "cpc", "obd", "acm", "nox",
    "spn", "fmi", "dtc", "fault code", "fault codes",
    "r730", "d13", "dc13", "dc09", "dc16", "grs", "oc9", "g380", "g400",
    "tbsa", "tbis", "torque spec", "tightening torque",
    "service manual", "workshop manual", "repair manual",
    "oil leak", "coolant leak", "won't start", "no start",
])


def _needs_scania_rewrite(question: str) -> bool:
    """True only when the question contains Scania service-manual vocabulary."""
    q = question.lower()
    return any(sig in q for sig in _SCANIA_SERVICE_SIGNALS)


def _rewrite_query(llm, question: str) -> str:
    """
    Rewrite `question` into a Scania-optimised dense search query.
    Skipped for questions that don't contain Scania service-manual signals —
    the Scania rewriter adds EGR/DPF/fault-code vocabulary that hurts retrieval
    for PPTX/presentation content (spatial diagrams, speaker, pillar lists).
    Returns original question unchanged on any failure or when rewrite is not needed.
    """
    if llm is None:
        return question
    if not _needs_scania_rewrite(question):
        logger.info(f"  [QueryRewrite] skipped (no Scania signals): '{question[:80]}'")
        return question
    try:
        rewritten = llm.invoke(
            system    = _SCANIA_REWRITE_SYSTEM,
            prompt    = _SCANIA_REWRITE_PROMPT.format(question=question),
            max_tokens = 120,
            temperature = 0.0,
        )
        rewritten = rewritten.strip()
        if rewritten and len(rewritten) > 5:
            logger.info(f"  [QueryRewrite] '{question[:60]}' → '{rewritten[:80]}'")
            return rewritten
    except Exception as e:
        logger.warning(f"  [QueryRewrite] failed: {e}")
    return question


def _query_chunks(
    colls: dict,
    emb: List[float],
    where: Optional[dict],
    top_k: int,
    seen_ids: set,
    query_text: str = "",
    section_ids: Optional[set] = None,
) -> List[Dict]:
    """
    Query text + table + image collections with the given where filter.
    Score = 0.40·dense + 0.25·salience + 0.10·evidence_quality + 0.25·bm25
            + 0.10 section_boost (if chunk's section_id or structure_unit_id
              matches a retrieved section — soft signal, not hard filter).
    Using doc_id as the primary filter avoids the _sec_ vs _slide_ mismatch
    that occurs when PPT chunks use slide-based section IDs.
    """
    out = []
    mod_map = {"text": "text", "tables": "table", "images_text": "image"}
    for coll_name, modality in mod_map.items():
        coll = colls.get(coll_name)
        if coll is None:
            continue
        try:
            count = coll.count()
            if count == 0:
                continue
            n_req = min(top_k * 5, count)
            r = coll.query(
                query_embeddings=[emb],
                n_results=n_req,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            candidates = []
            for cid, text, meta, dist in zip(
                r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]
            ):
                if cid in seen_ids:
                    continue
                # For table chunks: read full HTML from disk using chunk_id to
                # reconstruct the file path. ChromaDB only stores 500 chars of
                # HTML (header only); the disk file has all data rows.
                chunk_text = (
                    _enrich_table_text(text, meta, chunk_id=cid)
                    if modality == "table"
                    else (text or "")
                )
                candidates.append({
                    "chunk_id":    cid,
                    "text":        chunk_text,
                    "modality":    modality,
                    "dense_score": round(max(0.0, 1.0 - float(dist)), 4),
                    "meta":        meta,
                })

            # BM25 scores over this candidate set (exact-term signal)
            bm25_scores = _bm25_score_candidates(query_text, candidates) if query_text else [0.0] * len(candidates)

            for chunk, bm25 in zip(candidates, bm25_scores):
                dense    = chunk["dense_score"]
                salience = float(chunk["meta"].get("salience_score", 0.5))
                eq       = float(chunk["meta"].get("evidence_quality",
                                   chunk["meta"].get("evidence_role_confidence", 0.7)))
                # Section boost: chunk belongs to a semantically relevant section
                sec_hit  = section_ids and (
                    chunk["meta"].get("section_id")        in section_ids or
                    chunk["meta"].get("structure_unit_id") in section_ids
                )
                sec_boost = 0.10 if sec_hit else 0.0
                # Image-type boost: layout diagrams and flowcharts contain structured
                # visual information (station names, process flows) that text metadata
                # cannot capture — give them a score lift to compensate for missing
                # salience/eq metadata fields that text chunks have.
                img_type_boost = 0.0
                if chunk["modality"] == "image":
                    img_type = chunk["meta"].get("image_type", "")
                    if img_type in ("layout_diagram", "flowchart",
                                    "architecture_diagram", "process_flow"):
                        img_type_boost = 0.12
                chunk["bm25_score"] = round(bm25, 4)
                chunk["score"]      = round(
                    0.40 * dense + 0.25 * salience + 0.10 * eq
                    + 0.25 * bm25 + sec_boost + img_type_boost, 4
                )
                out.append(chunk)
                seen_ids.add(chunk["chunk_id"])
        except Exception:
            pass
    return out


def retrieve_r1a(question: str, embedder, colls: dict,
                 doc_filter: Optional[str] = None,
                 top_k: int = DEFAULT_TOP_K,
                 llm=None) -> List[Dict]:
    """
    Hierarchical retrieval with hybrid scoring:
      1. Query rewrite  — Scania-aware LLM rewrite expands vocab + abbreviations
      2. Document layer — resolve target docs by semantic similarity
      3. Section layer  — narrow to most relevant sections within docs
      4. Chunk layer    — text/table/image, scored by dense + BM25 + metadata
    Score = 0.40·dense + 0.25·salience + 0.10·evidence_quality + 0.25·bm25
    """
    # Stage 0: rewrite query to match Scania technical vocabulary
    search_query = _rewrite_query(llm, question) if llm else question
    emb = embedder.embed_query(search_query)
    if emb is None:
        return []

    results:  List[Dict] = []
    seen_ids: set        = set()
    is_meta = _is_metadata_question(question)

    # ── STEP 1: Document layer ────────────────────────────────────────────────
    target_doc_ids: List[str] = []

    if doc_filter:
        # Expand to include all versions of the same source (e.g., PDF + PPTX)
        target_doc_ids = _expand_doc_filter(doc_filter, colls)
    elif colls.get("documents"):
        dc    = colls["documents"]
        count = dc.count()
        if count > 0:
            try:
                # Retrieve ALL documents — let section + chunk layers rank by relevance
                r = dc.query(
                    query_embeddings=[emb],
                    n_results=count,
                    include=["metadatas", "distances"],
                )
                target_doc_ids = r["ids"][0]
            except Exception:
                pass

    # Metadata questions get the full doc card injected at score=1.0
    if is_meta and target_doc_ids:
        _add_doc_cards(target_doc_ids, colls, emb, results, seen_ids)

    # ── STEP 2: Section layer ─────────────────────────────────────────────────
    target_section_ids: List[str] = []

    sc = colls.get("sections")
    if sc and target_doc_ids:
        count = sc.count()
        if count > 0:
            try:
                where = _make_where("doc_id", target_doc_ids)
                # 8 sections per targeted doc for broader section coverage
                n_sec = min(8 * len(target_doc_ids), count)
                r = sc.query(
                    query_embeddings=[emb],
                    n_results=n_sec,
                    where=where,
                    include=["metadatas", "distances"],
                )
                target_section_ids = r["ids"][0]
            except Exception:
                pass

    # ── STEP 3: Chunk layer — doc-scoped primary, section as score boost ────────
    # Always filter by doc_id (guaranteed to work for both PDF and PPT).
    # Section IDs are used as a score multiplier inside _query_chunks, NOT as
    # a hard filter — this avoids the _sec_ vs _slide_ mismatch in PPT ingestion.
    doc_where = _make_where("doc_id", target_doc_ids) if target_doc_ids else None
    chunk_results = _query_chunks(
        colls, emb, doc_where, top_k * 2, seen_ids,
        query_text    = search_query,
        section_ids   = set(target_section_ids),
    )

    results.extend(chunk_results)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ── Doc-card backfill utility ─────────────────────────────────────────────────

def build_doc_cards(embedder, colls: dict) -> int:
    """
    Ensure every document in documents_collection has a record.
    For documents that exist in text_chunks but not in documents_collection,
    create a minimal entry from their early-page text chunks.
    Returns number of new records created.
    """
    text_coll = colls.get("text")       # → text_chunks
    doc_coll  = colls.get("documents")  # → documents_collection
    if not text_coll or not doc_coll:
        return 0

    # Fetch all text metadata (no embeddings needed here)
    try:
        all_r = text_coll.get(limit=5000, include=["metadatas", "documents"])
    except Exception:
        return 0

    # Group chunks by doc_id, keep text alongside
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for meta, doc in zip(all_r.get("metadatas", []), all_r.get("documents", [])):
        doc_id = meta.get("doc_id", "")
        if not doc_id:
            continue
        groups[doc_id].append({
            "arxiv_id":    meta.get("arxiv_id", doc_id),
            "page":        int(meta.get("page_number") or 99),
            "chunk_index": int(meta.get("chunk_index") or 99),
            "text":        doc or "",
        })

    # Which doc_ids already exist in documents_collection?
    existing: set = set()
    try:
        existing_r = doc_coll.get(limit=2000, include=["metadatas"])
        # documents_collection uses doc_id as the record id — pull from ids
        for eid in existing_r.get("ids", []):
            existing.add(eid)
    except Exception:
        pass

    built = 0
    for doc_id, chunks in groups.items():
        if doc_id in existing:
            continue
        early     = sorted(chunks, key=lambda c: (c["page"], c["chunk_index"]))[:4]
        arxiv_id  = early[0]["arxiv_id"]
        card_text = f"Document: {arxiv_id}\n\n" + "\n\n".join(
            c["text"] for c in early if c["text"]
        )
        card_emb = embedder.embed_query(card_text[:2000])
        if card_emb is None:
            continue
        doc_coll.upsert(
            ids=[doc_id],
            embeddings=[card_emb],
            documents=[card_text[:2000]],
            metadatas=[{
                "doc_title":     arxiv_id,
                "author":        "",
                "source_id":     arxiv_id,
                "document_type": "Research Paper",
                "file_format":   "pdf",
                "language":      "English",
                "doc_summary":   "",
                "chunk_count":   len(chunks),
                "total_pages":   0,
            }],
        )
        built += 1
    return built


# ── Generation (R2-G) ─────────────────────────────────────────────────────────

_DIRECT_EXTRACT_SYS = (
    "You are a precise document information extractor. "
    "Answer using ONLY the provided document text. Be specific and direct."
)
_DIRECT_EXTRACT_PROMPT = """\
Extract the answer to this question from the document text below.
Be direct and specific. If the information is not present, say so.

Question: {question}

Document text:
{evidence}

Answer:"""


def _generate_direct_extraction(llm, question: str, results: List[Dict]) -> dict:
    """
    Direct LLM extraction for metadata questions (author, title, year…).
    Bypasses R2-G Stage-1 chunk filter which marks factual/header text as NEUTRAL
    and causes abstention. Instead, extracts directly from doc card + top chunks.
    """
    texts        = [r["text"]        for r in results]
    chunk_ids    = [r["chunk_id"]    for r in results]
    modalities   = [r["modality"]    for r in results]
    dense_scores = [r["dense_score"] for r in results]

    # Prefer doc-card text; fall back to all evidence
    card_texts  = [r["text"] for r in results if r.get("meta", {}).get("is_doc_card")]
    other_texts = [r["text"] for r in results if not r.get("meta", {}).get("is_doc_card")]
    evidence    = "\n\n---\n\n".join((card_texts + other_texts[:3]))[:3000]

    try:
        answer = llm.invoke(
            system=_DIRECT_EXTRACT_SYS,
            prompt=_DIRECT_EXTRACT_PROMPT.format(question=question, evidence=evidence),
            max_tokens=250,
            temperature=0,
        ).strip()
    except Exception as e:
        answer = f"Extraction failed: {e}"

    return {
        "answer":             answer,
        "claims":             [],
        "coverage_score":     1.0,
        "unsupported_count":  0,
        "contradicted_count": 0,
        "abstained":          False,
        "error":              "",
        "texts":              texts,
        "chunk_ids":          chunk_ids,
        "modalities":         modalities,
        "dense_scores":       dense_scores,
        "generation_path":    "direct_extraction",
    }


def generate_r2g(llm, question: str, results: List[Dict]) -> dict:
    """
    R2-G: two-stage chunk-filter + claim ArgRAG generation.
    Metadata questions (author, title, year…) are routed to direct extraction
    to avoid R2-G Stage-1 classifying factual header text as NEUTRAL → abstain.
    """
    has_doc_card = any(r.get("meta", {}).get("is_doc_card") for r in results)
    if _is_metadata_question(question) and has_doc_card:
        return _generate_direct_extraction(llm, question, results)

    from rq2_ablation.conditions import r2g_two_stage

    texts        = [r["text"]        for r in results]
    chunk_ids    = [r["chunk_id"]    for r in results]
    modalities   = [r["modality"]    for r in results]
    dense_scores = [r["dense_score"] for r in results]

    output = r2g_two_stage.run(llm, question, texts, chunk_ids, modalities)
    return {
        "answer":             output.answer,
        "claims":             output.claims,
        "coverage_score":     float(output.coverage_score),
        "unsupported_count":  output.unsupported_count,
        "contradicted_count": output.contradicted_count,
        "abstained":          output.abstained,
        "error":              output.error,
        "texts":              texts,
        "chunk_ids":          chunk_ids,
        "modalities":         modalities,
        "dense_scores":       dense_scores,
        "generation_path":    "r2g",
    }


# ── Confidence scoring ────────────────────────────────────────────────────────

def score_answer(llm, question: str, evidence_text: str, gen: dict) -> dict:
    """
    Compute C = w_faith·S2 + w_evstr·S4 + w_comp·S3 + w_direct·Direct.
    Weights are configurable via sidebar sliders (SPIQA-calibrated defaults).

    S2 faith_new  — atomic claim faithfulness (SUPPORTED=1/PARTIAL=0.5/UNSUPPORTED=0)
    S3 comp_new   — reference-free completeness (requirements from question, coverage by answer)
    S4 evstr      — evidence_support_ratio: fraction of question requirements supported by evidence
    Direct        — bonus when evidence fully covers the question (ev_support_ratio >= 0.80)
    """
    judge = _run_judge_new(llm, question, evidence_text, gen["answer"])

    faith    = judge["faith_new"]
    comp     = judge["comp_new"]
    evstr    = judge["evidence_support_ratio"]
    w_faith, w_evstr, w_comp, w_direct = _get_weights()
    direct   = w_direct if evstr >= COVERAGE_DIRECT_THR else 0.0

    # Exact weighted sum — weights must sum to 1.0 (sidebar enforces this with a warning).
    # Direct bonus (w_direct) is part of the weight budget; it fires when S4 >= threshold.
    # When direct does not fire, max C = w_faith + w_evstr + w_comp (< 1.0 by design —
    # full coverage is required to reach 1.0). Formula matches display exactly.
    raw = w_faith * faith + w_evstr * evstr + w_comp * comp + direct

    # HallucinationCap: if abstained or >50% claims unsupported/contradicted → cap at 0.35
    # Both counts from the same judge decomposition — consistent and correct
    n_claims = judge["n_claims"]
    n_unsupp = judge["n_unsupported_from_judge"]
    if gen["abstained"] or (n_claims > 0 and n_unsupp / max(n_claims, 1) >= 0.5):
        raw = min(raw, HAL_CAP)

    c = round(max(0.0, min(1.0, raw)), 4)

    return {
        "c_score":               c,
        "faithfulness":          faith,          # faith_new
        "evidence_strength":     evstr,          # evidence_support_ratio
        "completeness":          comp,           # comp_new
        "direct_bonus":          round(direct, 4),
        "n_claims":              n_claims,
        "n_requirements":        judge["n_requirements"],
        "n_supported_reqs":      judge["n_supported_reqs"],
        "missing_requirements":  judge["missing_requirements"],  # gap_repair seeds
        "requirements_detail":   judge.get("requirements_detail", []),  # full detail for UI
        "faith_summary":         judge["faith_summary"],
        "comp_summary":          judge["comp_summary"],
        "reliability":           judge["reliability"],
        "is_faithful":           faith >= 0.7,
        "verdict":               judge["faith_summary"],
    }


# ── Self-correction repair ────────────────────────────────────────────────────

class _StoreAdapter:
    """Minimal adapter: exposes .collections dict so repairs.py / multiquery.py work."""
    def __init__(self, colls: dict):
        self.collections = colls


def _infer_doc_id(chunk_ids: List[str], colls: dict) -> str:
    """
    Infer doc_id from chunk metadata in ChromaDB.
    More reliable than parsing chunk_id strings since formats vary by pipeline.
    Falls back to string-splitting if ChromaDB lookup fails.
    """
    # Try ChromaDB metadata first (most reliable)
    for coll_name in ("text", "tables", "images_text"):
        coll = colls.get(coll_name)
        if not coll or not chunk_ids:
            continue
        try:
            r = coll.get(ids=chunk_ids[:3], include=["metadatas"])
            for meta in r.get("metadatas") or []:
                doc_id = meta.get("doc_id", "") or meta.get("arxiv_id", "")
                if doc_id:
                    return doc_id
        except Exception:
            continue
    # Fallback: parse from chunk_id string
    from rq3_extended.multiquery import _extract_doc_id
    return _extract_doc_id(chunk_ids) or ""


# ── HyDE helper ───────────────────────────────────────────────────────────────

def _hyde_retrieve(
    llm, embedder, colls: dict,
    question: str, doc_id: str,
    existing_chunk_ids: list,
    top_k: int = 10,
) -> tuple:
    """
    Hypothetical Document Embedding (HyDE):
    1. Generate a short hypothetical ideal answer with the LLM
    2. Embed the hypothesis (not the question)
    3. Retrieve chunks similar to the hypothesis
    Returns (new_texts, new_chunk_ids, new_modalities).
    """
    hyde_system = (
        "You are a technical domain expert. Write a concise, specific answer to the "
        "question as if you had perfect knowledge of the document. Be precise and use "
        "domain-appropriate terminology — include part numbers, torque values, procedure "
        "steps, component names, or technical specifications where relevant."
    )
    hyde_prompt = (
        f"Question: {question}\n\n"
        "Write a 2-3 sentence hypothetical ideal answer. Include specific technical "
        "details such as values, steps, component names, or codes if relevant."
    )
    try:
        hypothesis = llm.invoke(
            system=hyde_system, prompt=hyde_prompt, max_tokens=250, temperature=0.3
        )
        logger.info(f"  [HyDE] hypothesis: {hypothesis[:120]}...")
    except Exception as e:
        logger.warning(f"  [HyDE] hypothesis generation failed: {e}")
        return [], [], []

    from rq3_extended.multiquery import multiquery_retrieve
    store = _StoreAdapter(colls)
    return multiquery_retrieve(
        llm, embedder, store, hypothesis, doc_id,
        existing_chunk_ids=existing_chunk_ids,
        n_variants=1,
        top_k=top_k,
    )


# ── Query Decomposition helper ─────────────────────────────────────────────────

def _decompose_query(
    llm, embedder, colls: dict,
    question: str, doc_id: str,
    existing_chunk_ids: list,
    top_k: int = 8,
) -> tuple:
    """
    Query Decomposition:
    1. LLM breaks complex question into 2-3 focused sub-questions
    2. Retrieve chunks per sub-question
    3. Merge deduplicated results
    Returns (new_texts, new_chunk_ids, new_modalities).
    """
    import json as _json
    import re as _re

    decomp_system = (
        "You are an expert at decomposing complex technical questions into focused "
        "sub-questions for document retrieval. "
        "Return ONLY a JSON array of strings. No markdown, no explanation."
    )
    decomp_prompt = (
        f"Break this question into 2-3 simpler sub-questions that together "
        f"cover all aspects needed to fully answer it. Each sub-question should be "
        f"retrievable from a single section of a technical document "
        f"(procedure, specification, or component description).\n\n"
        f"Question: {question}\n\n"
        f"Return ONLY: [\"sub-question 1\", \"sub-question 2\", ...]"
    )
    sub_qs = [question]
    try:
        resp  = llm.invoke(
            system=decomp_system, prompt=decomp_prompt, max_tokens=200, temperature=0
        )
        clean = _re.sub(r"```(?:json)?", "", resp).strip()
        s, e  = clean.find("["), clean.rfind("]") + 1
        if s >= 0 and e > s:
            parsed = _json.loads(clean[s:e])
            if isinstance(parsed, list) and parsed:
                sub_qs = [q for q in parsed if isinstance(q, str) and q.strip()][:3]
        logger.info(f"  [QueryDecomp] sub-questions: {sub_qs}")
    except Exception as ex:
        logger.warning(f"  [QueryDecomp] decomposition failed: {ex}")

    from rq3_extended.multiquery import multiquery_retrieve
    store = _StoreAdapter(colls)

    all_texts, all_ids, all_mods = [], [], []
    seen = set(existing_chunk_ids)

    for sub_q in sub_qs:
        nt, ni, nm = multiquery_retrieve(
            llm, embedder, store, sub_q, doc_id,
            existing_chunk_ids=list(seen),
            n_variants=1,
            top_k=top_k,
        )
        for t, i, m in zip(nt, ni, nm):
            if i not in seen:
                all_texts.append(t)
                all_ids.append(i)
                all_mods.append(m)
                seen.add(i)

    logger.info(f"  [QueryDecomp] {len(all_ids)} new chunks from {len(sub_qs)} sub-questions")
    return all_texts, all_ids, all_mods


def run_repair(repair_type: str, llm, embedder, colls: dict,
               question: str, doc_id_str: str, gen: dict, score: dict) -> dict:
    from rq3_extended import repairs
    from rq2_ablation.conditions.base import format_evidence

    texts      = gen["texts"]
    chunk_ids  = gen["chunk_ids"]
    modalities = gen["modalities"]
    store      = _StoreAdapter(colls)
    t0         = time.time()

    # Auto-infer doc_id when blank — prevents MultiQuery from searching all documents
    effective_doc_id = doc_id_str.strip() or _infer_doc_id(chunk_ids, colls)

    try:
        if repair_type == "faith_repair":
            output = repairs.faith_repair(llm, question, texts, chunk_ids, modalities)

        elif repair_type == "evidence_repair":
            output = repairs.evidence_repair(
                llm, embedder, store, question, effective_doc_id,
                texts, chunk_ids, modalities,
            )

        elif repair_type == "gap_repair":
            # Prefer missing requirements from comp_new judge (more targeted than weak claims)
            # These are the specific aspects the question needs but the answer didn't cover
            missing_reqs = score.get("missing_requirements", [])
            if missing_reqs:
                gap_seeds = missing_reqs[:4]
            else:
                # Fallback: weak ArgRAG claims (old behaviour)
                gap_seeds = [c.text for c in gen["claims"]
                             if getattr(c, "status", "") in ("unsupported", "weak")][:4]
            output = repairs.gap_repair(
                llm, embedder, store, question, effective_doc_id,
                texts, chunk_ids, modalities, weak_claim_texts=gap_seeds,
            )

        elif repair_type == "full_escalation":
            output = repairs.full_escalation(
                llm, embedder, store, question, effective_doc_id,
                texts, chunk_ids, modalities,
            )

        elif repair_type == "hyde_repair":
            new_texts, new_ids, new_mods = _hyde_retrieve(
                llm, embedder, colls, question, effective_doc_id,
                existing_chunk_ids=chunk_ids,
            )
            merged_t = list(texts) + new_texts
            merged_i = list(chunk_ids) + new_ids
            merged_m = list(modalities) + new_mods
            logger.info(
                f"  [HyDE] {len(texts)} original + {len(new_texts)} hypothesis-matched "
                f"= {len(merged_t)} total chunks"
            )
            output = repairs._run_argrag(
                llm, question, merged_t, merged_i, merged_m,
                condition_label="hyde_repair",
            )

        elif repair_type == "query_decomp":
            new_texts, new_ids, new_mods = _decompose_query(
                llm, embedder, colls, question, effective_doc_id,
                existing_chunk_ids=chunk_ids,
            )
            merged_t = list(texts) + new_texts
            merged_i = list(chunk_ids) + new_ids
            merged_m = list(modalities) + new_mods
            logger.info(
                f"  [QueryDecomp] {len(texts)} original + {len(new_texts)} decomposed "
                f"= {len(merged_t)} total chunks"
            )
            output = repairs._run_argrag(
                llm, question, merged_t, merged_i, merged_m,
                condition_label="query_decomp",
            )

        else:
            return {}

    except Exception as e:
        return {"error": str(e)}

    latency = round((time.time() - t0) * 1000, 1)

    new_texts = getattr(output, "_texts",      texts)
    new_ids   = getattr(output, "_chunk_ids",  chunk_ids)
    new_mods  = getattr(output, "_modalities", modalities)

    # Repairs don't go through retrieve_r1a, so we don't have fresh dense scores.
    # Re-use the existing dense scores from the original gen (conservative but honest).
    new_gen = {
        "answer":             output.answer,
        "claims":             output.claims,
        "coverage_score":     float(output.coverage_score),
        "unsupported_count":  output.unsupported_count,
        "contradicted_count": output.contradicted_count,
        "abstained":          output.abstained,
        "error":              output.error,
        "texts":              new_texts,
        "chunk_ids":          new_ids,
        "modalities":         new_mods,
        "dense_scores":       gen.get("dense_scores", []),
    }
    evidence       = format_evidence(new_texts, new_ids, new_mods)
    new_score      = score_answer(llm, question, evidence, new_gen)
    new_score["latency_ms"] = latency
    return {"gen": new_gen, "score": new_score}


# ── UI component helpers ──────────────────────────────────────────────────────

def _score_bar(label: str, value: float):
    filled = int(round(value * 20))
    bar    = "█" * filled + "░" * (20 - filled)
    color  = _score_color(value)
    col1, col2, col3 = st.columns([2.5, 5, 1])
    col1.caption(label)
    col2.markdown(
        f"<span style='color:{color};font-family:monospace;font-size:14px'>{bar}</span>",
        unsafe_allow_html=True,
    )
    col3.caption(f"`{value:.3f}`")


def render_c_score(score: dict):
    c     = score["c_score"]
    color = _score_color(c)
    rel   = score.get("reliability", "")
    rel_color = {"HIGH": "#10b981", "MODERATE": "#f59e0b", "LOW": "#ef4444"}.get(rel, "#888")
    wf, we, wc, wd = _get_weights()

    st.markdown(
        f"<div style='text-align:center'>"
        f"<span style='font-size:48px;font-weight:bold;color:{color}'>{c:.4f}</span>"
        f"<br><span style='color:#888;font-size:12px'>"
        f"C = {wf:.2f}·S2(faith) + {we:.2f}·S4(ev_support) + {wc:.2f}·S3(comp) + {wd:.2f}·Direct</span>"
        f"<br><span style='color:{rel_color};font-size:11px;font-weight:600'>"
        f"{rel} RELIABILITY</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    _score_bar(f"S2 faith_new  ×{wf:.2f}", score["faithfulness"])
    _score_bar(f"S4 ev_support ×{we:.2f}", score["evidence_strength"])
    _score_bar(f"S3 comp_new   ×{wc:.2f}", score["completeness"])
    _score_bar(f"Direct Bonus  ×{wd:.2f}",
               score["direct_bonus"] / wd if wd > 0 else 0.0)

    # Claim / requirement counts
    n_cl  = score.get("n_claims", 0)
    n_req = score.get("n_requirements", 0)
    n_sup = score.get("n_supported_reqs", 0)
    if n_cl or n_req:
        st.caption(
            f"Claims verified: **{n_cl}**  ·  "
            f"Requirements: **{n_req}** ({n_sup} supported by evidence)"
        )

    if score.get("faith_summary"):
        st.caption(f"Faithfulness: _{score['faith_summary']}_")
    if score.get("comp_summary"):
        st.caption(f"Completeness: _{score['comp_summary']}_")

    if not score.get("is_faithful", True):
        st.warning("⚠️ Answer contains unsupported claims (faith_new < 0.70)")


def render_routing(score: dict) -> Optional[str]:
    from rq3_extended.router import route, describe_route
    repair = route(score["faithfulness"], score["evidence_strength"], score["completeness"])
    desc   = describe_route(repair)
    if repair is None:
        st.success(f"✅ **No repair needed** — {desc}")
    elif repair == "full_escalation":
        st.error(f"🚨 Recommended: **{repair}**  \n_{desc}_")
    else:
        st.warning(f"⚡ Recommended: **{repair}**  \n_{desc}_")
    return repair


def render_chunk_card(r: dict, rank: int):
    mod      = r["modality"]
    meta     = r.get("meta", {})
    is_card  = bool(meta.get("is_doc_card", 0))
    icon     = "🗂️" if is_card else MODALITY_ICON.get(mod, "📄")
    doc      = meta.get("arxiv_id", meta.get("doc_id", "—"))
    page     = meta.get("page_number", "?")
    role     = meta.get("evidence_role", "")
    sal      = float(meta.get("salience_score", 0.0))
    score    = r.get("score", 0.0)
    anchor   = meta.get("semantic_anchor", "")
    rc       = "#0ea5e9" if is_card else ROLE_COLOR.get(role, "#6b7280")
    card_tag = " · **DOC CARD**" if is_card else ""

    with st.expander(
        f"{icon}  **#{rank}**  {doc}{card_tag} · p.{page}  [{mod}]  score={score:.3f}",
        expanded=(rank <= 3),
    ):
        if is_card:
            st.markdown(
                "<span style='background:#0ea5e9;color:white;padding:2px 10px;"
                "border-radius:4px;font-size:12px'>Document metadata card</span>",
                unsafe_allow_html=True,
            )
            st.caption("Contains title, authors, and abstract area — used for metadata questions.")
            st.markdown("---")
        elif anchor:
            st.markdown(f"**Anchor:** _{anchor}_")
        if role and not is_card:
            st.markdown(
                f"<span style='background:{rc};color:white;padding:2px 10px;"
                f"border-radius:4px;font-size:12px;margin-right:8px'>{role}</span>"
                f"<small>salience={sal:.2f} | dense={r['dense_score']:.3f}</small>",
                unsafe_allow_html=True,
            )
        st.markdown("---")
        preview = r["text"][:700] + ("…" if len(r["text"]) > 700 else "")
        st.text(preview)


def render_claims(claims: list):
    STATUS_ICON = {
        "supported":   "✅", "strong": "✅", "moderate": "🟡",
        "contested":   "🔶", "unsupported": "❌", "weak": "⚠️",
        "contradicted": "🚫",
    }
    for c in claims:
        icon     = STATUS_ICON.get(getattr(c, "status", ""), "•")
        status   = getattr(c, "status", "")
        raw_sup  = getattr(c, "raw_support_score", None)
        raw_att  = getattr(c, "raw_attack_score",  None)
        strength = getattr(c, "support_score", 0.0)
        # Show raw LLM scores when available, fall back to composite strength
        if raw_sup is not None:
            score_label = f"sup={raw_sup:.2f} att={raw_att:.2f} strength={strength:.2f}"
        else:
            score_label = f"strength={strength:.2f}"
        st.markdown(
            f"{icon} **[{status}]** {c.text}  "
            f"<small style='color:#888'>{score_label}</small>",
            unsafe_allow_html=True,
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple:
    with st.sidebar:
        st.markdown("## 🔬 HEMMIR")
        st.caption("Hierarchical Evidence-Guided Multimodal Retrieval")
        st.markdown("---")

        # LLM provider
        st.markdown("### ⚙️ LLM Settings")
        provider = st.selectbox("Provider", ["bedrock", "anthropic"], index=0)
        api_key  = ""
        if provider == "anthropic":
            api_key = st.text_input(
                "Anthropic API Key",
                value=os.environ.get("ANTHROPIC_API_KEY", ""),
                type="password",
            )
        else:
            st.caption("Using AWS Bedrock (IAM auth)")

        # ChromaDB — force-reset to chroma_app if session has a stale value
        st.markdown("### 🗃️ ChromaDB")
        if st.session_state.get("_chroma_default_set") != DEFAULT_CHROMA_DIR:
            st.session_state["chroma_dir_path"]    = DEFAULT_CHROMA_DIR
            st.session_state["_chroma_default_set"] = DEFAULT_CHROMA_DIR

        chroma_dir = st.text_input(
            "Path", key="chroma_dir_path",
            help="Shared DB — same for all sessions. Change only to connect a different store.",
        )
        os.makedirs(chroma_dir, exist_ok=True)

        col_ref, col_build = st.columns(2)
        if col_ref.button("🔄 Refresh", use_container_width=True):
            get_db.clear()

        try:
            _, _colls = get_db(chroma_dir)
            APP_COLLS = [
                ("documents",   "🗂️", "docs"),
                ("sections",    "📑", "sections"),
                ("text",        "📝", "chunks"),
                ("tables",      "📊", "chunks"),
                ("images_text", "🖼️", "chunks"),
                ("images_clip", "🎨", "clips"),
            ]
            for key, icon, label in APP_COLLS:
                n = _colls[key].count()
                st.caption(f"{icon} **{key}**: {n} {label}")

            # Warn if documents_collection is empty but text chunks exist
            if _colls["documents"].count() == 0 and _colls["text"].count() > 0:
                st.warning("⚠️ documents_collection empty — click **Build Cards** to enable metadata questions.")

            if col_build.button("🗂️ Build Cards", use_container_width=True,
                                help="Create doc_meta cards from existing text chunks (no re-ingest needed)"):
                _emb = get_embedder()
                with st.spinner("Building doc cards…"):
                    n_built = build_doc_cards(_emb, _colls)
                get_db.clear()
                st.success(f"✅ {n_built} doc card(s) created.")
                st.rerun()
        except Exception as e:
            st.caption(f"⚠️ {e}")

        # ── Confidence Weight Sliders ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### ⚖️ Confidence Weights")
        st.caption("SPIQA-calibrated defaults (S4≫S3≫S2 by AUROC)")

        # Preset selector
        preset = st.selectbox(
            "Preset",
            ["Scania-calibrated (thesis)", "SPIQA-calibrated", "Equal weights", "Faith-first (original)"],
            key="weight_preset",
        )
        if preset == "Scania-calibrated (thesis)":
            # From thesis grid-search: W1=0.25·S1, W2=0·S2, W3=0.20·S3, W4=0.55·S4
            # Mapped to app signals: faith=0 (S2 zero weight), evstr=0.55 (S4),
            # comp=0.20 (S3), direct=0.25 (S1 proxy — fires when evidence covers ≥80% reqs)
            pf, pe, pc, pd = 0.00, 0.55, 0.20, 0.25
        elif preset == "SPIQA-calibrated":
            pf, pe, pc, pd = 0.15, 0.35, 0.35, 0.15
        elif preset == "Equal weights":
            pf, pe, pc, pd = 0.25, 0.25, 0.25, 0.25
        else:
            pf, pe, pc, pd = 0.45, 0.25, 0.20, 0.10

        wf = st.slider("S2 faith_new",  0.0, 0.6, pf, 0.05,
                       help="Faithfulness — ceiling effect on SPIQA (AUROC 0.46)")
        we = st.slider("S4 ev_support", 0.0, 0.6, pe, 0.05,
                       help="Evidence support ratio — strongest signal (AUROC 0.75***)")
        wc = st.slider("S3 comp_new",   0.0, 0.6, pc, 0.05,
                       help="Completeness — second strongest (AUROC 0.73***)")
        wd = st.slider("Direct bonus",  0.0, 0.3, pd, 0.05,
                       help="Bonus when ev_support >= 0.80 (full coverage)")

        total = round(wf + we + wc + wd, 2)
        if abs(total - 1.0) > 0.01:
            st.warning(f"Weights sum to {total:.2f} — should be 1.00")
        else:
            st.caption(f"Sum = {total:.2f} ✓")

        st.session_state["w_faith"]  = wf
        st.session_state["w_evstr"]  = we
        st.session_state["w_comp"]   = wc
        st.session_state["w_direct"] = wd

        # Session summary
        st.markdown("---")
        st.markdown("### 📋 Session")
        if "retrieved" in st.session_state:
            st.caption(f"Retrieved: {len(st.session_state['retrieved'])} chunks")
        if "c_score_breakdown" in st.session_state:
            c = st.session_state["c_score_breakdown"].get("c_score", "—")
            st.caption(f"C score: **{c}**")
        if "repair_history" in st.session_state:
            h = st.session_state["repair_history"]
            st.caption(f"Repairs run: {len(h)}")

        if st.button("🗑️ Clear session"):
            # Increment reset token so all widget keys change → browser state ignored
            token  = st.session_state.get("_session_reset", 0) + 1
            chroma = st.session_state.get("chroma_dir_path", DEFAULT_CHROMA_DIR)
            chroma_set = st.session_state.get("_chroma_default_set", "")
            st.session_state.clear()
            st.session_state["_session_reset"]    = token
            st.session_state["chroma_dir_path"]   = chroma
            st.session_state["_chroma_default_set"] = chroma_set
            st.rerun()

    return provider, api_key, chroma_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    provider, api_key, chroma_dir = render_sidebar()

    # Reset token — changes whenever Clear Session is pressed, forcing new widget keys
    _tok = st.session_state.get("_session_reset", 0)

    # Initialise shared pipeline resources (cached)
    try:
        embedder        = get_embedder()
        llm             = get_llm(provider, api_key)
        _, colls        = get_db(chroma_dir)
    except Exception as e:
        st.error(f"Pipeline initialisation failed: {e}")
        st.stop()

    tab_ingest, tab_retrieve, tab_generate, tab_repair, tab_history = st.tabs([
        "📥 Ingest",
        "🔍 Retrieve",
        "🧠 Generate & Score",
        "🔧 Self-Correct",
        "📚 History",
    ])

    # ── TAB 1: INGEST ─────────────────────────────────────────────────────────
    with tab_ingest:
        st.header("📥 Ingest Documents")
        st.caption(
            "Upload PDF, PPTX, or PPT files — Docling extracts text / tables / images, "
            "LLM enriches each chunk (semantic anchor, evidence role, salience), "
            "then embeds and indexes into the shared ChromaDB."
        )

        uploaded = st.file_uploader(
            "Choose PDF / PowerPoint files",
            type=["pdf", "pptx", "ppt"],
            accept_multiple_files=True,
        )

        col_btn, col_info = st.columns([1, 3])
        start = col_btn.button("🚀 Ingest", disabled=not uploaded)
        if uploaded:
            col_info.caption(f"{len(uploaded)} file(s) ready")

        if start and uploaded:
            # Read bytes in main thread (UploadedFile not thread-safe)
            file_data = [
                (Path(uf.name).stem, Path(uf.name).suffix.lower(), uf.read())
                for uf in uploaded
            ]

            log_q       = queue.Queue()
            log_lines   = []
            log_ph      = st.empty()
            progress    = st.progress(0.0)
            status_ph   = st.empty()
            n_docs      = len(file_data)
            done_count  = [0]

            def _ingest_worker():
                for doc_name, file_ext, content in file_data:
                    try:
                        ingest_document(content, doc_name, file_ext, provider, chroma_dir, log_q)
                    except Exception as exc:
                        log_q.put(f"❌ {doc_name} failed: {exc}")
                        log_q.put("__DONE__error")
                log_q.put("__ALL_DONE__")

            thread = threading.Thread(target=_ingest_worker, daemon=True)
            thread.start()

            while True:
                try:
                    msg = log_q.get(timeout=0.5)
                except queue.Empty:
                    if not thread.is_alive():
                        break
                    continue

                if msg.startswith("__DONE__"):
                    done_count[0] += 1
                    progress.progress(done_count[0] / n_docs)
                elif msg == "__ALL_DONE__":
                    break
                else:
                    log_lines.append(msg)
                    status_ph.caption(msg)

                log_ph.code("\n".join(log_lines[-80:]))

            thread.join()
            progress.progress(1.0)
            status_ph.success(f"✅ {n_docs} document(s) indexed into {chroma_dir}")
            # Refresh stats
            get_db.clear()
            st.rerun()

    # ── TAB 2: RETRIEVE ───────────────────────────────────────────────────────
    with tab_retrieve:
        st.header("🔍 Retrieve Evidence")
        st.caption(
            "R1-A: metadata-aware multi-source dense retrieval "
            "(combined score = 0.5·dense + 0.3·salience + 0.2·evidence_quality)"
        )

        question = st.text_input(
            "Question",
            key=f"question_input_{_tok}",
            placeholder="What methods are used to evaluate the model?",
        )

        # Document filter selector — build display_label → doc_id map from metadata.
        # Two-pass: first collect unique (doc_id → {source_id, file_format}), then
        # build labels — appending format when two docs share the same source_id.
        doc_options = ["— all documents —"]
        _source_to_docid: dict = {}
        try:
            metas = colls["text"].get(limit=2000, include=["metadatas"])["metadatas"] or []
            # Pass 1: unique docs
            doc_info: dict = {}  # doc_id → {sid, fmt}
            for m in metas:
                sid = m.get("source_id", "") or m.get("arxiv_id", "")
                did = m.get("doc_id", "")
                fmt = m.get("file_format", "")
                if sid and did and did not in doc_info:
                    doc_info[did] = {"sid": sid, "fmt": fmt}
            # Pass 2: detect source_id collisions, build labels
            from collections import Counter
            sid_counts = Counter(v["sid"] for v in doc_info.values())
            for did, info in doc_info.items():
                sid, fmt = info["sid"], info["fmt"]
                label = f"{sid} ({fmt})" if sid_counts[sid] > 1 and fmt else sid
                _source_to_docid[label] = did
            doc_options += sorted(_source_to_docid.keys())
        except Exception:
            pass

        col_doc, col_k = st.columns([3, 1])
        with col_doc:
            sel_doc = st.selectbox("Filter by document (optional)", doc_options, key=f"sel_doc_{_tok}")
        with col_k:
            top_k = st.number_input("Top K", min_value=1, max_value=30, value=DEFAULT_TOP_K, key=f"top_k_{_tok}")

        # Use actual doc_id from metadata map — avoids MD5 mismatch with _doc_id()
        doc_filter = _source_to_docid.get(sel_doc) if sel_doc != "— all documents —" else None

        if st.button("🔍 Retrieve", disabled=not question):
            if _is_metadata_question(question):
                st.info("🗂️ Metadata question detected — document card will be included in evidence.")
            with st.spinner("Retrieving…"):
                hits = retrieve_r1a(question, embedder, colls, doc_filter, int(top_k), llm=llm)
            st.session_state["retrieved"]         = hits
            st.session_state["current_question"]  = question
            st.session_state["current_doc_id"]    = doc_filter
            # Clear downstream state when re-retrieving
            for k in ["generated", "c_score_breakdown", "repair_history",
                      "recommended_repair"]:
                st.session_state.pop(k, None)
            st.rerun()

        if "retrieved" in st.session_state:
            hits = st.session_state["retrieved"]
            q    = st.session_state.get("current_question", "")
            st.markdown(f"**{len(hits)} chunks** retrieved for: _{q}_")
            st.markdown("---")

            mods = [h["modality"] for h in hits]
            n_text  = mods.count("text")
            n_table = mods.count("table")
            n_image = mods.count("image")

            _left, _right = st.columns([3, 1])
            with _right:
                _slices = [(l, v, c) for l, v, c in [
                    ("Text",   n_text,  "#4C9BE8"),
                    ("Tables", n_table, "#F4A259"),
                    ("Images", n_image, "#6DBF82"),
                ] if v > 0]
                _total = sum(v for _, v, _ in _slices) or 1
                # Build conic-gradient stops
                _stops, _pct = [], 0.0
                for _lbl, _val, _clr in _slices:
                    _end = _pct + _val / _total * 100
                    _stops.append(f"{_clr} {_pct:.1f}% {_end:.1f}%")
                    _pct = _end
                _gradient = "conic-gradient(" + ", ".join(_stops) + ")"
                # Legend rows
                _legend_rows = "".join(
                    f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">'
                    f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{_clr};flex-shrink:0;"></span>'
                    f'<span style="font-size:12px;color:#555;">{_lbl}: <b>{_val}</b></span></div>'
                    for _lbl, _val, _clr in _slices
                )
                _html = (
                    f'<div style="display:flex;flex-direction:column;align-items:center;padding:6px 0;">'
                    f'<div style="width:120px;height:120px;border-radius:50%;'
                    f'background:{_gradient};'
                    f'-webkit-mask:radial-gradient(circle,transparent 38%,black 39%);'
                    f'mask:radial-gradient(circle,transparent 38%,black 39%);"></div>'
                    f'<div style="margin-top:10px;text-align:left;">{_legend_rows}</div>'
                    f'</div>'
                )
                st.markdown(_html, unsafe_allow_html=True)
            st.markdown("---")

            for i, h in enumerate(hits, 1):
                render_chunk_card(h, i)

    # ── TAB 3: GENERATE & SCORE ───────────────────────────────────────────────
    with tab_generate:
        st.header("🧠 Generate Answer & Score")
        st.caption(
            "R2-G two-stage: chunk-level filter → claim ArgRAG. "
            "C = w_faith·S2 + w_evstr·S4 + w_comp·S3 + w_direct·S1_bonus — "
            "weights set in sidebar; score is normalized to [0, 1]."
        )

        if "retrieved" not in st.session_state:
            st.info("Run retrieval first (Retrieve tab).")
        else:
            q    = st.session_state.get("current_question", "")
            hits = st.session_state["retrieved"]
            st.markdown(f"**Question:** {q}  \n**Evidence pool:** {len(hits)} chunks")

            is_meta_q = _is_metadata_question(q)
            btn_label = "⚡ Extract (Direct)" if is_meta_q else "⚡ Generate (R2-G)"
            if is_meta_q:
                st.info("🗂️ Metadata question — will use direct extraction (bypasses ArgRAG chunk filter).")

            if st.button(btn_label, type="primary"):
                spinner_msg = "Extracting from document card…" if is_meta_q else "Running R2-G ArgRAG generation…"
                with st.spinner(spinner_msg):
                    gen = generate_r2g(llm, q, hits)
                from rq2_ablation.conditions.base import format_evidence
                ev_text = format_evidence(gen["texts"], gen["chunk_ids"], gen["modalities"])
                with st.spinner("Scoring faithfulness + C score…"):
                    score = score_answer(llm, q, ev_text, gen)
                st.session_state["generated"]          = gen
                st.session_state["c_score_breakdown"]  = score
                st.session_state["repair_history"]     = []
                st.session_state.pop("recommended_repair", None)

                # ── Save to history ───────────────────────────────────────────
                import uuid as _uuid
                _rid = str(_uuid.uuid4())[:16]
                st.session_state["current_record_id"] = _rid
                _rec = _build_history_record(
                    record_id      = _rid,
                    question       = q,
                    gen            = gen,
                    score          = score,
                    hits           = hits,
                    repair_history = [],
                    was_repaired   = False,
                )
                save_history_record(_rec)

                st.rerun()

            if "generated" in st.session_state:
                gen   = st.session_state["generated"]
                score = st.session_state["c_score_breakdown"]

                col_ans, col_score = st.columns([3, 2])

                with col_ans:
                    st.subheader("Answer")
                    path = gen.get("generation_path", "r2g")
                    if path == "direct_extraction":
                        st.caption("🗂️ _Direct extraction path — metadata question answered from document card_")
                    if gen.get("abstained"):
                        st.warning("⚠️ Model abstained — insufficient evidence to answer confidently")
                    if gen.get("error"):
                        st.caption(f"Note: {gen['error']}")
                    st.markdown(gen["answer"])

                    if gen.get("claims"):
                        with st.expander(
                            f"📋 ArgRAG claims breakdown ({len(gen['claims'])} claims)", expanded=False
                        ):
                            render_claims(gen["claims"])

                with col_score:
                    st.subheader("Confidence Score")
                    render_c_score(score)
                    st.markdown("---")
                    st.subheader("Routing")
                    recommended = render_routing(score)
                    st.session_state["recommended_repair"] = recommended

                # Source attribution table
                st.markdown("---")
                st.subheader("Source Attribution")
                rows = []
                for h in hits:
                    m = h.get("meta", {})
                    rows.append({
                        "Rank":       hits.index(h) + 1,
                        "Doc":        m.get("arxiv_id", m.get("doc_id", "—")),
                        "Page":       m.get("page_number", "?"),
                        "Modality":   MODALITY_ICON.get(h["modality"], h["modality"]),
                        "Role":       m.get("evidence_role", "—"),
                        "Salience":   f"{m.get('salience_score', 0):.2f}",
                        "Score":      f"{h['score']:.3f}",
                        "Anchor":     m.get("semantic_anchor", "")[:80],
                    })
                st.dataframe(rows, use_container_width=True, height=300)

                # Requirements Analysis
                req_detail = score.get("requirements_detail", [])
                if req_detail:
                    st.markdown("---")
                    st.subheader("Requirements Analysis")
                    n_full    = sum(1 for r in req_detail if r["coverage"] == "FULL")
                    n_partial = sum(1 for r in req_detail if r["coverage"] == "PARTIAL")
                    n_missing = sum(1 for r in req_detail if r["coverage"] == "MISSING")
                    st.caption(
                        f"Completeness judge extracted **{len(req_detail)} requirements** from your question — "
                        f"✅ {n_full} fully covered · ⚠️ {n_partial} partial · ❌ {n_missing} missing. "
                        "Click any requirement to see which chunks support it."
                    )

                    _COV_ICON = {"FULL": "✅", "PARTIAL": "⚠️", "MISSING": "❌"}
                    _cov_order = {"MISSING": 0, "PARTIAL": 1, "FULL": 2, "UNKNOWN": 3}
                    _sorted_reqs = sorted(req_detail, key=lambda r: _cov_order.get(r["coverage"], 3))

                    _chunk_texts = [h.get("text", "") for h in hits]

                    def _bm25_top_chunks(query: str, texts: list, top_k: int = 3):
                        """Return (chunk_index, bm25_score) pairs for top-k matching chunks."""
                        try:
                            from rank_bm25 import BM25Okapi
                            tokenized = [t.lower().split() for t in texts]
                            bm25 = BM25Okapi(tokenized)
                            raw_scores = bm25.get_scores(query.lower().split())
                            ranked = sorted(range(len(raw_scores)),
                                            key=lambda i: raw_scores[i], reverse=True)[:top_k]
                            return [(i, raw_scores[i]) for i in ranked if raw_scores[i] > 0]
                        except Exception:
                            return []

                    for req in _sorted_reqs:
                        icon  = _COV_ICON.get(req["coverage"], "❓")
                        label = f"{icon} **{req['id']}** ({req['coverage']}) — {req['text']}"
                        with st.expander(label, expanded=(req["coverage"] == "MISSING")):
                            if req.get("cov_reason"):
                                st.markdown(f"*{req['cov_reason']}*")

                            matches = _bm25_top_chunks(req["text"], _chunk_texts, top_k=3)
                            if matches:
                                st.markdown("**Chunks most relevant to this requirement:**")
                                for chunk_idx, bm25_score in matches:
                                    h = hits[chunk_idx]
                                    m = h.get("meta", {})
                                    doc_id = m.get("arxiv_id", m.get("doc_id", f"#{chunk_idx+1}"))
                                    page   = m.get("page_number", "?")
                                    modal  = MODALITY_ICON.get(h.get("modality", ""), "📄")
                                    st.markdown(
                                        f"{modal} **Chunk #{chunk_idx+1}** — `{doc_id}` · p.{page} "
                                        f"· retrieval score {h.get('score', 0):.3f} · BM25 {bm25_score:.1f}"
                                    )
                                    snippet = h["text"][:350]
                                    if len(h["text"]) > 350:
                                        snippet += "…"
                                    st.markdown(f"> {snippet}")
                            else:
                                st.caption("No strong BM25 chunk match found for this requirement.")

    # ── TAB 4: SELF-CORRECT ───────────────────────────────────────────────────
    with tab_repair:
        st.header("🔧 Self-Correction")
        st.caption(
            "Component-aware routing: router inspects C score components and "
            "recommends the best repair. You choose when to fire it."
        )

        if "c_score_breakdown" not in st.session_state:
            st.info("Generate an answer first (Generate & Score tab).")
        else:
            score = st.session_state["c_score_breakdown"]
            gen   = st.session_state["generated"]
            q     = st.session_state.get("current_question", "")
            doc_id_default = st.session_state.get("current_doc_id") or ""

            # Current scores — thresholds from config
            from rq3_extended.config import THR_FAITH_LOW, THR_EVSTR_LOW, THR_COMP_LOW
            from rq3_extended.router import route, describe_route, ceiling_diagnosis

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("C score",        f"{score['c_score']:.3f}")
            c2.metric("faith_new",      f"{score['faithfulness']:.3f}",
                      delta="✓" if score["faithfulness"] >= THR_FAITH_LOW else f"↓ < {THR_FAITH_LOW}")
            c3.metric("ev_support",     f"{score['evidence_strength']:.3f}",
                      delta="✓" if score["evidence_strength"] >= THR_EVSTR_LOW else f"↓ < {THR_EVSTR_LOW}")
            c4.metric("comp_new",       f"{score['completeness']:.3f}",
                      delta="✓" if score["completeness"] >= THR_COMP_LOW else f"↓ < {THR_COMP_LOW}")

            st.markdown("---")

            # ── Live routing — always re-computed from current score ─────────
            # (not stale session_state — fixes the "faith_repair stalls" bug)
            recommended = route(
                score["faithfulness"],
                score["evidence_strength"],
                score["completeness"],
            )
            # Update stored recommendation so repair buttons show correct highlight
            st.session_state["recommended_repair"] = recommended

            # Ceiling diagnosis — warn when faith_repair has hit its limit
            repair_hist_so_far = st.session_state.get("repair_history", [])
            ceiling_msg = ceiling_diagnosis(
                score["faithfulness"],
                score["evidence_strength"],
                score["completeness"],
                repair_hist_so_far,
            )
            if ceiling_msg:
                st.warning(f"⚠️ {ceiling_msg}")

            if recommended:
                st.info(f"⚡ **Auto-route suggests:** `{recommended}` — "
                        f"_{describe_route(recommended)}_")
            else:
                st.success("✅ No repair needed — all signals above thresholds")

            # Routing thresholds legend
            with st.expander("ℹ️ Routing logic (worst-signal-first)", expanded=False):
                st.markdown(f"""
| Signal | Threshold | Repair triggered |
|---|---|---|
| ev_support < {THR_EVSTR_LOW} | Evidence missing question requirements | `evidence_repair` — MultiQuery re-retrieval |
| faith_new < {THR_FAITH_LOW} | Unsupported claims in answer | `faith_repair` — tighten ArgRAG to 0.70 |
| comp_new < {THR_COMP_LOW} | Answer misses covered requirements | `gap_repair` — missing-requirement retrieval |
| All three very low | Systemic failure | `full_escalation` — MultiQuery + K=15 |
| coverage_score low | Indirect / abstract question | `hyde_repair` — hypothetical answer embedding |
| Multi-aspect question | Question has several sub-parts | `query_decomp` — decompose into sub-questions |

**Priority:** worst-signal-first. If ev_support is the binding constraint, evidence_repair
fires before faith_repair — faith_repair cannot add information that isn't in the evidence pool.

**HyDE:** generates a hypothetical ideal answer, embeds it, and retrieves chunks similar
to the hypothesis — effective when the question is indirect and the question embedding
alone doesn't capture the required content.

**Query Decomposition:** splits a complex multi-aspect question into 2-3 sub-questions,
retrieves separately for each, then re-generates on the merged evidence pool.
""")
            st.markdown("---")

            # Doc ID for re-retrieval — auto-infer when blank
            auto_doc_id = _infer_doc_id(gen.get("chunk_ids", []), colls)
            doc_id_input = st.text_input(
                "doc_id for re-retrieval",
                value=doc_id_default or auto_doc_id,
                help="Filters MultiQuery to this document. Auto-inferred from chunk metadata.",
            )
            if auto_doc_id and not doc_id_default:
                st.caption(f"🔍 Auto-inferred doc_id: `{auto_doc_id}` — evidence_repair will search only this document")

            # Repair buttons — row 1: original 4
            st.markdown("**Choose repair action:**")
            b1, b2, b3, b4 = st.columns(4)
            repair_type = None

            def _btn_label(base: str, key: str) -> str:
                return base + ("  ← suggested" if recommended == key else "")

            if b1.button(_btn_label("⚡ Faith Repair",     "faith_repair"),
                         use_container_width=True, type="primary" if recommended == "faith_repair" else "secondary"):
                repair_type = "faith_repair"
            if b2.button(_btn_label("🔍 Evidence Repair",  "evidence_repair"),
                         use_container_width=True, type="primary" if recommended == "evidence_repair" else "secondary"):
                repair_type = "evidence_repair"
            if b3.button(_btn_label("🧩 Gap Repair",       "gap_repair"),
                         use_container_width=True, type="primary" if recommended == "gap_repair" else "secondary"):
                repair_type = "gap_repair"
            if b4.button(_btn_label("🚀 Full Escalation",  "full_escalation"),
                         use_container_width=True, type="primary" if recommended == "full_escalation" else "secondary"):
                repair_type = "full_escalation"

            # Row 2: advanced repairs
            st.markdown("**Advanced repairs:**")
            b5, b6, _ = st.columns([1, 1, 2])
            if b5.button("🧪 HyDE Repair",
                         use_container_width=True,
                         help="Generate hypothetical answer → embed → retrieve similar chunks",
                         type="secondary"):
                repair_type = "hyde_repair"
            if b6.button("🔀 Query Decomp",
                         use_container_width=True,
                         help="Split complex question into sub-questions → retrieve per sub → merge",
                         type="secondary"):
                repair_type = "query_decomp"

            # Show missing requirements from comp_new judge (gap seed transparency)
            missing_reqs = score.get("missing_requirements", [])
            if missing_reqs:
                with st.expander(f"🧩 comp_new gap seeds ({len(missing_reqs)} missing requirements)", expanded=False):
                    st.caption("These are used as targeted retrieval queries in Gap Repair:")
                    for i, req in enumerate(missing_reqs, 1):
                        st.markdown(f"**{i}.** {req}")

            if repair_type:
                with st.spinner(f"Running {repair_type}…"):
                    result = run_repair(
                        repair_type, llm, embedder, colls,
                        q, doc_id_input, gen, score,
                    )

                if "error" in result and "gen" not in result:
                    st.error(f"Repair failed: {result['error']}")
                else:
                    new_score = result["score"]
                    new_gen   = result["gen"]
                    delta_c   = new_score["c_score"] - score["c_score"]
                    improved  = delta_c > 0.02

                    hist = st.session_state.get("repair_history", [])
                    hist.append({
                        "repair_type": repair_type,
                        "before_c":    score["c_score"],
                        "after_c":     new_score["c_score"],
                        "delta_c":     round(delta_c, 4),
                        "improved":    improved,
                        "latency_ms":  new_score.get("latency_ms", 0),
                    })
                    st.session_state["repair_history"] = hist

                    if improved:
                        st.success(f"✅ Improved! ΔC = {delta_c:+.4f}  "
                                   f"({score['c_score']:.3f} → {new_score['c_score']:.3f})")
                        st.session_state["generated"]          = new_gen
                        st.session_state["c_score_breakdown"]  = new_score
                        st.session_state.pop("recommended_repair", None)

                        # ── Update saved history record with repaired answer ───
                        _rid = st.session_state.get("current_record_id")
                        if _rid:
                            _upd = _build_history_record(
                                record_id      = _rid,
                                question       = q,
                                gen            = new_gen,
                                score          = new_score,
                                hits           = st.session_state.get("retrieved", []),
                                repair_history = hist,
                                was_repaired   = True,
                            )
                            update_history_record(_rid, _upd)

                    else:
                        st.warning(f"No meaningful improvement (ΔC = {delta_c:+.4f}) — original kept")
                    st.rerun()

            # Repair history table
            hist = st.session_state.get("repair_history", [])
            if hist:
                st.markdown("---")
                st.markdown("#### Repair History")
                for row in hist:
                    delta = row.get("delta_c", 0)
                    if row["improved"]:
                        icon = "✅"
                        note = ""
                    elif delta < -0.01:
                        icon = "🔴"
                        note = " ← made worse (original kept)"
                    else:
                        icon = "❌"
                        note = " ← no improvement (original kept)"
                    st.markdown(
                        f"{icon} **{row['repair_type']}** — "
                        f"C: {row['before_c']:.3f} → {row['after_c']:.3f}  "
                        f"(ΔC={delta:+.4f}, {row['latency_ms']:.0f} ms){note}"
                    )

            # Current best answer (updated after each improvement)
            if "generated" in st.session_state:
                st.markdown("---")
                with st.expander("📄 Current best answer", expanded=True):
                    cur_gen = st.session_state["generated"]
                    if cur_gen.get("abstained"):
                        st.warning("Abstained — insufficient evidence")
                    st.markdown(cur_gen["answer"])


    # ── TAB 5: HISTORY ────────────────────────────────────────────────────────
    with tab_history:
        st.header("📚 Answer History")
        st.caption(
            "Every generated answer is saved here. "
            "If a repair improved the score, the record is automatically updated "
            "with the repaired answer and repair trail."
        )

        history = load_history()

        if not history:
            st.info("No history yet — generate an answer in the **Generate & Score** tab.")
        else:
            # ── Top stats bar ─────────────────────────────────────────────────
            total      = len(history)
            repaired   = sum(1 for r in history if r.get("was_repaired"))
            abstained  = sum(1 for r in history if r.get("abstained"))
            mean_c     = sum(r.get("c_score", 0) for r in history) / max(1, total)

            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.metric("Total Q&A",       total)
            hc2.metric("Repaired",         repaired)
            hc3.metric("Abstained",        abstained)
            hc4.metric("Mean C score",     f"{mean_c:.3f}")

            st.markdown("---")

            # ── Search / filter ───────────────────────────────────────────────
            search_col, filter_col, sort_col = st.columns([3, 2, 2])
            with search_col:
                search_q = st.text_input(
                    "🔍 Search questions",
                    placeholder="Type keywords…",
                    label_visibility="collapsed",
                )
            with filter_col:
                filter_opt = st.selectbox(
                    "Filter",
                    ["All", "Repaired only", "Abstained only", "High confidence (C≥0.7)"],
                    label_visibility="collapsed",
                )
            with sort_col:
                sort_opt = st.selectbox(
                    "Sort",
                    ["Newest first", "Oldest first", "C score ↓", "C score ↑"],
                    label_visibility="collapsed",
                )

            # Apply search
            display = history
            if search_q.strip():
                kw = search_q.strip().lower()
                display = [r for r in display if kw in r.get("question", "").lower()
                           or kw in r.get("answer", "").lower()]

            # Apply filter
            if filter_opt == "Repaired only":
                display = [r for r in display if r.get("was_repaired")]
            elif filter_opt == "Abstained only":
                display = [r for r in display if r.get("abstained")]
            elif filter_opt == "High confidence (C≥0.7)":
                display = [r for r in display if r.get("c_score", 0) >= 0.7]

            # Apply sort
            if sort_opt == "Oldest first":
                display = list(reversed(display))
            elif sort_opt == "C score ↓":
                display = sorted(display, key=lambda r: r.get("c_score", 0), reverse=True)
            elif sort_opt == "C score ↑":
                display = sorted(display, key=lambda r: r.get("c_score", 0))

            st.caption(f"Showing **{len(display)}** of {total} records")
            st.markdown("---")

            # ── Export button ─────────────────────────────────────────────────
            if display:
                export_lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in display)
                st.download_button(
                    "⬇️ Export shown records (JSONL)",
                    data=export_lines,
                    file_name="hemmir_history.jsonl",
                    mime="application/jsonlines",
                )

            # ── Per-record cards ──────────────────────────────────────────────
            for rec in display:
                c_val   = rec.get("c_score", 0.0)
                c_color = _score_color(c_val)
                repaired_tag = " 🔧 repaired" if rec.get("was_repaired") else ""
                abstain_tag  = " ⚠️ abstained" if rec.get("abstained") else ""
                ts = rec.get("timestamp", "")[:19].replace("T", " ")

                label = (
                    f"[{ts}]  C={c_val:.3f}  "
                    f"{rec.get('question', '?')[:90]}"
                    f"{repaired_tag}{abstain_tag}"
                )

                with st.expander(label, expanded=False):
                    # Question + answer
                    st.markdown(f"**Question:** {rec.get('question','')}")
                    st.markdown("**Answer:**")
                    if rec.get("abstained"):
                        st.warning(rec.get("answer", "Abstained"))
                    else:
                        st.markdown(rec.get("answer", ""))

                    # Signal breakdown
                    st.markdown("---")
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("C score",       f"{c_val:.3f}")
                    sc2.metric("S2 faith",      f"{rec.get('faithfulness',0):.3f}")
                    sc3.metric("S4 ev_support", f"{rec.get('evidence_strength',0):.3f}")
                    sc4.metric("S3 comp",       f"{rec.get('completeness',0):.3f}")

                    extra_cols = st.columns(3)
                    extra_cols[0].caption(f"Path: `{rec.get('generation_path','—')}`")
                    extra_cols[1].caption(
                        f"Claims: {rec.get('n_claims',0)}  |  "
                        f"Reqs: {rec.get('n_requirements',0)} "
                        f"({rec.get('n_supported_reqs',0)} supported)"
                    )
                    extra_cols[2].caption(f"Record ID: `{rec.get('record_id','—')}`")

                    # Repair trail
                    repairs = rec.get("repair_history", [])
                    if repairs:
                        st.markdown("**Repair trail:**")
                        for rh in repairs:
                            icon = "✅" if rh.get("improved") else "❌"
                            st.caption(
                                f"{icon} `{rh.get('repair_type','?')}` — "
                                f"C: {rh.get('before_c',0):.3f} → {rh.get('after_c',0):.3f}  "
                                f"(ΔC={rh.get('delta_c',0):+.4f})"
                            )

                    # Retrieved chunks
                    chunks = rec.get("chunks_used", [])
                    all_chunks = rec.get("retrieved_chunks", [])
                    if chunks:
                        with st.expander(f"🔎 Chunks used in generation ({len(chunks)})", expanded=False):
                            for ch in chunks:
                                mod_icon = MODALITY_ICON.get(ch.get("modality","text"), "📄")
                                st.markdown(
                                    f"{mod_icon} `{ch.get('chunk_id','?')}` [{ch.get('modality','?')}]"
                                )
                                st.text(ch.get("text_preview","")[:200])
                                st.markdown("---")
                    if all_chunks:
                        with st.expander(f"📋 All retrieved chunks ({len(all_chunks)})", expanded=False):
                            for ch in all_chunks:
                                mod_icon = MODALITY_ICON.get(ch.get("modality","text"), "📄")
                                st.markdown(
                                    f"{mod_icon} **{ch.get('doc','?')}** p.{ch.get('page','?')}  "
                                    f"score=`{ch.get('score',0):.3f}` [{ch.get('modality','?')}]  "
                                    f"`{ch.get('chunk_id','')[:30]}`"
                                )
                                st.text(ch.get("text_preview","")[:200])
                                st.markdown("---")

            # ── Danger zone: clear all history ───────────────────────────────
            st.markdown("---")
            with st.expander("🗑️ Clear all history", expanded=False):
                st.warning("This permanently deletes all saved Q&A records.")
                confirm = st.text_input("Type **DELETE** to confirm", key="hist_delete_confirm")
                if st.button("🗑️ Delete history file", type="primary"):
                    if confirm.strip() == "DELETE":
                        if HISTORY_FILE.exists():
                            HISTORY_FILE.unlink()
                        st.success("History cleared.")
                        st.rerun()
                    else:
                        st.error("Type DELETE exactly to confirm.")


if __name__ == "__main__":
    main()
