# HEMMIR - Hierarchical Explainable Multimodal Modality-Information Retrieval

**Industrial RAG System for Manufacturing Documents**


---

## What This System Does

HEMMIR is a complete Retrieval-Augmented Generation (RAG) pipeline built for industrial documents — specifically Scania assembly manuals, risk assessments, tightening sequences, and technical reports. It does the following in sequence:

1. **Ingest** — Upload a PDF or PPTX file. The pipeline parses every page and extracts text chunks, tables, and images separately.
2. **Enrich** — A Claude LLM adds a semantic anchor, evidence role, and salience score to each chunk so retrieval knows what the chunk is about.
3. **Embed & Index** — All chunks are converted to 1536-dimensional dense vectors using OpenAI `text-embedding-3-small` and stored in ChromaDB across three separate collections (text / tables / images).
4. **Retrieve** — When a question is asked, the system runs BM25 sparse retrieval and dense vector retrieval in parallel, merges the results, and re-ranks them with a cross-encoder model.
5. **Generate** — ArgRAG produces a grounded two-stage answer: a draft is generated, each claim is verified against the evidence, and a final answer is synthesised only from supported claims.
6. **Score** — Four confidence signals (S2, S3, S4, Direct) produce a composite C score that measures how well-grounded and complete the answer is.
7. **Self-Correct** — If any signal falls below its threshold, the system automatically diagnoses the failure type and triggers the appropriate repair action (re-retrieval, re-synthesis, or gap filling).

---

## System Architecture — 7-Layer Pipeline

### Layer 1 — Ingestion (`ingestion_layer/`)

Parses uploaded files and converts them into structured chunks. Supported formats: **PDF** and **PPTX/PPT**.

- **PDF** — extracted with `PDFExtractorAdapter` (Docling-based), which detects and separates text blocks, tables, and embedded images on each page.
- **PPTX** — extracted with `PptxExtractor`, which maps slides to text chunks, table chunks, and image regions.

Each chunk carries structured metadata: document ID, page number, section title, modality (`text` / `table` / `image`), and bounding box coordinates.

Chunk size is bounded: minimum 80 characters, maximum 1200 characters. Chunks below the minimum are discarded; large blocks are split at sentence boundaries.

### Layer 2 — Encoding (`encoding_layer/`)

Converts raw extracted chunks into retrieval-ready representations.

- Builds a `retrieval_text` field for each chunk by combining the chunk body with promoted metadata fields (section title, document title, page number, semantic anchor if available).
- Constructs `promoted_fields` that the retrieval layer uses to boost relevant metadata during BM25 matching.
- Produces a `ScoreBreakdown` structure per chunk to carry all scoring signals through the pipeline.

### Layer 3 — Enrichment (`enrichment_layer/`)

Each chunk is passed to a Claude LLM that adds three fields:

| Field | What it is |
|-------|------------|
| `semantic_anchor` | One sentence capturing the core claim or finding of the chunk |
| `evidence_role` | Whether this chunk is a definition, procedure, specification, result, etc. |
| `salience` | 0–1 score: how informative this chunk is compared to surrounding chunks |

For **image chunks**, Claude Sonnet 4.5 generates a full vision description from the image bytes — so images become searchable text.

For **table chunks**, Claude Haiku 4.5 generates a table summary and a semantic anchor describing what question the table answers.

**Models used:**
- Text enrichment: `claude-haiku-4-5-20251001` (fast, low cost, structured JSON output)
- Vision / image description: `claude-sonnet-4-5-20250929` (better multimodal reasoning)
- Via **AWS Bedrock** (default, EU region, IAM auth) or **Anthropic API** (API key)

### Layer 4 — Embedding (`embedding_layer/`)

Converts enriched `retrieval_text` into dense vectors for semantic search.

- **Model**: OpenAI `text-embedding-3-small` — 1536-dimensional vectors
- **API key required**: `OPENAI_API_KEY` environment variable
- Embeds text chunks, table chunks (semantic pass on caption + summary), and image chunks (on vision description) separately
- Batch size: 32 chunks per API call, with automatic retry on rate-limit errors

### Layer 5 — Indexing (`indexing_layer/`)

Stores all embedded chunks into **ChromaDB** — a local vector database.

Three separate collections are maintained per corpus:

| Collection name | Contents |
|----------------|----------|
| `text` | Text chunks from PDF pages and PPTX slides |
| `tables` | Table chunks with semantic anchor + HTML cell content |
| `images_text` | Image chunks with vision description as searchable text |

Metadata filters on `doc_id` allow per-document retrieval scoping. The ChromaDB directory path is configurable from the sidebar (default: `chroma_app/`).

### Layer 6 — Retrieval (`retrieval_layer/`)

Implements **R1-A**: hybrid BM25 + dense retrieval with cross-encoder re-ranking.

**Step 1 — Parallel retrieval across all three collections:**
- **Dense**: cosine similarity search in ChromaDB using the embedded question vector
- **BM25**: `rank-bm25` sparse retrieval over `retrieval_text` fields
- Both run independently on text, table, and image collections
- Results are merged and deduplicated by chunk ID

**Step 2 — Cross-encoder re-ranking:**
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, no API key needed, runs on CPU or GPU)
- Each (question, chunk) pair is scored by the cross-encoder
- Final score = 0.70 × dense similarity + 0.30 × cross-encoder score
- Top K chunks (default: 10) are returned

**Scania Query Rewriter (automatic):**
When a question contains service-manual vocabulary — including terms like EGR, DPF, fault codes, tightening torque, maintenance intervals, SCR, AdBlue, injector, camshaft, gearbox — the question is automatically reformulated into a precise technical query before retrieval. This improves recall for vague natural-language questions about industrial procedures. The rewriter uses Claude Haiku and only activates on Scania-specific signals; it does not trigger for general questions.

### Layer 7 — Generation (`generation_layer/` + `rq2_ablation/`)

Implements **R2-G ArgRAG**: a two-stage argumentation-based generation pipeline.

**Stage 1 — Draft generation:**
Claude receives the question and the top-K retrieved chunks. It generates an initial answer using all available evidence.

**Stage 2 — Claim verification:**
Each atomic claim in the draft is independently checked against the retrieved chunks:
- Claims with strong evidence support (entailment score ≥ 0.65) are kept as `STRONG`
- Claims with partial support (≥ 0.35) are kept as `WEAK`
- Claims with no support are marked `UNSUPPORTED` and filtered out

**Stage 3 — Final answer synthesis:**
A final answer is generated using only the verified claims. If hallucinations (unsupported claims) were detected, the confidence score is hard-capped at 0.35.

The `rq2_ablation/conditions/` module manages the two-stage prompt construction and evidence formatting. The `rq2_ablation/conditions/base.py` module formats retrieved evidence into structured blocks for the prompt.

---

## Confidence Scoring — The 4 Signals

After generation, the answer is evaluated by four signals that together form the composite C score.

### S2 — Faithfulness (`faith_new`)

The LLM reads each atomic claim in the answer and checks whether it is directly supported by the retrieved evidence chunks.

- Score = fraction of claims that are entailed by at least one chunk
- AUROC on SPIQA benchmark: **0.460** (near-chance — ceiling effect: most good answers are fully faithful)
- Default weight in composite: **0.15**

### S3 — Completeness (`comp_new`)

The LLM extracts the atomic requirements from the question (what aspects does the question ask for?), then checks whether each requirement is covered in the answer.

- Score = fraction of evidence-supported requirements that the answer addresses
- AUROC on SPIQA benchmark: **0.733**
- Default weight in composite: **0.35**

### S4 — Evidence Support Ratio (`ev_support`)

Measures what fraction of the question's requirements have at least one supporting chunk in the retrieved evidence pool — regardless of what ended up in the answer.

- This signal diagnoses retrieval quality: a low S4 means the evidence pool itself is insufficient, not just the answer
- AUROC on SPIQA benchmark: **0.752** (strongest single signal)
- Default weight in composite: **0.35**

### Direct Bonus

A binary reward added when S4 ≥ 0.80, rewarding answers where the evidence pool covers at least 80% of question requirements.

- Default weight: **0.15**

### Composite Formula

```
C = 0.15 × S2 + 0.35 × S3 + 0.35 × S4 + 0.15 × Direct
```

If hallucinations were detected during claim verification, C is **hard-capped at 0.35** regardless of the formula result.

C is always in the range [0.0, 1.0]. Higher is better. A score above 0.75 is considered reliable.

---

## Self-Correction System — Signal-to-Action Mapping

The self-correction pipeline (`rq3_extended/`) reads the per-signal breakdown and decides whether and how to repair the answer. It runs automatically after generation if any signal is below its threshold. Maximum 2 repair iterations. A repair is only accepted if C improves by at least **0.02**.

### Thresholds

| Signal | LOW threshold | VLOW threshold | Interpretation at LOW |
|--------|:------------:|:--------------:|----------------------|
| S2 Faithfulness | **0.75** | **0.55** | Unsupported claims dominate the answer |
| S4 Evidence Support | **0.60** | **0.40** | < 60% of requirements have evidence in the pool |
| S3 Completeness | **0.40** | **0.25** | Answer covers < 40% of answerable requirements |

### Repair Actions

#### Full Escalation — all three signals simultaneously below VLOW
Condition: `S2 < 0.55 AND S4 < 0.40 AND S3 < 0.25`

This is a systemic failure — retrieval, grounding, and coverage have all broken down at the same time. The system triggers MultiQuery retrieval with an expanded pool (K=15, 4 query variants) followed by a full R2-G regeneration from scratch.

#### Evidence Repair — S4 is the binding bottleneck
Condition: `S4 < 0.60` and S4 has the largest normalised deficit among the three signals

The evidence pool does not cover enough of the question's requirements. The system generates 4 query variants from the original question, retrieves top-10 chunks per variant, merges all results, and re-runs R2-G with the expanded evidence pool.

**Why this is prioritised over faith repair:** faith_repair re-synthesises from the same evidence. If S4 is already the bottleneck, tightening the claim threshold only removes claims without adding new information — the answer hits a ceiling. Expanding the evidence pool first unlocks both faithfulness and completeness improvements in one step.

#### Faith Repair — answer has unsupported claims, evidence is sufficient
Condition: `S2 < 0.75` and S4 is not the dominant bottleneck

The evidence is there but the answer contains hallucinated or poorly-grounded claims. The system re-runs Stage 2 of ArgRAG with a stricter claim entailment threshold (**0.70** instead of the normal 0.65), filtering out more marginal claims before the final synthesis.

#### Gap Repair — answer is missing key aspects
Condition: `S3 < 0.40` and neither S2 nor S4 is below threshold

The evidence covers the question's requirements but the answer failed to address all of them. The system identifies which specific requirements are missing (from the S3 judge output) and runs targeted retrieval using those missing requirements as query seeds. The additional evidence is appended and the answer is re-generated.

#### No Repair
Condition: all three signals above their LOW thresholds

The answer is considered reliable. No self-correction is triggered.

### Routing Logic

The router uses a **worst-deficit-first** strategy — not a fixed priority order. For each signal it computes:

```
deficit = max(0, threshold_LOW - signal_value) / threshold_LOW
```

The signal with the largest normalised deficit determines the repair type. Full escalation is checked first (before deficit comparison) because it requires all three to be in VLOW simultaneously.

### Ceiling Detection

The system detects when repair is stalling and explains why:

- **Faith repair stall** — if faith_repair has been tried but S4 is still below 0.60, the system warns that evidence_repair is the real fix needed
- **Evidence degradation** — if evidence_repair made the C score worse, the system concludes the document does not contain information about the questioned aspect and recommends accepting the current answer or trying a different document
- **Systemic ceiling** — if 2 or more repairs all failed to improve the score, the system reports the best achievable C score from the available evidence

---

## Directory Structure

```
HEMMIR_SCANIA_THESIS_FINAL/
│
├── app.py                        ← Main Streamlit application (all UI + pipeline wiring)
├── __init__.py                   ← Package marker
│
├── ingestion_layer/              ← Layer 1: PDF / PPTX parser
│   ├── main_multiformat.py       ← Entry point, file format router
│   ├── loaders/                  ← Source document loaders
│   └── extractors/               ← PDF extractor adapter, PPTX extractor, base class
│
├── encoding_layer/               ← Layer 2: Retrieval text construction
│   ├── encoding_pipeline.py      ← Orchestrates encoding for a document
│   ├── retrieval_text_builders.py← Builds retrieval_text from chunk + metadata
│   ├── promoted_fields_builder.py← Constructs BM25-boosted metadata fields
│   ├── section_linker.py         ← Links chunks to their parent sections
│   └── models.py                 ← Encoding data models
│
├── enrichment_layer/             ← Layer 3: LLM chunk enrichment
│   ├── enrichment_pipeline.py    ← Orchestrates enrichment for a document
│   ├── enrichers/                ← Text, table, and image enrichers
│   └── utils/
│       └── llm_client.py         ← Claude Haiku (text) + Claude Sonnet (vision) client
│                                    Supports AWS Bedrock and Anthropic API
│
├── embedding_layer/              ← Layer 4: Dense vector embedding
│   ├── embedding_pipeline.py     ← Orchestrates embedding for a document
│   └── embedders/
│       └── text_embedder.py      ← OpenAI text-embedding-3-small (1536-dim)
│                                    Requires OPENAI_API_KEY env variable
│
├── indexing_layer/               ← Layer 5: ChromaDB storage
│   ├── indexing_pipeline.py      ← Orchestrates indexing for a document
│   └── utils/                    ← Collection helpers, metadata serialisation
│
├── cross_reference_layer/        ← Cross-document reference detection
│   ├── cross_referencer.py       ← Finds references between documents in the DB
│   └── utils/                    ← Reference extraction utilities
│
├── retrieval_layer/              ← Layer 6: Hybrid retrieval + re-ranking
│   ├── retrieval_pipeline.py     ← Orchestrates retrieval for a query
│   ├── RAG/                      ← Dense vector retrieval (ChromaDB queries)
│   ├── reranker/
│   │   └── cross_encoder_reranker.py ← cross-encoder/ms-marco-MiniLM-L-6-v2
│   ├── modality_plugin/          ← Per-modality retrieval logic (text/table/image)
│   └── utils/                    ← BM25 helpers, result merging, deduplication
│
├── generation_layer/             ← Layer 7: ArgRAG answer generation
│   ├── generator.py              ← Draft generation + claim verification
│   ├── verifier.py               ← Claim-level entailment verifier
│   ├── argrag.py                 ← ArgRAG two-stage pipeline
│   ├── evidence.py               ← Evidence block formatter
│   ├── fuse.py                   ← Evidence fusion utilities
│   ├── attribution.py            ← Source attribution to answer claims
│   └── uncertainty.py            ← Uncertainty signal helpers
│
├── rq2_ablation/                 ← Two-stage generation conditions + evidence formatting
│   ├── conditions/               ← r2g_two_stage: ArgRAG prompt construction
│   │   └── base.py               ← format_evidence: structures chunks into prompt blocks
│   ├── judge.py                  ← LLM-as-judge for claim verification
│   └── stats.py                  ← Aggregation utilities
│
├── rq3_extended/                 ← Self-correction pipeline
│   ├── router.py                 ← Signal → repair action routing (worst-deficit-first)
│   ├── config.py                 ← All thresholds: LOW/VLOW per signal, max iterations
│   ├── multiquery.py             ← Multi-query retrieval (4 variants, expanded K)
│   ├── repairs.py                ← faith_repair / evidence_repair / gap_repair logic
│   ├── pipeline.py               ← Self-correction loop orchestrator
│   └── evaluator.py              ← Per-repair C score delta evaluation
│
├── shared/                       ← Shared Pydantic data models
│   └── models/
│       ├── pipeline_models.py    ← RetrievedChunk, ScoreBreakdown, GenerationResult
│       ├── metadata_models.py    ← DocumentMetadata, ChunkMetadata
│       └── multiformat_models.py ← Multi-format ingestion models
│
├── app_workspace/                ← Scania source documents (23 folders, PDF + PPTX)
│   ├── 2025-10-09_Tightening-Sequence/
│   ├── 2025-10-24_SFL-update-ACG/
│   ├── 2025-11-26_Inbolt-Tests/
│   ├── 2025-12-11_Position-deviation-compensation-in-assembly-INBOLT/
│   ├── AprisoConfigurations/
│   ├── Assembly Steps/
│   ├── Flex Line plan/
│   ├── Flexible Assembly Line/
│   ├── HRC_QualityStation_Doc/
│   ├── Layout/
│   ├── Phase1_FlexLine/
│   ├── Risk Assessment for station/
│   ├── Risk Identification Tightening Automation/
│   ├── Vanguard/
│   └── ... (23 Scania documents total)
│
├── chroma_app/                   ← Default pre-indexed ChromaDB (99 MB)
│                                    app.py uses this path by default on startup
├── chroma_scania/                ← Scania-specific ChromaDB (66 MB)
│                                    Switch to this path in the sidebar if needed
└── logs/                         ← Runtime log files written by the pipeline
```

---

## Prerequisites

### Python Version
Python **3.12** is required. The pipeline uses features and type annotations that are incompatible with Python 3.10 or older.

### API Keys Required

| Service | Purpose | How to provide |
|---------|---------|---------------|
| **AWS Bedrock** (default) | LLM for enrichment + generation (Claude Haiku + Sonnet) | IAM role or `AWS_DEFAULT_REGION` + credentials — no API key needed |
| **Anthropic API** (alternative) | Same LLM tasks via Anthropic direct | `ANTHROPIC_API_KEY` environment variable or enter in sidebar |
| **OpenAI** | Embeddings only (`text-embedding-3-small`) | `OPENAI_API_KEY` environment variable |

Set environment variables before running:

```bash
export OPENAI_API_KEY="sk-..."
# If using Anthropic instead of Bedrock:
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Install Dependencies

```bash
pip install streamlit chromadb openai anthropic boto3 botocore \
            sentence-transformers rank-bm25 \
            pydantic loguru tqdm Pillow python-dotenv \
            open-clip-torch torch docling
```

Or install from the layer-level requirements files:

```bash
pip install -r enrichment_layer/requirements.txt
pip install -r embedding_layer/requirements.txt
pip install -r cross_reference_layer/requirements.txt
pip install -r indexing_layer/requirements.txt
```

---

## How to Run

```bash
cd /path/to/HEMMIR_SCANIA_THESIS_FINAL
streamlit run app.py --server.port 8501
```

Open your browser at: `http://localhost:8501`

The app loads immediately. The default ChromaDB (`chroma_app/`) is pre-indexed with Scania documents — you can start querying right away without re-ingesting.

---

## UI Tabs

### 📥 Ingest
Upload one or more PDF or PPTX files. The full 7-layer pipeline runs automatically: parse → enrich → embed → index. Progress is shown chunk-by-chunk. After ingestion, documents appear in the sidebar document list and are immediately queryable.

### 🔍 Retrieve
Run a retrieval-only query. Useful for inspecting what chunks the system finds before generation. Shows each retrieved chunk with its modality icon (text / table / image), source document, page number, retrieval score, and a preview of the text.

### 🧠 Generate & Score
The main query interface. Enter a question, optionally filter by document, and set Top K. The system runs the full R2-G ArgRAG pipeline and displays:
- The final answer with source attributions
- The composite C score with a colour-coded badge
- Individual signal values (S2 Faithfulness, S3 Completeness, S4 Evidence Support)
- The claim verification breakdown (which claims were STRONG / WEAK / UNSUPPORTED)

### 🔧 Self-Correct
Shows the self-correction decision for the most recent answer. Displays which signals triggered which repair action, the C score before and after each repair attempt, and the ceiling diagnosis if repair stalled. The repaired answer replaces the original if the C score improved by at least 0.02.

### 📚 History
Full history of all questions asked in this session, with their answers, C scores, and signal breakdowns. Supports search by keyword, filter by document, and sort by score or date. Individual history entries can be expanded for full detail or deleted. The entire history can be cleared with a DELETE confirmation prompt.

---

## Sidebar Configuration

| Control | Default | Description |
|---------|---------|-------------|
| **Provider** | `bedrock` | LLM provider: `bedrock` (IAM auth, no key) or `anthropic` (API key) |
| **API Key** | — | Only shown when `anthropic` is selected |
| **ChromaDB Path** | `chroma_app/` | Path to the ChromaDB directory. Change to `chroma_scania/` to use the Scania-specific index |
| **Top K** | 10 | Number of chunks retrieved per query (1–30) |
| **Weight Presets** | SPIQA-calibrated | Quick presets: SPIQA-calibrated, Vectara-calibrated, Equal weights |
| **S2 faith_new** | 0.15 | Slider 0.0–0.6, step 0.05 |
| **S4 ev_support** | 0.35 | Slider 0.0–0.6, step 0.05 |
| **S3 comp_new** | 0.35 | Slider 0.0–0.6, step 0.05 |
| **Direct bonus** | 0.15 | Slider 0.0–0.3, step 0.05 |

Weight changes take effect immediately on the next query. They do not retroactively change history scores.

### Choosing the Right ChromaDB

| DB | Path | Use when |
|----|------|----------|
| `chroma_app/` | Default | General use — contains whatever documents were last indexed via the Ingest tab |
| `chroma_scania/` | Set in sidebar | Scania industrial documents specifically — pre-indexed, ready to query |

---

## Pre-Indexed Databases

Two ChromaDB databases are included and ready to use without any re-ingestion:

**`chroma_app/` (99 MB)** — the default database that `app.py` loads on startup. Contains Scania documents indexed with the full enrichment pipeline.

**`chroma_scania/` (66 MB)** — a Scania-specific database. To use it, enter `chroma_scania` in the ChromaDB Path field in the sidebar.

Both databases contain three collections each: `text`, `tables`, `images_text`.

---

## Thesis Context

This system is the implementation artifact of the MSc thesis:

> **HEMMIR: Hierarchical Explainable Multimodal Modality-Integrated Reasoning for Industrial Document Question Answering**
> Ravindra Kumar — MSc Computer Science, Umeå University, 2025
> External Supervisor: Swathi Rao (Scania CV AB)

The thesis evaluates the HEMMIR confidence signals (S2, S3, S4, C) across three corpora:
- **SPIQA** — scientific figure QA (218 questions, multimodal academic papers)
- **Vectara** — diverse web QA benchmark
- **Scania** — 103 industrial questions with dual-verifier labels (GPT-4o AND Gemma 3 27B must both agree an answer is correct)

Key empirical findings validated by this system:
- S4 (Evidence Support Ratio) is the strongest individual signal across all three corpora (AUROC 0.752 on SPIQA)
- Composite C score outperforms all individual signals (AUROC 0.775 on SPIQA with R2-A retrieval)
- LLM-based signals outperform transformer-only baselines on industrial text (Scania); transformer S2/S3 are competitive on academic text (SPIQA)
- Platt-scaled calibration reduces ECE from 0.398 to 0.137 on SPIQA
