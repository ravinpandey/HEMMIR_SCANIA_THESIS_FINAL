"""
rq3_extended/config.py

Threshold and hyperparameter configuration for RQ3 Extended
self-correction loop.

Signals (updated to faith_new / comp_new / evidence_support_ratio):
──────────────────────────────────────────────────────────────────────
faith_new thresholds:
  LOW  = 0.75  — below this, unsupported atomic claims dominate synthesis.
                 From SPIQA interim results: mean faith_new ≈ 0.916 (R2-A baseline).
                 0.75 ≈ 1.8σ below mean — grounding is genuinely breaking down.
  VLOW = 0.55  — structurally unreliable; most claims unsupported.

evidence_support_ratio thresholds:
  LOW  = 0.60  — fewer than 60% of question requirements are covered by evidence.
                 Previously cosine-distance proxy (mean=0.85); new signal directly
                 measures whether evidence pool satisfies question needs.
                 0.60 is a principled floor: if >40% of requirements have no evidence,
                 MultiQuery re-retrieval is warranted.
  VLOW = 0.40  — evidence pool covers less than half of what the question requires;
                 retrieval has essentially failed for this question.

comp_new thresholds:
  LOW  = 0.40  — answer covers fewer than 40% of evidence-supported requirements.
                 From RAVI_THESIS: comp_new mean on Vectara ≈ 0.55 (R2-A).
                 0.40 ≈ 1.5σ below mean — key aspects are systematically missing.
  VLOW = 0.25  — answer covers less than 1 in 4 of the answerable requirements.

Full-escalation condition:
  All three simultaneously below VLOW — systemic failure across retrieval,
  grounding, and coverage; MultiQuery + expanded K is the only viable path.

Signal → repair mapping (one-to-one):
  faith_new < LOW             → faith_repair    (tighten ArgRAG claim threshold)
  evidence_support_ratio < LOW → evidence_repair (MultiQuery re-retrieval)
  comp_new < LOW              → gap_repair      (targeted missing-requirement retrieval)
                                  gap seeds = missing requirements from comp_new judge
                                  (previously: weak ArgRAG claims — less targeted)
  all three < VLOW            → full_escalation

MIN_C_IMPROVEMENT = 0.02:
  Guard against accepting marginal repairs. 0.02 is ~1% of scale — deliberate
  conservatism to prevent runaway iteration.
"""

# ── Repair routing thresholds ─────────────────────────────────────────────────
THR_FAITH_LOW  = 0.75
THR_FAITH_VLOW = 0.55   # tightened: faith_new < 0.55 = structurally unreliable

THR_EVSTR_LOW  = 0.60   # lowered from 0.70: ev_support_ratio has different distribution
THR_EVSTR_VLOW = 0.40   # lowered from 0.60: <40% requirements covered = retrieval failed

THR_COMP_LOW   = 0.40   # raised from 0.35: comp_new mean ≈ 0.55 on Vectara
THR_COMP_VLOW  = 0.25   # unchanged: <25% coverage is systemic incompleteness

# ── Improvement guard ─────────────────────────────────────────────────────────
MIN_C_IMPROVEMENT = 0.02

# ── MultiQuery defaults ───────────────────────────────────────────────────────
DEFAULT_N_VARIANTS   = 4
DEFAULT_TOP_K        = 10   # chunks per query variant
DEFAULT_EXPANDED_K   = 15   # used during full escalation
DEFAULT_MAX_ITER     = 2

# ── ArgRAG thresholds (stricter for faith repair) ────────────────────────────
FAITH_REPAIR_STRONG_THR = 0.70   # vs normal R2-G 0.65
NORMAL_STRONG_THR       = 0.65

# ── C score weights (mirrors rq3_scorer.py) ───────────────────────────────────
W_FAITH   = 0.45
W_EVSTR   = 0.25
W_COMP    = 0.20
W_DIRECT  = 0.10
HAL_CAP   = 0.35
COVERAGE_DIRECT_THR = 0.80
