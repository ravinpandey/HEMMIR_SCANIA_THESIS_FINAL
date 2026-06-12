"""
rq2_ablation/stats.py

Statistical tests for RQ2 paired comparison.

All tests are paired — each question has outputs from both Direct RAG and
each ArgRAG condition, so within-question correlation is the dominant effect.

Tests implemented:
  1. McNemar test        — binary faithful/unfaithful outcome per question
  2. Wilcoxon signed-rank — graded faithfulness scores (non-parametric)
  3. Bootstrap CI         — 95% CI on metric differences (mean faithfulness delta)

Faithful binary threshold: 0.7 (from judge.py FAITHFUL_THRESHOLD)

Usage:
    from rq2_ablation.stats import run_all_tests
    results = run_all_tests(baseline_scores, condition_scores)
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple


FAITHFUL_THRESHOLD = 0.7


# ── McNemar test ──────────────────────────────────────────────────────────────

def mcnemar_test(
    baseline:   List[float],
    condition:  List[float],
    threshold:  float = FAITHFUL_THRESHOLD,
) -> Dict:
    """
    McNemar test for paired binary faithfulness outcomes.

    Contingency table:
        b = questions where baseline=faithful, condition=unfaithful
        c = questions where baseline=unfaithful, condition=faithful

    H0: P(b) = P(c)  — no difference in faithfulness rates
    H1: P(b) != P(c) — argumentation changes faithfulness

    Uses continuity-corrected chi-squared: χ² = (|b-c| - 1)² / (b+c)
    p-value from chi-squared distribution with df=1.
    Returns exact Fisher if b+c < 25.
    """
    assert len(baseline) == len(condition), "Lengths must match"

    b_bin = [1 if s >= threshold else 0 for s in baseline]
    c_bin = [1 if s >= threshold else 0 for s in condition]

    # Count discordant pairs
    b_count = sum(1 for bb, cc in zip(b_bin, c_bin) if bb == 1 and cc == 0)
    c_count = sum(1 for bb, cc in zip(b_bin, c_bin) if bb == 0 and cc == 1)
    n_total = b_count + c_count

    if n_total == 0:
        return {
            "test":      "McNemar",
            "b":         0,
            "c":         0,
            "chi2":      0.0,
            "p_value":   1.0,
            "significant": False,
            "note":      "no discordant pairs",
        }

    if n_total < 25:
        # Exact binomial p-value: P(X >= max(b,c)) under H0: p=0.5
        bigger = max(b_count, c_count)
        p_value = 2 * _binomial_tail(n_total, bigger, 0.5)
        method = "exact_binomial"
    else:
        # Chi-squared with continuity correction
        chi2 = (abs(b_count - c_count) - 1) ** 2 / n_total
        p_value = _chi2_p(chi2, df=1)
        method = "chi2_continuity"

    baseline_faithful  = sum(b_bin) / max(1, len(b_bin))
    condition_faithful = sum(c_bin) / max(1, len(c_bin))

    return {
        "test":               "McNemar",
        "method":             method,
        "b":                  b_count,
        "c":                  c_count,
        "baseline_faithful":  round(baseline_faithful, 4),
        "condition_faithful": round(condition_faithful, 4),
        "p_value":            round(p_value, 6),
        "significant":        p_value < 0.05,
        "direction":          "condition_better" if c_count > b_count else "baseline_better",
    }


# ── Wilcoxon signed-rank test ─────────────────────────────────────────────────

def wilcoxon_test(
    baseline:  List[float],
    condition: List[float],
) -> Dict:
    """
    Wilcoxon signed-rank test for paired graded faithfulness scores.
    Non-parametric — no normality assumption required.

    H0: median difference = 0
    H1: median difference != 0

    Uses normal approximation when n >= 10 (with tie correction).
    Returns exact p-value (exhaustive sign enumeration) when n < 10.
    """
    assert len(baseline) == len(condition)

    diffs = [c - b for b, c in zip(baseline, condition)]

    # Remove zeros
    nonzero = [(abs(d), 1 if d > 0 else -1) for d in diffs if d != 0.0]
    n = len(nonzero)

    if n == 0:
        return {
            "test":      "Wilcoxon",
            "n":         0,
            "W":         0.0,
            "p_value":   1.0,
            "significant": False,
            "note":      "all differences are zero",
            "mean_diff": 0.0,
        }

    # Rank |diffs|
    ranked = sorted(nonzero, key=lambda x: x[0])
    # Handle ties: assign average rank
    ranks = _average_ranks([x[0] for x in ranked])

    W_plus  = sum(r for (_, sign), r in zip(ranked, ranks) if sign > 0)
    W_minus = sum(r for (_, sign), r in zip(ranked, ranks) if sign < 0)
    W = min(W_plus, W_minus)

    if n < 10:
        p_value = _wilcoxon_exact(n, W)
        method = "exact"
    else:
        # Normal approximation with tie correction
        n_total = n * (n + 1) * (2 * n + 1) / 6
        # Tie correction
        vals = [x[0] for x in ranked]
        tie_groups = {}
        for v in vals:
            tie_groups[v] = tie_groups.get(v, 0) + 1
        tie_corr = sum(t ** 3 - t for t in tie_groups.values()) / 48
        variance = (n_total - tie_corr) / 4
        if variance <= 0:
            p_value = 1.0
        else:
            z = (W - 0.5) / math.sqrt(variance)
            p_value = 2 * _normal_cdf(-abs(z))
        method = "normal_approx"

    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0

    return {
        "test":        "Wilcoxon",
        "method":      method,
        "n":           n,
        "W_plus":      round(W_plus, 2),
        "W_minus":     round(W_minus, 2),
        "W":           round(W, 2),
        "p_value":     round(p_value, 6),
        "significant": p_value < 0.05,
        "mean_diff":   round(mean_diff, 4),
        "direction":   "condition_better" if mean_diff > 0 else "baseline_better",
    }


# ── Bootstrap confidence intervals ───────────────────────────────────────────

def bootstrap_ci(
    baseline:    List[float],
    condition:   List[float],
    n_bootstrap: int   = 2000,
    ci_level:    float = 0.95,
    seed:        int   = 42,
) -> Dict:
    """
    Bootstrap 95% CI on mean faithfulness difference (condition - baseline).

    Resamples paired observations with replacement n_bootstrap times.
    Reports: mean_diff, ci_lower, ci_upper, effect_size (Cohen's d for
    the paired differences).
    """
    assert len(baseline) == len(condition)
    rng = random.Random(seed)
    pairs = list(zip(baseline, condition))
    n = len(pairs)

    diffs_obs = [c - b for b, c in pairs]
    mean_obs  = sum(diffs_obs) / n

    boot_means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(pairs) for _ in range(n)]
        diffs  = [c - b for b, c in sample]
        boot_means.append(sum(diffs) / n)

    boot_means.sort()
    alpha = 1 - ci_level
    lo_idx = int(alpha / 2 * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap)
    ci_lower = boot_means[lo_idx]
    ci_upper = boot_means[min(hi_idx, n_bootstrap - 1)]

    # Cohen's d for paired differences
    if n > 1:
        var = sum((d - mean_obs) ** 2 for d in diffs_obs) / (n - 1)
        sd  = math.sqrt(var) if var > 0 else 1e-9
        cohens_d = mean_obs / sd
    else:
        cohens_d = 0.0

    return {
        "test":       "Bootstrap_CI",
        "n":          n,
        "n_bootstrap": n_bootstrap,
        "mean_diff":  round(mean_obs, 4),
        "ci_lower":   round(ci_lower, 4),
        "ci_upper":   round(ci_upper, 4),
        "ci_level":   ci_level,
        "cohens_d":   round(cohens_d, 4),
        "effect_size": _interpret_effect(abs(cohens_d)),
        "significant": ci_lower > 0 or ci_upper < 0,   # CI excludes 0
    }


# ── Run all three tests ───────────────────────────────────────────────────────

def run_all_tests(
    baseline:  List[float],
    condition: List[float],
    label:     str = "",
) -> Dict:
    """
    Run McNemar + Wilcoxon + Bootstrap on a baseline vs. condition pair.
    Returns combined dict with all test results.
    """
    return {
        "condition":  label,
        "n":          len(baseline),
        "mcnemar":    mcnemar_test(baseline, condition),
        "wilcoxon":   wilcoxon_test(baseline, condition),
        "bootstrap":  bootstrap_ci(baseline, condition),
    }


# ── Math helpers ──────────────────────────────────────────────────────────────

def _binomial_tail(n: int, k: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    total = 0.0
    for i in range(k, n + 1):
        total += _binom_pmf(n, i, p)
    return min(1.0, total)


def _binom_pmf(n: int, k: int, p: float) -> float:
    log_p = (
        _log_factorial(n)
        - _log_factorial(k)
        - _log_factorial(n - k)
        + k * math.log(p + 1e-300)
        + (n - k) * math.log(1 - p + 1e-300)
    )
    return math.exp(log_p)


def _log_factorial(n: int) -> float:
    return sum(math.log(i) for i in range(1, n + 1)) if n > 0 else 0.0


def _chi2_p(chi2: float, df: int = 1) -> float:
    """Approximate p-value from chi-squared distribution using regularized gamma."""
    return _regularized_gamma_upper(df / 2, chi2 / 2)


def _regularized_gamma_upper(a: float, x: float, max_iter: int = 200) -> float:
    """Upper regularized incomplete gamma function via continued fraction."""
    if x < 0:
        return 1.0
    if x == 0:
        return 1.0
    # Use series expansion for small x
    if x < a + 1:
        return 1.0 - _gamma_series(a, x)
    return _gamma_cf(a, x)


def _gamma_series(a: float, x: float) -> float:
    if x == 0:
        return 0.0
    ln_gamma_a = math.lgamma(a)
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(200):
        ap += 1
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-10:
            break
    return total * math.exp(-x + a * math.log(x) - ln_gamma_a)


def _gamma_cf(a: float, x: float) -> float:
    ln_gamma_a = math.lgamma(a)
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 201):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return math.exp(-x + a * math.log(x) - ln_gamma_a) * h


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _average_ranks(values: List[float]) -> List[float]:
    n = len(values)
    ranks = list(range(1, n + 1))
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[j] == values[j + 1]:
            j += 1
        avg = (i + j + 2) / 2
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    return ranks


def _wilcoxon_exact(n: int, W: float) -> float:
    """Exact Wilcoxon p-value by exhaustive enumeration for small n."""
    max_w = n * (n + 1) // 2
    counts = [0] * (max_w + 1)
    # Enumerate all 2^n sign assignments
    for mask in range(1 << n):
        w = sum((rank + 1) for rank in range(n) if mask & (1 << rank))
        if w <= max_w:
            counts[w] += 1
    total = 1 << n
    # Two-sided: count cases where W_stat <= W
    p = sum(counts[i] for i in range(int(W) + 1)) / total
    return min(1.0, 2 * p)


def _interpret_effect(d: float) -> str:
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"
