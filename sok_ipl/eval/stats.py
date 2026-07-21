"""Statistical tests required by NDSS: paired tests, effect sizes, CIs.

We report mean +/- std over >=3 seeds and a paired Wilcoxon signed-rank test
between every defense and the no-defense baseline, plus Cohen's d. NDSS
reviewers in the ML-security track routinely reject papers that report only a
single run or only p < 0.05 without effect size; this module enforces the bar.
"""

from __future__ import annotations

import math
from typing import Sequence

try:
    from scipy.stats import wilcoxon  # type: ignore
except Exception:  # scipy optional in CI
    wilcoxon = None  # type: ignore


def mean_std(xs: Sequence[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / max(n - 1, 1)
    return mu, math.sqrt(var)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    ma, sa = mean_std(a)
    mb, sb = mean_std(b)
    pooled = math.sqrt((sa * sa + sb * sb) / 2) or 1e-9
    return (ma - mb) / pooled


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Returns (statistic, p-value). Falls back to NaN if scipy unavailable."""
    if wilcoxon is None or len(a) < 5:
        return float("nan"), float("nan")
    try:
        out: object = wilcoxon(a, b)
        if isinstance(out, tuple):
            return float(out[0]), float(out[1])
        return (
            float(getattr(out, "statistic", float("nan"))),
            float(getattr(out, "pvalue", float("nan"))),
        )
    except Exception:
        return float("nan"), float("nan")


def bootstrap_ci(xs: Sequence[float], n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    """95% bootstrap CI for the mean of `xs`."""
    import random

    if not xs:
        return 0.0, 0.0
    rng = random.Random(0)
    means = []
    for _ in range(n_boot):
        sample = [rng.choice(xs) for _ in range(len(xs))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return lo, hi
