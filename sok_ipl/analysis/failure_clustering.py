"""Failure-mode clustering (contribution 3).

Given the per-(defense, attack) AttackResults with `failure_root_cause` labels,
cluster failures into root-cause families. The labeling itself is done by the
root_cause module; this module performs the aggregation and reports, for each
defense class, which root causes it is susceptible to.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..types import AttackResult


def cluster_by_root_cause(results: Iterable[AttackResult]) -> dict[str, dict[str, int]]:
    """{defense_name: {root_cause: count_of_successful_attacks}}.

    Counts only successful attacks (defenses that were broken), grouped by
    their labeled root cause.
    """
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        # Skip benign (L0) and non-L0 attacks that did not succeed.
        if r.attack_level.value == "L0_benign":
            continue
        if not r.success:
            continue
        cause = r.failure_root_cause or "unlabeled"
        out[r.defense_name][cause] += 1
    return {d: dict(v) for d, v in out.items()}


def cluster_by_defense_class(results: Iterable[AttackResult], defense_class_of: dict[str, str]) -> dict[str, dict[str, int]]:
    """Aggregate to the defense-class level: {defense_class: {root_cause: count}}."""
    per_def = cluster_by_root_cause(results)
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for d, causes in per_def.items():
        cls = defense_class_of.get(d, "unknown")
        for c, n in causes.items():
            out[cls][c] += n
    return {c: dict(v) for c, v in out.items()}


def dominant_root_causes(results: Iterable[AttackResult], defense_class_of: dict[str, str]) -> dict[str, str]:
    """For each defense class, the single most frequent root cause."""
    clustered = cluster_by_defense_class(results, defense_class_of)
    out = {}
    for cls, causes in clustered.items():
        if causes:
            out[cls] = max(causes.items(), key=lambda kv: kv[1])[0]
        else:
            out[cls] = "none"
    return out
