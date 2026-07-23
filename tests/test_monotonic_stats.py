"""Unit tests for monotonic aggregation + paired statistics.
Pure Python. Run:  python3 -m tests.test_monotonic_stats
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sok_ipl.native.monotonic_stats import (
    analyze, robust_asr, per_unit_summary, group_by_unit,
    mcnemar_exact, paired_bootstrap_delta,
)


def _mk(defense, ut, it, kind, succ, util=True, status="complete"):
    return {"target_defense": defense, "seed": 0, "user_task_id": ut,
            "injection_task_id": it, "candidate_kind": kind,
            "candidate_id": kind, "attack_success": succ, "utility": util,
            "attempt_status": status}


def test_robust_asr_ge_l1():
    """Robust ASR must always be >= ASR_L1 (L1 is in the set)."""
    attempts = []
    for ut in range(4):
        attempts += [
            _mk("d", f"u{ut}", "i0", "l1", ut in (0, 1)),
            _mk("d", f"u{ut}", "i0", "matched", False),
            _mk("d", f"u{ut}", "i0", "generic", ut == 2),
            _mk("d", f"u{ut}", "i0", "control", False),
            _mk("d", f"u{ut}", "i0", "paraphrase", ut == 3),
        ]
    units = group_by_unit(attempts)
    rows = per_unit_summary(units)
    r = robust_asr(rows)
    assert r["Robust_ASR_L2set"] >= r["ASR_L1"], r
    # L1 succeeds on 2/4, generic adds u2, paraphrase adds u3 -> all 4
    assert r["ASR_L1"] == 0.5
    assert r["Robust_ASR_L2set"] == 1.0
    print("OK test_robust_asr_ge_l1 (L1=0.5, robust=1.0)")


def test_monotonicity_enforced():
    """If a synthetic dataset violated monotonicity, per_unit_summary asserts.
    Here L1 success is always included in the OR, so it can never violate."""
    attempts = [
        _mk("d", "u0", "i0", "l1", True),      # L1 success
        _mk("d", "u0", "i0", "matched", False),
        _mk("d", "u0", "i0", "generic", False),
        _mk("d", "u0", "i0", "control", False),
        _mk("d", "u0", "i0", "paraphrase", False),
    ]
    rows = per_unit_summary(group_by_unit(attempts))
    assert rows[0]["l1_success"] and rows[0]["l2_set_success"]
    print("OK test_monotonicity_enforced (L1 success -> set success)")


def test_mcnemar_bounds():
    assert mcnemar_exact(0, 0) == 1.0
    # all discordant in one direction -> small p
    assert mcnemar_exact(0, 10) < 0.01
    assert 0.0 <= mcnemar_exact(3, 5) <= 1.0
    print("OK test_mcnemar_bounds")


def test_incomplete_units_excluded():
    """Units missing L1 or with errored attempts are excluded, not counted as fail."""
    attempts = [
        _mk("d", "u0", "i0", "matched", True, status="api_error"),
        # no L1 for u0 -> unit skipped
        _mk("d", "u1", "i0", "l1", False),
        _mk("d", "u1", "i0", "matched", True),
    ]
    rows = per_unit_summary(group_by_unit(attempts))
    ids = {r["user_task_id"] for r in rows}
    assert ids == {"u1"}, ids
    print("OK test_incomplete_units_excluded")


def test_paired_bootstrap_shape():
    rows = [{"l1_success": i % 2 == 0, "l2_set_success": True,
             "user_task_id": f"u{i%3}", "injection_task_id": "i0"}
            for i in range(12)]
    d = paired_bootstrap_delta(rows, n_boot=2000, seed=1)
    assert d["lo"] <= d["delta"] <= d["hi"]
    assert d["n"] == 12
    print("OK test_paired_bootstrap_shape (delta=%.2f [%.2f,%.2f])"
          % (d["delta"], d["lo"], d["hi"]))


def test_analyze_end_to_end():
    attempts = []
    for d in ["spotlighting", "task_shield"]:
        for ut in range(5):
            for it in range(2):
                attempts += [
                    _mk(d, f"u{ut}", f"i{it}", "l1", (ut + it) % 3 == 0),
                    _mk(d, f"u{ut}", f"i{it}", "matched", False),
                    _mk(d, f"u{ut}", f"i{it}", "generic", ut == 4),
                    _mk(d, f"u{ut}", f"i{it}", "control", False),
                    _mk(d, f"u{ut}", f"i{it}", "paraphrase", ut == 3),
                ]
    res = analyze(attempts, n_boot=500)
    for d in ["spotlighting", "task_shield"]:
        pd = res["per_defense"][d]
        assert pd["Robust_ASR_L2set"] >= pd["ASR_L1"]
        assert pd["monotonicity_violations_b"] == 0
    assert len(res["per_candidate"]) > 0
    print("OK test_analyze_end_to_end")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print("\nALL %d STATS TESTS PASSED" % len(fns))


if __name__ == "__main__":
    _run_all()
