"""Fix W2 results in-place: recompute utility (L0) and root_cause (L1-L4) from
the existing checkpoint, without re-running any API calls.

Usage: python -m scripts.fix_results --checkpoint results/checkpoint.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sok_ipl.analysis.root_cause import label as label_root_cause
from sok_ipl.analysis import summary_table, dominant_root_causes
from sok_ipl.types import AttackLevel, AttackResult


DEFENSE_CLASS_OF = {
    "none": "none",
    "spotlighting": "input_encoding",
    "spotlighting_quoting": "input_encoding",
    "spotlighting_base64": "input_encoding",
    "struq": "structured_query",
    "task_shield": "runtime_checking",
    "attention_tracker": "internal_probing",
    "ipiguard": "runtime_checking",
    "polymorphic": "input_encoding",
    "mixture_encodings": "input_encoding",
    "fath": "runtime_checking",
    "p1_unobservable_signal": "runtime_checking",
    "p2_orthogonal_signals": "runtime_checking",
    "p3_task_invariant": "runtime_checking",
    "p4_least_privilege": "architecture",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/checkpoint.jsonl")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    lines = Path(args.checkpoint).read_text().strip().split("\n")
    print(f"Loaded {len(lines)} checkpoint entries")

    fixed_results = []
    n_util_fixed = 0
    n_cause_fixed = 0

    for line in lines:
        r = json.loads(line)
        level = AttackLevel(r["attack_level"])

        # Fix 1: L0 utility = not blocked (was incorrectly always False)
        if level == AttackLevel.L0_BENIGN:
            old_util = r["utility_preserved"]
            r["utility_preserved"] = not r["blocked_by_defense"]
            if old_util != r["utility_preserved"]:
                n_util_fixed += 1

        # Fix 2: root_cause for successful attacks (was empty due to labeling bug)
        if r["success"] and level != AttackLevel.L0_BENIGN:
            defense_class = DEFENSE_CLASS_OF.get(r["defense_name"], "unknown")
            old_cause = r.get("failure_root_cause")
            # Construct a minimal AttackResult for the label function
            temp = AttackResult(
                task_id=r["task_id"],
                attack_level=level,
                defense_name=r["defense_name"],
                backbone=r["backbone"],
                seed=r["seed"],
                success=r["success"],
                blocked_by_defense=r["blocked_by_defense"],
                flagged_by_defense=r["flagged_by_defense"],
                utility_preserved=r["utility_preserved"],
                failure_root_cause=None,
                attack_strategy=defense_class,
            )
            r["failure_root_cause"] = label_root_cause(temp, defense_class, llm=None)
            if old_cause != r["failure_root_cause"]:
                n_cause_fixed += 1

        # Reconstruct AttackResult for summary functions
        fixed_results.append(AttackResult(
            task_id=r["task_id"],
            attack_level=level,
            defense_name=r["defense_name"],
            backbone=r["backbone"],
            seed=r["seed"],
            success=r["success"],
            blocked_by_defense=r["blocked_by_defense"],
            flagged_by_defense=r["flagged_by_defense"],
            utility_preserved=r["utility_preserved"],
            failure_root_cause=r.get("failure_root_cause"),
            attack_strategy=r.get("attack_strategy", ""),
        ))

    print(f"Fixed {n_util_fixed} utility entries (L0)")
    print(f"Fixed {n_cause_fixed} root_cause entries (L1-L4)")

    # Write fixed results.json
    out_dir = Path(args.out)
    rows = []
    for r in fixed_results:
        d = {
            "task_id": r.task_id,
            "attack_level": r.attack_level.value,
            "defense_name": r.defense_name,
            "backbone": r.backbone,
            "seed": r.seed,
            "success": r.success,
            "blocked_by_defense": r.blocked_by_defense,
            "flagged_by_defense": r.flagged_by_defense,
            "utility_preserved": r.utility_preserved,
            "failure_root_cause": r.failure_root_cause,
            "attack_strategy": r.attack_strategy,
        }
        rows.append(d)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2, default=str))

    # Recompute summary table
    summary = summary_table(fixed_results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # CSV
    import csv
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        headers = ["defense", "USR", "ASR_L1", "ASR_L2", "ASR_L3", "ASR_L4", "FPR", "R-ASR"]
        w.writerow(headers)
        for d, m in summary.items():
            w.writerow([d] + [f"{m[h]:.3f}" for h in headers[1:]])

    # Root causes
    dom = dominant_root_causes(fixed_results, DEFENSE_CLASS_OF)
    (out_dir / "root_causes.json").write_text(json.dumps(dom, indent=2))

    # Print results
    print(f"\n=== FIXED SUMMARY TABLE ({len(summary)} defenses) ===")
    print(f"{'defense':<28s} {'USR':>5s} {'L1':>5s} {'L2':>5s} {'L3':>5s} {'L4':>5s} {'FPR':>5s} {'RASR':>5s}")
    for d, m in summary.items():
        print(f"{d:<28s} {m['USR']:5.2f} {m['ASR_L1']:5.2f} {m['ASR_L2']:5.2f} {m['ASR_L3']:5.2f} {m['ASR_L4']:5.2f} {m['FPR']:5.2f} {m['R-ASR']:5.2f}")

    print(f"\n=== DOMINANT ROOT CAUSE PER DEFENSE CLASS ===")
    for cls, cause in dom.items():
        print(f"  {cls:<25s} -> {cause}")

    # Detailed failure clustering
    from sok_ipl.analysis.failure_clustering import cluster_by_defense_class
    clustered = cluster_by_defense_class(fixed_results, DEFENSE_CLASS_OF)
    print(f"\n=== FAILURE CLUSTERING (counts) ===")
    for cls, causes in clustered.items():
        if causes:
            print(f"  {cls:<25s} -> {causes}")


if __name__ == "__main__":
    main()
