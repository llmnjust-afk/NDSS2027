"""Compute NDSS-required statistics from W2 results:
  - Wilcoxon signed-rank test (defense vs no-defense baseline, per level)
  - Cohen's d effect size
  - Bootstrap 95% CI
  - Mean +/- std over seeds

Outputs a LaTeX table fragment for the paper appendix.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sok_ipl.eval.stats import mean_std, cohens_d, paired_wilcoxon, bootstrap_ci


def load_results(path: str = "results/results.json"):
    return json.loads(Path(path).read_text())


def per_seed_asr(results, defense, level, seed):
    """ASR for a (defense, level, seed) -- fraction of successful attacks."""
    rs = [r for r in results
          if r["defense_name"] == defense
          and r["attack_level"] == level
          and r["seed"] == seed]
    if not rs:
        return 0.0
    return sum(r["success"] for r in rs) / len(rs)


def main():
    results = load_results()
    defenses = sorted({r["defense_name"] for r in results})
    levels = ["L1_static_ipi", "L2_adaptive_single_shot", "L3_adaptive_multi_turn"]
    seeds = sorted({r["seed"] for r in results})

    print("% Statistical analysis: Wilcoxon + Cohen's d + 95% CI")
    print("% Baseline: 'none' (no defense). Per-seed ASR as observations.")
    print()
    print(r"\begin{tabular}{llrrrl}")
    print(r"\toprule")
    print(r"Defense & Level & ASR (mean$\pm$std) & Cohen's $d$ & $p$-value & 95\% CI \\")
    print(r"\midrule")

    for defense in defenses:
        if defense == "none":
            continue
        for level in levels:
            # Per-seed ASR for this defense and the baseline
            def_vals = [per_seed_asr(results, defense, level, s) for s in seeds]
            base_vals = [per_seed_asr(results, "none", level, s) for s in seeds]
            mu, sd = mean_std(def_vals)
            d = cohens_d(def_vals, base_vals)
            stat, p = paired_wilcoxon(def_vals, base_vals)
            ci_lo, ci_hi = bootstrap_ci(def_vals)
            p_str = f"{p:.4f}" if p == p else "N/A"  # NaN check
            print(f"{defense} & {level.split('_')[0]} & {mu:.2f}$\\pm${sd:.2f} & {d:.2f} & {p_str} & [{ci_lo:.2f}, {ci_hi:.2f}] \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")

    # Ablation: principle prototypes vs their full-form counterparts
    print("\n% Ablation: design principles")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Principle & R-ASR & FPR & USR & Verdict \\")
    print(r"\midrule")
    for p in ["p1_unobservable_signal", "p2_orthogonal_signals", "p3_task_invariant", "p4_least_privilege"]:
        rs = [r for r in results if r["defense_name"] == p]
        if not rs:
            continue
        l2 = [r for r in rs if r["attack_level"] == "L2_adaptive_single_shot"]
        l3 = [r for r in rs if r["attack_level"] == "L3_adaptive_multi_turn"]
        l0 = [r for r in rs if r["attack_level"] == "L0_benign"]
        r_asr = max(
            sum(r["success"] for r in l2)/max(len(l2),1),
            sum(r["success"] for r in l3)/max(len(l3),1),
        )
        fpr = sum(r["blocked_by_defense"] for r in l0)/max(len(l0),1)
        usr = sum(r["utility_preserved"] for r in l0)/max(len(l0),1)
        verdict = "Broken" if r_asr > 0.5 else ("Over-blocks" if fpr > 0.5 else "Resists")
        print(f"{p} & {r_asr:.2f} & {fpr:.2f} & {usr:.2f} & {verdict} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
