"""Analyze results.json and emit LaTeX tables + plots for the paper.

Produces:
  - figures/asr_by_level.pdf      (main result: ASR L0->L4 per defense)
  - figures/root_cause_heatmap.pdf (failure-mode clustering)
  - figures/adaptive_delta.pdf     (L2-L1 delta, the headline failure)
  - tables/main_table.tex          (the SoK main table)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: str = "results/results.json"):
    return json.loads(Path(path).read_text())


def main():
    results = load()
    by_def = defaultdict(list)
    for r in results:
        by_def[r["defense_name"]].append(r)

    defenses = sorted(by_def.keys())
    levels = ["L0_benign", "L1_static_ipi", "L2_adaptive_single_shot", "L3_adaptive_multi_turn", "L4_backdoor_augmented"]
    asr_matrix = np.zeros((len(defenses), len(levels)))
    for i, d in enumerate(defenses):
        for j, lv in enumerate(levels):
            rs = [r for r in by_def[d] if r["attack_level"] == lv]
            if rs:
                # For L0, report utility preservation instead (inverted scale)
                if lv == "L0_benign":
                    asr_matrix[i, j] = 1 - sum(r["utility_preserved"] for r in rs) / len(rs)
                else:
                    asr_matrix[i, j] = sum(r["success"] for r in rs) / len(rs)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(asr_matrix, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(defenses)))
    ax.set_yticklabels(defenses, fontsize=9)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(["L0\n(1-USR)", "L1", "L2", "L3", "L4"], fontsize=9)
    ax.set_title("Attack success rate by level (red = defense fails)")
    plt.colorbar(im, ax=ax, fraction=0.025)
    Path("figures").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig("figures/asr_by_level.pdf")
    print("wrote figures/asr_by_level.pdf")

    # Headline: L2 - L1 delta
    deltas = asr_matrix[:, 2] - asr_matrix[:, 1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(defenses, deltas, color="darkred")
    ax.set_xlabel("ASR(L2 adaptive) - ASR(L1 static)")
    ax.set_title("Adaptive-attack delta (Zhan et al. effect, all defenses)")
    ax.axvline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("figures/adaptive_delta.pdf")
    print("wrote figures/adaptive_delta.pdf")

    # LaTeX main table
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Defense & USR$\uparrow$ & ASR$_{L1}$ & ASR$_{L2}$ & ASR$_{L3}$ & ASR$_{L4}$ & R-ASR$\downarrow$ \\",
        r"\midrule",
    ]
    for i, d in enumerate(defenses):
        usr = 1 - asr_matrix[i, 0]
        row = f"{d.replace('_', '- ')} & {usr:.2f} & " + " & ".join(f"{asr_matrix[i, j]:.2f}" for j in range(1, 5))
        r_asr = max(asr_matrix[i, 2:5])
        row += f" & {r_asr:.2f} \\\\"
        lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    Path("figures").mkdir(exist_ok=True)
    Path("figures/main_table.tex").write_text("\n".join(lines))
    print("wrote figures/main_table.tex")


if __name__ == "__main__":
    main()
