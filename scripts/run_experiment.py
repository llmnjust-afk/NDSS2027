"""Main experiment driver.

Reads a YAML config, materializes the experiment matrix
(defense x attack_level x backbone x seed x benchmark), runs it, and writes
results + the summary table to disk.

Usage:
    python -m scripts.run_experiment --config configs/experiments.yaml
    python -m scripts.run_experiment --config configs/experiments.yaml --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import yaml

# Ensure the package is importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sok_ipl.attacks import AdaptiveAttackFramework  # noqa: E402
from sok_ipl.defenses import (  # noqa: E402
    NoDefense, Spotlighting, SpotlightingQuoting, SpotlightingBase64, StruQ,
    TaskShield, AttentionTracker, IPIGuard, PolymorphicPrompt,
    MixtureOfEncodings, FATH,
)
from sok_ipl.principles import (  # noqa: E402
    P1UnobservableSignal, P2OrthogonalSignals, P3TaskInvariant, P4LeastPrivilege,
)
from sok_ipl.eval import (  # noqa: E402
    AgentDojoAdapter, InjecAgentAdapter, ASBAdapter, AdaptiveIPIBenchAdapter,
    Runner, mean_std, cohens_d, paired_wilcoxon,
)
from sok_ipl.analysis import summary_table, dominant_root_causes, label_root_cause  # noqa: E402
from sok_ipl.llm import build_backbone, StubBackend  # noqa: E402
from sok_ipl.registry import get as get_defense, list_names  # noqa: E402
from sok_ipl.types import AttackLevel  # noqa: E402


DEFENSE_FACTORY = {
    "none": NoDefense,
    "spotlighting": Spotlighting,
    "spotlighting_quoting": SpotlightingQuoting,
    "spotlighting_base64": SpotlightingBase64,
    "struq": StruQ,
    "task_shield": TaskShield,
    "attention_tracker": AttentionTracker,
    "ipiguard": IPIGuard,
    "polymorphic": PolymorphicPrompt,
    "mixture_encodings": MixtureOfEncodings,
    "fath": FATH,
    "p1_unobservable_signal": P1UnobservableSignal,
    "p2_orthogonal_signals": P2OrthogonalSignals,
    "p3_task_invariant": P3TaskInvariant,
    "p4_least_privilege": P4LeastPrivilege,
}

BENCHMARK_FACTORY = {
    "agentdojo": AgentDojoAdapter,
    "injecagent": InjecAgentAdapter,
    "asb": ASBAdapter,
    "adaptive_ipi_bench": AdaptiveIPIBenchAdapter,
}

LEVELS = {
    "L0": AttackLevel.L0_BENIGN,
    "L1": AttackLevel.L1_STATIC,
    "L2": AttackLevel.L2_ADAPTIVE_SINGLE,
    "L3": AttackLevel.L3_ADAPTIVE_MULTI,
    "L4": AttackLevel.L4_BACKDOOR_AUGMENTED,
}

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


def build_llm(spec: str):
    if spec == "stub":
        return StubBackend(response="CALL send_email {\"to\": \"user@example.com\"}")
    return build_backbone(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--smoke", action="store_true", help="tiny matrix for CI")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.smoke:
        cfg["defenses"] = ["none", "struq"]
        cfg["backbones"] = ["stub"]
        cfg["levels"] = ["L0", "L1"]
        cfg["seeds"] = [0]
        cfg["benchmarks"] = ["asb"]
        cfg["max_tasks_per_benchmark"] = 3

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    all_results = []
    max_iters = cfg.get("adaptive_max_iters", 2)

    # Checkpoint: resume from previous partial run if results/checkpoint.jsonl exists.
    checkpoint_path = out_dir / "checkpoint.jsonl"
    done_keys: set[str] = set()
    if checkpoint_path.exists():
        import json as _json
        from sok_ipl.types import AttackResult as _AR
        with checkpoint_path.open() as f:
            for line in f:
                try:
                    r = _json.loads(line)
                    key = f"{r['benchmark']}|{r['backbone']}|{r['seed']}|{r['defense_name']}|{r['attack_level']}|{r['task_id']}"
                    done_keys.add(key)
                    # Reconstruct AttackResult object so summary functions work.
                    all_results.append(_AR(
                        task_id=r["task_id"],
                        attack_level=AttackLevel(r["attack_level"]),
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
                except Exception:
                    pass
        print(f"[resume] loaded {len(done_keys)} completed results from checkpoint")

    ckpt_file = checkpoint_path.open("a")
    n_new = 0
    for bench_name in cfg["benchmarks"]:
        # AgentDojoAdapter takes suite_name + max_per_suite; others take nothing.
        if bench_name == "agentdojo":
            adapter = BENCHMARK_FACTORY[bench_name](suite_name="workspace", max_per_suite=cfg.get("max_tasks_per_benchmark", 20))
        else:
            adapter = BENCHMARK_FACTORY[bench_name]()
        tasks = adapter.load_tasks()
        if cfg.get("max_tasks_per_benchmark"):
            tasks = tasks[: cfg["max_tasks_per_benchmark"]]
        print(f"[{bench_name}] {len(tasks)} tasks")

        for backbone_spec in cfg["backbones"]:
            llm = build_llm(backbone_spec)

            for seed in cfg["seeds"]:
                runner = Runner(llm=llm, seed=seed, max_iters=max_iters)
                for defense_name in cfg["defenses"]:
                    DefenseCls = DEFENSE_FACTORY[defense_name]
                    try:
                        defense = DefenseCls(llm=llm)
                    except TypeError:
                        defense = DefenseCls()
                    for level_key in cfg["levels"]:
                        level = LEVELS[level_key]
                        for task in tasks:
                            key = f"{bench_name}|{getattr(llm,'name','stub')}|{seed}|{defense_name}|{level.value}|{task.task_id}"
                            if key in done_keys:
                                continue
                            res = runner.run_one(task, defense, level)
                            res.seed = seed
                            res.failure_root_cause = (
                                label_root_cause(res, DEFENSE_CLASS_OF[defense_name], llm)
                                if res.success else None
                            )
                            all_results.append(res)
                            n_new += 1
                            # Checkpoint every result (cheap, append-only).
                            ckpt_file.write(__import__("json").dumps({
                                "task_id": res.task_id,
                                "benchmark": bench_name,
                                "attack_level": res.attack_level.value,
                                "defense_name": res.defense_name,
                                "backbone": res.backbone,
                                "seed": res.seed,
                                "success": res.success,
                                "blocked_by_defense": res.blocked_by_defense,
                                "flagged_by_defense": res.flagged_by_defense,
                                "utility_preserved": res.utility_preserved,
                                "failure_root_cause": res.failure_root_cause,
                                "attack_strategy": res.attack_strategy,
                            }) + "\n")
                            ckpt_file.flush()
                            if (len(all_results)) % 50 == 0:
                                print(f"  ...{len(all_results)} results so far ({n_new} new)")
    ckpt_file.close()

    # Persist raw results (deep copy so we do not mutate the live AttackResults)
    import copy as _copy
    rows = []
    for r in all_results:
        d = _copy.copy(r.__dict__)
        d["attack_level"] = r.attack_level.value
        rows.append(d)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2, default=str))

    # Summary table
    summary = summary_table(all_results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # CSV for easy copy into LaTeX
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        headers = ["defense", "USR", "ASR_L1", "ASR_L2", "ASR_L3", "ASR_L4", "FPR", "R-ASR"]
        w.writerow(headers)
        for d, m in summary.items():
            w.writerow([d] + [f"{m[h]:.3f}" for h in headers[1:]])

    # Dominant root cause per defense class
    dom = dominant_root_causes(all_results, DEFENSE_CLASS_OF)
    (out_dir / "root_causes.json").write_text(json.dumps(dom, indent=2))

    print(f"\nWrote {len(all_results)} results to {out_dir}/")
    print(f"Summary table ({len(summary)} defenses):")
    for d, m in summary.items():
        print(f"  {d:28s} USR={m['USR']:.2f} L1={m['ASR_L1']:.2f} "
              f"L2={m['ASR_L2']:.2f} L3={m['ASR_L3']:.2f} L4={m['ASR_L4']:.2f} "
              f"FPR={m['FPR']:.2f} R-ASR={m['R-ASR']:.2f}")
    print("\nDominant root cause per defense class:")
    for cls, cause in dom.items():
        print(f"  {cls:25s} -> {cause}")


if __name__ == "__main__":
    main()
