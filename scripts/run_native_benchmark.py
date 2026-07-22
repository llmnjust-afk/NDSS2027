"""Native AgentDojo benchmark runner (Round-B faithful evaluation).

Replaces the synthetic string-matching runner with REAL end-to-end evaluation:
each (defense, attack, seed, user_task, injection_task) trial runs the agent
through AgentDojo's native pipeline, executes real tools, and is scored by
AgentDojo's native ``utility`` (task completion) and ``security_from_traces``
(injection executed) evaluators.

Outputs, per experiment directory:
  - checkpoint.jsonl : one JSON record per trial (resumable)
  - summary.csv      : aggregated per (defense, attack) with ASR/utility/CIs

Usage:
  PYTHONPATH=/data/lab/NDSS2027 python3 scripts/run_native_benchmark.py \
      --defenses none spotlighting struq task_shield p2 \
      --attacks important_instructions ignore_previous tool_knowledge \
      --suite workspace --n-user-tasks 10 --n-injection-tasks 3 \
      --seeds 0 1 2 --model gpt-4o-mini-2024-07-18 \
      --outdir /data/lab/NDSS2027/results_native_pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# --- pydantic forward-ref fix for agentdojo 0.1.35 TaskResults --------------- #
from agentdojo.functions_runtime import FunctionCall  # noqa: F401
import agentdojo.benchmark as _bm

try:
    _bm.TaskResults.model_rebuild()
except Exception as _e:  # pragma: no cover
    print("TaskResults.model_rebuild note:", _e, file=sys.stderr)

from agentdojo.task_suite import get_suite
from agentdojo.benchmark import run_task_with_injection_tasks
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.logging import OutputLogger

sys.path.insert(0, "/data/lab/NDSS2027")
from sok_ipl.native.pipeline_builder import build_pipeline  # noqa: E402

API_KEY = os.environ.get(
    "CHATANYWHERE_KEY", "sk-9ZuUA9MWpglHgMJhKyBsPUDnGDm95ygy9yN4YqVoLc7GsRp0"
)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_done(checkpoint: Path) -> set[str]:
    done = set()
    if checkpoint.exists():
        with checkpoint.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["trial_id"])
                except Exception:
                    continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defenses", nargs="+", required=True)
    ap.add_argument("--attacks", nargs="+", default=["important_instructions"])
    ap.add_argument("--suite", default="workspace")
    ap.add_argument("--benchmark-version", default="v1.2.2")
    ap.add_argument("--n-user-tasks", type=int, default=10)
    ap.add_argument("--n-injection-tasks", type=int, default=3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--p2-mode", default="combined")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir = outdir / "runs"
    checkpoint = outdir / "checkpoint.jsonl"
    done = load_done(checkpoint)

    suite = get_suite(args.benchmark_version, args.suite)
    ut_ids = list(suite.user_tasks.keys())[: args.n_user_tasks]
    it_ids = list(suite.injection_tasks.keys())[: args.n_injection_tasks]

    total = len(args.defenses) * len(args.attacks) * len(args.seeds) * len(ut_ids) * len(it_ids)
    print(f"Total trials planned: {total} (already done: {len(done)})", flush=True)

    ck = checkpoint.open("a")
    n_run = 0
    t_start = time.time()
    for defense in args.defenses:
        for attack_name in args.attacks:
            for seed in args.seeds:
                # Build a fresh pipeline per (defense, seed) so element caches reset.
                pipeline = build_pipeline(
                    defense, args.model, API_KEY,
                    threshold=args.threshold, p2_mode=args.p2_mode,
                )
                # Keep pipeline.name = model (so AgentDojo attacks address the model
                # by prose name), but isolate on-disk traces per (defense, mode, seed)
                # via a dedicated logdir subtree so different defenses never overwrite
                # each other's trace JSON files.
                tag = defense + (f"-{args.p2_mode}" if defense == "p2" else "")
                trial_logdir = logdir / tag / f"s{seed}"
                trial_logdir.mkdir(parents=True, exist_ok=True)
                attack = load_attack(attack_name, suite, pipeline)
                for ut_id in ut_ids:
                    for it_id in it_ids:
                        trial_id = f"{defense}|{attack_name}|s{seed}|{ut_id}|{it_id}"
                        if trial_id in done:
                            continue
                        rec = {
                            "trial_id": trial_id, "defense": defense,
                            "attack": attack_name, "seed": seed,
                            "user_task": ut_id, "injection_task": it_id,
                            "model": args.model, "threshold": args.threshold,
                            "p2_mode": args.p2_mode if defense == "p2" else None,
                        }
                        try:
                            with OutputLogger(str(trial_logdir)):
                                ut, sec = run_task_with_injection_tasks(
                                    suite, pipeline,
                                    suite.get_user_task_by_id(ut_id),
                                    attack, trial_logdir, True, [it_id],
                                    args.benchmark_version,
                                )
                            rec["utility"] = bool(list(ut.values())[0])
                            rec["security_breached"] = bool(list(sec.values())[0])
                            rec["error"] = None
                        except Exception as e:
                            rec["utility"] = None
                            rec["security_breached"] = None
                            rec["error"] = f"{type(e).__name__}: {e}"[:300]
                        ck.write(json.dumps(rec) + "\n")
                        ck.flush()
                        n_run += 1
                        if n_run % 5 == 0:
                            el = time.time() - t_start
                            print(f"  {n_run} trials run ({el:.0f}s, "
                                  f"{el / n_run:.1f}s/trial)", flush=True)
    ck.close()
    print(f"Done. Ran {n_run} new trials.", flush=True)
    aggregate(checkpoint, outdir / "summary.csv")


def aggregate(checkpoint: Path, out_csv: Path):
    """Aggregate per (defense, attack): ASR = mean(security_breached),
    utility = mean(utility), with Wilson CIs, over successful trials."""
    from collections import defaultdict

    groups = defaultdict(lambda: {"sec": [], "util": [], "err": 0})
    with checkpoint.open() as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["defense"], rec["attack"])
            if rec.get("error"):
                groups[key]["err"] += 1
                continue
            if rec.get("security_breached") is not None:
                groups[key]["sec"].append(1 if rec["security_breached"] else 0)
            if rec.get("utility") is not None:
                groups[key]["util"].append(1 if rec["utility"] else 0)

    rows = []
    for (defense, attack), g in sorted(groups.items()):
        n_sec = len(g["sec"]); k_sec = sum(g["sec"])
        n_util = len(g["util"]); k_util = sum(g["util"])
        asr = k_sec / n_sec if n_sec else 0.0
        util = k_util / n_util if n_util else 0.0
        asr_lo, asr_hi = wilson_ci(k_sec, n_sec)
        util_lo, util_hi = wilson_ci(k_util, n_util)
        rows.append({
            "defense": defense, "attack": attack,
            "n_trials": n_sec, "ASR": round(asr, 4),
            "ASR_ci_lo": round(asr_lo, 4), "ASR_ci_hi": round(asr_hi, 4),
            "utility": round(util, 4),
            "util_ci_lo": round(util_lo, 4), "util_ci_hi": round(util_hi, 4),
            "errors": g["err"],
        })

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["defense", "attack", "n_trials", "ASR"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote summary: {out_csv}", flush=True)
    for r in rows:
        print(f"  {r['defense']:14s} {r['attack']:22s} "
              f"ASR={r['ASR']:.2f}[{r['ASR_ci_lo']:.2f},{r['ASR_ci_hi']:.2f}] "
              f"util={r['utility']:.2f} n={r['n_trials']} err={r['errors']}",
              flush=True)


if __name__ == "__main__":
    main()
