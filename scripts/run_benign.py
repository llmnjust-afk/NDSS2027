"""Benign L0 runner: measure utility WITHOUT injection attacks.

This is critical for separating three failure sources:
  - U_benign,no-defense  : base model's raw task capability
  - U_attack,no-defense  : model capability under attack (currently 0.19)
  - U_benign,defense     : defense's over-blocking on benign tasks
  - U_attack,defense     : defense's utility under attack (currently in main table)

Genuine over-blocking = U_benign,no-defense - U_benign,defense
Attack-induced failure = U_benign,no-defense - U_attack,no-defense
Model weakness = 1 - U_benign,no-defense

Usage:
  PYTHONPATH=/data/lab/NDSS2027 python3 scripts/run_benign.py \
      --defenses none spotlighting struq_prompt polymorphic mixture_encodings \
                 fath ipiguard p1 p2 p3 task_shield \
      --suite workspace --n-user-tasks 10 --seeds 0 1 2 \
      --model gpt-4o-mini-2024-07-18 \
      --outdir /data/lab/NDSS2027/results_benign
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

from agentdojo.functions_runtime import FunctionCall  # noqa: F401
import agentdojo.benchmark as _bm

try:
    _bm.TaskResults.model_rebuild()
except Exception as _e:
    print("TaskResults.model_rebuild note:", _e, file=sys.stderr)

from agentdojo.task_suite import get_suite
from agentdojo.benchmark import run_task_without_injection_tasks
from agentdojo.logging import OutputLogger

sys.path.insert(0, "/data/lab/NDSS2027")
from sok_ipl.native.pipeline_builder import build_pipeline  # noqa: E402

API_KEY = os.environ.get(
    "CHATANYWHERE_KEY", "sk-9ZuUA9MWpglHgMJhKyBsPUDnGDm95ygy9yN4YqVoLc7GsRp0"
)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_done(checkpoint):
    done = set()
    if checkpoint.exists():
        with checkpoint.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["trial_id"])
                except Exception:
                    continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defenses", nargs="+", required=True)
    ap.add_argument("--suite", default="workspace")
    ap.add_argument("--benchmark-version", default="v1.2.2")
    ap.add_argument("--n-user-tasks", type=int, default=10)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--p2-mode", default="combined")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = outdir / "checkpoint.jsonl"
    done = load_done(checkpoint)

    suite = get_suite(args.benchmark_version, args.suite)
    ut_ids = list(suite.user_tasks.keys())[: args.n_user_tasks]

    total = len(args.defenses) * len(args.seeds) * len(ut_ids)
    print(f"Total benign trials planned: {total} (done: {len(done)})", flush=True)

    ck = checkpoint.open("a")
    n_run = 0
    t0 = time.time()
    for defense in args.defenses:
        for seed in args.seeds:
            pipeline = build_pipeline(
                defense, args.model, API_KEY,
                threshold=args.threshold, p2_mode=args.p2_mode,
            )
            tag = defense + (f"-{args.p2_mode}" if defense == "p2" else "")
            trial_logdir = outdir / "runs" / tag / f"s{seed}"
            trial_logdir.mkdir(parents=True, exist_ok=True)
            for ut_id in ut_ids:
                trial_id = f"{defense}|benign|s{seed}|{ut_id}"
                if trial_id in done:
                    continue
                rec = {
                    "trial_id": trial_id, "defense": defense,
                    "attack": "benign", "seed": seed, "user_task": ut_id,
                }
                try:
                    with OutputLogger(str(trial_logdir)):
                        utility, security = run_task_without_injection_tasks(
                            suite, pipeline,
                            suite.get_user_task_by_id(ut_id),
                            trial_logdir, True, args.benchmark_version,
                        )
                    rec["utility"] = bool(utility)
                    rec["error"] = None
                except Exception as e:
                    rec["utility"] = None
                    rec["error"] = f"{type(e).__name__}: {e}"[:300]
                ck.write(json.dumps(rec) + "\n")
                ck.flush()
                n_run += 1
                if n_run % 5 == 0:
                    el = time.time() - t0
                    print(f"  {n_run} trials ({el:.0f}s, {el/n_run:.1f}s/trial)", flush=True)
    ck.close()
    print(f"Done. Ran {n_run} new trials.", flush=True)
    aggregate(checkpoint, outdir / "summary.csv")


def aggregate(checkpoint, out_csv):
    from collections import defaultdict
    groups = defaultdict(lambda: {"util": [], "err": 0})
    with checkpoint.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("error"):
                groups[rec["defense"]]["err"] += 1
                continue
            if rec.get("utility") is not None:
                groups[rec["defense"]]["util"].append(1 if rec["utility"] else 0)
    rows = []
    for defense, g in sorted(groups.items()):
        n = len(g["util"]); k = sum(g["util"])
        util = k / n if n else 0.0
        lo, hi = wilson_ci(k, n)
        rows.append({
            "defense": defense, "n": n,
            "benign_utility": round(util, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "errors": g["err"],
        })
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["defense"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n=== BENIGN L0 RESULTS ===", flush=True)
    for r in rows:
        print(f"  {r['defense']:16s} benign_util={r['benign_utility']:.2f} "
              f"[{r['ci_lo']:.2f},{r['ci_hi']:.2f}] n={r['n']} err={r['errors']}",
              flush=True)


if __name__ == "__main__":
    main()
