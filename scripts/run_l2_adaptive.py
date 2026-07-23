"""L2 adaptive attack runner: defense-class-matched evasion strategies.

For each defense, selects the matched adaptive attack from DEFENSE_TO_ATTACK
and runs it natively. This produces the L2 (adaptive) results that, combined
with the existing L1 (important_instructions, static) results, form the
L1-vs-L2 comparison that is the core of the paper's "Adaptive Evaluation"
contribution.

Usage:
  PYTHONPATH=/data/lab/NDSS2027 python3 scripts/run_l2_adaptive.py \
      --suite workspace --n-user-tasks 10 --n-injection-tasks 3 --seeds 0 1 2 \
      --model gpt-4o-mini-2024-07-18 \
      --outdir /data/lab/NDSS2027/results_l2_adaptive
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
from agentdojo.benchmark import run_task_with_injection_tasks
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.logging import OutputLogger

# Register adaptive attacks
import sok_ipl.native.adaptive_attacks  # noqa: F401
from sok_ipl.native.adaptive_attacks import DEFENSE_TO_ATTACK

sys.path.insert(0, "/data/lab/NDSS2027")
from sok_ipl.native.pipeline_builder import build_pipeline  # noqa: E402

API_KEY = os.environ.get(
    "CHATANYWHERE_KEY", "sk-9ZuUA9MWpglHgMJhKyBsPUDnGDm95ygy9yN4YqVoLc7GsRp0"
)

DEFENSES = list(DEFENSE_TO_ATTACK.keys())


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
    ap.add_argument("--defenses", nargs="+", default=DEFENSES)
    ap.add_argument("--suite", default="workspace")
    ap.add_argument("--benchmark-version", default="v1.2.2")
    ap.add_argument("--n-user-tasks", type=int, default=10)
    ap.add_argument("--n-injection-tasks", type=int, default=3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
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
    it_ids = list(suite.injection_tasks.keys())[: args.n_injection_tasks]

    total = len(args.defenses) * len(args.seeds) * len(ut_ids) * len(it_ids)
    print(f"Total L2 adaptive trials planned: {total} (done: {len(done)})", flush=True)

    ck = checkpoint.open("a")
    n_run = 0
    t0 = time.time()
    for defense in args.defenses:
        attack_name = DEFENSE_TO_ATTACK[defense]
        for seed in args.seeds:
            pipeline = build_pipeline(
                defense, args.model, API_KEY,
                threshold=args.threshold, p2_mode=args.p2_mode,
            )
            tag = defense + (f"-{args.p2_mode}" if defense == "p2" else "")
            trial_logdir = outdir / "runs" / tag / f"s{seed}"
            trial_logdir.mkdir(parents=True, exist_ok=True)
            attack = load_attack(attack_name, suite, pipeline)
            for ut_id in ut_ids:
                for it_id in it_ids:
                    trial_id = f"{defense}|{attack_name}|s{seed}|{ut_id}|{it_id}"
                    if trial_id in done:
                        continue
                    rec = {
                        "trial_id": trial_id, "defense": defense,
                        "attack": attack_name, "attack_type": "L2_adaptive",
                        "seed": seed, "user_task": ut_id, "injection_task": it_id,
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
                        el = time.time() - t0
                        print(f"  {n_run} trials ({el:.0f}s, {el/n_run:.1f}s/trial)", flush=True)
    ck.close()
    print(f"Done. Ran {n_run} new trials.", flush=True)
    aggregate(checkpoint, outdir / "summary.csv")


def aggregate(checkpoint, out_csv):
    from collections import defaultdict
    groups = defaultdict(lambda: {"sec": [], "util": [], "err": 0, "attack": ""})
    with checkpoint.open() as f:
        for line in f:
            rec = json.loads(line)
            key = rec["defense"]
            groups[key]["attack"] = rec.get("attack", "")
            if rec.get("error"):
                groups[key]["err"] += 1
                continue
            if rec.get("security_breached") is not None:
                groups[key]["sec"].append(1 if rec["security_breached"] else 0)
            if rec.get("utility") is not None:
                groups[key]["util"].append(1 if rec["utility"] else 0)
    rows = []
    for defense, g in sorted(groups.items()):
        n = len(g["sec"]); k = sum(g["sec"]); ku = sum(g["util"]); nu = len(g["util"])
        asr = k / n if n else 0.0
        util = ku / nu if nu else 0.0
        lo, hi = wilson_ci(k, n)
        ulo, uhi = wilson_ci(ku, nu)
        rows.append({
            "defense": defense, "attack": g["attack"], "n": n,
            "ASR": round(asr, 4), "ASR_lo": round(lo, 4), "ASR_hi": round(hi, 4),
            "utility": round(util, 4), "util_lo": round(ulo, 4), "util_hi": round(uhi, 4),
            "errors": g["err"],
        })
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["defense"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n=== L2 ADAPTIVE RESULTS ===", flush=True)
    for r in rows:
        print(f"  {r['defense']:16s} [{r['attack']:30s}] "
              f"ASR={r['ASR']:.2f}[{r['ASR_lo']:.2f},{r['ASR_hi']:.2f}] "
              f"util={r['utility']:.2f} n={r['n']} err={r['errors']}",
              flush=True)


if __name__ == "__main__":
    main()
