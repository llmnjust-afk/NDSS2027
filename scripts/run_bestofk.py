"""Capability-monotonic best-of-k L2 runner (native AgentDojo, end-to-end).

For each unit (defense, seed, user_task, injection_task) this runner:
  1. builds the candidate set from sok_ipl.native.attack_candidates
     (L1 + matched + generic + length-control + k paraphrases);
  2. runs EACH candidate through AgentDojo's native pipeline, scoring with
     security_from_traces (attack_success) and user_task.utility (utility);
  3. writes one attempt record per candidate (attempt-level JSONL);
  4. after all attempts, monotonic_stats.analyze() computes best-of-k /
     Robust ASR, McNemar, paired + cluster bootstrap.

Because the candidate set always contains the exact L1 payload, the reported
Robust ASR is guaranteed >= ASR_L1 (asserted in analyze()).

The runner also supports a matched-vs-mismatched transfer matrix via
--matrix-source-defenses: a payload rendered for a SOURCE defense is injected
UNCHANGED into every TARGET defense.

Credentials: read from CHATANYWHERE_KEY env var (no hard-coded fallback).

Usage (on Lab):
  CHATANYWHERE_KEY=... PYTHONPATH=/data/lab/NDSS2027 \
    python3 scripts/run_bestofk.py \
      --defenses none spotlighting polymorphic mixture_encodings struq_prompt \
                 task_shield ipiguard fath p1 p2 p3 \
      --suite workspace --n-user-tasks 10 --n-injection-tasks 3 --seeds 0 1 2 \
      --k-paraphrases 3 --model gpt-4o-mini-2024-07-18 \
      --outdir /data/lab/NDSS2027/results_bestofk
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Repo root on sys.path (portable; no hard-coded /data/lab path).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agentdojo.functions_runtime import FunctionCall  # noqa: F401,E402
import agentdojo.benchmark as _bm  # noqa: E402

try:
    _bm.TaskResults.model_rebuild()
except Exception as _e:  # pragma: no cover
    print("TaskResults.model_rebuild note:", _e, file=sys.stderr)

from agentdojo.task_suite import get_suite  # noqa: E402
from agentdojo.benchmark import run_task_with_injection_tasks  # noqa: E402
from agentdojo.attacks.base_attacks import BaseAttack  # noqa: E402
from agentdojo.logging import OutputLogger  # noqa: E402

from sok_ipl.native.pipeline_builder import build_pipeline  # noqa: E402
from sok_ipl.native.attack_candidates import (  # noqa: E402
    AttackContext, build_candidate_set, render_source_payload,
    normalize_defense, EVALUATED_DEFENSES,
)
from sok_ipl.native.monotonic_stats import analyze  # noqa: E402


def _require_key() -> str:
    key = os.environ.get("CHATANYWHERE_KEY")
    if not key:
        print("ERROR: set CHATANYWHERE_KEY environment variable.", file=sys.stderr)
        sys.exit(2)
    return key


class PayloadAttack(BaseAttack):
    """Injects a FIXED, pre-rendered payload string into every injection vector
    of the user task. This lets us evaluate any candidate payload built by the
    candidate layer, decoupled from AgentDojo's own attack templates."""

    name = "payload"

    def __init__(self, task_suite, target_pipeline, payload: str):
        super().__init__(task_suite, target_pipeline)
        self._payload = payload

    def attack(self, user_task, injection_task) -> dict:
        return {inj: self._payload for inj in self.get_injection_candidates(user_task)}


def load_done(checkpoint: Path) -> set[str]:
    done = set()
    if checkpoint.exists():
        with checkpoint.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["attempt_id"])
                except Exception:
                    continue
    return done


def _run_one(suite, pipeline, ut_id, it_id, payload, logdir, bench_ver):
    """Run one candidate payload; return (utility, attack_success)."""
    attack = PayloadAttack(suite, pipeline, payload)
    with OutputLogger(str(logdir)):
        ut, sec = run_task_with_injection_tasks(
            suite, pipeline, suite.get_user_task_by_id(ut_id),
            attack, logdir, True, [it_id], bench_ver,
        )
    key = (ut_id, it_id)
    utility = bool(ut[key]) if key in ut else bool(list(ut.values())[0])
    attack_success = bool(sec[key]) if key in sec else bool(list(sec.values())[0])
    return utility, attack_success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defenses", nargs="+", default=EVALUATED_DEFENSES)
    ap.add_argument("--suite", default="workspace")
    ap.add_argument("--benchmark-version", default="v1.2.2")
    ap.add_argument("--n-user-tasks", type=int, default=10)
    ap.add_argument("--n-injection-tasks", type=int, default=3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--k-paraphrases", type=int, default=3)
    ap.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--p2-mode", default="combined")
    ap.add_argument("--matrix-source-defenses", nargs="*", default=[],
                    help="If set, also run transfer matrix: these source payloads "
                         "injected into every target defense.")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    api_key = _require_key()
    defenses = [normalize_defense(d) for d in args.defenses]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = outdir / "attempts.jsonl"
    done = load_done(checkpoint)

    suite = get_suite(args.benchmark_version, args.suite)
    ut_ids = list(suite.user_tasks.keys())[: args.n_user_tasks]
    it_ids = list(suite.injection_tasks.keys())[: args.n_injection_tasks]

    # Save the manifest so every candidate/analysis uses the same task set.
    (outdir / "manifest.json").write_text(json.dumps({
        "benchmark_version": args.benchmark_version, "suite": args.suite,
        "model": args.model, "user_tasks": ut_ids, "injection_tasks": it_ids,
        "seeds": args.seeds, "k_paraphrases": args.k_paraphrases,
        "defenses": defenses, "p2_mode": args.p2_mode, "threshold": args.threshold,
    }, indent=2))

    ck = checkpoint.open("a")
    n_run = 0
    t0 = time.time()

    def emit(rec):
        nonlocal n_run
        ck.write(json.dumps(rec) + "\n")
        ck.flush()
        n_run += 1
        if n_run % 5 == 0:
            el = time.time() - t0
            print(f"  {n_run} attempts ({el:.0f}s, {el/max(n_run,1):.1f}s/attempt)", flush=True)

    # ---- main best-of-k candidate sets ----
    for defense in defenses:
        for seed in args.seeds:
            pipeline = build_pipeline(defense, args.model, api_key,
                                      threshold=args.threshold, p2_mode=args.p2_mode)
            for ut_id in ut_ids:
                ut_prompt = suite.get_user_task_by_id(ut_id).PROMPT
                for it_id in it_ids:
                    goal = suite.get_injection_task_by_id(it_id).GOAL
                    ctx = AttackContext(
                        target_defense=defense, seed=seed,
                        user_task_id=ut_id, injection_task_id=it_id,
                        user_task_prompt=ut_prompt, goal=goal,
                    )
                    for cand in build_candidate_set(ctx, k_paraphrases=args.k_paraphrases):
                        attempt_id = f"{defense}|s{seed}|{ut_id}|{it_id}|{cand.candidate_id}"
                        if attempt_id in done:
                            continue
                        logdir = outdir / "runs" / defense / f"s{seed}" / cand.candidate_id
                        logdir.mkdir(parents=True, exist_ok=True)
                        rec = {
                            "attempt_id": attempt_id, "target_defense": defense,
                            "seed": seed, "user_task_id": ut_id, "injection_task_id": it_id,
                            "candidate_kind": cand.candidate_kind,
                            "candidate_id": cand.candidate_id,
                            "source_strategy": cand.source_strategy,
                            "suffix_chars": cand.suffix_chars,
                            "paraphrase_index": cand.paraphrase_index,
                            "control_for": cand.control_for,
                            "match_relation": "matched" if cand.candidate_kind != "l1" else "l1",
                        }
                        try:
                            u, a = _run_one(suite, pipeline, ut_id, it_id,
                                            cand.payload, logdir, args.benchmark_version)
                            rec.update(utility=u, attack_success=a,
                                       attempt_status="complete", error=None)
                        except Exception as e:
                            rec.update(utility=None, attack_success=None,
                                       attempt_status="runner_error",
                                       error=f"{type(e).__name__}: {e}"[:300])
                        emit(rec)

    # ---- optional matched-vs-mismatched transfer matrix ----
    for src in [normalize_defense(d) for d in args.matrix_source_defenses]:
        for defense in defenses:
            for seed in args.seeds:
                pipeline = build_pipeline(defense, args.model, api_key,
                                          threshold=args.threshold, p2_mode=args.p2_mode)
                for ut_id in ut_ids:
                    ut_prompt = suite.get_user_task_by_id(ut_id).PROMPT
                    for it_id in it_ids:
                        goal = suite.get_injection_task_by_id(it_id).GOAL
                        ctx = AttackContext(
                            target_defense=defense, seed=seed,
                            user_task_id=ut_id, injection_task_id=it_id,
                            user_task_prompt=ut_prompt, goal=goal,
                        )
                        payload = render_source_payload(src, ctx)
                        attempt_id = f"MATRIX|src={src}|tgt={defense}|s{seed}|{ut_id}|{it_id}"
                        if attempt_id in done:
                            continue
                        logdir = outdir / "matrix" / f"src_{src}" / defense / f"s{seed}"
                        logdir.mkdir(parents=True, exist_ok=True)
                        rec = {
                            "attempt_id": attempt_id, "target_defense": defense,
                            "source_defense": src, "seed": seed,
                            "user_task_id": ut_id, "injection_task_id": it_id,
                            "candidate_kind": "matrix",
                            "candidate_id": f"matrix.src_{src}",
                            "match_relation": "matched" if src == defense else "mismatched",
                        }
                        try:
                            u, a = _run_one(suite, pipeline, ut_id, it_id,
                                            payload, logdir, args.benchmark_version)
                            rec.update(utility=u, attack_success=a,
                                       attempt_status="complete", error=None)
                        except Exception as e:
                            rec.update(utility=None, attack_success=None,
                                       attempt_status="runner_error",
                                       error=f"{type(e).__name__}: {e}"[:300])
                        emit(rec)

    ck.close()
    print(f"Done. Ran {n_run} new attempts.", flush=True)

    # ---- analysis ----
    attempts = [json.loads(l) for l in checkpoint.open()]
    main_attempts = [r for r in attempts if r.get("candidate_kind") != "matrix"]
    res = analyze(main_attempts, n_boot=10000)
    (outdir / "analysis.json").write_text(json.dumps(res, indent=2))
    print("\n=== ROBUST ASR (best-of-k, includes L1) ===", flush=True)
    for d, v in sorted(res["per_defense"].items()):
        db = v["delta_bootstrap"]
        print(f"  {d:16s} ASR_L1={v['ASR_L1']:.2f} matched={v['ASR_matched_only']} "
              f"Robust={v['Robust_ASR_L2set']:.2f} "
              f"Δ={db['delta']:+.2f}[{db['lo']:+.2f},{db['hi']:+.2f}] "
              f"McNemar_p={v['mcnemar_p']}", flush=True)


if __name__ == "__main__":
    main()
