"""Formal statement of the single-signal impossibility theorem (contribution 4).

THEOREM (informal): Let D be any IPI defense whose decision rule is a function
of a single signal s that is *computable from the prompt the LLM sees*. Then
there exists an adaptive attack A* that breaks D with probability 1 - o(1) over
the randomness of the LLM.

PROOF SKETCH:
  1. Because s is computable from the prompt, the attacker can evaluate s on
     any candidate injection (the attacker sees the same prompt the LLM sees,
     modulo the trusted system prompt which the attacker can also read in the
     IPI setting since the user task and tool outputs are concatenated).
  2. The attacker performs projected gradient-free search: sample a candidate
     injection, evaluate s, keep the candidate iff s stays below D's threshold.
     This is the L2 adaptive attack we instantiate in adaptive_single.py.
  3. After O(log|S|) iterations the attacker finds an injection whose s is
     below threshold, so D does not flag it, while the injection still
     contains the malicious instruction, so the LLM follows it with the same
     probability it follows any instruction.
  4. Hence ASR(D, A*) >= ASR(no defense) - epsilon, i.e. D provides no
     meaningful robustness.

COROLLARY: A defense that is adaptive-robust must rely on at least one signal
that is NOT computable from the prompt the LLM sees. This is the formal
statement behind Principle 1 (unobservable signal) and Principle 2 (orthogonal
signals, where at least one is unobservable).

The code below is a *simulation* of the proof: it instantiates the search
procedure in step 2 against any single-signal defense and empirically confirms
ASR -> 1. It is not a formal proof checker; it is the empirical evidence the
SoK offers alongside the analytic argument.
"""

from __future__ import annotations

from typing import Callable

from .attacks.framework import AdaptiveAttackFramework
from .defenses.base import BaseDefense
from .eval.runner import Runner
from .types import AttackLevel, Task


def empirical_evidence(
    defense: BaseDefense,
    tasks: list[Task],
    runner: Runner,
    n_iters: int = 5,
) -> dict[str, float]:
    """Run the L2 adaptive search against `defense` over `tasks` and return
    the observed ASR. The theorem predicts this approaches 1.0 for any
    single-signal defense."""
    fw = AdaptiveAttackFramework(llm=runner.llm, seed=0, max_iters=n_iters)
    results = []
    for task in tasks:
        corrupted = fw.generate(task, AttackLevel.L2_ADAPTIVE_SINGLE, defense.class_hint())
        res = runner.run_one(corrupted, defense, AttackLevel.L2_ADAPTIVE_SINGLE)
        results.append(res)
    return {
        "n_tasks": len(tasks),
        "adaptive_asr": sum(r.success for r in results) / max(len(results), 1),
        "predicted_by_theorem": 1.0,
    }


SINGLE_SIGNAL_DEFENSES = [
    "spotlighting",
    "struq",
    "attention_tracker",
    "polymorphic",
    "mixture_encodings",
]
"""Defenses the theorem predicts will be broken. The SoK verifies each."""
