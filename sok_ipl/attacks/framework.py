"""The unified adaptive attack generator (contribution 2).

This is the orchestrator that, given a defense and a task, picks the matching
attack level and strategy and returns the corrupted task. It is the single
entry point the evaluator calls, so that adding a new attack never requires
touching the eval harness.

The generator is *composable*: an attack is a (level, strategy) pair, and the
generator materializes the right instance from the registry. This is what lets
us emit a per-defense adaptive adversary -- the same machinery Zhan et al.
applied to 3 defenses, scaled to all 15 in the SoK.
"""

from __future__ import annotations

from typing import Optional

from ..llm import LLMBackend
from ..registry import get as get_attack
from ..types import AttackLevel, Task
from .base import BaseAttack


class AdaptiveAttackFramework:
    """Pick + run the right attack for (defense_class, level)."""

    LEVEL_TO_NAME = {
        AttackLevel.L0_BENIGN: None,  # no attack
        AttackLevel.L1_STATIC: "static_injection",
        AttackLevel.L2_ADAPTIVE_SINGLE: "adaptive_single",
        AttackLevel.L3_ADAPTIVE_MULTI: "adaptive_multi",
        AttackLevel.L4_BACKDOOR_AUGMENTED: "backdoor_augmented",
    }

    def __init__(self, llm: Optional[LLMBackend] = None, seed: int = 0, max_iters: int = 3):
        self.llm = llm
        self.seed = seed
        self.max_iters = max_iters
        self._cache: dict[str, BaseAttack] = {}

    def _instance(self, name: str) -> BaseAttack:
        if name not in self._cache:
            cls = get_attack("attack", name)
            self._cache[name] = cls(llm=self.llm, seed=self.seed, max_iters=self.max_iters)
        return self._cache[name]

    def generate(
        self,
        task: Task,
        level: AttackLevel,
        defense_class: Optional[str] = None,
    ) -> Task:
        """Return a corrupted task for the given level. L0 returns the task as-is."""
        if level is AttackLevel.L0_BENIGN:
            return task
        name = self.LEVEL_TO_NAME[level]
        attack = self._instance(name)
        return attack.attack(task, defense_hint=defense_class)

    def all_levels(self) -> list[AttackLevel]:
        return [
            AttackLevel.L0_BENIGN,
            AttackLevel.L1_STATIC,
            AttackLevel.L2_ADAPTIVE_SINGLE,
            AttackLevel.L3_ADAPTIVE_MULTI,
            AttackLevel.L4_BACKDOOR_AUGMENTED,
        ]
