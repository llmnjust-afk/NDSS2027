"""Base class for all attacks.

An attack is a function from a benign Task to a corrupted Task whose
tool_outputs carry an injection. The defense never sees the difference between
a benign and an attacked task at the type level; it only sees ToolOutput.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

from ..llm import LLMBackend
from ..types import AttackLevel, Task, ToolOutput


class BaseAttack(ABC):
    """All attacks share: a level, a name, and an `attack(task)` method.

    Adaptive attacks (L2+) take an optional `defense_hint` so they can tailor the
    injection to a *specific* defense class. This is the key modeling decision
    that makes the framework a faithful adaptive-attack generator: an adaptive
    attack is not "a stronger injection" but "an injection crafted with knowledge
    of the defense's detection signal".
    """

    level: AttackLevel
    name: str

    def __init__(self, llm: Optional[LLMBackend] = None, seed: int = 0, max_iters: int = 3):
        self.llm = llm
        self.seed = seed
        self.max_iters = max_iters
        self._rng = random.Random(seed)

    @abstractmethod
    def attack(self, task: Task, defense_hint: Optional[str] = None) -> Task:
        """Return a copy of `task` with one or more tool outputs injected."""
        raise NotImplementedError

    def _make_malicious(self, task: Task, contents: list[str], payload: str = "") -> Task:
        """Convenience: replace benign tool outputs with malicious ones.

        Preserves task_spec (including injection_goal) so the runner can still
        judge whether the attack achieved its goal.
        """
        outputs = [
            ToolOutput(
                tool_name=task.benign_tool_outputs[i].tool_name if i < len(task.benign_tool_outputs) else f"tool_{i}",
                content=c,
                is_malicious=True,
                injection_payload=payload or c,
            )
            for i, c in enumerate(contents)
        ]
        return Task(
            task_id=task.task_id,
            benchmark=task.benchmark,
            scenario=task.scenario,
            user_task=task.user_task,
            expected_tool=task.expected_tool,
            expected_args=task.expected_args,
            injection_point=task.injection_point,
            benign_tool_outputs=outputs,
            task_spec=task.task_spec,
        )
