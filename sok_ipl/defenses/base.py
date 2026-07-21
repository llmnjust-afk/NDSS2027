"""Base class for all defenses.

A defense sits between the tool outputs and the LLM. It sees an AgentContext
and either (a) rewrites it, (b) blocks the resulting action, or (c) flags it.
Every defense declares its class so the adaptive attack framework can pick the
matching adversary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..llm import LLMBackend
from ..types import AgentAction, AgentContext, DefenseClass, DefenseDecision


class BaseDefense(ABC):
    """A defense has: a name, a class, and a `process(context)` method.

    `inspect_action` is optional and runs after the LLM proposes an action; it
    lets runtime-checking defenses (Task Shield, IPIGuard) block a suspicious
    action even if the context rewrite did not catch it.
    """

    name: str
    defense_class: DefenseClass = DefenseClass.INPUT_ENCODING
    trust_assumption: DefenseClass = DefenseClass.TRUST_NONE

    def __init__(self, llm: Optional[LLMBackend] = None):
        self.llm = llm

    @abstractmethod
    def process(self, context: AgentContext) -> DefenseDecision:
        raise NotImplementedError

    def inspect_action(self, context: AgentContext, action: AgentAction) -> DefenseDecision:
        """Default: no action-level check. Override for runtime-checking defenses."""
        return DefenseDecision(rewritten_context=context, blocked=False, flagged=False)

    def class_hint(self) -> str:
        """String used by the adaptive attack framework to pick a strategy."""
        return self.defense_class.value
