"""No-op defense for the L0 baseline (no defense applied)."""

from __future__ import annotations

from ..registry import register
from ..types import AgentContext, DefenseDecision
from .base import BaseDefense


@register("defense", "none")
class NoDefense(BaseDefense):
    name = "none"

    def process(self, context: AgentContext) -> DefenseDecision:
        return DefenseDecision(rewritten_context=context, blocked=False, flagged=False)
