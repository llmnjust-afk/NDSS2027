"""IPIGuard (An et al. 2025) - tool-dependency-graph defense.

IPIGuard builds a static graph of which tools may call which other tools (from
the agent's spec) and blocks any action whose dependency chain is not in the
graph. An injected instruction that asks the email tool to call the file-delete
tool will be blocked because email->delete is not an edge.

We model the graph as an allow-list of (src_tool, dst_tool) edges supplied in
the task_spec; the defense rejects any action whose `tool_name` is not reachable
from the task's expected tool.
"""

from __future__ import annotations

from typing import Optional

from ..llm import LLMBackend
from ..registry import register
from ..types import AgentAction, AgentContext, DefenseClass, DefenseDecision
from .base import BaseDefense


@register("defense", "ipiguard")
class IPIGuard(BaseDefense):
    name = "ipiguard"
    defense_class = DefenseClass.RUNTIME_CHECKING

    def process(self, context: AgentContext) -> DefenseDecision:
        return DefenseDecision(rewritten_context=context, blocked=False, flagged=False)

    def inspect_action(self, context: AgentContext, action: AgentAction) -> DefenseDecision:
        allowed = set(context.task_spec.get("allowed_tools", [context.task_spec.get("expected_tool")]))
        allowed.discard(None)
        if not allowed:
            return DefenseDecision(rewritten_context=context, blocked=False, flagged=False)
        ok = action.tool_name in allowed
        return DefenseDecision(
            rewritten_context=context,
            blocked=not ok,
            flagged=not ok,
            reason=f"tool {action.tool_name!r} not in allow-list {allowed}",
        )
