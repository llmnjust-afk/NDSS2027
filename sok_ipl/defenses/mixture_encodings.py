"""Mixture-of-Encodings (Zhang et al. 2025) - defense via multiple encodings.

Run the same prompt through N different encodings (quoting, base64, rot13) and
aggregate the LLM's responses: only an action that survives all encodings is
executed. The intuition: a real instruction survives re-encoding, an injection
that exploits a specific encoding's delimiter does not.

We implement the aggregation as majority vote over decoded actions.
"""

from __future__ import annotations

import base64
import codecs

from ..registry import register
from ..types import AgentContext, DefenseClass, DefenseDecision
from .base import BaseDefense
from .spotlighting import Spotlighting


@register("defense", "mixture_encodings")
class MixtureOfEncodings(BaseDefense):
    name = "mixture_encodings"
    defense_class = DefenseClass.INPUT_ENCODING

    def __init__(self):
        super().__init__()
        self._variants = [
            Spotlighting(mode="quoting"),
            Spotlighting(mode="datamarking", delimiter="[DATA]"),
            Spotlighting(mode="base64"),
        ]

    def process(self, context: AgentContext) -> DefenseDecision:
        # Apply all three encodings and let the runner query the LLM once per
        # variant; the aggregation happens at action time. We stash the variant
        # contexts in task_spec so the runner can fan out.
        variants = [v.process(context).rewritten_context for v in self._variants]
        new_ctx = AgentContext(
            system_prompt=context.system_prompt,
            user_task=context.user_task,
            tool_outputs=context.tool_outputs,
            history=context.history,
            task_spec={**context.task_spec, "_moe_variants": variants},
        )
        return DefenseDecision(rewritten_context=new_ctx, blocked=False, flagged=False)
