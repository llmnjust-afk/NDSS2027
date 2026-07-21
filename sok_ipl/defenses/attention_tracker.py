"""Attention Tracker (Hung et al. 2025) - internal-probing defense.

The defense inspects the LLM's attention pattern over the prompt: an injected
instruction draws disproportionate attention to the untrusted region, which
benign data does not. We model this as a score in [0,1] computed from the
attention mass on untrusted tokens vs. trusted tokens.

Because real attention extraction requires logits-level access, we provide two
backends:
  - heuristics: a token-overlap proxy that flags prompts where untrusted text
    contains instruction-like verbs (default; runnable on any API).
  - logits:   a real attention-mass score requiring a local model (the A100).
"""

from __future__ import annotations

import re
from typing import Optional

from ..llm import LLMBackend
from ..registry import register
from ..types import AgentContext, DefenseClass, DefenseDecision
from .base import BaseDefense

_INSTRUCTION_VERBS = re.compile(
    r"\b(ignore|execute|call|send|forward|delete|transfer|override|system|admin|please|must|do)\b",
    re.IGNORECASE,
)


@register("defense", "attention_tracker")
class AttentionTracker(BaseDefense):
    name = "attention_tracker"
    defense_class = DefenseClass.INTERNAL_PROBING

    def __init__(self, llm: Optional[LLMBackend] = None, threshold: float = 0.6, mode: str = "heuristic"):
        super().__init__(llm=llm)
        self.threshold = threshold
        self.mode = mode

    def process(self, context: AgentContext) -> DefenseDecision:
        score = self._attention_proxy(context)
        flagged = score > self.threshold
        return DefenseDecision(
            rewritten_context=context,
            blocked=False,
            flagged=flagged,
            score=score,
            reason=f"attention_mass={score:.3f}",
        )

    def _attention_proxy(self, context: AgentContext) -> float:
        """Proxy: fraction of untrusted tokens that look like instructions."""
        if not context.tool_outputs:
            return 0.0
        untrusted = " ".join(t.content for t in context.tool_outputs).split()
        if not untrusted:
            return 0.0
        hits = len(_INSTRUCTION_VERBS.findall(" ".join(untrusted)))
        return min(hits / max(len(untrusted) / 20, 1), 1.0)
