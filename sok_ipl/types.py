"""Core data structures shared across attacks, defenses, and the evaluator.

The design decouples *what* an agent sees (ToolOutput) from *how* a defense
rewrites the agent context (AgentContext). This separation is what lets us drop
any defense in front of any attack without touching attack code, and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class AttackLevel(str, Enum):
    """Five-level attack-strength taxonomy used throughout the evaluation.

    L0 benign establishes the defense's utility ceiling (USR); L1-L4 progressively
    weaken the defense. The jump from L1 to L2 is the single most informative
    measurement in the framework: it is the delta at which most prior defenses
    collapse, and it is the delta that prior SoK-style work under-reported.
    """

    L0_BENIGN = "L0_benign"
    L1_STATIC = "L1_static_ipi"
    L2_ADAPTIVE_SINGLE = "L2_adaptive_single_shot"
    L3_ADAPTIVE_MULTI = "L3_adaptive_multi_turn"
    L4_BACKDOOR_AUGMENTED = "L4_backdoor_augmented"


class DefenseClass(str, Enum):
    """The two-dimensional defense taxonomy (contribution 1).

    axis 1 - layer position: where in the agent pipeline the defense intervenes.
    axis 2 - trust assumption: what the defense assumes it can trust.
    """

    INPUT_ENCODING = "input_encoding"
    STRUCTURED_QUERY = "structured_query"
    RUNTIME_CHECKING = "runtime_checking"
    INTERNAL_PROBING = "internal_probing"
    TRAINING_BASED = "training_based"
    ARCHITECTURE = "architecture"

    TRUST_NONE = "trust_none"
    TRUST_FORMAL_BOUNDARY = "trust_formal_boundary"
    TRUST_TEE = "trust_tee"


@dataclass
class ToolOutput:
    """Untrusted content returned by an external tool.

    `is_malicious` is ground truth known only to the evaluator, never exposed to
    the defense or the agent. This is the single source of truth for ASR.
    """

    tool_name: str
    content: str
    is_malicious: bool = False
    injection_payload: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    """A discrete action the agent proposes to take via a tool call."""

    tool_name: str
    arguments: dict[str, Any]
    raw_text: str = ""

    def matches(self, expected_tool: str, expected_args: dict[str, Any]) -> bool:
        if self.tool_name != expected_tool:
            return False
        for k, v in expected_args.items():
            if self.arguments.get(k) != v:
                return False
        return True


@dataclass
class AgentContext:
    """The full context a defense rewrites before the LLM sees it.

    system_prompt: trusted, set by the developer.
    user_task: trusted, supplied by the end user.
    tool_outputs: untrusted, returned by external tools (attack surface).
    history: prior turns, partially untrusted (carries tool output residues).
    """

    system_prompt: str
    user_task: str
    tool_outputs: list[ToolOutput] = field(default_factory=list)
    history: list[dict[str, str]] = field(default_factory=list)
    task_spec: dict[str, Any] = field(default_factory=dict)

    def untrusted_text(self) -> str:
        return "\n".join(t.content for t in self.tool_outputs)


@dataclass
class DefenseDecision:
    """Output of a defense: the (possibly rewritten) context + verdict.

    A defense may (a) rewrite the context (encoding/structuring), (b) block the
    action (runtime check), or (c) flag the context as suspicious (detector).
    `blocked` and `flagged` are reported separately so the evaluator can measure
    false-positive rate independently of attack-success rate.
    """

    rewritten_context: AgentContext
    blocked: bool = False
    flagged: bool = False
    score: float = 0.0
    reason: str = ""


@dataclass
class AttackResult:
    """Outcome of a single (attack, defense, task) trial."""

    task_id: str
    attack_level: AttackLevel
    defense_name: str
    backbone: str
    seed: int
    success: bool
    blocked_by_defense: bool
    flagged_by_defense: bool
    utility_preserved: bool
    n_llm_calls: int = 0
    n_tokens: int = 0
    latency_ms: float = 0.0
    failure_root_cause: Optional[str] = None
    attack_strategy: str = ""
    notes: str = ""


@dataclass
class Task:
    """A benchmark task with a ground-truth expected action."""

    task_id: str
    benchmark: str
    scenario: str
    user_task: str
    expected_tool: str
    expected_args: dict[str, Any]
    injection_point: str = "tool_output"
    benign_tool_outputs: list[ToolOutput] = field(default_factory=list)
    task_spec: dict[str, Any] = field(default_factory=dict)
