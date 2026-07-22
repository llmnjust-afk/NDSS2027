"""§3: Unknown-defense black-box attack (Setting A) + no-feedback single-shot (Setting B).

Setting A: Attacker does NOT know the defense class. Uses a universal attack
  payload (the <INFORMATION> template without defense-specific evasion).
  This is exactly our L1 static attack — we compare L1 ASR (unknown defense)
  vs L2 ASR (known defense) to measure the value of defense-class knowledge.

Setting B: No-feedback single-shot. The attacker places one injection without
  any iterative optimization. This is our L2 without the LLM-in-the-loop
  optimizer (max_iters=0). We compare L2(max_iters=2) vs L2(max_iters=0)
  to measure the value of feedback.

We also add Setting C: Cross-defense transfer. Payloads generated against
  defense A are tested against defense B, to distinguish matched adaptive
  attacks from universal stronger prompts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sok_ipl.attacks import AdaptiveAttackFramework
from sok_ipl.defenses import (
    NoDefense, Spotlighting, StruQ, TaskShield,
    AttentionTracker, IPIGuard, PolymorphicPrompt, MixtureOfEncodings, FATH,
)
from sok_ipl.eval import AgentDojoAdapter, Runner
from sok_ipl.llm import build_backbone
from sok_ipl.types import AttackLevel, Task


DEFENSES = {
    "none": NoDefense,
    "spotlighting": Spotlighting,
    "struq": StruQ,
    "task_shield": TaskShield,
    "attention_tracker": AttentionTracker,
    "polymorphic": PolymorphicPrompt,
    "mixture_encodings": MixtureOfEncodings,
}


def main():
    llm = build_backbone("openai:gpt-4o-mini")
    tasks = AgentDojoAdapter(suite_name="workspace", max_per_suite=15).load_tasks()
    runner = Runner(llm=llm, seed=0, max_iters=0)  # max_iters=0 for no-feedback
    fw_known = AdaptiveAttackFramework(llm=llm, seed=0, max_iters=2)
    fw_nofb = AdaptiveAttackFramework(llm=llm, seed=0, max_iters=0)

    print(f"=== §3 Supplemental Experiments ({len(tasks)} tasks, GPT-4o-mini) ===\n")

    results = {"setting_A": {}, "setting_B": {}, "setting_C": {}}

    # === Setting A: Unknown-defense (L1 static, no defense knowledge) ===
    print("--- Setting A: Unknown-defense (L1 static) vs Known-defense (L2 adaptive) ---")
    for name, DefCls in DEFENSES.items():
        try:
            defense = DefCls(llm=llm)
        except TypeError:
            defense = DefCls()

        l1_hits = l2_hits = 0
        for t in tasks:
            # L1: universal payload, no defense knowledge
            r1 = runner.run_one(fw_known.generate(t, AttackLevel.L1_STATIC, None), defense, AttackLevel.L1_STATIC)
            l1_hits += r1.success
            # L2: matched payload with defense knowledge
            r2 = runner.run_one(fw_known.generate(t, AttackLevel.L2_ADAPTIVE_SINGLE, defense.class_hint()), defense, AttackLevel.L2_ADAPTIVE_SINGLE)
            l2_hits += r2.success

        n = len(tasks)
        results["setting_A"][name] = {"ASR_unknown": l1_hits/n, "ASR_known": l2_hits/n}
        print(f"  {name:20s} unknown(L1)={l1_hits/n:.2f}  known(L2)={l2_hits/n:.2f}  delta={ (l2_hits-l1_hits)/n:+.2f}")

    # === Setting B: No-feedback (max_iters=0) vs feedback (max_iters=2) ===
    print("\n--- Setting B: No-feedback (iters=0) vs Feedback (iters=2) ---")
    for name, DefCls in DEFENSES.items():
        try:
            defense = DefCls(llm=llm)
        except TypeError:
            defense = DefCls()

        nofb_hits = fb_hits = 0
        for t in tasks:
            r_nofb = runner.run_one(fw_nofb.generate(t, AttackLevel.L2_ADAPTIVE_SINGLE, defense.class_hint()), defense, AttackLevel.L2_ADAPTIVE_SINGLE)
            nofb_hits += r_nofb.success
            r_fb = runner.run_one(fw_known.generate(t, AttackLevel.L2_ADAPTIVE_SINGLE, defense.class_hint()), defense, AttackLevel.L2_ADAPTIVE_SINGLE)
            fb_hits += r_fb.success

        n = len(tasks)
        results["setting_B"][name] = {"ASR_no_feedback": nofb_hits/n, "ASR_feedback": fb_hits/n}
        print(f"  {name:20s} no_feedback={nofb_hits/n:.2f}  feedback={fb_hits/n:.2f}  delta={ (fb_hits-nofb_hits)/n:+.2f}")

    # === Setting C: Cross-defense transfer ===
    print("\n--- Setting C: Cross-defense transfer (payload from A tested on B) ---")
    # Generate payloads against spotlighting, test on struq, task_shield, attention_tracker
    source_def = Spotlighting()
    target_defs = {"struq": StruQ(), "task_shield": TaskShield(llm=llm), "attention_tracker": AttentionTracker()}

    for tname, tdef in target_defs.items():
        transfer_hits = matched_hits = 0
        for t in tasks:
            # Generate payload for spotlighting (source)
            corrupted = fw_known.generate(t, AttackLevel.L2_ADAPTIVE_SINGLE, source_def.class_hint())
            # Test on target defense
            r_transfer = runner.run_one(corrupted, tdef, AttackLevel.L2_ADAPTIVE_SINGLE)
            transfer_hits += r_transfer.success
            # Matched attack on target for comparison
            r_matched = runner.run_one(fw_known.generate(t, AttackLevel.L2_ADAPTIVE_SINGLE, tdef.class_hint()), tdef, AttackLevel.L2_ADAPTIVE_SINGLE)
            matched_hits += r_matched.success

        n = len(tasks)
        results["setting_C"][tname] = {"ASR_transfer": transfer_hits/n, "ASR_matched": matched_hits/n}
        print(f"  spotlighting->{tname:20s} transfer={transfer_hits/n:.2f}  matched={matched_hits/n:.2f}  transfer_ratio={transfer_hits/max(matched_hits,1):.2f}")

    # Save
    Path("results_supplemental").mkdir(exist_ok=True)
    with open("results_supplemental/settings_ABC.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results_supplemental/settings_ABC.json")


if __name__ == "__main__":
    main()
