"""P2: SecAlign evaluation script.

Loads Mistral-7B-Instruct-v0.1 + SecAlign LoRA adapter and runs the IPI
evaluation against it, comparing SecAlign's robustness to StruQ and no-defense.

This requires the local GPU (A100) and the SecAlign LoRA adapter downloaded
via `python setup.py --instruct` in the SecAlign repo.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sok_ipl.attacks import AdaptiveAttackFramework
from sok_ipl.defenses import NoDefense, StruQ
from sok_ipl.eval import AgentDojoAdapter, Runner
from sok_ipl.types import AttackLevel


SEALIGN_ADAPTER_PATH = "/data/lab/SecAlign/mistralai/Mistral-7B-Instruct-v0.1_dpo_NaiveCompletion_2025-03-12-12-01-27"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"


class SecAlignBackend:
    """LLM backend wrapping Mistral-7B + SecAlign LoRA adapter."""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.name = "sealign-mistral-7b"
        self.usage = type("Usage", (), {"n_calls": 0, "n_tokens": 0, "latency_ms": 0.0})()

        print(f"Loading base model {BASE_MODEL}...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
        )
        print(f"Applying SecAlign LoRA adapter from {SEALIGN_ADAPTER_PATH}...")
        self.model = PeftModel.from_pretrained(self.model, SEALIGN_ADAPTER_PATH)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print("SecAlign model ready.")

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        import torch

        t0 = time.time()
        messages = [{"role": "user", "content": prompt}]
        # Mistral uses [INST] format
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        self.usage.n_calls += 1
        self.usage.n_tokens += inputs["input_ids"].shape[1] + len(text.split())
        self.usage.latency_ms += (time.time() - t0) * 1000
        return text


class SecAlignDefense:
    """SecAlign is a training-based defense: the model itself is trained to
    resist injections. So 'processing' the context is a no-op — the defense
    is baked into the model weights."""

    name = "sealign"
    defense_class = None  # Will be set dynamically

    def __init__(self, llm=None):
        from sok_ipl.types import DefenseClass
        self.defense_class = DefenseClass.TRAINING_BASED
        self.llm = llm

    def process(self, context):
        from sok_ipl.types import DefenseDecision
        return DefenseDecision(rewritten_context=context, blocked=False, flagged=False)

    def inspect_action(self, context, action):
        from sok_ipl.types import DefenseDecision
        return DefenseDecision(rewritten_context=context, blocked=False, flagged=False)

    def class_hint(self):
        return "training_based"


def main():
    print("=== SecAlign Evaluation ===")
    print(f"Base: {BASE_MODEL}")
    print(f"Adapter: {SEALIGN_ADAPTER_PATH}")
    print()

    # Load SecAlign model
    sealign_llm = SecAlignBackend()

    # Also need a vanilla Mistral for comparison (no defense baseline with same backbone)
    # For now, use SecAlign as both the "SecAlign defense" and the backbone
    tasks = AgentDojoAdapter(suite_name="workspace", max_per_suite=15).load_tasks()
    runner = Runner(llm=sealign_llm, seed=0, max_iters=2)
    fw = AdaptiveAttackFramework(llm=sealign_llm, seed=0, max_iters=2)

    defenses = [
        ("no_defense", NoDefense()),
        ("sealign", SecAlignDefense(llm=sealign_llm)),
    ]

    results = {}
    for name, defense in defenses:
        l0_ok = l1_hits = l2_hits = l3_hits = 0
        for t in tasks:
            r0 = runner.run_one(t, defense, AttackLevel.L0_BENIGN)
            l0_ok += r0.utility_preserved

            t1 = fw.generate(t, AttackLevel.L1_STATIC, defense.class_hint())
            r1 = runner.run_one(t1, defense, AttackLevel.L1_STATIC)
            l1_hits += r1.success

            t2 = fw.generate(t, AttackLevel.L2_ADAPTIVE_SINGLE, defense.class_hint())
            r2 = runner.run_one(t2, defense, AttackLevel.L2_ADAPTIVE_SINGLE)
            l2_hits += r2.success

            t3 = fw.generate(t, AttackLevel.L3_ADAPTIVE_MULTI, defense.class_hint())
            r3 = runner.run_one(t3, defense, AttackLevel.L3_ADAPTIVE_MULTI)
            l3_hits += r3.success

        n = len(tasks)
        results[name] = {
            "USR": l0_ok / n,
            "ASR_L1": l1_hits / n,
            "ASR_L2": l2_hits / n,
            "ASR_L3": l3_hits / n,
            "R-ASR": max(l2_hits, l3_hits) / n,
        }
        print(f"{name:15s} USR={l0_ok/n:.2f} L1={l1_hits/n:.2f} L2={l2_hits/n:.2f} L3={l3_hits/n:.2f} R-ASR={max(l2_hits,l3_hits)/n:.2f}")

    # Save results
    Path("results_sealign").mkdir(exist_ok=True)
    with open("results_sealign/summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results_sealign/summary.json")


if __name__ == "__main__":
    main()
