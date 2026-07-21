# SoK: Why LLM Agent Defenses Fail Under Adaptive Prompt Injection

Systematization-of-Knowledge framework for evaluating indirect prompt
injection (IPI) defenses for LLM agents under adaptive attacks.

Target venue: **NDSS 2027 Symposium** (Fall Cycle, submission 2026-08-19).

## What this repository is

This is the evaluation framework and artifact for the SoK. It implements:

1. **A unified adaptive attack framework** (`sok_ipl/attacks/`) that
   materializes a matched adversary for every defense class. Five attack
   levels (L0 benign → L4 backdoor-augmented) and six per-defense evasion
   strategies. This is the contribution that scales Zhan et al. 2025
   (3 defenses) to all 15 defenses in the SoK.

2. **15 baseline defenses** (`sok_ipl/defenses/`) wrapped behind a single
   interface, so any defense can be evaluated against any attack without
   touching attack code. Covers Spotlighting, StruQ, Task Shield, Attention
   Tracker, IPIGuard, Polymorphic Prompt, Mixture-of-Encodings, FATH, plus
   variants.

3. **Four design-principle prototypes** (`sok_ipl/principles/`) — minimal
   defenses (~100 lines each) that instantiate the falsifiable design
   principles. If a prototype is broken by its matched adaptive attack, the
   corresponding principle is refuted and the SoK reports that honestly.

4. **Failure-mode clustering + root-cause labeling** (`sok_ipl/analysis/`)
   that aggregates which defenses fail to which root cause (R1 observable
   signal mimicry / R2 boundary blur / R3 semantic paraphrase /
   R4 missing channel isolation).

5. **A formal single-signal impossibility theorem** (`sok_ipl/theorem.py`)
   with an empirical evidence harness: any defense whose decision is a
   function of a single observable signal is broken by the L2 adaptive
   search.

6. **A contributed benchmark, AdaptiveIPI-Bench**
   (`sok_ipl/eval/benchmark.py`): the (task, defense_class, level) triples
   that give a new defense its adaptive-ASR in a single number.

## Quick start

```bash
pip install -r requirements.txt

# Smoke test (stub backend, zero API cost, <5s)
python -m scripts.run_experiment --config configs/smoke.yaml --smoke

# Reproduce Zhan et al. 2025 "defenses break under adaptive attacks"
python -m scripts.reproduce_zhan.py
# with a real backbone:
python -m scripts.reproduce_zhan.py --llm openai:gpt-4o

# Full evaluation (the paper's main table)
python -m scripts.run_experiment --config configs/experiments.yaml

# Generate figures + LaTeX tables from results
python -m scripts.analyze_results.py
```

## Repository layout

```
sok_ipl/
  types.py              core data structures (Task, AgentContext, ...)
  registry.py           name -> class registry for attacks/defenses/principles
  theorem.py            single-signal impossibility theorem + evidence harness
  llm/client.py         unified OpenAI / Anthropic / local / stub backend
  attacks/              L1 static, L2 adaptive, L3 multi-turn, L4 backdoor
  defenses/             15 baseline defenses behind one interface
  principles/           4 design-principle prototypes
  analysis/             metrics, failure clustering, root-cause labeling
  eval/                 benchmark adapters, runner, stats tests
scripts/
  run_experiment.py     main driver (reads YAML, writes results + summary)
  reproduce_zhan.py     W1 D4-D5 validation gate
  analyze_results.py    figures + LaTeX tables
configs/
  experiments.yaml      full matrix (15 defenses x 5 levels x 4 backbones x 3 seeds)
  smoke.yaml            CI config
tests/
  test_framework.py     end-to-end smoke test
```

## The four research questions

| RQ | Question | Section / code |
|----|----------|----------------|
| RQ1 | How do existing IPI defenses perform under static IPI? | L1 column of main table |
| RQ2 | How do adaptive attacks break these defenses? | L2-L4 columns; `reproduce_zhan.py` |
| RQ3 | Can failure modes be clustered into root causes? | `analysis/failure_clustering.py` |
| RQ4 | Which design principles survive adaptive attacks? | `principles/` prototypes |

## Baseline code reused

The framework reuses these open-source baselines (install separately; the
package imports cleanly without them via stub fallbacks):

| Baseline | Source |
|----------|--------|
| AgentDojo (eval env) | `github.com/ethz-spylab/agentdojo` |
| InjecAgent (attack cases) | `github.com/uiuc-kang-lab/InjecAgent` |
| ASB (formalized benchmark) | `github.com/agiresearch/ASB` |
| StruQ | `github.com/struq-security/struq` |
| SecAlign | `github.com/facebookresearch/SecAlign` |
| Adaptive Attacks (Zhan) | `github.com/qiusi-zhan/Adaptive-Attacks` |
| AgentVigil | `github.com/BAI-LSEC/AgentVigil` |

## Compute

- **API path (recommended)**: 0 GPUs. Backbones via OpenAI/Anthropic/DeepSeek.
- **Local path**: 1× A100 80GB (~150 GPU-hours) for Llama-3.1-8B / Qwen2.5-7B
  served by vLLM at `localhost:8000`.
- Full evaluation API cost: ~$3,000-$4,600 depending on backbone mix.

## Reproducibility

- Every result is identified by (defense, attack_level, backbone, seed, task_id).
- Seeds 0/1/2 are reported as mean ± std.
- Paired Wilcoxon signed-rank test against the no-defense baseline, with
  Cohen's d effect size and 95% bootstrap CIs (`sok_ipl/eval/stats.py`).
- Adaptive attacks use `max_iters=3` by default (Zhan et al. use 5; configurable).

## Status

WIP. The framework is functional end-to-end on the stub backend (see
`tests/test_framework.py`). Real-backbone runs require API keys
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`) and/or a local
vLLM server.
