"""sok_ipl: Systematization of Knowledge on Indirect Prompt Injection Defenses.

A unified framework for evaluating and analyzing the failure modes of LLM agent
defenses against indirect prompt injection (IPI) under adaptive attacks.

Four research questions:
  RQ1: How do existing IPI defenses perform under static IPI?
  RQ2: How do adaptive attacks break these defenses?
  RQ3: Can failure modes be clustered into root causes?
  RQ4: Which design principles survive adaptive attacks?

The package is organized around three orthogonal axes:
  - attacks/:  a composable adaptive attack generator (contribution 2)
  - defenses/: 15 baseline defenses wrapped behind a unified interface
  - principles/: minimal-surface defense prototypes that instantiate the
    formalized design principles (contribution 4)
  - analysis/: failure-mode clustering and root-cause formalization
  - eval/: unified runner over AgentDojo / InjecAgent / ASB / AdaptiveIPI-Bench
"""

__version__ = "0.1.0"
