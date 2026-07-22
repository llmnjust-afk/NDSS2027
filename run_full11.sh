#!/bin/bash
cd /data/lab/NDSS2027
PYTHONPATH=/data/lab/NDSS2027 python3 scripts/run_native_benchmark.py \
  --defenses none spotlighting struq_prompt polymorphic mixture_encodings fath ipiguard p1 p2 p3 task_shield \
  --attacks important_instructions \
  --suite workspace --n-user-tasks 10 --n-injection-tasks 3 --seeds 0 1 2 \
  --model gpt-4o-mini-2024-07-18 \
  --outdir /data/lab/NDSS2027/results_native_full11
