#!/usr/bin/env bash
# AIME-2025 base CoTs at the raised 16,384-token cap (all 30 questions x 3 CoTs).
# Purpose: measure the real </think> terminate rate at 16k, which the pilot could
# not observe (41/90 = 46% at the old 8,192 cap).
# Prompt/sampling are UNCHANGED from the pilot; only max_tokens/max_model_len move.
set -euo pipefail
cd /playpen-ssd/shared/llm-probing-project
source env.sh
# env.sh's .outlines_cache/cache.db is owned by rishp (mode 644) -- unwritable by us.
export OUTLINES_CACHE_DIR="${REPO_DIR}/.outlines_cache_yuvraj"

export CUDA_VISIBLE_DEVICES=0

"$PY" src/generation/run_eval.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset aime2025 \
  --output-dir runs/goal1/full \
  --num-examples 30 \
  --samples-per-example 3 \
  --seed 42 \
  --temperature 0.7 \
  --top-p 0.95 \
  --max-tokens 16384 \
  --max-model-len 20480 \
  --sample-batch-size 256 \
  --max-num-seqs 128 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --resume
