#!/usr/bin/env bash
# Spread the leftover phase-C work (old shards 0 and 3) across every free GPU.
# num_shards=32: indices {0,8,16,24} == old shard 0 (id%8==0); {3,11,19,27} == old
# shard 3. Resume scans all shard files, so finished chunks are never redone.
set -uo pipefail
cd /playpen-ssd/shared/llm-probing-project
source env.sh
export OUTLINES_CACHE_DIR="${REPO_DIR}/.outlines_cache_yuvraj"
"$PY" src/generation/gpu_pool.py \
  --run-dir runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B \
  --dataset gpqa_diamond --k 25 \
  --num-shards 32 --indices 0,3,8,11,16,19,24,27 \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-tokens 16384 --max-model-len 24576 \
  --max-n 4 --max-num-seqs 64 \
  --subset-mod 5 --subset-keep 1,2,3,4
