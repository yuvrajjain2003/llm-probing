#!/usr/bin/env bash
# Cover the gap left by old shard 4 (example_id%8==4), split 8 ways.
# num_shards=64 indices {4,12,20,28,36,44,52,60} == exactly id%8==4.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source env.sh
"$PY" src/generation/gpu_pool.py \
  --run-dir runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B \
  --dataset gpqa_diamond --k 25 \
  --num-shards 64 --indices 4,12,20,28,36,44,52,60 \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-tokens 16384 --max-model-len 24576 \
  --max-n 4 --max-num-seqs 64 \
  --subset-mod 5 --subset-keep 1,2,3,4
