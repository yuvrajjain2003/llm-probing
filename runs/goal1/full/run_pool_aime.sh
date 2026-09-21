#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source env.sh
"$PY" src/generation/gpu_pool.py \
  --run-dir runs/goal1/full/aime2025/DeepSeek-R1-Distill-Qwen-7B \
  --dataset aime2025 --k 100 \
  --num-shards 16 --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-tokens 16384 --max-model-len 24576 \
  --max-n 4 --max-num-seqs 64
