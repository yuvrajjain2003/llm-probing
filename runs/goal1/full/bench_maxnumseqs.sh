#!/usr/bin/env bash
# max_num_seqs sweep on one GPU. Same slice, same K, fixed wall time each; compare
# vLLM's own output tok/s. 48 was tuned when K=10; with n=100 per request vLLM
# shares the prefix KV across all 100 continuations, so the old bound may be low.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source env.sh
BASE=runs/goal1/full/math500/DeepSeek-R1-Distill-Qwen-7B
for MNS in 48 128; do
  echo "=== max_num_seqs=$MNS ==="
  CUDA_VISIBLE_DEVICES=0 timeout 420 "$PY" src/generation/continuations.py \
    --base-dir "$BASE" \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset math500 \
    --output-dir "runs/goal1/bench/mns${MNS}" \
    --continuations-per-prefix 100 \
    --prefix-batch-size 8 \
    --max-tokens 8192 --max-model-len 24576 \
    --max-num-seqs "$MNS" --gpu-memory-utilization 0.85 \
    --max-n-per-request 20 --num-shards 64 --shard-index 0 \
    > "runs/goal1/bench/mns${MNS}.log" 2>&1
  echo "  exit=$? (124 = hit the time limit, expected)"
done
