#!/usr/bin/env bash
# Re-run phase C shards that died on vLLM's scheduler assertion
# (assert len(seqs)==1 -- fires when an n>1 request group is preempted).
# Mitigation: smaller request groups + lower concurrency => less preemption.
# Resumable, so this picks up exactly where each shard stopped.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source env.sh
RUN=runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B
for i in 0 3; do
  CUDA_VISIBLE_DEVICES=$i "$PY" src/generation/continuations.py \
    --base-dir "$RUN" --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset gpqa_diamond --continuations-per-prefix 25 \
    --fractions 0.1 0.3 0.5 0.7 0.9 --prefix-batch-size 32 \
    --temperature 0.7 --top-p 0.95 \
    --max-tokens 16384 --max-model-len 24576 \
    --max-n-per-request 4 --max-num-seqs 64 \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --swap-space 32 \
    --subset-mod 5 --subset-keep 1,2,3,4 \
    --num-shards 8 --shard-index $i --resume \
    > "runs/goal1/full/logs/cont_gpqa_diamond_pCfix_shard${i}.log" 2>&1 &
  echo "  repair shard $i -> GPU $i (pid $!)"
done
wait
echo "repair shards finished"
