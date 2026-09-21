#!/usr/bin/env bash
# Prefix-conditioned continuations for one dataset, sharded across N single-GPU workers.
#   usage: launch_cont_sharded.sh <dataset> <K> <max_tokens> <max_model_len> [num_shards]
#
# Settings established empirically on this box (8x RTX A6000, 48GB):
#   --max-num-seqs 128        1.93x faster than the pilot's 48 (139s -> 72s per chunk)
#   --max-n-per-request 20    K=100 as one n=100 group needs 2.3x the 420k-token KV
#                             cache -> vLLM swaps and the GPU sits at 0%. Splitting
#                             keeps every group schedulable; prefix caching keeps
#                             the shared prefix hot across the sub-requests.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source env.sh

DATASET="$1"; K="$2"; MAXTOK="$3"; MAXLEN="$4"; NSHARD="${5:-8}"; GPU0="${6:-0}"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
BASE="runs/goal1/full/${DATASET}/DeepSeek-R1-Distill-Qwen-7B"
LOGDIR="runs/goal1/full/logs"; mkdir -p "$LOGDIR"

for i in $(seq 0 $((NSHARD-1))); do
  CUDA_VISIBLE_DEVICES=$((GPU0+i)) "$PY" src/generation/continuations.py \
    --base-dir "$BASE" --model "$MODEL" --dataset "$DATASET" \
    --continuations-per-prefix "$K" \
    --fractions 0.1 0.3 0.5 0.7 0.9 \
    --prefix-batch-size 32 \
    --temperature 0.7 --top-p 0.95 \
    --max-tokens "$MAXTOK" --max-model-len "$MAXLEN" \
    --max-n-per-request ${MAXN:-20} --max-num-seqs 128 \
    --subset-mod ${SUBSET_MOD:-1} --subset-keep ${SUBSET_KEEP:-0} \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --swap-space 32 \
    --num-shards "$NSHARD" --shard-index "$i" --resume \
    > "$LOGDIR/cont_${DATASET}${LOGTAG:-}_shard${i}.log" 2>&1 &
  echo "  shard $i -> GPU $((GPU0+i)) (pid $!)"
done
wait
echo "all $NSHARD continuation shards finished for $DATASET"
