#!/usr/bin/env bash
# Base CoTs for one dataset, sharded across N single-GPU vLLM workers.
#   usage: launch_base_sharded.sh <dataset> <num_examples> <max_tokens> <max_model_len> [num_shards]
# Data-parallel (TP=1 per worker), NOT tensor-parallel: the 7B is 14GB fp16 and
# fits one A6000 with room for KV, so sharding a model that already fits would
# only add an all-reduce per layer.
set -euo pipefail
cd /playpen-ssd/shared/llm-probing-project
source env.sh
export OUTLINES_CACHE_DIR="${REPO_DIR}/.outlines_cache_yuvraj"

DATASET="$1"; NUM_EX="$2"; MAXTOK="$3"; MAXLEN="$4"; NSHARD="${5:-8}"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
LOGDIR="runs/goal1/full/logs"; mkdir -p "$LOGDIR"

for i in $(seq 0 $((NSHARD-1))); do
  CUDA_VISIBLE_DEVICES=$i "$PY" src/generation/run_eval.py \
    --model "$MODEL" --dataset "$DATASET" \
    --output-dir runs/goal1/full \
    --num-examples "$NUM_EX" --samples-per-example 3 --seed 42 \
    --temperature 0.7 --top-p 0.95 \
    --max-tokens "$MAXTOK" --max-model-len "$MAXLEN" \
    --sample-batch-size 256 --max-num-seqs 128 \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
    --num-shards "$NSHARD" --shard-index "$i" --resume \
    > "$LOGDIR/base_${DATASET}_shard${i}.log" 2>&1 &
  echo "  shard $i -> GPU $i (pid $!)"
done
wait
echo "all $NSHARD shards finished"
