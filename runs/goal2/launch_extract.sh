#!/usr/bin/env bash
# GPQA activation extraction, one worker per free GPU.
# Shard by example_id so all CoTs of a question stay together.
set -uo pipefail
cd /playpen-ssd/shared/llm-probing-project
export HF_HUB_CACHE=/playpen-ssd/yuvraj/.cache/huggingface/hub
export HF_HOME="$PWD/.hf_home"
export HF_HUB_OFFLINE=1
GPUS="${GPUS:-3,4,5,6,7}"
IFS=',' read -ra G <<< "$GPUS"
N=${#G[@]}
LOG=runs/goal2/logs; mkdir -p "$LOG"
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES="${G[$i]}" .venv-easysteer/bin/python src/probing/extract_activations.py \
    --num-shards "$N" --shard-index "$i" --stride 4 --batch-size 2 \
    > "$LOG/extract_shard${i}.log" 2>&1 &
  echo "  shard $i -> GPU ${G[$i]} (pid $!)"
done
wait
echo "ALL SHARDS DONE"
