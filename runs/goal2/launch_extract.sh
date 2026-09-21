#!/usr/bin/env bash
# SUPERSEDED. This drives extract_activations.py (stride-4, layers 14/21/28 only).
# The current extractor is extract_checkpoints.py (all 28 layers, exact checkpoint
# token); on Sycamore run it through runs/gpqa25/05_extract.sh. Kept because it
# reproduces the old stride-4 store that probe_linear.py was run on.
#
# One worker per GPU on a single box you own outright -- NOT for SLURM.
# Shard by example_id so all CoTs of a question stay together.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env.sh
export HF_HUB_OFFLINE=1
GPUS="${GPUS:-3,4,5,6,7}"
IFS=',' read -ra G <<< "$GPUS"
N=${#G[@]}
LOG=runs/goal2/logs; mkdir -p "$LOG"
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES="${G[$i]}" "$PY_ES" src/probing/extract_activations.py \
    --num-shards "$N" --shard-index "$i" --stride 4 --batch-size 2 \
    > "$LOG/extract_shard${i}.log" 2>&1 &
  echo "  shard $i -> GPU ${G[$i]} (pid $!)"
done
wait
echo "ALL SHARDS DONE"
