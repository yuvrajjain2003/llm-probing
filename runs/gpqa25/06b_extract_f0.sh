#!/bin/bash
# 0% checkpoint activations: all 28 layers at the LAST PROMPT TOKEN, one vector per
# question (198). Input dir made by src/probing/make_f0_input.py. Writes a NEW dir;
# never touches activations_ckpt/.
#SBATCH --job-name=gpqa25-extract-f0
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
set -euo pipefail
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"
need_py "$PY_ES"; need_weights
export HF_TOKEN="${HF_TOKEN:-dummy}"   # cache/huggingface/token is unreadable for us; offline anyway
IN="$RUNS_ROOT/goal2/$RUN_TAG/_f0_input"
OUT="$RUNS_ROOT/goal2/$RUN_TAG/activations_ckpt_f0/$DATASET/$MODEL_SLUG"
[ -f "$IN/raw/checkpoints.jsonl" ] || "$PY" src/probing/make_f0_input.py --run-dir "$RUN_DIR" --out-dir "$IN"
mkdir -p "$LOG_DIR"; exec >>"$LOG_DIR/extract_f0_job${SLURM_JOB_ID:-local}.log" 2>&1
"$PY_ES" src/probing/extract_checkpoints.py --run-dir "$IN" --out-dir "$OUT" \
  --batch-size 2 --gpu-memory-utilization 0.55 --max-model-len 4096 \
  --num-shards 1 --shard-index 0
