#!/bin/bash
# Baseline 1: sentence-embed every checkpoint prefix (question + choices + prefix) on 1 GPU.
#   source env.sh && EMB_MODEL=Qwen/Qwen3-Embedding-8B EMB_NAME=qwen3_emb_8b sbatch runs/gpqa25/10_embed.sh
# Resumable: rerun the same command after a timeout; finished chunks are skipped.
#SBATCH --job-name=gpqa25-embed
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
set -euo pipefail
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"
export HF_TOKEN="${HF_TOKEN:-dummy}"   # cache/huggingface/token is unreadable for us; offline anyway
mkdir -p "$LOG_DIR"; exec >>"$LOG_DIR/embed_${EMB_NAME:?}_job${SLURM_JOB_ID:-local}.log" 2>&1
"$PY" src/probing/embed_prefixes.py --model "${EMB_MODEL:?}" --name "$EMB_NAME" ${EMB_EXTRA:-}
