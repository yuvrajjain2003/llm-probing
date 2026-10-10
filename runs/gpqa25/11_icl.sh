#!/bin/bash
# R1: in-context-learning baseline on vLLM (current vLLM, env envs/icl). One model per job; the engine is
# loaded once and all k values run in it (resumable per (k, fold): finished files are skipped).
#   source env.sh
#   ICL_MODEL=r1qwen7b   ICL_KS="0 10 30 40" sbatch runs/gpqa25/11_icl.sh
#   ICL_MODEL=qwen35_9b  ICL_KS="0 10 30 92" sbatch runs/gpqa25/11_icl.sh                  # add ICL_EXTRA=--yarn for 1M ctx
#   ICL_MODEL=gemma4_31b ICL_KS="0 10 30 93" sbatch --gres=gpu:h100:2 --mem=192G runs/gpqa25/11_icl.sh   # TP=2
# "max" k per model comes from `icl_predict.py plan` (runs/goal2/gpqa25/icl/plan_<model>.csv).
# Smoke test first: ICL_EXTRA="--max-rows 20" ICL_KS="10" (files get a _smoke suffix).
# ICL_EXTRA also takes --subsample-cots 0 (all 25 CoTs) and --thinking on.
#SBATCH --job-name=gpqa25-icl
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -euo pipefail
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"
export HF_TOKEN="${HF_TOKEN:-dummy}" HF_HUB_OFFLINE=1
export CUDA_HOME=/nas/sycamore/apps/cuda/13.2 PATH=/nas/sycamore/apps/cuda/13.2/bin:$PATH   # flashinfer JIT needs nvcc (module cuda/13.2 = driver CUDA)
PY_ICL="${PY_ICL:-$ENVS_DIR/icl/bin/python}"; need_py "$PY_ICL"
mkdir -p "$LOG_DIR/../round2"; exec >>"$LOG_DIR/../round2/icl_${ICL_MODEL:?}_job${SLURM_JOB_ID:-local}.log" 2>&1
"$PY_ICL" src/baselines/icl_predict.py run --model "$ICL_MODEL" --k ${ICL_KS:?} ${ICL_EXTRA:-}
