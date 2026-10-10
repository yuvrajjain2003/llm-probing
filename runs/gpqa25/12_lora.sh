#!/bin/bash
# R8: LoRA + MLP-head probe on our model, ONE model per (checkpoint fraction, fold) (src/probing/lora_probe.py), 1 GPU.
#   source env.sh
#   MODE=check sbatch -t 00:40:00 runs/gpqa25/12_lora.sh   # layer-21 state vs stored activations 
#   MODE=pilot sbatch -t 04:00:00 runs/gpqa25/12_lora.sh   # fraction 0.5, fold 0, 1 epoch  -> runs/goal2/gpqa25/lora/pilot
#   sbatch runs/gpqa25/12_lora.sh              # all fractions x 5 folds -> runs/goal2/gpqa25/lora/main
# Subset / split across jobs: FRACTIONS="0.1 0.3" sbatch runs/gpqa25/12_lora.sh (outputs share lora/main).
# Resumable per (fraction, fold): rerun the same command; finished ones are skipped.
#SBATCH --job-name=gpqa25-lora
#SBATCH --account=rc_ssriva_pi
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
set -euo pipefail
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"
export HF_TOKEN="${HF_TOKEN:-dummy}"
LOG="$RUNS_ROOT/logs/round2"; mkdir -p "$LOG"
exec >>"$LOG/lora_${MODE:-main}_job${SLURM_JOB_ID:-local}.log" 2>&1
case "${MODE:-main}" in
  check) "$PY" src/probing/lora_probe.py --check ;;
  pilot) "$PY" src/probing/lora_probe.py --fractions 0.5 --folds-run 0 --epochs 1 --out-dir runs/goal2/gpqa25/lora/pilot ${LORA_EXTRA:-} ;;
  main)  "$PY" src/probing/lora_probe.py --fractions ${FRACTIONS:-0.1 0.3 0.5 0.7 0.9} --epochs "${EPOCHS:-3}" --out-dir runs/goal2/gpqa25/lora/main ${LORA_EXTRA:-} ;;
esac
