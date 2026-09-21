#!/bin/bash
# Stage 4: prefix-conditioned continuations -- 5 checkpoints x K=25 per CoT, uniform K
# over the whole dataset. This is ~all of the GPU cost of the run (618,750
# continuations vs 4,950 base CoTs), so it is the stage where worker count matters.
#
# One WORKER job = 1 GPU pulling shards from the queue (run_queue in common.sh) until
# none are left. Long jobs, not arrays: fair-share is charged per job. Walltime capped at
# 48 h; resubmit workers until `status.sh` shows every shard done.
#
# continuations.py resumes by count across ALL continuations.shard*.jsonl files, so a
# timed-out worker loses nothing, and raising K later only generates the shortfall.
# CPU/mem are GUESSES (no H100 timing yet).
#SBATCH --job-name=gpqa25-cont
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"

need_py "$PY"; need_weights; need_data
[ -f "$RUN_DIR/raw/sample_records.jsonl" ] || die "no merged base CoTs at $RUN_DIR/raw -- run 02_merge_base.sh first"
log_worker cont

run_queue cont "$NSHARDS" cont_shard
