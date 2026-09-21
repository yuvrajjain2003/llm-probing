#!/bin/bash
# Stage 1: base CoTs -- 198 questions x 25 independent CoTs, in NSHARDS shards by
# example_id.  One WORKER job = 1 GPU; it pulls shards from the queue (see run_queue in
# common.sh) until none are left, then exits and frees the GPU.  Submit several copies
# for parallelism (submit_all.sh does: NWORKERS, default 4).  Long jobs, not arrays:
# fair-share is charged per job, so few long jobs beat many short ones.
#
# Walltime is capped at 48 h (a reported cluster policy: jobs beyond 2 days get timed
# out). If a worker times out mid-shard nothing is lost: the shard resumes by count
# and its stale lock is released by the next worker. Just resubmit.
# CPU/mem are GUESSES (no H100 timing exists yet).
#SBATCH --job-name=gpqa25-base
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"

need_py "$PY"; need_weights; need_data
log_worker base

run_queue base "$NSHARDS" base_shard
