#!/bin/bash
# Stage 7: validate the extracted activations -- the deliverable of the whole run. All
# shards present; vectors finite, not all-zero, right shape and dtype; 1:1 with
# checkpoint_labels.csv; positions monotonic within each CoT; different checkpoints of a
# CoT have different vectors. CPU only (src/analysis/check_run.py). Fails the job, and
# so is visible in `squeue`/`sacct`, if anything is wrong.
#SBATCH --job-name=gpqa25-check-acts
#SBATCH --partition=small    # the batch QoS demands >=48 CPUs per job; small allows 2-192
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"
need_py "$PY"
SHARD=0; log_to check_acts

check_activations
