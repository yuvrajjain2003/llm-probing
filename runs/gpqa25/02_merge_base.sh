#!/bin/bash
# Stage 2: merge the base-CoT shards. continuations.py and verify_capture.py read
# the MERGED sample_records.jsonl, not the shard files, so this sits between the
# stages. Also refuses to let the chain proceed on an incomplete base run.
#SBATCH --job-name=gpqa25-merge-base
#SBATCH --partition=small    # the batch QoS demands >=48 CPUs per job; small allows 2-192
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"
need_py "$PY"
SHARD=0; log_to merge_base

merge_base_and_check
