#!/bin/bash
# Stage 5: merge continuation shards and compute the behavioural labels (the ground
# truth): checkpoint_labels.{csv,parquet} + commitment.{csv,parquet}. CPU only.
# Use answer_entropy_bits_mm as the entropy target (Miller-Madow); both estimators
# are written so the gap stays auditable.
#
# Then a completeness check, so the chain stops rather than extracting on a partial
# run: every row must hold exactly K continuations, and every question that has a
# terminated base CoT must appear in the labels.
#
# Memory: labels.py loads every continuation (text included) into one DataFrame. That
# is ~10 GB of text by my estimate at this scale, hence the generous request; unmeasured.
#SBATCH --job-name=gpqa25-labels
#SBATCH --partition=small    # the batch QoS demands >=48 CPUs per job; small allows 2-192
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -euo pipefail
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"
need_py "$PY"
SHARD=0; log_to labels

merge_labels_and_check
