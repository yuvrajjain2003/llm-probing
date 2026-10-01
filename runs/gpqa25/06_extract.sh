#!/bin/bash
# Stage 6: activation capture. All 28 decoder layers at the EXACT checkpoint token,
# one fp16 vector per (example_id, cot_id, fraction) -- 1:1 with checkpoint_labels.csv.
# ~24.7k vectors, ~4.6 GiB in total. Uses the EasySteer env ($PY_ES), NOT $PY.
#
# One forward pass per CoT (its longest prefix) yields every checkpoint's state,
# because the model is causal. Nothing is captured per continuation: all K
# continuations of a prefix share it, so their activations would be identical.
#
# WORKER job (1 GPU) pulling shards from the queue (run_queue in common.sh) until none
# are left. Long jobs, not arrays: fair-share is charged per job. Walltime capped at 48 h.
#
# extract_checkpoints.py keeps a shard's vectors in RAM and writes only when the shard
# finishes (a finished shard is skipped on rerun; a crashed one restarts from zero).
# Hence EXTRACT_SHARDS=32 small shards (~6 questions x 25 CoTs each): a timeout costs at
# most one small shard, and the queue re-runs it.
#
# Depends on 03_verify_capture.sh (capture conventions confirmed) and 05_merge_label.sh
# (which produces the merged checkpoints.jsonl this reads). CPU/mem are GUESSES.
#SBATCH --job-name=gpqa25-extract
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"

need_py "$PY_ES"; need_weights
[ -f "$RUN_DIR/raw/checkpoints.jsonl" ] || die "no merged checkpoints at $RUN_DIR/raw -- run 05_merge_label.sh first"
# The capture conventions must have been verified for THIS run at least once. Stage 03
# normally guarantees that via the job dependency, but a resume (START_AT>=6) can skip
# it, so re-check the artifact rather than assume.
grep -q "VERDICT: PASS" "$VERIFY_DIR/cmp.txt" 2>/dev/null \
  || die "capture never verified for this run ($VERIFY_DIR/cmp.txt) -- run 03_verify_capture.sh first"
log_worker extract

run_engines extract "$EXTRACT_SHARDS" extract_shard
