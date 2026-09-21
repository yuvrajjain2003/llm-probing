#!/bin/bash
# Stage 3: gate for activation capture. Runs verify_capture.py's three stages on one
# GPU and FAILS unless (a) the verdict is PASS and (b) the discovered alignment is the
# one extract_checkpoints.py hard-codes:
#     decoder layer L (1-indexed) -> EasySteer layer index L-1     (HF entry = ES layer + 1)
#     token position t            -> EasySteer position t+1        (HF pos   = ES pos   - 1)
# verify_capture.py's own PASS only checks that SOME alignment scores > 0.99, and it
# always exits 0, so the alignment is checked here. On the original box: min cosine
# 0.9969 over layers 0..26 (compare the number in the log).
#
# Needs the merged base run (uses its shortest terminated CoT as the test input), so
# it runs after 02 and in parallel with the continuations. 06_extract.sh depends on it.
#SBATCH --job-name=gpqa25-verify
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
set -euo pipefail
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"
need_py "$PY"; need_py "$PY_ES"; need_weights
SHARD=0; log_to verify

verify_capture_gate
