#!/bin/bash
# Round 7, item 7: all-token activation capture at one layer, for weighted-pooling probes.
# Writes to /work (NOT the group-shared project folder): the store is 30 to 170 GB.
#
#   cd <repo> && source env.sh
#   R7_TAG=gpqa25 R7_COT_MAX=5 R7_NUM_SHARDS=8 R7_SHARDS=0,1,2,3,4,5,6,7 \
#     sbatch runs/r7_tokens.sh
#
# Resumable: a shard whose tokens+index files exist is skipped, so a killed job
# can be resubmitted unchanged.
#SBATCH --job-name=r7-tokens
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=runs/logs/slurm/slurm-%j.out
set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
[ -f "$REPO_DIR/env.sh" ] || { echo "ERROR: source env.sh in the repo before sbatch" >&2; exit 2; }
cd "$REPO_DIR"; source env.sh
export HF_TOKEN="${HF_TOKEN:-dummy}"

TAG="${R7_TAG:-gpqa25}"
case "$TAG" in
  gpqa25)      DS=gpqa_diamond ;;
  mmlupro_v1)  DS=mmlu_pro ;;
  globalop_v1) DS=global_opinions ;;
  *) echo "unknown R7_TAG=$TAG" >&2; exit 2 ;;
esac
MS=DeepSeek-R1-Distill-Qwen-7B
RUN_DIR="$REPO_DIR/runs/goal1/$TAG/$DS/$MS"
# the bulk store lives on /work, which is NFS-mounted so compute nodes see it
TOK_ROOT="${R7_TOK_ROOT:-/work/users/y/u/yuvraj/llm-probing/runs/goal2}"
SUB="${R7_SUB:-activations_tokens}"
OUT="$TOK_ROOT/$TAG/$SUB/$DS/$MS"
mkdir -p "$OUT" "$REPO_DIR/runs/logs/round7"
LOG="$REPO_DIR/runs/logs/round7/tokens_${TAG}_${SUB}_job${SLURM_JOB_ID:-local}.log"
exec >>"$LOG" 2>&1
echo "=== $(date) tag=$TAG out=$OUT shards=${R7_SHARDS:-0}/${R7_NUM_SHARDS:-1} ==="
df -h "$TOK_ROOT" | tail -1
"$PY_ES" src/probing/extract_tokens.py \
  --run-dir "$RUN_DIR" --out-dir "$OUT" \
  --layers "${R7_LAYERS:-21}" --stride "${R7_STRIDE:-1}" \
  ${R7_COT_MAX:+--cot-max $R7_COT_MAX} \
  --batch-size "${R7_BATCH:-2}" --max-model-len 24576 \
  --gpu-memory-utilization "${R7_GPU_UTIL:-0.55}" \
  --num-shards "${R7_NUM_SHARDS:-1}" --shard-index "${R7_SHARDS:-0}"
echo "=== $(date) done ==="
du -sh "$OUT"
