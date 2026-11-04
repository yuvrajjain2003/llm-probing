#!/bin/bash
# Round 7, item 7: the four poolings, scored identically.
#
#   cd <repo> && source env.sh
#   R7_TAG=gpqa25 R7_VARIANT=pilot5 R7_TOK_SUB=activations_tokens_pilot5 R7_COT_MAX=5 \
#     R7_MODE=pool sbatch runs/r7_pooling.sh
#
# R7_MODE: pool (CPU, I/O heavy)  ridge (CPU)  adam (GPU)  table (CPU)
# For R7_MODE=adam, R7_POOLING selects attn|mean|max|last.
#SBATCH --job-name=r7-pool
#SBATCH --output=runs/logs/slurm/slurm-%j.out
set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
[ -f "$REPO_DIR/env.sh" ] || { echo "ERROR: source env.sh in the repo before sbatch" >&2; exit 2; }
cd "$REPO_DIR"; source env.sh
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}" HF_TOKEN="${HF_TOKEN:-dummy}"

TAG="${R7_TAG:-gpqa25}"; VAR="${R7_VARIANT:-full}"
TOK_SUB="${R7_TOK_SUB:-activations_tokens}"
case "$TAG" in
  gpqa25) DS=gpqa_diamond ;; mmlupro_v1) DS=mmlu_pro ;; globalop_v1) DS=global_opinions ;;
  *) echo "unknown R7_TAG=$TAG" >&2; exit 2 ;;
esac
MS=DeepSeek-R1-Distill-Qwen-7B
SW="runs/goal2/$TAG/sweep/r7_pooling"
TOK="${R7_TOK_ROOT:-/work/users/y/u/yuvraj/llm-probing/runs/goal2}/$TAG/$TOK_SUB/$DS/$MS"
ACTOUT="$SW/_act_$VAR"
mkdir -p "$SW" "$REPO_DIR/runs/logs/round7"
LOG="$REPO_DIR/runs/logs/round7/pooling_${TAG}_${VAR}_${R7_MODE}${R7_POOLING:+_$R7_POOLING}_job${SLURM_JOB_ID:-local}.log"
exec >>"$LOG" 2>&1
echo "=== $(date) tag=$TAG variant=$VAR mode=$R7_MODE ==="

case "${R7_MODE}" in
pool)
  "$PY" src/probing/attn_probe.py pool --tag "$TAG" --tok-sub "$TOK_SUB"
  "$PY" src/probing/attn_probe.py actdir --tag "$TAG" --tok-sub "$TOK_SUB" \
      --act-out "$ACTOUT" ${R7_COT_MAX:+--cot-max $R7_COT_MAX}
  ;;
ridge)
  CFG="configs/sweep_r7_pooling_${TAG}_${VAR}.yaml"
  EXTRA=""
  if [ "$TAG" != "gpqa25" ]; then EXTRA=", answer_entropy_bits_mm_norm"; fi
  cat > "$CFG" <<YAML
# Round 7 item 7: ridge/logistic on the LAST-token, MEAN-pooled and MAX-pooled
# layer-21 states, on exactly the rows the all-token store covers. Same grids,
# folds and fold_seed as configs/sweep_ref*.yaml, so the numbers are comparable
# with every published approach. act_dir is a filtered VIEW of the published
# store (symlinked .npy, subset index), made by attn_probe.py actdir.
act_dir: $ACTOUT
run_dir: runs/goal1/$TAG/$DS/$MS
out_dir: $SW/ridge_$VAR
folds: 5
fixed_folds: true
fold_seed: 0
fractions: [0.1, 0.3, 0.5, 0.7, 0.9]
targets: [answer_entropy_bits_mm, p_mode, rem_len_mean, rem_len_std, committed_by_here$EXTRA]

features:
  act_L21:  {type: act, layers: [21]}
  mean_L21: {type: npy, path: $TOK/pooled_mean.npy, keys: $TOK/pooled_keys.parquet}
  max_L21:  {type: npy, path: $TOK/pooled_max.npy,  keys: $TOK/pooled_keys.parquet}

probes:
  ridge_act: {class: ridge,    grid: {alpha: {logspace: [0, 6, 13]}}}
  logistic:  {class: logistic, grid: {C: {logspace: [-4, 1, 6]}}}

runs:
  - {features: [act_L21, mean_L21, max_L21], probes: [ridge_act, logistic]}
YAML
  "$PY" src/probing/sweep.py --config "$CFG" ${R7_SHARD:+--shard $R7_SHARD}
  ;;
adam)
  "$PY" src/probing/attn_probe.py train --tag "$TAG" --tok-sub "$TOK_SUB" \
      --pooling "${R7_POOLING:?set R7_POOLING}" --out-dir "$SW/adam_$VAR" \
      --epochs "${R7_EPOCHS:-100}" --log-every "${R7_LOG_EVERY:-20}" \
      --feat-norm "${R7_FEAT_NORM:-perdim}" \
      --tok-budget "${R7_TOK_BUDGET:-65536}" \
      ${R7_STATES:+--save-states $R7_STATES} \
      ${R7_STATE_TARGETS:+--state-targets $R7_STATE_TARGETS}
  ;;
attnridge)
  "$PY" src/probing/attn_ridge.py --tag "$TAG" --tok-sub "$TOK_SUB" \
      --states "${R7_STATES:-$SW/states_$VAR}" --out "$SW/adam_$VAR"
  ;;
attnstats)
  "$PY" src/probing/r7_attn_stats.py --states "${R7_STATES:?set R7_STATES}"
  ;;
loglen)
  "$PY" src/probing/r7_loglen.py --tag "$TAG" --tok-sub "$TOK_SUB" \
      --out "$SW/loglen_$VAR" ${R7_STATES:+--states $R7_STATES}
  ;;
table)
  NE=""; [ "$TAG" != "gpqa25" ] && NE="--norm-entropy"
  "$PY" src/viz/r7_pooling_table.py --tag "$TAG" --variant "$VAR" $NE
  # a second table over the two headline targets only, which is the widest set the
  # attention-pooled RIDGE readout covers on the big datasets
  "$PY" src/viz/r7_pooling_table.py --tag "$TAG" --variant "$VAR" --suffix _2t \
      --targets answer_entropy_bits_mm rem_len_mean
  ;;
*) echo "set R7_MODE to pool|ridge|adam|attnridge|attnstats|loglen|table" >&2; exit 2 ;;
esac
echo "=== $(date) done ==="
