#!/bin/bash
# Correctness gate for a vLLM engine config, before it is used on real data:
#
#   cd <repo> && source env.sh
#   CFG_NAME=v2_mstep8 CFG_FLAGS="--use-v2-block-manager --num-scheduler-steps 8" \
#     sbatch runs/gpqa25/validate_config.sh
#
# Speed is not enough.  BlockSpaceManagerV2 changes KV block reuse (a bug there
# corrupts text) and multi-step changes when stop conditions are observed (a bug
# there shifts remaining_reasoning_tokens -- a Goal 1 LABEL, not just a cost).
# So: regenerate continuations for prefixes the real run ALREADY completed under
# the baseline config, then compare the two distributions, paired per prefix.
# Nothing is written into the real run's directories.
#SBATCH --job-name=gpqa25-validate
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=runs/logs/slurm/slurm-%j.out
set -uo pipefail
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"
need_py "$PY"; need_weights; need_data
: "${CFG_NAME:?set CFG_NAME}"; : "${CFG_FLAGS:?set CFG_FLAGS}"
GEN_SECS="${GEN_SECS:-4500}"          # ~75 min of generation, inside the 2 h walltime

OUT="$RUNS_ROOT/bench/validate/$CFG_NAME"
SHARD=0; log_to "validate_$CFG_NAME"
echo "config: $CFG_NAME   flags: $CFG_FLAGS"

# Shard 0 of the real 16-way split is COMPLETE under the baseline config, so every
# prefix we regenerate here has 25 baseline continuations to compare against.
BASE_JSONL="$RUN_DIR/raw/continuations.shard00.jsonl"
[ -s "$BASE_JSONL" ] || die "no completed baseline shard at $BASE_JSONL"
rm -rf "$OUT"; mkdir -p "$OUT"

timeout --signal=INT "$GEN_SECS" "$PY" src/generation/continuations.py \
    --base-dir "$RUN_DIR" --output-dir "$OUT" --model "$MODEL" --dataset "$DATASET" \
    --continuations-per-prefix "$K" --fractions "${FRACTIONS[@]}" \
    --prefix-batch-size 32 --temperature 0.7 --top-p 0.95 \
    --max-tokens "$MAX_TOKENS" --max-model-len "$MAX_MODEL_LEN" \
    --max-n-per-request "$MAXN" --max-num-seqs "$MAX_NUM_SEQS" \
    --tensor-parallel-size 1 --gpu-memory-utilization "$GPU_UTIL" --swap-space "$SWAP_GB" \
    --num-shards "$NSHARDS" --shard-index 0 --log-stats $CFG_FLAGS 2>&1 | tee "$OUT/gen.log"
echo "generation stopped (SIGINT at the time box is expected)"

# Operational health at PRODUCTION context lengths, which the short speed benchmark
# cannot show: it runs only the first few minutes of a chunk, when sequences are
# still short and the KV cache is nearly empty.  Here sequences grow to full length,
# so this is where cache pressure and preemption would appear.
echo; echo "---- engine health (config: $CFG_NAME) ----"
ST=$(tr '\r' '\n' < "$OUT/gen.log" | grep -a "Avg generation throughput")
echo "$ST" | sed -nE 's/.*GPU KV cache usage: ([0-9.]+)%.*/\1/p' \
  | sort -n | awk '{a[NR]=$1} END{if(NR)printf "  KV cache usage: median %.0f%%, max %.0f%% (over %d samples)\n", a[int((NR+1)/2)], a[NR], NR}'
echo "$ST" | sed -nE 's/.*Running: ([0-9]+) reqs.*/\1/p' \
  | sort -n | awk '{a[NR]=$1} END{if(NR)printf "  running request groups: median %d, max %d\n", a[int((NR+1)/2)], a[NR]}'
sw=$(tr '\r' '\n' < "$OUT/gen.log" | grep -aci "preempt")
echo "  preemption events in the log: $sw  (non-zero means the KV cache is too small for this concurrency)"

CAND="$OUT/raw/continuations.shard00.jsonl"
[ -s "$CAND" ] || die "candidate produced no continuations -- see the log above"
echo; echo "############ $CFG_NAME vs baseline ############"
"$PY" src/analysis/compare_configs.py --baseline "$BASE_JSONL" --candidate "$CAND"
rc=$?
echo; [ $rc -eq 0 ] && echo "VALIDATION PASSED for $CFG_NAME" || echo "VALIDATION FAILED for $CFG_NAME"
exit $rc
