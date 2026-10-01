#!/bin/bash
# Benchmark vLLM engine-CPU knobs on ONE GPU:  sbatch runs/gpqa25/bench_engine.sh
#
# WHY.  A py-spy profile of the first GPQA continuation run showed the engine is
# CPU-bound, not GPU-bound: GPU ~30-37% busy while the engine's single main thread
# sat at 97% of one core.  Per engine step, ~37% went to the Python scheduler
# (BlockSpaceManagerV1.access_all_blocks_in_seq walks EVERY KV block of EVERY
# running sequence EVERY step) and ~17% to output processing/detokenisation.
# Two stock vLLM 0.5.5 knobs target exactly that:
#   use_v2_block_manager  -- V2's access_all_blocks_in_seq is O(1) (one timestamp
#                            per sequence, deferred to free()) instead of a loop.
#   num_scheduler_steps N -- run N decode steps per scheduler call, amortising the
#                            scheduler + output-processing cost over N GPU steps.
#
# HOW.  Each config generates from the SAME slice of real prefixes (read-only from
# the real run's base CoTs) into its own scratch dir under $RUNS_ROOT/bench, for a
# fixed wall budget.  Throughput comes from vLLM's own periodic stats line, so it
# does not depend on a JSONL flush landing inside the window.  Nothing here touches
# the real run's data.
#SBATCH --job-name=gpqa25-bench
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:15:00   # short, to backfill into gaps on a busy partition
set -uo pipefail
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"
need_py "$PY"; need_weights; need_data

BENCH_ROOT="$RUNS_ROOT/bench/engine"
SECS="${BENCH_SECS:-300}"          # generation budget per config, after model load
SLICE_SHARDS="${SLICE_SHARDS:-50}" # --num-shards 50, index 0 => example_id % 50 == 0 (4 questions)
mkdir -p "$BENCH_ROOT"
# Append, and keep a per-job copy: a later round must not erase an earlier one
# (round 3 truncated round 2's table before its numbers had been written down).
RESULT="$BENCH_ROOT/results_job${SLURM_JOB_ID:-local}.tsv"
ALL_RESULTS="$BENCH_ROOT/results_all.tsv"
: > "$RESULT"

SHARD=0; log_to bench
echo "budget ${SECS}s per config; slice = 1/${SLICE_SHARDS} of questions; base CoTs read from $RUNS_ROOT/goal1/gpqa25"

# run_cfg <name> [@engines=N] <extra continuations.py flags...>
# @engines=N launches N engine PROCESSES on the one GPU, each on a different slice
# and with the memory budget split N ways, and sums their throughput.  This is the
# only way to use more of the card here: the engine is CPU-bound on one Python
# thread and the GIL means extra threads cannot help, while vLLM 0.5.5 refuses
# multi-step outside AsyncLLMEngine ("Multiple scheduler steps (multi-step) are only
# supported through AsyncLLMEngine").
run_cfg() {
  local name="$1"; shift
  local engines=1 a args=()
  for a in "$@"; do
    case "$a" in @engines=*) engines="${a#@engines=}" ;; *) args+=("$a") ;; esac
  done
  set -- "${args[@]:-}"
  local out="$BENCH_ROOT/$name" log="$LOG_DIR/bench_${name}.log"
  rm -rf "$out"; mkdir -p "$out"
  echo; echo "=== [$name] engines=$engines $* ==="
  # Sample GPU utilisation alongside the run.
  ( while true; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; sleep 5; done ) > "$out/gpu.txt" 2>/dev/null &
  local sampler=$!
  local t0=$(date +%s) util="$GPU_UTIL" pids=() e
  [ "$engines" -gt 1 ] && util=$(awk -v u="$GPU_UTIL" -v n="$engines" 'BEGIN{printf "%.3f",(u-0.03)/n}')
  for e in $(seq 0 $((engines - 1))); do
    # Engines on one GPU must LOAD one at a time: vLLM sizes its KV cache from
    # device-wide free memory, so two engines profiling at once each count the
    # other's weights as their own and one dies with "No available memory for the
    # cache blocks".  Wait for the previous engine's ready-file (written once its
    # cache is allocated) before starting the next; they then generate concurrently.
    local rdy="$out/ready.e$e"
    if [ "$e" -gt 0 ]; then
      local prev="$out/ready.e$((e - 1))" w=0
      while [ ! -f "$prev" ] && [ $w -lt 900 ]; do sleep 3; w=$((w + 3)); done
      echo "    engine $e starting (previous engine ready after ${w}s)"
    fi
    # Different --shard-index per engine so they do disjoint work, as in production.
    timeout --signal=INT $((SECS + 300)) "$PY" src/generation/continuations.py \
        --ready-file "$rdy" \
        --base-dir "$RUN_DIR" --output-dir "$out/e$e" --model "$MODEL" --dataset "$DATASET" \
        --continuations-per-prefix "$K" --fractions "${FRACTIONS[@]}" \
        --prefix-batch-size 32 --temperature 0.7 --top-p 0.95 \
        --max-tokens "$MAX_TOKENS" --max-model-len "$MAX_MODEL_LEN" \
        --max-n-per-request "$MAXN" --max-num-seqs "$MAX_NUM_SEQS" \
        --tensor-parallel-size 1 --gpu-memory-utilization "$util" --swap-space "$SWAP_GB" \
        --num-shards "$SLICE_SHARDS" --shard-index "$e" --log-stats "$@" \
        > "${log%.log}_e$e.log" 2>&1 &
    pids+=($!)
  done
  local pid=${pids[0]}
  cat /dev/null > "$log"
  # Stop the generator once it has produced SECS seconds of stats past model load.
  ( while kill -0 $pid 2>/dev/null; do
      n=$(tr '\r' '\n' < "${log%.log}_e$((engines-1)).log" 2>/dev/null | grep -ac "Avg generation throughput"); [ "${n:-0}" -ge $((SECS / 5)) ] && { kill -INT "${pids[@]}" 2>/dev/null; break; }; sleep 5
    done ) &
  local stopper=$!
  for e in "${pids[@]}"; do wait "$e" 2>/dev/null; done
  kill $sampler $stopper 2>/dev/null; wait $sampler $stopper 2>/dev/null
  cat "${log%.log}"_e*.log > "$log" 2>/dev/null
  local elapsed=$(( $(date +%s) - t0 ))

  # Throughput: median of the stats lines, skipping the first 6 (warm-up/prefill burst).
  local stats; stats=$(tr '\r' '\n' < "$log" | grep -a "Avg generation throughput" | tail -n +7)
  local tps run kv gpu
  # Throughput is the SUM over engines of each engine's median (they share the GPU,
  # so the card's total output is what matters).
  tps=$(for e in $(seq 0 $((engines - 1))); do
          tr '\r' '\n' < "${log%.log}_e$e.log" 2>/dev/null | grep -a "Avg generation throughput" | tail -n +7 \
            | sed -nE 's/.*Avg generation throughput: ([0-9.]+) tokens\/s.*/\1/p' \
            | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}'
        done | awk '{t+=$1} END{if(t>0)printf "%.0f", t; else print "n/a"}')
  run=$(echo "$stats" | sed -nE 's/.*Running: ([0-9]+) reqs.*/\1/p' | sort -n | awk '{a[NR]=$1} END{if(NR)printf "%.0f", a[int((NR+1)/2)]; else print "n/a"}')
  kv=$(echo "$stats"  | sed -nE 's/.*GPU KV cache usage: ([0-9.]+)%.*/\1/p' | sort -n | awk '{a[NR]=$1} END{if(NR)printf "%.0f", a[int((NR+1)/2)]; else print "n/a"}')
  gpu=$(sort -n "$out/gpu.txt" 2>/dev/null | awk '{a[NR]=$1} END{if(NR)printf "%.0f", a[int((NR+1)/2)]; else print "n/a"}')
  local recs=$(cat "$out"/e*/raw/continuations*.jsonl 2>/dev/null | wc -l)
  # We stop each config with SIGINT, so a KeyboardInterrupt traceback is expected.
  # Flag only real failures.
  local err=""
  grep -aqE "CUDA out of memory|AssertionError|ValueError|RuntimeError|NotImplementedError|lack of CPU swap" "$log" && err="ERRORS-IN-LOG"
  [ "${tps:-n/a}" = "n/a" ] && err="${err:+$err,}NO-STATS(config may be unsupported)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "${tps:-n/a}" "${gpu:-n/a}" "${run:-n/a}" "${kv:-n/a}" "$recs" "$elapsed" "$err" | tee -a "$RESULT" >> "$ALL_RESULTS"
  echo "  -> ${tps:-n/a} tok/s | GPU ${gpu:-n/a}% | running ${run:-n/a} reqs | KV ${kv:-n/a}% | $recs records | ${elapsed}s $err"
}

# Speed only.  The winner is validated for CORRECTNESS separately, paired against
# the 387k continuations the real run already produced under the baseline config.
#
# BENCH_CONFIGS lets a second round sweep something else on top of the winner, e.g.
#   BENCH_CONFIGS='seqs256|--use-v2-block-manager --num-scheduler-steps 8 --max-num-seqs 256'
# (entries separated by ';', name and flags by '|').
if [ -n "${BENCH_CONFIGS:-}" ]; then
  IFS=';' read -ra CFGS <<< "$BENCH_CONFIGS"
  for c in "${CFGS[@]}"; do run_cfg "${c%%|*}" ${c#*|}; done
else
  run_cfg baseline
  run_cfg v2blocks            --use-v2-block-manager
  run_cfg v2_mstep8           --use-v2-block-manager --num-scheduler-steps 8
  run_cfg v2_mstep16          --use-v2-block-manager --num-scheduler-steps 16
fi

echo; echo "================ ENGINE KNOB BENCHMARK ================"
printf "%-16s %10s %7s %9s %6s %9s %7s %s\n" config tok/s GPU% running KV% records secs notes
awk -F'\t' '{printf "%-16s %10s %7s %9s %6s %9s %7s %s\n", $1,$2,$3,$4,$5,$6,$7,$8}' "$RESULT"
b=$(awk -F'\t' '$1=="baseline"{print $2}' "$RESULT")
[ -n "$b" ] && [ "$b" != "n/a" ] && awk -F'\t' -v b="$b" 'NR>0 && $2!="n/a"{printf "  %-16s %.2fx baseline\n", $1, $2/b}' "$RESULT"
echo "======================================================="
