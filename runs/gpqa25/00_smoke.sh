#!/bin/bash
# End-to-end smoke test: the WHOLE pipeline in miniature, on ONE GPU, in one job.
#
#   sbatch runs/gpqa25/00_smoke.sh        # then read $RUNS_ROOT/logs/smoke/smoke_summary_job<ID>.txt
#
# 4 questions x 2 CoTs x 5 checkpoints x K=3, in 2 shards -- so it exercises the real
# sharding, the shard queue, both merges, the count/label gates, the capture
# verification, activation extraction and the activation gate, using the exact same shard
# commands as the real run (they live in common.sh). Everything goes to RUN_TAG=smoke,
# wiped at the start, never touching the real run's directories.
#
# What it is FOR: (1) does vLLM 0.5.5 actually load and generate on the H100; (2) does
# every stage hand valid output to the next; (3) real per-stage timings, tokens/s and KV
# cache size, to replace the guessed walltimes and the A6000-derived vLLM knobs.
# What it is NOT: evidence about the science (3 continuations per prefix is noise).
#
# Base CoTs use the real 16384-token budget: GPQA reasoning is long, and a small cap
# would truncate nearly every CoT and leave nothing to test downstream.
#SBATCH --job-name=gpqa25-smoke
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
export RUN_TAG=smoke N_EXAMPLES=4 N_COTS=2 K=3 NSHARDS=2 EXTRACT_SHARDS=2
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"

need_py "$PY"; need_py "$PY_ES"; need_weights; need_data
SHARD=0; log_to smoke
LOGFILE="$LOG_DIR/smoke_shard00_job${SLURM_JOB_ID:-local}.log"

# Fresh state: resume-by-count would otherwise skip work from a previous smoke run.
for d in "$G1_ROOT" "$RUNS_ROOT/goal2/$RUN_TAG" "$QUEUE_ROOT"; do
  case "$d" in */smoke|*/smoke/*) rm -rf "$d" ;; *) die "refusing to delete $d" ;; esac
done

declare -A SEC=()
ORDER=()
summary() {
  echo; echo "================ SMOKE TEST SUMMARY (job ${SLURM_JOB_ID:-local}, node $(hostname)) ================"
  echo "stage timings (wall seconds; includes vLLM model load, ~1-3 min per shard):"
  for s in "${ORDER[@]}"; do printf "  %-14s %6ss\n" "$s" "${SEC[$s]}"; done
  echo
  echo "vLLM KV cache, goal-1 env at gpu_util=$GPU_UTIL (the old A6000s held ~420k tokens; 16 tokens/block):"
  grep -ah "# GPU blocks" "$LOGFILE" | sed -E 's/.*# GPU blocks: ([0-9]+), # CPU blocks: ([0-9]+).*/\1 \2/' | sort -u \
    | awk '{printf "  GPU %d blocks = %d tokens; CPU swap %d blocks = %d tokens\n", $1, $1*16, $2, $2*16}' || true
  echo "capture env (gpu_util 0.55-0.60):"
  grep -ah "GPU KV cache size" "$LOGFILE" | sed -E 's/^ *//; s/.*(GPU KV cache size: .*)/  \1/' | sort -u || true
  echo
  SEC_BASE="${SEC[base]:-0}" SEC_CONT="${SEC[cont]:-0}" "$PY" - "$RUN_DIR" <<'PY' || true
import os, sys, pandas as pd
run = sys.argv[1]
sb, sc = float(os.environ["SEC_BASE"]), float(os.environ["SEC_CONT"])
try:
    b = pd.read_json(f"{run}/raw/sample_records.jsonl", lines=True)
    term = b.raw_completion.str.contains("</think>")
    print(f"base CoTs: {len(b)}, terminated {int(term.sum())} ({term.mean():.0%}; original GPQA run truncated 1.85%), "
          f"parse_success {b.parse_success.mean():.0%}")
    if "reasoning_tokens" in b and sb > 0:
        tps = b.reasoning_tokens.sum() / sb
        print(f"  {b.reasoning_tokens.sum():,} reasoning tokens in {sb:.0f}s = {tps:,.0f} tok/s "
              f"(mean {b.reasoning_tokens.mean():,.0f}/CoT)")
        full = 198 * 25
        print(f"  ROUGH full base stage: {full:,} CoTs ~ {full * b.reasoning_tokens.mean() / tps / 3600:.1f} GPU-hours")
except Exception as e:
    print("base summary unavailable:", e)
try:
    if not os.path.exists(f"{run}/raw/continuations.jsonl"):
        raise FileNotFoundError("not produced (test stopped before that stage)")
    c = pd.read_json(f"{run}/raw/continuations.jsonl", lines=True)
    print(f"continuations: {len(c)}, terminated {c.terminated.mean():.0%}, parse_success {c.parse_success.mean():.0%} "
          f"(original: 0.79% parse failures)")
    if sc > 0:
        t = c.remaining_reasoning_tokens
        tps = t.sum() / sc
        print(f"  {t.sum():,} reasoning tokens in {sc:.0f}s = {tps:,.0f} tok/s (mean {t.mean():,.0f}/continuation)")
        n = 198 * 25 * 5 * 25
        h = n * t.mean() / tps / 3600
        print(f"  ROUGH full continuation stage: {n:,} x {t.mean():,.0f} tok = {h:,.0f} GPU-hours "
              f"= {h/4:,.0f} h wall with 4 workers")
        print("  (pessimistic: this run is tiny, so model load and the long-sequence tail dominate the timing,"
              " but it has too few samples to trust the mean length either -- treat as an order of magnitude)")
except Exception as e:
    print("continuation summary unavailable:", e)
PY
}

stage() {   # stage <name> <command...>   -- stops the whole test on the first failure
  local name="$1"; shift; local t; t=$(date +%s); ORDER+=("$name")
  echo; echo "=== [$name] start $(date '+%F %T')"
  if "$@"; then SEC[$name]=$(( $(date +%s) - t )); echo "=== [$name] OK in ${SEC[$name]}s"
  else          SEC[$name]=$(( $(date +%s) - t )); echo "=== [$name] FAILED after ${SEC[$name]}s"
                summary | tee "$LOG_DIR/smoke_summary_job${SLURM_JOB_ID:-local}.txt"
                echo "SMOKE TEST FAILED at stage: $name"; exit 1
  fi
}

stage base       run_queue base "$NSHARDS" base_shard
stage merge_base merge_base_and_check
stage verify     verify_capture_gate
stage cont       run_queue cont "$NSHARDS" cont_shard
stage labels     merge_labels_and_check
stage extract    run_queue extract "$EXTRACT_SHARDS" extract_shard
stage check_acts check_activations

summary | tee "$LOG_DIR/smoke_summary_job${SLURM_JOB_ID:-local}.txt"
echo; echo "SMOKE TEST PASSED (all stages, incl. activation gate)"
