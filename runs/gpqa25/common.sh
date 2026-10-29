# shellcheck shell=bash
# Sourced by every stage script in this directory; not runnable on its own.
#
# Run design (agreed 2026-09-18): GPQA-diamond, Qwen-7B, 25 independent CoTs per
# question, 5 checkpoints (10/30/50/70/90% -- NO 0% checkpoint), K=25 rollouts per
# prefix, whole dataset at one uniform K.  198 x 25 x 5 x 25 = 618,750 continuations.
# Scope: labels + activations only; no probes.
#
# HOW A STAGE FINDS THE REPO.  SLURM runs a COPY of the submitted script from its
# spool directory, so BASH_SOURCE / $0 cannot locate the repo.  Instead:
#     cd <repo> && source env.sh && sbatch runs/gpqa25/01_base.sh
# sbatch exports the environment by default, so REPO_DIR / RUNS_ROOT / PY / PY_ES
# ride along into the job.  SLURM_SUBMIT_DIR is the fallback when a job is
# submitted from the repo root without env.sh sourced.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [ ! -f "$REPO_DIR/env.sh" ]; then
  echo "ERROR: cannot find the repo (REPO_DIR='$REPO_DIR'). 'source env.sh' before sbatch." >&2
  exit 2
fi
export REPO_DIR
source "$REPO_DIR/env.sh"
cd "$REPO_DIR"

# ── What this run is ──────────────────────────────────────────────────────────
QUEUE_ROOT="${QUEUE_ROOT:-$RUNS_ROOT/queue/${RUN_TAG:-gpqa25}}"
DATASET="${DATASET:-gpqa_diamond}"      # any key of DATASETS in src/generation/run_eval.py
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
MODEL_SLUG="${MODEL##*/}"
RUN_TAG="${RUN_TAG:-gpqa25}"            # keep separate from the old 3-CoT design's runs/

N_EXAMPLES="${N_EXAMPLES:-198}"
N_COTS="${N_COTS:-25}"
K="${K:-25}"
FRACTIONS=(0.1 0.3 0.5 0.7 0.9)
MAX_TOKENS="${MAX_TOKENS:-16384}"        # same budget as the original GPQA run
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"

# Shard counts = queue lengths for the worker jobs (see run_queue below), independent of
# how many workers you submit. Sharding is by example_id % N, so keep N fixed across
# resubmits of a stage.
# Shard count is PINNED PER RUN in the queue directory. Changing it mid-run would
# re-partition the work by `example_id % N` and regenerate finished questions under new
# shard numbers, so a live run keeps whatever it started with and only new runs pick up
# the default. 64 (not 16) because with 16 shards over 198 questions one unlucky shard
# held the 2026-09-26 run hostage for a full extra day.
_pin="${QUEUE_ROOT:-$RUNS_ROOT/queue/$RUN_TAG}/cont/.nshards"
if [ -z "${NSHARDS:-}" ] && [ -r "$_pin" ]; then NSHARDS="$(cat "$_pin")"; fi
NSHARDS="${NSHARDS:-64}"                 # base CoTs + continuations
EXTRACT_SHARDS="${EXTRACT_SHARDS:-32}"   # extract_checkpoints.py writes only when a shard
                                         # finishes, so small shards bound what a crash loses

# vLLM knobs.  MAXN/GPU_UTIL/SWAP_GB are the values measured on 8x A6000 48 GB (see
# .claude/SETUP.md sec. 7).  MAX_NUM_SEQS and the block manager were re-measured on
# the H100 (benchmark jobs 4522034 / 4527004, 2026-09-25, 5 min per config, throughput
# from vLLM's own stats, control reproducible to 0.2%):
#
#   V1 blocks, 128 seqs  3,487 tok/s  (the first run's settings)
#   V2 blocks, 128 seqs  4,116        1.18x  -- V1 walks every KV block of every
#                                               sequence every step; V2 is O(1)
#   V2 blocks, 256 seqs  4,943        1.42x  <- what we now run
#   V2, 2 engines x 128  4,431        1.27x  -- same 256 sequences split over two
#                                               processes: 12% SLOWER, because each
#                                               process re-reads the 15 GB of weights
#                                               per step.  GPU "utilisation" reads
#                                               higher (84% vs 55%) while doing less
#                                               useful work -- utilisation is not
#                                               throughput.
#   multi-step (any N)   unsupported  -- NotImplementedError: only via AsyncLLMEngine
#
# Why not more than 256: at production context lengths (~2,530 tokens/sequence) 256
# sequences already need ~70% of the 923k-token cache.  Starving the cache is what
# made the 3-engine config thrash into CPU swap and hang (each engine got 68k tokens
# for sequences needing 115k).
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
USE_V2_BLOCKS="${USE_V2_BLOCKS:-1}"      # 0 restores the first run's V1 block manager
# Built as a variable, not "${USE_V2_BLOCKS:+...}": that form expands for ANY
# non-empty value, so USE_V2_BLOCKS=0 would still have passed the flag.
V2_FLAG=""; [ "$USE_V2_BLOCKS" = "1" ] && V2_FLAG="--use-v2-block-manager"
MAXN="${MAXN:-8}"                        # --max-n-per-request; 8 fit at a 16384-token budget
SWAP_GB="${SWAP_GB:-32}"

# ── Where things live (all under $RUNS_ROOT, i.e. /work, never home) ──────────
G1_ROOT="$RUNS_ROOT/goal1/$RUN_TAG"
RUN_DIR="$G1_ROOT/$DATASET/$MODEL_SLUG"                       # raw/ labels/ config/
ACT_DIR="$RUNS_ROOT/goal2/$RUN_TAG/activations_ckpt/$DATASET/$MODEL_SLUG"
VERIFY_DIR="$RUNS_ROOT/goal2/$RUN_TAG/_verify"
LOG_DIR="$RUNS_ROOT/logs/$RUN_TAG"

# ── Per-job environment ───────────────────────────────────────────────────────
if [ -n "${SLURM_JOB_ID:-}" ]; then
  # Compute nodes may have no outbound network; weights must already be cached
  # (env/fetch_models.sh on the login node). 00_gpu_check.sh tests this.
  export HF_HUB_OFFLINE=1
  # outlines' cache is a sqlite DB. Several concurrent jobs sharing one on NFS is a
  # locking hazard, so each job gets a private node-local one.
  export OUTLINES_CACHE_DIR="${TMPDIR:-/tmp}/outlines_${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
  mkdir -p "$OUTLINES_CACHE_DIR"
  trap 'rm -rf "$OUTLINES_CACHE_DIR"' EXIT
fi
SHARD="${SHARD:-${SLURM_ARRAY_TASK_ID:-}}"   # single-shard stages set SHARD=0 themselves

# ── Helpers ───────────────────────────────────────────────────────────────────
die() { echo "ERROR: $*" >&2; exit 1; }

need_py() {
  [ -x "$1" ] || die "interpreter not found: $1 (build it: env/setup_goal1.sh, or runs/goal2/setup_easysteer.sh)"
}
need_weights() {
  local d="$HF_HUB_CACHE/models--${MODEL//\//--}/snapshots"
  [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ] \
    || die "weights for $MODEL not in $HF_HUB_CACHE -- run env/fetch_models.sh on the login node"
}
need_data() {
  [ -d "$REPO_DIR/data/$DATASET" ] \
    || die "dataset missing: run '\"\$PY\" src/setup/download_dataset.py $DATASET' on the login node"
}
# ── Long-lived workers instead of job arrays ──────────────────────────────────
# The scheduler's fair-share is charged per JOB (user's finding, 2026-09-20): one long
# job carries far more priority than several short ones, and every array task is its
# own job. So the GPU stages are a few long single-GPU WORKER jobs that pull shard
# indices from a queue until none are left, then exit -- a long walltime is a limit,
# never a reservation, so a finished worker releases its GPU immediately.
# Walltime is 48 h: the scheduler accepted 5-day requests on h100_all at submission,
# but a colleague reports jobs beyond 2 days get timed out, so we stay under it.
# Workers are 1 GPU each: on 2026-09-20 a 1-GPU job was projected to start in ~14 h but
# a whole 4-GPU node only in ~2.9 days. Submit more workers for more parallelism.
#
# Queue protocol, all under $QUEUE_ROOT/<stage>/ on shared storage:
#   done.<n>        shard n finished successfully -- skipped forever after
#   lock.<n>/owner  a live worker is running shard n (mkdir is atomic, incl. on NFS)
# A lock whose owner job is no longer in squeue is stale (timeout, scancel, node
# failure) and is released; the shard resumes by count, so nothing already generated
# is redone.

# Owner ids are "<jobid>.<slot>" (slot distinguishes concurrent engine loops inside
# one job), so strip the slot before asking SLURM whether the job still exists.
_job_alive() { squeue -a -h -j "${1%%.*}" 2>/dev/null | grep -q .; }

# run_queue <stage> <n_shards> <fn>      fn is called as `fn <shard>`; end it with
# `|| return 1` on every command (set -e is suspended inside a function used in `if`).
# Set ENGINE_SLOT to distinguish concurrent loops within one job (see run_engines).
run_queue() {
  local stage="$1" n="$2" fn="$3" q="$QUEUE_ROOT/$1"
  local me="${SLURM_JOB_ID:-local}.${ENGINE_SLOT:-0}"
  local l owner s ran=0 fails=0
  mkdir -p "$q"
  for l in "$q"/lock.*; do
    [ -d "$l" ] || continue
    owner="$(cat "$l/owner" 2>/dev/null || true)"
    # Stale if: its owner's JOB is gone; or it is MY exact id, slot included (a
    # requeued job restarts with the same id, and each loop runs one shard at a time,
    # so any lock bearing my own slot is from before); or it has no owner file and is
    # old (a fresh empty lock is a claim still being written).
    if { [ -n "$owner" ] && { [ "$owner" = "$me" ] || ! _job_alive "$owner"; }; } \
       || { [ -z "$owner" ] && [ -n "$(find "$l" -maxdepth 0 -mmin +5 2>/dev/null)" ]; }; then
      rm -rf "$l"; echo "[queue] released stale $(basename "$l") (owner ${owner:-none})"
    fi
  done
  # Longest shard first. With shards taken in index order the most expensive one tends
  # to start last and run alone; starting it first leaves a short shard as the tail.
  mkdir -p "$q"; [ -e "$q/.nshards" ] || echo "$n" > "$q/.nshards"
  local order
  order=$("$PY" src/analysis/plan_shards.py order --run-dir "$RUN_DIR" --num-shards "$n" 2>/dev/null) \
    || order=$(seq 0 $((n - 1)))
  [ -z "$order" ] && order=$(seq 0 $((n - 1)))

  for s in $order; do
    [ -e "$q/done.$s" ] && continue
    mkdir "$q/lock.$s" 2>/dev/null || continue           # another worker has it
    echo "$me" > "$q/lock.$s/owner"
    SHARD="$s"; echo "[$(date '+%F %T')] [queue] $stage shard $s/$n: start"
    if "$fn" "$s"; then
      touch "$q/done.$s"; ran=$((ran + 1)); echo "[$(date '+%F %T')] [queue] $stage shard $s: done"
    else
      # One retry at half the concurrency. The failure we actually hit --
      # `assert len(seqs) == 1` in vLLM's scheduler -- fires when an n>1 request group
      # is PREEMPTED, so it is a symptom of too large a running set for this shard's
      # contexts and usually clears with fewer sequences in flight.
      echo "[$(date '+%F %T')] [queue] $stage shard $s: failed; retrying at half concurrency"
      if ( MAX_NUM_SEQS=$(( ${MAX_NUM_SEQS:-128} / 2 )); export MAX_NUM_SEQS; "$fn" "$s" ); then
        touch "$q/done.$s"; ran=$((ran + 1))
        echo "[$(date '+%F %T')] [queue] $stage shard $s: done on retry"
      else
        fails=$((fails + 1)); echo "[$(date '+%F %T')] [queue] $stage shard $s: FAILED twice"
      fi
    fi
    rm -rf "$q/lock.$s"
  done
  local left; left=$(( n - $(ls "$q"/done.* 2>/dev/null | wc -l) ))
  echo "[queue] $stage: this worker ran $ran shard(s), $fails failed; $left of $n not done yet"
  [ "$fails" -eq 0 ]
}

# One durable log per worker job (all of its shards, in order).
log_worker() {   # log_worker <stage>
  mkdir -p "$LOG_DIR"
  local f="$LOG_DIR/${1}_worker_job${SLURM_JOB_ID:-local}.log"
  exec >>"$f" 2>&1
  echo "[$(date '+%F %T')] $1 worker node=$(hostname) job=${SLURM_JOB_ID:-local} gpu=${CUDA_VISIBLE_DEVICES:-none}"
}

# Redirect the rest of the script into a durable per-shard log under $LOG_DIR
# (slurm-*.out stays in the submit directory and holds only scheduler-level errors).
log_to() {       # log_to <stage>
  mkdir -p "$LOG_DIR"
  local f; f="$(printf '%s/%s_shard%02d_job%s.log' "$LOG_DIR" "$1" "${SHARD:-0}" "${SLURM_JOB_ID:-local}")"
  exec >>"$f" 2>&1
  echo "[$(date '+%F %T')] $1 shard=${SHARD:-n/a} node=$(hostname) job=${SLURM_JOB_ID:-local} gpu=${CUDA_VISIBLE_DEVICES:-none}"
}

# ── Stage bodies, shared by the worker scripts and by 00_smoke.sh ──────────────
# Each takes a shard index and ends every command with `|| return 1` (see run_queue).
base_shard() {
  "$PY" src/generation/run_eval.py \
    --model "$MODEL" --dataset "$DATASET" --output-dir "$G1_ROOT" \
    --num-examples "$N_EXAMPLES" --samples-per-example "$N_COTS" --seed 42 \
    --temperature 0.7 --top-p 0.95 \
    --max-tokens "$MAX_TOKENS" --max-model-len "$MAX_MODEL_LEN" \
    --sample-batch-size 256 --max-num-seqs "$MAX_NUM_SEQS" \
    --tensor-parallel-size 1 --gpu-memory-utilization "${GPU_UTIL_EFFECTIVE:-$GPU_UTIL}" \
    --num-shards "$NSHARDS" --shard-index "$1" --resume || return 1
}
cont_shard() {
  # Concurrency is sized from THIS shard's prefix lengths, not a constant. A fixed
  # max_num_seqs is what broke the 2026-09-26 run: 256 was tuned on ~1,200-token
  # sequences and applied to a shard averaging ~5,700, giving a working set 1.6x the KV
  # cache -- continuous preemption, 3.5x slower, and whole shards killed by vLLM's
  # n>1-preemption assertion. MAX_NUM_SEQS in the environment (e.g. the retry path)
  # still wins.
  local seqs="${MAX_NUM_SEQS_OVERRIDE:-${MAX_NUM_SEQS:-}}"
  if [ -z "${MAX_NUM_SEQS_FIXED:-}" ]; then
    seqs=$("$PY" src/analysis/plan_shards.py concurrency --run-dir "$RUN_DIR" \
             --num-shards "$NSHARDS" --shard-index "$1" --gen-tokens "${GEN_TOKENS:-2800}" \
             --fallback "${MAX_NUM_SEQS:-128}" --max-seqs "${MAX_NUM_SEQS:-128}" --explain 2>&1 \
             | tail -1) || seqs="${MAX_NUM_SEQS:-128}"
    case "$seqs" in ''|*[!0-9]*) seqs="${MAX_NUM_SEQS:-128}" ;; esac
    echo "[concurrency] shard $1: max_num_seqs=$seqs (cap ${MAX_NUM_SEQS:-128})"
  fi
  MAX_NUM_SEQS="$seqs"
  run_gated "${TMPDIR:-/tmp}/ready_cont_${ENGINE_SLOT:-0}_$1" \
    "$PY" src/generation/continuations.py \
    --ready-file "${TMPDIR:-/tmp}/ready_cont_${ENGINE_SLOT:-0}_$1" \
    --base-dir "$RUN_DIR" --model "$MODEL" --dataset "$DATASET" \
    --continuations-per-prefix "$K" --fractions "${FRACTIONS[@]}" \
    --prefix-batch-size 32 \
    --temperature 0.7 --top-p 0.95 \
    --max-tokens "$MAX_TOKENS" --max-model-len "$MAX_MODEL_LEN" \
    --max-n-per-request "$MAXN" --max-num-seqs "$MAX_NUM_SEQS" \
    $V2_FLAG \
    --tensor-parallel-size 1 --gpu-memory-utilization "${GPU_UTIL_EFFECTIVE:-$GPU_UTIL}" --swap-space "$SWAP_GB" \
    --subset-mod "${SUBSET_MOD:-1}" --subset-keep "${SUBSET_KEEP:-0}" \
    --num-shards "$NSHARDS" --shard-index "$1" --resume || return 1
}
extract_shard() {
  "$PY_ES" src/probing/extract_checkpoints.py \
    --run-dir "$RUN_DIR" --out-dir "$ACT_DIR" \
    --batch-size 2 --gpu-memory-utilization 0.55 --max-model-len "$MAX_MODEL_LEN" \
    --num-shards "$EXTRACT_SHARDS" --shard-index "$1" || return 1
}

# Merge + completeness gates (src/analysis/check_run.py exits nonzero on a bad stage).
merge_base_and_check() {
  "$PY" src/generation/merge_shards.py --run-dir "$RUN_DIR" || return 1
  "$PY" src/analysis/check_run.py base --run-dir "$RUN_DIR" --n-cots "$N_COTS" || return 1
}
merge_labels_and_check() {
  "$PY" src/generation/merge_shards.py --run-dir "$RUN_DIR" --no-metrics || return 1
  "$PY" src/analysis/labels.py --continuations "$RUN_DIR/raw/continuations.jsonl" || return 1
  "$PY" src/analysis/check_run.py labels --run-dir "$RUN_DIR" --k "$K" || return 1
}
check_activations() {
  "$PY" src/analysis/check_run.py activations --run-dir "$RUN_DIR" --act-dir "$ACT_DIR" \
    --n-shards "$EXTRACT_SHARDS" || return 1
}

# Capture gate. verify_capture.py's own verdict always exits 0 and only checks that SOME
# layer/position alignment scores > 0.99, so require the exact alignment that
# extract_checkpoints.py hard-codes:
#     decoder layer L (1-indexed) -> EasySteer layer index L-1     (HF entry = ES layer + 1)
#     token position t            -> EasySteer position t+1        (HF pos   = ES pos   - 1)
# On the original box: min cosine 0.9969 over layers 0..26 (compare the number in the log).
verify_capture_gate() {
  export VERIFY_RUN_DIR="$RUN_DIR" VERIFY_OUT="$VERIFY_DIR"
  mkdir -p "$VERIFY_DIR"
  "$PY"    src/probing/verify_capture.py --stage hf  || return 1     # stock HF reference
  "$PY_ES" src/probing/verify_capture.py --stage es  || return 1     # EasySteer, separate process
  "$PY"    src/probing/verify_capture.py --stage cmp | tee "$VERIFY_DIR/cmp.txt" || return 1
  grep -q "VERDICT: PASS" "$VERIFY_DIR/cmp.txt" \
    || { echo "ERROR: capture verification did not PASS -- do NOT extract activations"; return 1; }
  grep -q "HF_entry = ES_layer + 1,  HF_pos = ES_pos + -1" "$VERIFY_DIR/cmp.txt" \
    || { echo "ERROR: alignment differs from what extract_checkpoints.py assumes (see $VERIFY_DIR/cmp.txt)"; return 1; }
  echo "[$(date '+%F %T')] capture verified: PASS and alignment matches extract_checkpoints.py"
}


# ── Serialising engine start-up on a shared GPU ───────────────────────────────
# vLLM sizes its KV cache as `total_gpu_memory * gpu_memory_utilization - peak_memory`,
# and derives peak_memory from torch.cuda.mem_get_info(), which reports the WHOLE
# DEVICE.  So while an engine is loading, any sibling that profiles at the same moment
# counts those weights as its own overhead and dies with "No available memory for the
# cache blocks" (observed, benchmark job 4527004).  Once a sibling is fully allocated
# and stable it is invisible to the calculation, because each process records its own
# baseline at start-up.  Hence: only ONE engine may be in its loading phase at a time.
# This applies to every launch, not just the first -- a worker relaunches the engine
# for each shard it picks up.
_gate_dir() { echo "${TMPDIR:-/tmp}/vllm_start_gate_${SLURM_JOB_ID:-local}"; }

_gate_acquire() {
  local g; g="$(_gate_dir)" ; local waited=0
  while ! mkdir "$g" 2>/dev/null; do
    # Force-release a gate held far longer than any model load (crashed holder).
    if [ -n "$(find "$g" -maxdepth 0 -mmin +20 2>/dev/null)" ]; then
      echo "[gate] releasing a stale start-up gate"; rm -rf "$g"; continue
    fi
    sleep 3; waited=$((waited + 3))
    [ "$waited" -ge 1800 ] && { echo "[gate] giving up waiting after ${waited}s"; return 0; }
  done
}
_gate_release() { rm -rf "$(_gate_dir)" 2>/dev/null; }

# run_gated <ready_file> <cmd...>   Hold the start-up gate only until this engine has
# allocated its cache (it writes <ready_file>), then release it so the next engine can
# load while this one generates.
run_gated() {
  local rf="$1"; shift
  [ "${ENGINES_PER_GPU:-1}" -le 1 ] && { "$@"; return $?; }
  rm -f "$rf"; _gate_acquire
  "$@" & local pid=$! waited=0
  while kill -0 "$pid" 2>/dev/null && [ ! -f "$rf" ] && [ "$waited" -lt 1200 ]; do
    sleep 3; waited=$((waited + 3))
  done
  _gate_release
  wait "$pid"
}

# ── Several engine processes on ONE GPU ───────────────────────────────────────
# The engine is CPU-bound, not GPU-bound: one Python thread does all scheduling and
# detokenisation, and the GIL means extra THREADS cannot help -- only extra
# PROCESSES.  Measured on the first GPQA run: GPU 30-37% busy, engine main thread at
# 97% of one core, every other thread idle.  Two engines on one card therefore use
# two cores against a GPU that is idle roughly two-thirds of the time.
#
# Cost: each process loads its own copy of the weights (~15 GB), so the KV cache is
# split.  At ENGINES=2 each gets gpu_memory_utilization ~0.42 => ~19 GB of KV, about
# 325k tokens; the first run peaked at ~18% of a 923k-token cache, so this is ample.
#
# Each loop claims its own shards and gets its own ENGINE_SLOT, so the queue's
# stale-lock logic never mistakes a sibling's live lock for its own.
run_engines() {   # run_engines <stage> <n_shards> <fn>
  local n="${ENGINES_PER_GPU:-1}"
  # One engine: clear any split budget so the stage uses the full $GPU_UTIL.
  if [ "$n" -le 1 ]; then unset GPU_UTIL_EFFECTIVE; run_queue "$@"; return $?; fi
  # Split the memory budget; keep a little headroom for per-process CUDA context.
  export GPU_UTIL_EFFECTIVE
  GPU_UTIL_EFFECTIVE=$(awk -v u="$GPU_UTIL" -v n="$n" 'BEGIN{printf "%.3f", (u-0.03)/n}')
  echo "[engines] $n engine processes on this GPU, gpu_memory_utilization=$GPU_UTIL_EFFECTIVE each"
  local pids=() i rc=0
  for i in $(seq 1 "$n"); do
    ( ENGINE_SLOT="$i" run_queue "$@" ) & pids+=($!)
  done
  for i in "${pids[@]}"; do wait "$i" || rc=1; done
  return $rc
}
