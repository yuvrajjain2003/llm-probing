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
DATASET=gpqa_diamond
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
NSHARDS="${NSHARDS:-16}"                 # base CoTs + continuations
EXTRACT_SHARDS="${EXTRACT_SHARDS:-32}"   # extract_checkpoints.py writes only when a shard
                                         # finishes, so small shards bound what a crash loses

# vLLM knobs. These defaults are the values MEASURED ON 8x A6000 48 GB (see
# .claude/SETUP.md sec. 7). They are known-safe, not tuned for H100: expect to
# raise MAXN / MAX_NUM_SEQS once a first shard has been timed.
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
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
QUEUE_ROOT="${QUEUE_ROOT:-$RUNS_ROOT/queue/$RUN_TAG}"

_job_alive() { squeue -a -h -j "$1" 2>/dev/null | grep -q .; }

# run_queue <stage> <n_shards> <fn>      fn is called as `fn <shard>`; end it with
# `|| return 1` on every command (set -e is suspended inside a function used in `if`).
run_queue() {
  local stage="$1" n="$2" fn="$3" q="$QUEUE_ROOT/$1" me="${SLURM_JOB_ID:-local}"
  local l owner s ran=0 fails=0
  mkdir -p "$q"
  for l in "$q"/lock.*; do
    [ -d "$l" ] || continue
    owner="$(cat "$l/owner" 2>/dev/null || true)"
    # Stale if: its owner is gone; or it is MY id (a requeued job restarts with the
    # same id, and I run one shard at a time so any lock of mine is from before); or
    # it has no owner file and is old (a fresh empty lock is a claim being written).
    if { [ -n "$owner" ] && { [ "$owner" = "$me" ] || ! _job_alive "$owner"; }; } \
       || { [ -z "$owner" ] && [ -n "$(find "$l" -maxdepth 0 -mmin +5 2>/dev/null)" ]; }; then
      rm -rf "$l"; echo "[queue] released stale $(basename "$l") (owner ${owner:-none})"
    fi
  done
  for s in $(seq 0 $((n - 1))); do
    [ -e "$q/done.$s" ] && continue
    mkdir "$q/lock.$s" 2>/dev/null || continue           # another worker has it
    echo "$me" > "$q/lock.$s/owner"
    SHARD="$s"; echo "[$(date '+%F %T')] [queue] $stage shard $s/$n: start"
    if "$fn" "$s"; then
      touch "$q/done.$s"; ran=$((ran + 1)); echo "[$(date '+%F %T')] [queue] $stage shard $s: done"
    else
      fails=$((fails + 1)); echo "[$(date '+%F %T')] [queue] $stage shard $s: FAILED"
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
    --tensor-parallel-size 1 --gpu-memory-utilization "$GPU_UTIL" \
    --num-shards "$NSHARDS" --shard-index "$1" --resume || return 1
}
cont_shard() {
  "$PY" src/generation/continuations.py \
    --base-dir "$RUN_DIR" --model "$MODEL" --dataset "$DATASET" \
    --continuations-per-prefix "$K" --fractions "${FRACTIONS[@]}" \
    --prefix-batch-size 32 \
    --temperature 0.7 --top-p 0.95 \
    --max-tokens "$MAX_TOKENS" --max-model-len "$MAX_MODEL_LEN" \
    --max-n-per-request "$MAXN" --max-num-seqs "$MAX_NUM_SEQS" \
    --tensor-parallel-size 1 --gpu-memory-utilization "$GPU_UTIL" --swap-space "$SWAP_GB" \
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
