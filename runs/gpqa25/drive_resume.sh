#!/usr/bin/env bash
# Unattended driver: validate the new engine config, and ONLY if it passes, resume the
# run -- then keep it alive across the 48 h job cap.
#
#   cd <repo> && source env.sh
#   tmux new-session -d -s gpqa25 'runs/gpqa25/drive_resume.sh'
#   tmux attach -t gpqa25          # to watch;  Ctrl-b d to detach
#
# It exists because of two things that actually happened: the continuation workers hit
# their 48 h cap with work left and the chain sat idle ~60 h with nobody watching, and
# because the config change (V2 block manager + 256 sequences) alters KV block reuse,
# which must be shown not to change the DATA before it generates ~80 GPU-hours of it.
#
# SKIP_VALIDATION=1 resumes immediately without the correctness gate (not advised).
#
# RUN IT AS A SLURM JOB, not on the login node:
#     cd <repo> && source env.sh && sbatch runs/gpqa25/drive_resume.sh
# A tmux session on the login node died on 2026-09-26 and took the driver with it,
# stranding the chain. A batch job survives tmux, ssh and the login node, and this
# script resubmits ITSELF before its walltime expires if work remains, so the
# supervision never lapses. (Verified a job may sbatch/squeue/scancel: job 4549559.)
#SBATCH --job-name=gpqa25-driver
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=46:00:00
#SBATCH --open-mode=append
#SBATCH --output=runs/logs/slurm/slurm-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}" || exit 1
[ -f env.sh ] || cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
source env.sh
source runs/gpqa25/common.sh

LOG="$LOG_DIR/drive_resume.log"; mkdir -p "$LOG_DIR"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
alive() { squeue -a -h -j "$1" 2>/dev/null | grep -q .; }
undone() { echo $(( NSHARDS - $(ls "$QUEUE_ROOT/cont"/done.* 2>/dev/null | wc -l) )); }

CFG_NAME="${CFG_NAME:-v2blocks}"   # validation label only; the RUN config lives in common.sh
CFG_FLAGS="${CFG_FLAGS:---use-v2-block-manager}"
NWORKERS="${NWORKERS:-4}"
MAX_ROUNDS="${MAX_ROUNDS:-4}"          # auto-resubmits after a 48 h timeout

# Resubmit self and exit while there is still work, rather than dying at the walltime.
DRIVER_START=$(date +%s)
DRIVER_BUDGET="${DRIVER_BUDGET:-160000}"     # ~44.5 h of a 46 h allocation
relay_if_near_walltime() {
  [ -n "${SLURM_JOB_ID:-}" ] || return 0
  [ $(( $(date +%s) - DRIVER_START )) -lt "$DRIVER_BUDGET" ] && return 0
  say "near walltime with work remaining -- handing over to a fresh driver job"
  local n; n=$(SKIP_VALIDATION=1 sbatch --parsable --export=ALL "$0" 2>&1)
  say "successor driver job: $n"
  exit 0
}

say "=== driver start === config '$CFG_NAME' ($CFG_FLAGS), $NWORKERS workers"
say "continuation shards still to do: $(undone)/$NSHARDS"

# ── 1. Correctness gate ───────────────────────────────────────────────────────
if [ "${SKIP_VALIDATION:-0}" = 1 ]; then
  say "SKIP_VALIDATION=1 -- proceeding WITHOUT the correctness gate"
else
  VJ=$(CFG_NAME="$CFG_NAME" CFG_FLAGS="$CFG_FLAGS" sbatch --parsable --export=ALL \
        runs/gpqa25/validate_config.sh) || { say "ABORT: could not submit validation"; exit 1; }
  say "validation job $VJ submitted; waiting (~2 h once it starts)"
  while alive "$VJ"; do sleep 60; done
  # validate_config.sh exits with compare_configs.py's status, so the SLURM state IS
  # the verdict: COMPLETED = distributionally indistinguishable from the baseline.
  ST=$(sacct -j "$VJ" -n -X -o State | head -1 | tr -d ' ')
  say "validation job $VJ finished: $ST"
  grep -aE "VERDICT|PAIRED|mean difference|KV cache usage|preemption events|VALIDATION" \
      "$LOG_DIR/validate_${CFG_NAME}_shard00_job${VJ}.log" 2>/dev/null | tee -a "$LOG"
  if [ "$ST" != "COMPLETED" ]; then
    say "ABORT: validation did not pass -- the config changes the data, or the job failed."
    say "Nothing was resubmitted. Inspect the log above, then decide."
    exit 1
  fi
  say "validation PASSED -- the new config is indistinguishable from the baseline data"
fi

# ── 2. Clear the timed-out chain's jobs, which can never run ──────────────────
# Everything of mine pending right now is left over from the chain that timed out on
# 2026-09-22 (its stage 05 is DependencyNeverSatisfied and 06/07 sit behind it).
STALE=$(squeue -a -u "$USER" -h -t PD -o "%i" | tr '\n' ' ')
[ -n "$STALE" ] && { say "cancelling the timed-out chain's leftover jobs: $STALE"; scancel $STALE; sleep 5; }

# ── 3. Resume, and keep resuming across the 48 h cap ──────────────────────────
# Wait while anything can still make progress. A job PENDING with
# DependencyNeverSatisfied never will -- its parent already failed or timed out -- so
# waiting on those would hang here forever (they are cancelled and the round restarts).
# Progress is judged from the CONTINUATION queue and from jobs that can still run --
# never from "is anything in the queue", because a job PENDING with
# DependencyNeverSatisfied never will. Its parent already failed, and `afterok` is not
# satisfied by a later successful retry, so the whole tail of the chain has to be
# resubmitted rather than waited on. (Worker 4527210 failed a shard on 2026-09-26 and
# stranded stages 05-07 exactly this way.)
cont_jobs_live() { squeue -a -u "$USER" -h -t R,PD -o "%j %r" 2>/dev/null \
    | awk '$1 ~ /^continuations/ && $2 != "DependencyNeverSatisfied"' | grep -q .; }
stranded() { squeue -a -u "$USER" -h -o "%i %j %r" 2>/dev/null \
    | awk '$3=="DependencyNeverSatisfied" || $2 ~ /^(extract|gpqa25-labels|gpqa25-check)/ {print $1}' | tr '\n' ' '; }

for round in $(seq 1 "$MAX_ROUNDS"); do
  left=$(undone)
  if [ "$left" -gt 0 ] && ! cont_jobs_live; then
    say "--- round $round/$MAX_ROUNDS: $left continuation shard(s) left, no worker running -- submitting stage 04"
    START_AT=4 NWORKERS="$NWORKERS" runs/gpqa25/submit_all.sh 2>&1 | tee -a "$LOG"
    sleep 120
  fi
  while cont_jobs_live; do
    relay_if_near_walltime
    sleep 300
    say "  cont: $(ls "$QUEUE_ROOT/cont"/done.* 2>/dev/null | wc -l)/$NSHARDS shards, $(cat "$RUN_DIR"/raw/continuations.shard*.jsonl 2>/dev/null | wc -l) records, $(squeue -a -h -u "$USER" -t R -o '%j' | grep -c '^continuations') worker(s) running"
  done
  left=$(undone)
  say "--- round $round: no continuation worker left; $left shard(s) unfinished"
  [ "$left" -le 0 ] && break
  say "workers stopped early (48 h cap, or a shard failed) -- looping to resubmit"
done

if [ "$(undone)" -gt 0 ]; then
  say "GIVING UP after $MAX_ROUNDS rounds with $(undone) shard(s) unfinished -- needs a human"
  exit 1
fi

# ── 4. Continuations complete: rebuild the tail of the chain ──────────────────
# Stages 05-07 were submitted with `afterok` on workers that have since failed, so they
# can never start. Cancel them and resubmit from stage 05 (capture verification already
# passed for this run, so stage 03 is correctly skipped; 06 re-checks that artifact).
say "all $NSHARDS continuation shards done ($(cat "$RUN_DIR"/raw/continuations.shard*.jsonl 2>/dev/null | wc -l) records)"
S=$(stranded)
[ -n "$S" ] && { say "cancelling stranded downstream jobs: $S"; scancel $S; sleep 5; }
say "submitting labels -> extraction -> activation check"
START_AT=5 NWORKERS="$NWORKERS" runs/gpqa25/submit_all.sh 2>&1 | tee -a "$LOG"
sleep 60
# Exclude THIS job: the driver is itself in squeue, so matching on "any job that is
# not DependencyNeverSatisfied" would wait on itself forever (observed 2026-09-28,
# job 4549563 span after everything had finished).
tail_pending() { squeue -a -u "$USER" -h -o "%i %r" 2>/dev/null \
    | awk -v me="${SLURM_JOB_ID:-0}" '$1 != me && $2 != "DependencyNeverSatisfied"' | grep -q .; }
while tail_pending; do
  relay_if_near_walltime
  sleep 300
  say "  tail: $(squeue -a -h -u "$USER" -t R | wc -l) running, activation shards $(ls "$ACT_DIR"/activations.shard*.npy 2>/dev/null | wc -l)/$EXTRACT_SHARDS"
done

say "=== driver done ==="
say "state: cont $(ls "$QUEUE_ROOT/cont"/done.* 2>/dev/null | wc -l)/$NSHARDS shards, $(cat "$RUN_DIR"/raw/continuations.shard*.jsonl 2>/dev/null | wc -l) continuations"
say "check with: runs/gpqa25/status.sh"
