#!/usr/bin/env bash
# Submit the whole GPQA run as one dependency chain:
#
#   01 base CoTs (workers) -> 02 merge+check -+-> 04 continuations (workers) -> 05 merge+labels -+
#                                             |                                                  +-> 06 extract (workers) -> 07 check activations
#                                             +-> 03 verify capture (1 GPU) --------------------+
#
#   cd <repo> && source env.sh && runs/gpqa25/submit_all.sh
#   DRY_RUN=1 runs/gpqa25/submit_all.sh          # print the plan, submit nothing
#   NWORKERS=8 runs/gpqa25/submit_all.sh         # more parallel GPUs (default 4)
#   START_AT=4 runs/gpqa25/submit_all.sh         # resume from the continuation stage
#   ENGINES_PER_GPU=2 runs/gpqa25/submit_all.sh  # engine processes per GPU (default 1)
#
# The GPU stages are NWORKERS long single-GPU jobs each (fair-share is charged per job,
# so few long jobs beat many short ones), pulling shards from a queue until none are
# left; walltime is capped at 48 h per job. If a stage needs more than 48 h of work,
# workers time out with their shards unfinished and the chain stalls in
# DependencyNeverSatisfied: `scancel` the waiting jobs, rerun this script, and finished
# shards are skipped while interrupted ones resume by count -- nothing is regenerated.
# Watch progress with runs/gpqa25/status.sh.
set -euo pipefail
: "${REPO_DIR:?run 'source env.sh' in the repo first}"
D="$REPO_DIR/runs/gpqa25"
NWORKERS="${NWORKERS:-4}"

sub() {
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "  would: sbatch $*" >&2; echo "<jobid>"; else sbatch --parsable "$@"; fi
}
# submit_workers <script> [dependency-spec]  ->  prints the worker job ids joined by ':'
submit_workers() {
  local script="$1" dep="${2:-}" ids=() i
  for i in $(seq 1 "$NWORKERS"); do
    if [ -n "$dep" ]; then ids+=("$(sub --dependency="$dep" --job-name="$(basename "$script" .sh | sed 's/^[0-9]*_//')-w$i" "$script")")
    else                   ids+=("$(sub --job-name="$(basename "$script" .sh | sed 's/^[0-9]*_//')-w$i" "$script")")
    fi
  done
  (IFS=:; echo "${ids[*]}")
}

# START_AT skips stages already finished, e.g. after a 48 h timeout in the middle of
# the continuation stage:  START_AT=4 runs/gpqa25/submit_all.sh
# This matters on a busy partition: re-submitting stage 01 would queue NWORKERS GPU
# jobs that find every shard already done and exit in seconds -- but stage 04 depends
# on them, so the real work would wait behind GPU allocations that do nothing.
START_AT="${START_AT:-1}"
dep() { [ -n "$1" ] && echo "--dependency=afterok:$1" || echo ""; }

j1=""; j2=""; j3=""; j4=""; j5=""; j6=""; j7=""
if [ "$START_AT" -le 1 ]; then
  j1=$(submit_workers "$D/01_base.sh");                                      echo "01 base (x$NWORKERS)       $j1"
fi
if [ "$START_AT" -le 2 ]; then
  j2=$(sub $(dep "$j1") "$D/02_merge_base.sh");                              echo "02 merge base             $j2"
fi
if [ "$START_AT" -le 3 ]; then
  j3=$(sub $(dep "$j2") "$D/03_verify_capture.sh");                          echo "03 verify                 $j3"
fi
if [ "$START_AT" -le 4 ]; then
  j4=$(submit_workers "$D/04_continuations.sh" "${j2:+afterok:$j2}");        echo "04 continuations (x$NWORKERS) $j4"
fi
if [ "$START_AT" -le 5 ]; then
  j5=$(sub $(dep "$j4") "$D/05_merge_label.sh");                             echo "05 merge+labels           $j5"
fi
if [ "$START_AT" -le 6 ]; then
  # Extraction needs BOTH the labels stage and a passing capture verification. If
  # stage 03 was skipped because it already passed, depend on stage 05 alone.
  j6=$(submit_workers "$D/06_extract.sh" "afterok:${j5}${j3:+:$j3}");        echo "06 extract (x$NWORKERS)       $j6"
fi
if [ "$START_AT" -le 7 ]; then
  j7=$(sub $(dep "$j6") "$D/07_check_activations.sh");                       echo "07 check activations      $j7"
fi

echo
echo "watch:  squeue -u \$USER     progress:  runs/gpqa25/status.sh"
