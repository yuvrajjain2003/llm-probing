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

j1=$(submit_workers "$D/01_base.sh");                                        echo "01 base (x$NWORKERS)       $j1"
j2=$(sub --dependency=afterok:"$j1" "$D/02_merge_base.sh");                  echo "02 merge base             $j2"
j3=$(sub --dependency=afterok:"$j2" "$D/03_verify_capture.sh");              echo "03 verify                 $j3"
j4=$(submit_workers "$D/04_continuations.sh" "afterok:$j2");                 echo "04 continuations (x$NWORKERS) $j4"
j5=$(sub --dependency=afterok:"$j4" "$D/05_merge_label.sh");                 echo "05 merge+labels           $j5"
j6=$(submit_workers "$D/06_extract.sh" "afterok:$j5:$j3");                   echo "06 extract (x$NWORKERS)       $j6"
j7=$(sub --dependency=afterok:"$j6" "$D/07_check_activations.sh");            echo "07 check activations      $j7"

echo
echo "watch:  squeue -u \$USER     progress:  runs/gpqa25/status.sh"
