#!/usr/bin/env bash
# Progress of the SLURM GPQA run.  Run on the login node:  runs/gpqa25/status.sh
# (Replaces the repo-root status.sh, which was for the tmux-driven 3-CoT run.)
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_DIR/runs/gpqa25/common.sh"
R="$RUN_DIR/raw"
lines() { cat "$@" 2>/dev/null | wc -l; }

echo "=================== GPQA-diamond, $N_COTS CoTs x 5 ckpts x K=$K  ($RUN_TAG) ==================="
echo "run dir: $RUN_DIR"
echo
echo "--- my jobs ---"
squeue -a -u "$USER" -h -o "%.10i %.9P %.18j %.3t %.10M %.10l %R" 2>/dev/null | head -30
[ -z "$(squeue -a -u "$USER" -h 2>/dev/null)" ] && echo "  (none queued or running)"
echo
echo "--- base CoTs:       $(lines "$R"/sample_records.shard*.jsonl) / $((N_EXAMPLES * N_COTS))"
np=$(lines "$R"/checkpoints.shard*.jsonl)
echo "--- prefixes:        $np  (up to $((N_EXAMPLES * N_COTS * 5)); fewer where a CoT is truncated)"
echo "--- continuations:   $(lines "$R"/continuations.shard*.jsonl) / $((np * K))   (target grows as shards write their prefixes)"
if [ -f "$RUN_DIR/labels/checkpoint_labels.csv" ]; then
  echo "--- labels:          $(($(lines "$RUN_DIR/labels/checkpoint_labels.csv") - 1)) prefixes labelled"
else
  echo "--- labels:          not built yet"
fi
# Worker queues (run_queue in common.sh): a shard is done once its done.<n> marker exists.
qs() { local n; n=$(ls "$QUEUE_ROOT/$1"/done.* 2>/dev/null | wc -l); \
       echo "$n / $2 shards done, $(ls -d "$QUEUE_ROOT/$1"/lock.* 2>/dev/null | wc -l) running"; }
echo "--- queues:          base $(qs base "$NSHARDS") | cont $(qs cont "$NSHARDS") | extract $(qs extract "$EXTRACT_SHARDS")"
na=$(ls "$ACT_DIR"/activations.shard*.npy 2>/dev/null | wc -l)
echo "--- activation shards: $na / $EXTRACT_SHARDS  ($ACT_DIR)"
[ -f "$VERIFY_DIR/cmp.txt" ] && echo "--- capture verify:  $(grep -E 'VERDICT' "$VERIFY_DIR/cmp.txt")"
echo
echo "--- errors in logs (vLLM KV/swap/OOM/disk, tracebacks) ---"
hits=$(grep -lE "No space left on device|lack of CPU swap space|CUDA out of memory|Traceback|ERROR:" "$LOG_DIR"/*.log 2>/dev/null | head -8)
if [ -n "$hits" ]; then echo "$hits"; else echo "  none"; fi
echo
echo "--- disk ---"
du -sh "$RUN_DIR" "$ACT_DIR" 2>/dev/null
df -h "$RUNS_ROOT" | tail -1
echo "=============================================================="
