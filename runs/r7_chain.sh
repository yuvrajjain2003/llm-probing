#!/bin/bash
# Round 7 item 7: submit the whole post-capture DAG for one dataset, with Slurm
# dependencies, so nothing needs babysitting.
#
#   cd <repo> && source env.sh
#   ./runs/r7_chain.sh <tag> <capture job ids, colon separated> <variant> <tok sub> [cot_max] [mem]
#
# The `pool` stage runs the exhaustive bitwise correctness gate and exits non-zero
# if it fails, and every later stage is `afterok`, so a failed gate BLOCKS the whole
# chain rather than letting a probe result be believed.
#
# The attention arm runs on CPU (`small`), not GPU: it is memory-bandwidth bound, not
# compute bound, `small` schedules immediately, and that keeps h100_all off the
# critical path for everything except the capture itself.
set -euo pipefail
TAG="$1"; DEP="$2"; VAR="${3:-full}"; SUB="${4:-activations_tokens}"
CM="${5:-}"; GMEM="${6:-96G}"; EP="${7:-60}"
# EP: epoch budget for the Adam arms. Selection is best-validation-epoch, so a smaller
# budget is a smaller search, not a biased one. The GPQA pilot plateaued by epoch 30
# (best val 0.7512 at 30, 0.7503 at 50), which is why 30 is enough on the big datasets.
STATES=/work/users/y/u/yuvraj/llm-probing/runs/goal2/$TAG/r7_states_$VAR
E="R7_TAG=$TAG R7_VARIANT=$VAR R7_TOK_SUB=$SUB"
sub() { eval "$1" sbatch --parsable "${@:2}" runs/r7_pooling.sh; }

J1=$(sub "$E R7_MODE=pool ${CM:+R7_COT_MAX=$CM}" --dependency=afterok:$DEP \
     -p small -c 8 --mem=96G -t 04:00:00)
echo "pool + GATE      $J1  (after $DEP)"
J2=$(sub "$E R7_MODE=ridge" --dependency=afterok:$J1 -p small -c 16 --mem=96G -t 1-00:00:00)
echo "ridge            $J2"
JA=""
for P in last mean max; do
  j=$(sub "$E R7_MODE=adam R7_POOLING=$P R7_EPOCHS=$EP R7_LOG_EVERY=20" \
      --dependency=afterok:$J1 -p small -c 8 --mem=64G -t 1-00:00:00)
  echo "adam $P         $j"; JA="$JA:$j"
done
J3=$(sub "$E R7_MODE=adam R7_POOLING=attn R7_EPOCHS=$EP R7_LOG_EVERY=20 R7_STATES=$STATES" \
    --dependency=afterok:$J1 -p small -c 48 --mem=$GMEM -t 2-00:00:00)
echo "adam attn        $J3"
J3b=$(sub "$E R7_MODE=adam R7_POOLING=meantok R7_EPOCHS=$EP R7_LOG_EVERY=20" \
     --dependency=afterok:$J1 -p small -c 48 --mem=$GMEM -t 2-00:00:00)
echo "adam meantok     $J3b"
J4=$(sub "$E R7_MODE=attnridge R7_STATES=$STATES" \
    --dependency=afterok:$J3 -p small -c 16 --mem=96G -t 1-00:00:00)
echo "attn_ridge       $J4"
J5=$(sub "$E R7_MODE=attnstats R7_STATES=$STATES" \
    --dependency=afterok:$J3 -p small -c 2 --mem=16G -t 01:00:00)
echo "attn stats       $J5"
J6=$(sub "$E R7_MODE=loglen R7_STATES=$STATES" \
    --dependency=afterok:$J3 -p small -c 16 --mem=96G -t 1-00:00:00)
echo "loglen raw+log1p $J6"
J7=$(sub "$E R7_MODE=table" --dependency=afterany:$J2$JA:$J3b:$J4 \
    -p small -c 8 --mem=48G -t 06:00:00)
echo "table            $J7"
