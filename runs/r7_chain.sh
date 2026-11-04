#!/bin/bash
# Round 7 item 7: submit the whole post-capture DAG for one dataset, with Slurm
# dependencies, so nothing needs babysitting.
#   cd <repo> && source env.sh && ./runs/r7_chain.sh <tag> <capture job ids, colon separated> <mem for the GPU arm>
set -euo pipefail
TAG="$1"; DEP="$2"; GMEM="${3:-200G}"
VAR=full; SUB=activations_tokens
STATES=/work/users/y/u/yuvraj/llm-probing/runs/goal2/$TAG/r7_states_$VAR
E="R7_TAG=$TAG R7_VARIANT=$VAR R7_TOK_SUB=$SUB"
sub() { eval "$1" sbatch --parsable "${@:2}" runs/r7_pooling.sh; }

J1=$(sub "$E R7_MODE=pool" --dependency=afterok:$DEP -p small -c 8 --mem=96G -t 06:00:00)
echo "pool            $J1  (after $DEP)"
J2=$(sub "$E R7_MODE=ridge" --dependency=afterok:$J1 -p small -c 16 --mem=96G -t 2-00:00:00)
echo "ridge           $J2"
JA=""
for P in last mean max; do
  j=$(sub "$E R7_MODE=adam R7_POOLING=$P R7_EPOCHS=80 R7_LOG_EVERY=20" \
      --dependency=afterok:$J1 -p small -c 8 --mem=64G -t 1-00:00:00)
  echo "adam $P        $j"; JA="$JA:$j"
done
J3=$(sub "$E R7_MODE=adam R7_POOLING=attn R7_EPOCHS=80 R7_LOG_EVERY=20 R7_STATES=$STATES" \
    --dependency=afterok:$J1 -p h100_all --gres=gpu:h100:1 -c 8 --mem=$GMEM -t 12:00:00)
echo "adam attn (GPU) $J3"
J4=$(sub "$E R7_MODE=attnridge R7_STATES=$STATES" \
    --dependency=afterok:$J3 -p small -c 16 --mem=96G -t 1-00:00:00)
echo "attn_ridge      $J4"
J5=$(sub "$E R7_MODE=table" --dependency=afterany:$J2$JA:$J4 -p small -c 8 --mem=48G -t 06:00:00)
echo "table           $J5"
