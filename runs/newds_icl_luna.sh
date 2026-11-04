#!/bin/bash
# Frontier-model ICL (GPT-6 Luna) on a NEW dataset. Login node, no GPU, no Slurm.
# Written 2026-11-01 (P3.1 of docs/plan_newdatasets.md). Resumable: re-running skips finished rows.
#
#   DS=mmlupro_v1  bash runs/newds_icl_luna.sh plan            # NO API calls: tokens + cost
#   DS=mmlupro_v1  bash runs/newds_icl_luna.sh run             # SPENDS MONEY
#   DS=mmlupro_v1  bash runs/newds_icl_luna.sh score
#
# Knobs:  DS (mmlupro_v1 | globalop_v1 | gpqa25), NQ (questions kept, 0 = all), WORKERS,
#         SPEC (version:k pairs, run in this order), BUDGET (hard stop PER SPEC, so worst case is
#         BUDGET x the number of specs).
# SPEC's default is the round-5/6 priority order: zero-shot first (best setting on GPQA), then k=10 as
# the "examples hurt" control, then blind k=0 for the hidden-depth test. Deliberately NOT all 8.
set -u
cd "$(dirname "$0")/.." || exit 1
CMD=${1:?plan|run|score}
DS=${DS:-mmlupro_v1}
NQ=${NQ:-500}
BUDGET=${BUDGET:-6}
SPEC=${SPEC:-ckpt:0 ckpt:10 blind:0}
WORKERS=${WORKERS:-8}
PYBIN=${PYBIN:-$PWD/envs/goal1/bin/python}      # openai 2.37.0 + tiktoken 0.13.0; the env that ran the
                                                # published GPQA Luna runs. Do NOT rely on env.sh's $PY.
case $DS in
  mmlupro_v1)  SUB=mmlu_pro;        NAME=MMLU-Pro;        NC="40 or 100"; NCOTS=5;  DOMCOL=category ;;
  globalop_v1) SUB=global_opinions; NAME=GlobalOpinionQA;  NC="30 or 100"; NCOTS=5;  DOMCOL=source ;;
  gpqa25)      SUB=gpqa_diamond;    NAME=GPQA-Diamond;     NC="25";        NCOTS=25; DOMCOL="High-level domain" ;;
  *) echo "unknown DS=$DS"; exit 1 ;;
esac

# ${VAR:-...} throughout, so an outer `export ICL_OUT=...` wins. ICL_OUT especially: pointing a
# gpqa25 plan run at runs/goal2/gpqa25/icl writes dryprompt/plan files into the published directory.
export ICL_RUN_DIR="${ICL_RUN_DIR:-runs/goal1/$DS/$SUB/DeepSeek-R1-Distill-Qwen-7B}"
export ICL_F0_INPUT="${ICL_F0_INPUT:-runs/goal2/$DS/_f0_input}"
export ICL_OUT="${ICL_OUT:-runs/goal2/$DS/icl}"
export ICL_DATASET_NAME="${ICL_DATASET_NAME:-$NAME}"
export ICL_N_CONT="${ICL_N_CONT:-$NC}"
export ICL_N_COTS="${ICL_N_COTS:-$NCOTS}"
export ICL_DOMAIN_COL="${ICL_DOMAIN_COL:-$DOMCOL}"
export ICL_REF_SWEEP="${ICL_REF_SWEEP:-runs/goal2/$DS/sweep/ref_fixed}"
if [ "$DS" = gpqa25 ]; then
  export ICL_DOMAIN_CSV="${ICL_DOMAIN_CSV:-data/gpqa_diamond/csv/train.csv}"   # not in examples.csv
else
  export ICL_DOMAIN_CSV="${ICL_DOMAIN_CSV:-$ICL_RUN_DIR/raw/examples.csv}"
fi
export HF_TOKEN=dummy                                          # cache/huggingface/token is unreadable

mkdir -p runs/logs/newds
LOG="runs/logs/newds/icl_luna_${DS}_${CMD}_$(date +%Y%m%d_%H%M%S).log"
{
  echo "=== $(date) $CMD DS=$DS NQ=$NQ BUDGET=\$$BUDGET/spec SPEC='$SPEC' ==="
  for s in $SPEC; do
    v=${s%%:*}; k=${s##*:}
    echo "--- $(date) $v k=$k ---"
    "$PYBIN" -I src/baselines/icl_openai.py "$CMD" --model gpt-6-luna --full --n-questions "$NQ" \
        --version "$v" --k "$k" --workers "$WORKERS" --budget "$BUDGET"
    rc=$?                                                      # own line: $? inside $( ) is the subshell's
    echo "--- $(date) $v k=$k EXIT CODE $rc ---"
  done
  echo "--- rows per config (the only honest completion signal; the log tail is not) ---"
  wc -l "$ICL_OUT"/raw/gpt6_luna_high*_k*.jsonl 2>/dev/null
} 2>&1 | tee "$LOG"
echo "log: $LOG"
