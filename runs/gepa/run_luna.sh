#!/bin/bash
# GEPA on the zero-shot Luna prompt: for one prompt version, optimise on each outer fold (in parallel),
# then score the seed and the optimised prompt on that fold's test rows. API calls only: runs on the
# login node. Resumable: rerun the same command. Usage (repo root):
#   nohup bash runs/gepa/run_luna.sh ckpt  > runs/logs/gepa/run_ckpt.log  2>&1 &
#   nohup bash runs/gepa/run_luna.sh blind > runs/logs/gepa/run_blind.log 2>&1 &
#   nohup bash runs/gepa/run_luna.sh ckpt --pct > runs/logs/gepa/run_ckpt_pct.log 2>&1 &   (percentile metric)
# then: source env.sh; $PY src/baselines/gepa_luna.py score --version <v>
set -u
V=$1
X="${*:2}"          # extra flags, e.g. --pct (logs then get a _pct suffix)
L=${V}${X:+_${X#--}}
export DSPY_CACHEDIR=$PWD/cache/dspy
PYD=envs/dspy/bin/python
for f in 0 1 2 3 4; do
  ( $PYD src/baselines/gepa_luna.py optimize --fold $f --version $V $X --threads 16 \
      > runs/logs/gepa/opt_${L}_fold$f.log 2>&1 \
    && $PYD src/baselines/gepa_luna.py test --fold $f --version $V $X --which seed gepa --threads 16 \
      >> runs/logs/gepa/opt_${L}_fold$f.log 2>&1
    echo "fold $f exit $?" ) &
done
wait
