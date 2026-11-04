#!/bin/bash
#SBATCH --job-name=newds-sweep
#SBATCH --output=runs/logs/slurm/slurm-%j.out
#SBATCH -A rc_ssriva_pi
#SBATCH -q normal
set -euo pipefail
PYBIN=./envs/goal1/bin/python
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
echo "host $(hostname) cpus $OMP_NUM_THREADS config $CFG shard ${SHARD:-0/1} extra ${EXTRA:-}"
date
$PYBIN src/probing/sweep.py --config "$CFG" --shard "${SHARD:-0/1}" ${EXTRA:-}
rc=$?
date
echo "EXIT CODE $rc"
