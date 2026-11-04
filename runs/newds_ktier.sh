#!/bin/bash
#SBATCH --job-name=nds-ktier
#SBATCH --output=runs/logs/slurm/slurm-%j.out
#SBATCH -A rc_ssriva_pi
#SBATCH -q normal
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
date
./envs/goal1/bin/python src/probing/newds_ktier.py --tag "$TAG" --ds "$DS" --cut "${CUT:-0.5}"
rc=$?
date
echo "EXIT CODE $rc"
