#!/bin/bash
# GEPA with Gemma-4-31B-it (task + reflection) on a local vLLM server. One job = one server (2 H100, TP=2)
# + the GEPA commands in $GEPA_CMDS, run in the repo against GEMMA_URL. Submit from the repo root:
#   source env.sh
#   GEPA_CMDS="envs/dspy/bin/python src/baselines/gepa_luna.py optimize --model gemma --all --rank --fold 0 --version ckpt" \
#     sbatch runs/gepa/gemma_job.sh
# Server log: runs/logs/gepa/vllm_job<id>.log; commands' output: runs/logs/gepa/gemma_job<id>.log
#SBATCH --job-name=gepa-gemma
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/logs/slurm/slurm-%j.out
set -uo pipefail
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"
export HF_HOME="$REPO_DIR/cache/huggingface" HF_TOKEN="${HF_TOKEN:-dummy}" HF_HUB_OFFLINE=1
export CUDA_HOME=/nas/sycamore/apps/cuda/13.2 PATH=/nas/sycamore/apps/cuda/13.2/bin:$PATH   # flashinfer JIT needs nvcc
export DSPY_CACHEDIR="$REPO_DIR/cache/dspy_gemma_job${SLURM_JOB_ID:-local}"   # one per job: a shared SQLite cache on NFS gets corrupted by jobs on two nodes
PORT=$((8000 + ${SLURM_JOB_ID:-0} % 1000)); export GEMMA_URL="http://localhost:$PORT/v1"
L=runs/logs/gepa; mkdir -p $L
envs/icl/bin/vllm serve google/gemma-4-31B-it --port $PORT --tensor-parallel-size 2 --max-model-len 32768 \
  --enable-prefix-caching --gpu-memory-utilization 0.92 --max-num-seqs 128 --max-num-batched-tokens 16384 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' > $L/vllm_job${SLURM_JOB_ID:-local}.log 2>&1 &
SERVER=$!
until curl -sf "$GEMMA_URL/models" > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "vLLM server died"; exit 1; }; sleep 15
done
echo "server up at $GEMMA_URL $(date)"
exec >> $L/gemma_job${SLURM_JOB_ID:-local}.log 2>&1
bash -c "${GEPA_CMDS:?}"
echo "commands exit $? $(date)"
kill $SERVER
