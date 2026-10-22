#!/bin/bash
# Diagnostic, run once before anything else:  sbatch runs/gpqa25/00_gpu_check.sh
# Then read slurm-<jobid>.out in the directory you submitted from.
#
# Answers what the login node cannot: which GPU and how much memory, which driver
# (this decides whether EasySteer can leave the v0.17.1 pin -- see
# runs/goal2/setup_easysteer.sh), whether a compute node has outbound network
# (decides whether HF_HUB_OFFLINE=1 is required), node-local scratch size, and
# whether both python environments actually see the GPU.
#
# Walltime, CPUs and memory below are guesses, not measurements.
#SBATCH --job-name=gpqa25-gpucheck
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=runs/logs/slurm/slurm-%j.out
set -uo pipefail          # deliberately no -e: report everything, even past a failure
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"

echo "== node =="; hostname; date
echo; echo "== GPU =="
nvidia-smi 2>&1 | head -20
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv 2>&1
echo; echo "== what SLURM gave this job =="
scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oE "(Partition|QOS|NumCPUs|TRES|TresPerNode|TimeLimit)=[^ ]+" | sort -u
echo; echo "== node-local scratch (TMPDIR=${TMPDIR:-unset}) =="; df -h "${TMPDIR:-/tmp}" | tail -1
echo; echo "== outbound network from a compute node =="
curl -sS -o /dev/null -w "huggingface.co -> HTTP %{http_code}\n" --max-time 8 https://huggingface.co 2>&1 || echo "no outbound network (keep HF_HUB_OFFLINE=1)"
echo; echo "== storage as seen from a compute node =="
echo "WORK_DIR=$WORK_DIR"; ls -ld "$WORK_DIR" "$RUNS_ROOT" "$HF_HUB_CACHE" 2>&1
( touch "$RUNS_ROOT/.write_test" && rm -f "$RUNS_ROOT/.write_test" && echo "write to RUNS_ROOT: ok" ) 2>&1 || echo "write to RUNS_ROOT: FAILED"
echo; echo "== weights cached? =="
ls "$HF_HUB_CACHE" 2>&1 | head

probe() {   # probe <label> <python>
  echo; echo "== $1 ($2) =="
  [ -x "$2" ] || { echo "  not built yet"; return; }
  "$2" - <<'PY' 2>&1 | tail -12
import sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| torch cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}  {p.total_memory/2**30:.1f} GiB  sm_{p.major}{p.minor}")
    x = torch.randn(2048, 2048, device="cuda"); print("matmul ok:", float((x @ x).sum()) != 0.0)
import vllm; print("vllm", vllm.__version__)
try:
    from easysteer.hidden_states import get_all_hidden_states_generate  # noqa: F401
    print("easysteer capture API: importable")
except ImportError:
    print("easysteer: not installed in this env")
PY
}
probe "Goal 1 env"  "$PY"
probe "Goal 2 env"  "$PY_ES"
echo; echo "[done]"
