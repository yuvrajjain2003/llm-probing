# Cluster bootstrap

Goal: get from `git clone` to "can run the pipeline". Budget roughly an hour,
most of it downloads and the EasySteer build.

## 0. The critical fact: you need TWO python environments

They cannot be merged, and picking the wrong one is the most common way to waste
an afternoon here.

| | Goal 1 env (`$PY`) | Goal 2 env (`$PY_ES`) |
|---|---|---|
| Used for | base CoTs, continuations, **all CPU analysis and probing** | activation capture **only** |
| vLLM | 0.5.5 | 0.17.1 fork (`vllm-steer`) |
| torch | 2.4.0+cu121 | 2.10.0+cu128 |
| python | 3.10.12 | 3.10.12 (the tag requires 3.10) |
| Built by | ordinary pip install | `runs/goal2/setup_easysteer.sh` |
| Freeze | `env/requirements-goal1.txt` | `env/requirements-goal2.txt` |

Only `src/probing/extract_activations.py`, `src/probing/extract_checkpoints.py`
and `src/probing/verify_capture.py --stage es` need `$PY_ES`. Everything else —
including `probe_checkpoints.py` — runs under `$PY`.

## 1. Environment variables

`env.sh` derives the repo root from its own location and takes every value from
the environment if set, so **a clone needs no edits to it**. Override what
differs on your cluster, then source it:

```bash
export PY=/path/to/goal1-venv/bin/python     # optional; see resolution order below
export HF_HUB_CACHE=/scratch/$USER/hf/hub    # put weights on fast local scratch
source env.sh
```

`$PY` resolves to, in order: whatever you exported → `$ENVS_DIR/goal1/bin/python`
(built by `env/setup_goal1.sh`) → `./.venv-goal1/bin/python` → the original box's
`/playpen-ssd/yuvraj/.venv/bin/python` → `python3` on PATH. `$PY_ES` likewise
prefers `$ENVS_DIR/easysteer`.

All bulk paths hang off `$WORK_DIR` (default: the user's `/work/users/<u>/<s>/<user>`
if it exists, else a gitignored `./.work`): `$ENVS_DIR`, `$CACHE_ROOT` (HF weights,
vLLM/Triton/inductor caches) and `$RUNS_ROOT` (run outputs). `env.sh` exports every
cache variable `run_eval.py` used to default to `/playpen-ssd/...` at import time.

`HF_HUB_OFFLINE` defaults to `0` (it was `1` on the original box, whose weights
were pre-cached). The SLURM stage scripts in `runs/gpqa25/` force it to `1`.

`OUTLINES_CACHE_DIR` defaults to `$CACHE_ROOT/outlines_$USER`. Keep it per-user: the
outlines cache is a sqlite DB the importing process must be able to **write**,
and a cache dir left by another user makes plain `import vllm` die with
`sqlite3.OperationalError: attempt to write a readonly database`. This bit the
original box, which is why its pipeline scripts all set the variable by hand.

## 2. Goal 1 environment

```bash
env/setup_goal1.sh        # conda python 3.10 -> $ENVS_DIR/goal1 (login node; no GPU needed)
```

(Was `python3.10 -m venv .venv-goal1` + `pip install -r env/requirements-goal1.txt`. Sycamore's
system python is 3.12 and only ships python as modules, so the script builds the env with
`module load anaconda`. It installs `env/requirements-goal1.in`, **not** the freeze -- the
freeze is not installable, and PyPI's `pyairports` breaks `import vllm`; both are explained
in [`GOTCHAS.md`](GOTCHAS.md).)

The freeze pins `vllm==0.5.5` / `torch==2.4.0+cu121`, which needs a CUDA 12.1-
compatible driver. If your cluster is newer and 0.5.5 has no usable wheel, the
generation code is not tied to that version — but re-validate `run_eval.py`'s
sampling path before trusting new numbers against old ones, and say so in the
run config.

## 3. Datasets

Not tracked (~89 MB). Re-fetch into `data/`:

```bash
source env.sh && "$PY" src/setup/download_dataset.py
```

Downloads AquaRAT, OpenBookQA, MMLU college/pro math, GPQA-diamond, AIME2025,
AIME 2022–2025, MATH500 as `save_to_disk` copies. `run_eval.load_examples` reads
these local copies, **not** the HF cache — so this step is mandatory even if the
HF cache is warm. Needs network and, for GPQA, an accepted HF gate on your token.

## 4. Model weights

```
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     # everything so far
deepseek-ai/DeepSeek-R1-Distill-Llama-8B    # in scope, never run
```

~14 GB fp16 each. `env/fetch_models.sh` pulls them into `$HF_HUB_CACHE` (login node); the jobs run offline.

## 5. Goal 2 environment (only when you need activations)

`runs/goal2/setup_easysteer.sh` is the recipe. It now derives the repo from its own
location, builds a conda python 3.10 env at `$ENVS_DIR/easysteer` and clones the
EasySteer source to `$WORK_DIR/src/easysteer`, and **refuses to overwrite** either
unless `FORCE=1` (it used to `rm -rf` them unconditionally). Run it on the login node.

It pins EasySteer `v0.17.1-Preview` deliberately: EasySteer `main` pins vLLM
0.26.0, whose wheels are built against CUDA 13. The original driver (555.42.06)
capped at CUDA 12.5, so those kernels could not launch —
`libcudart.so.13: cannot open shared object file`. vLLM moved to CUDA 13 at
0.22, so v0.17.1 (vLLM 0.17.1, torch 2.10.0 → cuda-runtime-cu12 12.8.90) was the
newest tag that ran.

**If your new cluster has a CUDA 13-capable driver, that constraint is gone and
a newer EasySteer may be preferable.** Check `nvidia-smi` first. If you do move
tags, re-run the capture verification in step 6 — the layer/position conventions
it pins down are version-specific, and they are silently wrong when they change.

`VLLM_PRECOMPILED_WHEEL_COMMIT` in that script must match the tag, or pip
compiles CUDA kernels from source (very slow) or links the wrong ABI.

## 6. Verify before trusting anything

```bash
source env.sh
"$PY"    src/probing/verify_capture.py --stage hf      # stock HF reference
"$PY_ES" src/probing/verify_capture.py --stage es      # EasySteer capture
"$PY"    src/probing/verify_capture.py --stage cmp     # cosine-similarity compare
```

Cross-checks EasySteer's hidden-state capture against stock HF transformers on
identical input. On the original box this matched at **min cosine 0.9969 across
layers 0–26**. Anything materially below that means the capture conventions
shifted and every downstream activation is suspect. Run this after any change to
the EasySteer version, the vLLM version, or the model.

Then the cheap end-to-end smoke test, which touches every stage on a handful of
questions before you commit GPU-days:

```bash
"$PY" src/generation/run_eval.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
      --dataset gpqa_diamond --output-dir runs/smoke \
      --num-examples 4 --samples-per-example 3 --max-tokens 4096
```

## 7. Cluster-specific things to re-tune

The batching constants in the launcher scripts were fitted to 8× A6000 48 GB and
are commented with the measurement that produced them. They are **not** portable
to different GPUs — re-measure rather than inheriting them:

- `--max-num-seqs 128` — 1.93× faster than the pilot's 48 on A6000.
- `--max-n-per-request 20` — K=100 as one `n=100` group needs 2.3× the 420k-token
  KV cache; vLLM then swaps to CPU and the GPU sits at 0%.
- `MAXN=8` for GPQA at a 16384-token budget — a 20-wide group needs ~1.17× the KV
  cache and vLLM dies on CPU swap. This is what killed the first AIME run.
- `--gpu-memory-utilization 0.85` (generation) / `0.55` (checkpoint extraction).
- Sharding is **data-parallel, TP=1 per worker** — the 7B is 14 GB fp16 and fits
  one A6000, so tensor-parallelism would only add an all-reduce per layer. On
  larger-memory GPUs, run more shards per node rather than raising TP.

`runs/goal1/full/bench_maxnumseqs.sh` is the benchmark that produced the 128.

## 8. Sycamore (UNC) — how the above maps onto this cluster

Facts below were read off the cluster on 2026-09-18, not assumed. Re-check anything
marked *unverified*.

**Storage.** `$HOME` is NFS with a ~71 GB / 1M-file quota — code and small results
only. `/work/users/<u>/<s>/<user>` is the bulk area (no per-user quota enabled;
*purge/backup policy unverified* — treat as scratch and copy labels + activations,
~5 GB, back to home once produced). `/proj` and `/datacommons` are group project
storage; an allocation is being arranged with the PI (*not confirmed*).

**Compute is SLURM only; the login node has no GPU.** The GPU partitions are hidden
from a bare `sinfo` — use `sinfo -a`. `h100_all` (4 nodes × 4 H100, 256 CPUs, ~1.5 TB
RAM each, 5-day max at submit; we cap jobs at 48 h) works under the default `normal` QoS. `h100` and the `h100*` QoS
entries are **not** granted to this account ("Invalid qos specification"). CPU-only
jobs must use `small` (`batch` requires ≥ 48 CPUs per job).

**Verified on a GPU node (job 4413892, 2026-09-20, 1 min 45 s):** NVIDIA **H100 80GB HBM3**
(79.2 GiB usable, sm_90), **driver 595.71.05, CUDA 13.2** — so the CUDA-12.5 driver cap that
forced the EasySteer v0.17.1 pin on the original box does not apply here (the pin still works
and stays; moving tags means re-running `03_verify_capture.sh`). Compute nodes **have outbound
network** (huggingface.co → 200), though the stage scripts still run offline. Node-local
`/tmp` is 202 GB; SLURM sets a per-job `$TMPDIR`. `/work` and the weights are visible and
writable from compute nodes. **Both environments see the GPU and run a CUDA matmul**: goal1
(torch 2.4.0+cu121, vllm 0.5.5) and capture (torch 2.10.0+cu128, easysteer importable).
**End-to-end smoke test passed (job 4413898, 2026-09-20, 19 min 28 s, `runs/gpqa25/00_smoke.sh`)** — 4
questions × 2 CoTs × 5 checkpoints × K=3 in 2 shards through *every* stage and gate. vLLM 0.5.5
loads and generates on the H100; capture verification PASS with the alignment
`extract_checkpoints.py` assumes (min cosine 0.9957 over layers 0–26 vs 0.9969 on the original
box); 40 prefixes each with exactly K continuations; 40 activation vectors, 28 layers × 3584,
fp16, finite, 1:1 with the labels. Stage wall times (incl. vLLM load, ~1–3 min per shard):
base 233 s, verify 78 s, continuations 631 s, extraction 192 s. **Goal-1 KV cache is 57,717
blocks ≈ 923k tokens at `gpu_util` 0.85 (2.2× the A6000s' ~420k)**, so `MAXN=8` /
`MAX_NUM_SEQS=128` have headroom — untested; raise only with the real run's numbers. Also:
`goal2/<tag>/_verify` holds ~0.9 GB of reference activations (safe to delete afterwards).

**Not measured, and it decides the schedule: saturated generation throughput.** The smoke run
had ≤ 60 sequences in flight and included model loads, giving ~570 tok/s — a floor, not an
estimate. Continuations are ~98% of the tokens: 618,750 × ~3,000 reasoning tokens ≈ 1.9 B
(mean from only 40 prefixes, so ± a lot). One 48 h round with 4 workers needs ~2,700 tok/s per
GPU; with 8 workers ~1,350. The first real base stage (128 sequences in flight) gives the
first honest throughput reading. More workers can be added any time (`sbatch` another copy of
`04_continuations.sh`; the queue makes it safe). Still *unverified*: whether `h100_all` preempts.

**The old launchers assume a box you own** (`CUDA_VISIBLE_DEVICES=$i`, tmux,
`gpu_pool.py` scanning `/proc`). They still work after the path fixes but are not for
SLURM. The SLURM equivalents live in `runs/gpqa25/`, which encodes the agreed GPQA
design (25 CoTs/question, 5 checkpoints, no 0%, K=25, labels + activations only).

**First-time order** (login node unless noted):

```bash
source env.sh
env/setup_goal1.sh                           # goal-1 env
runs/goal2/setup_easysteer.sh                # capture env
"$PY" src/setup/download_dataset.py gpqa_diamond   # needs an accepted HF gate + token
env/fetch_models.sh                          # ~15 GB into $HF_HUB_CACHE
sbatch runs/gpqa25/00_gpu_check.sh           # read slurm-<id>.out before going on
runs/gpqa25/submit_all.sh                    # DRY_RUN=1 to preview the dependency chain
runs/gpqa25/status.sh
```

`submit_all.sh` chains: base CoTs (workers) → merge+count check → {capture
verification ‖ continuations (workers)} → merge+labels+completeness check →
extraction (workers). SLURM runs a *copy* of each script, so the stages find the repo
through `REPO_DIR` from `source env.sh`, never via `$0`.

**Long worker jobs, not job arrays.** On this scheduler fair-share is charged **per job**
(a colleague's finding, 2026-09-20, consistent with the config: `PriorityWeightFairShare`
is 10000, every other factor ≤ 1000), so one long job carries far more priority than
several short ones — and every array task is its own job. Each GPU stage is therefore
`NWORKERS` (default 4) long **single-GPU** jobs that pull shard indices from a queue
(`run_queue` in `runs/gpqa25/common.sh`; `done.<n>` markers and atomic `mkdir` locks under
`$RUNS_ROOT/queue/`) until none are left, then exit and free the GPU. Walltime is **48 h**:
`h100_all` accepts up to 5 days at submission but a colleague reports jobs beyond 2 days get
timed out. Workers are 1 GPU because on 2026-09-20 a 1-GPU job was projected to start in
~14 h and a whole 4-GPU node only in ~2.9 days. If a stage needs more than 48 h, workers
time out with shards unfinished: stale locks are released by the next worker and
interrupted shards resume by count, so rerunning `submit_all.sh` loses nothing.
`NWORKERS=8 runs/gpqa25/submit_all.sh` for more GPUs.

`03_verify_capture.sh` is a stricter gate than `verify_capture.py`'s own verdict: that
script exits 0 regardless and only checks that *some* alignment scores > 0.99, so the
job additionally requires the alignment `extract_checkpoints.py` hard-codes.

**Walltimes, CPU and memory in the `#SBATCH` lines are guesses** — no H100 timing
exists yet. The vLLM knobs (`MAXN`, `MAX_NUM_SEQS`, `GPU_UTIL`) default to the values
measured on 48 GB A6000s, which are known-safe but not tuned for H100; time one shard
and raise them.
