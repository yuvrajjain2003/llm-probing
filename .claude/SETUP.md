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

`$PY` resolves to, in order: whatever you exported → `./.venv-goal1/bin/python`
→ the original box's `/playpen-ssd/yuvraj/.venv/bin/python` → `python3` on PATH.
On a new cluster, create `./.venv-goal1` and it is picked up automatically.

`HF_HUB_OFFLINE` now defaults to `0` (it was `1` on the original box, whose
weights were pre-cached). Leave it at 0 until the models have been pulled once.

`OUTLINES_CACHE_DIR` defaults to `.outlines_cache_$USER`. Keep it per-user: the
outlines cache is a sqlite DB the importing process must be able to **write**,
and a cache dir left by another user makes plain `import vllm` die with
`sqlite3.OperationalError: attempt to write a readonly database`. This bit the
original box, which is why its pipeline scripts all set the variable by hand.

## 2. Goal 1 environment

```bash
python3.10 -m venv .venv-goal1
.venv-goal1/bin/pip install --upgrade pip wheel setuptools
.venv-goal1/bin/pip install -r env/requirements-goal1.txt
```

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

~14 GB fp16 each. Pull once into `$HF_HUB_CACHE`, then set `HF_HUB_OFFLINE=1`.

## 5. Goal 2 environment (only when you need activations)

`runs/goal2/setup_easysteer.sh` is the recipe. **Read it before running it** —
it begins with `rm -rf .venv-easysteer runs/goal2/_src` and has the original
box's absolute path hardcoded in its `cd`. Change that `cd` to the clone path.

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
