# Things that will silently give you wrong numbers

Each of these cost real debugging time. None of them fail loudly.

## Activation capture (EasySteer / vLLM fork)

Pinned down by `src/probing/verify_capture.py` against stock HF transformers;
min cosine similarity 0.9969 over layers 0–26. **Both indexing conventions are
wrong by default:**

1. **Layer** — decoder layer L (1-indexed) is EasySteer index **L−1**.
2. **Token position** — position t is EasySteer position **t+1**, because vLLM
   emits a double-BOS.
3. **vLLM's final layer is pre-RMSNorm**, unlike HF's. The last layer is not
   comparable to an HF last-layer vector without applying the norm.

Off-by-one in either direction produces plausible-looking probe accuracies that
are quietly measuring the wrong token or layer. Re-run all three
`verify_capture.py` stages after any version change.

**Prefix caching and chunked prefill must both be OFF during capture** — either
one silently truncates the captured sequence. You get fewer positions back than
you asked for, with no error.

EasySteer already returns **all 28 layers on every call**; the original
`extract_activations.py` requested 3 and discarded 25. There is no cost saving in
asking for fewer.

## `env/requirements-goal1.txt` is a record, not an installable file

It is a raw `pip freeze` of the original box's venv and cannot be installed: it pins
`vllm==0.5.5` / `torch==2.4.0` alongside orphans from a newer vLLM (`torchaudio==2.11.0`
needs torch 2.11; `opencv-python-headless` 4.13 needs numpy>=2 while `numpy==1.26.4`
is pinned; also `flashinfer`, `xgrammar 0.2.1`, `mistral_common 1.11`, ...). A strict
`pip install -r` dies with `ResolutionImpossible`. `env/setup_goal1.sh` installs
`env/requirements-goal1.in` (the 13 packages `src/` actually imports) and uses the freeze
only as `-c` constraints. Result on Sycamore: 138 packages, every one at exactly the
freeze's version, 119 orphans not installed, `pip check` clean.

## `pyairports` on PyPI is a placeholder, and `import vllm` fails with it

`outlines==0.0.46` (pulled in by vllm 0.5.5) does `from pyairports.airports import
AIRPORT_LIST`. The only `pyairports` on PyPI is one release, 0.0.1 (2025-09-25, author
"John Doe"): a stock setuptools sample project with no airports data that also installs
top-level `sample/` and `tests/` packages and a `byted-wandb` console script. Its
`setup.py` is inert (reads the README), but it cannot satisfy the import, so
`import vllm` fails with `ModuleNotFoundError: No module named 'pyairports'`.
`env/shims/pyairports/` is a local stand-in (`AIRPORT_LIST = []`, which only empties
outlines' airport-code enum -- structured decoding is unused here); `setup_goal1.sh`
installs it first so pip never fetches the PyPI one. The original freeze also lists
`pyairports==0.0.1`; how imports worked on the original box is unexplained.

## `squeue` hides jobs on the GPU partitions — use `squeue -a`

`h100_all` is a hidden partition. Plain `squeue -u $USER` omits hidden-partition jobs and
prints nothing while GPU jobs are running (an explicit `squeue -j <id>` does show them).
Use `squeue -a -u $USER`. `runs/gpqa25/status.sh` and the queue's liveness check do.
Also NFS: a directory listing on the login node can lag a few tens of seconds behind a
compute node's `mkdir` (locks written by workers appear late) — wait and re-list before
concluding something is missing.

## vLLM 0.5.5 is CPU-bound on the H100 for this workload (GPU ~30-35% busy)

Measured on the live GPQA continuation run (2026-09-21): `nvidia-smi` utilisation steady at
28-37% on two workers, the engine's main thread at 97% of ONE core (all other threads idle),
no preemption. A 45 s `py-spy` profile (2,249 samples, non-blocking) of the engine step:
~43% execute_model (of which ~27% is `_random_sample`'s `random_samples.cpu()` = the CPU
waiting on the GPU, i.e. the genuinely GPU-busy part), ~37% scheduler, ~17% output
processing. Inside the scheduler, `BlockSpaceManagerV1.access_all_blocks_in_seq` (a Python
loop over EVERY KV block of EVERY running sequence, every step, guarded by prefix caching;
~19% incl.) and `get_block_table` (~17%) are O(running seqs x context blocks); detokenization is
~12%. With 128 sequences and contexts of thousands of tokens (prompt + long prefix + up to 16k
generated) that per-step Python cost is ~1.3x the GPU time, so one engine per GPU can never
use more than ~40% of it. Faster GPUs only speed up the GPU share (Amdahl): estimates scaled
from the A6000 numbers by memory bandwidth were ~3x too optimistic (measured ~2,186
continuations/GPU-h, ~280 GPU-h for 615k continuations). MEASURED remedies (benchmark jobs 4522034 / 4527004 / 4527023,
2026-09-25; 5 min per config, throughput from vLLM's own stats, control reproducible
to 0.2%): `use_v2_block_manager=True` 1.18x; **plus `max_num_seqs=256` 1.42x -- now the
default in `runs/gpqa25/common.sh`**; multi-step is unavailable (`NotImplementedError:
Multiple scheduler steps (multi-step) are only supported through AsyncLLMEngine`, and we
use the offline `LLM()` class).

**Two engines per GPU is SLOWER than one engine with the same total concurrency**
(4,431 vs 4,943 tok/s at 256 sequences), because each process re-reads the 15 GB of
weights every step, and decode at these batch sizes is bound by that read. It also
reports much HIGHER GPU utilisation (84% vs 55%) while doing less useful work --
**nvidia-smi utilisation is not throughput**, and optimising for it is a trap. Three
engines hung outright: each got 68k tokens of KV for sequences needing ~115k, and
thrashed into CPU swap.

Two further traps if you do run several engines on one GPU (the plumbing is in place,
`ENGINES_PER_GPU`): vLLM sizes its cache from `total_gpu_memory * gpu_memory_utilization
- peak_memory` where `peak_memory` comes from `torch.cuda.mem_get_info()`, which reports
the WHOLE DEVICE -- so engines must LOAD one at a time (`run_gated` in `common.sh`, using
`continuations.py --ready-file`) or one dies with "No available memory for the cache
blocks"; and a stable, fully-loaded sibling is invisible to that calculation, so an equal
`gpu_memory_utilization` split is correct once start-up is serialised.

## The two vLLM versions are mutually incompatible

`$PY` (0.5.5) and `$PY_ES` (0.17.1 fork) cannot share a venv. Running
`probe_checkpoints.py` under `$PY_ES`, or `extract_checkpoints.py` under `$PY`,
fails in confusing ways. See [`SETUP.md`](SETUP.md).

## Entropy: two estimators, and they disagree materially

`answer_entropy_bits` is **plug-in** (`run_eval._entropy`) and biased low at
small K. `answer_entropy_bits_mm` is **Miller–Madow** corrected
(`labels._entropy_mm`). Both are stored so the gap stays auditable.

**Use `_mm`.** On GPQA's K=25 slice, 66.5% of rows were biased beyond the stated
0.02-bit tolerance, with a mean correction of 0.022–0.053 bits. The K=25 decision
itself was justified only under Miller–Madow while `labels.py` was still emitting
plug-in — that inconsistency is the origin of the whole issue.

## GPQA's ground truth is not uniform in K

40 questions at K=100, 155 at K=25 (`example_id % 5 == 0` is the K=100 slice).
Any analysis that treats GPQA labels as homogeneous is comparing noisier
estimates against cleaner ones. Use `--k100-only` when that matters. Commitment
labels on the K=25 slice are ~82% stable.

## `prefix_len` is a genuinely strong baseline, not a formality

At the 0.1 checkpoint `prefix_len` alone gets R²=0.605 on `rem_len_mean` —
matching the best activation probe. It beats TF-IDF at the first three
checkpoints. **Any claim about predicting remaining length must be checked
against it**, or you are reporting token-counting as representation-reading.

## Grouped CV is mandatory

A question's 3 CoTs share the question text. `GroupKFold` by `example_id`
everywhere — ungrouped CV leaks and inflates every number. Under the go-forward
1-CoT design this matters less, but the grouping should stay.

## Mean-pool per (CoT, depth-bin), not per token

The first smoke probe pooled per token, so long CoTs dominated the fit. Each CoT
must contribute **one vector per bin**.

## Sentence splitting must not cross `</think>`

`checkpoint.py` splits only the pre-`</think>` span. R1-Distill models write a
plain-language summary after `</think>` that often states the answer outright;
including it fabricates surface-form leakage in late prefixes. See
[`PIPELINE.md`](PIPELINE.md) §2.

## vLLM KV-cache blowups look like a hang, not an error

Too-wide request groups exhaust the KV cache, vLLM swaps to CPU, and the GPU sits
at 0% indefinitely. This killed the first AIME run. `--max-n-per-request 20`
(K=100 as one `n=100` group needs 2.3× the 420k-token cache) and `MAXN=8` for
GPQA at a 16384-token budget. Symptom to watch for in logs:
`lack of CPU swap space`, `No space left on device`, `CUDA out of memory` —
`status.sh` greps for exactly these.

## The outlines cache must be writable by *you*

A cache dir left by another user makes plain `import vllm` fail with
`sqlite3.OperationalError: attempt to write a readonly database`. `env.sh` now
defaults `OUTLINES_CACHE_DIR` to `.outlines_cache_$USER`.

## Datasets come from `data/`, not the HF cache

`run_eval.load_examples` reads local `save_to_disk` copies. A warm HF cache does
not substitute for running `src/setup/download_dataset.py`.

## `archive/` paths are stale

`archive/` is prior Qwen3 logit-lens/probing work, **reference only**. It
hardcodes `/playpen-ssd/yuvraj/...` paths that were already dead before the move.

## Stale results still on disk (original box only, not pushed)

`probe_goal1_labels.csv` and the stride-4 `activations/` store are **superseded**
and were produced under a handicapped comparison. Don't cite them; see
[`STATE.md`](STATE.md).
