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
