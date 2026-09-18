# Running the pipeline

Every command assumes `source env.sh` first. Every GPU stage is **resumable**
(append-only JSONL + resume-by-count) — re-running continues rather than
restarting, so an interrupted job is not lost work. Do not add `--no-resume`
unless you intend to discard.

Modules run either as `python src/<pkg>/<mod>.py` or `python -m <pkg>.<mod>`;
cross-package imports put `src/` on `sys.path` themselves.

## Stage map

```
run_eval          base CoTs, N per question          GPU   $PY
   ↓ raw/sample_records.jsonl
checkpoint        sentence-split → prefixes          (library, no CLI)
continuations     K rollouts per prefix              GPU   $PY
   ↓ raw/continuations.jsonl
merge_shards      shard files → one JSONL            CPU   $PY
labels            → checkpoint_labels.csv            CPU   $PY
k_sufficiency     → is K big enough?                 CPU   $PY
robustness        parse / leakage / K catalogs       CPU   $PY
   ─────────────── Goal 2 ───────────────
extract_checkpoints  28 layers @ checkpoint token    GPU   $PY_ES ★
probe_checkpoints    per ckpt × layer × target       CPU   $PY
bootstrap_ci         CI on the act-minus-text gap    CPU   $PY
```

★ the only stage needing the EasySteer interpreter.

## Goal 1

### 1. Base CoTs

```bash
"$PY" src/generation/run_eval.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset gpqa_diamond --output-dir runs/goal1/full \
  --num-examples 198 --samples-per-example 3 --seed 42 \
  --temperature 0.7 --top-p 0.95 --max-tokens 16384 --max-model-len 24576 \
  --max-num-seqs 128 --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
  --num-shards 8 --shard-index 0 --resume
```

Sharded across GPUs by `runs/goal1/full/launch_base_sharded.sh <dataset>
<num_examples> <max_tokens> <max_model_len> [num_shards]`, one worker per GPU,
TP=1. Writes `raw/sample_records.shard*.jsonl` + `raw/examples.csv`.

Per the go-forward design, **new** runs use `--samples-per-example 1`.

### 2. Checkpoints

`src/generation/checkpoint.py` has no CLI — `continuations.py` calls it. It
splits with a **math-aware** sentence splitter (masks decimals, `$...$` spans,
abbreviations and single-letter choice labels, splits, then unmasks; every mask
is the same length as what it replaces so offsets survive).

It splits **only the pre-`</think>` span**. This is deliberate and load-bearing:
R1-Distill models often write a plain-language summary after `</think>` that
states the answer outright, so splitting the full `reasoning` field would smuggle
the answer into late prefixes and manufacture the very surface-form leakage the
robustness analysis exists to measure. Do not "fix" this.

### 3. Continuations

```bash
"$PY" src/generation/continuations.py \
  --base-dir runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --dataset gpqa_diamond \
  --continuations-per-prefix 100 --fractions 0.1 0.3 0.5 0.7 0.9 \
  --temperature 0.7 --top-p 0.95 --max-tokens 16384 --max-model-len 24576 \
  --max-n-per-request 20 --max-num-seqs 128 --swap-space 32 \
  --subset-mod 1 --subset-keep 0 \
  --num-shards 8 --shard-index 0 --resume
```

Feeds `prompt + prefix_text` raw to vLLM — **no chat template re-applied** to the
partial assistant turn. Prefix caching is on so the shared prefix stays hot.
`--subset-mod/--subset-keep` select a question slice (this is how GPQA's two-tier
K happened). Sharded by `launch_cont_sharded.sh`; `MAXN`, `SUBSET_MOD`,
`SUBSET_KEEP`, `LOGTAG` are read from the environment.

### 4. Merge, label, analyse

```bash
RUN=runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B
"$PY" src/generation/merge_shards.py --run-dir "$RUN"
"$PY" src/analysis/labels.py        --continuations "$RUN/raw/continuations.jsonl"
"$PY" src/analysis/k_sufficiency.py --run-dir "$RUN" --draws 200
"$PY" src/analysis/robustness.py    --run-dir "$RUN"
```

`labels.py` writes `checkpoint_labels.{csv,parquet}` (entropy plug-in **and**
Miller–Madow, P(mode), P(gold), remaining length mean/median/std, accuracy at
single / maj@5 / maj@10) and `commitment.{csv,parquet}` (earliest checkpoint
whose modal answer stays modal and ≥80% concentrated through every later one).

`k_sufficiency.py` tolerances: `--bias-tol 0.02`, `--sd-tol 0.05`,
`--agree-tol 0.99`. Note `k_precision` has been NaN on every dataset so far.

### The whole thing unattended

`runs/goal1/full/pipeline_gpqa.sh` is the reference end-to-end script: phase A at
K=100 on a 20% slice → measure K\* → phase C on the remaining 80% at K\*. It
hardcodes the original box's path in its `cd`; change that. `pipeline_aime_next.sh`
is the AIME variant. `status.sh` prints live progress (also path-hardcoded).

## Goal 2

### 5. Extract activations at checkpoints

```bash
"$PY_ES" src/probing/extract_checkpoints.py \
  --run-dir runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B \
  --out-dir runs/goal2/activations_ckpt/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B \
  --batch-size 2 --gpu-memory-utilization 0.55 --max-model-len 24576 \
  --num-shards 1 --shard-index 0
```

Re-feeds Goal 1's already-generated CoT text with `max_tokens=1`. Captures all 28
decoder layers at the **exact checkpoint token** (last token of each prefix), one
vector per `(example_id, cot_id, fraction)`, 1:1 with `checkpoint_labels.csv`.
Because the model is causal, **one forward pass over each CoT's longest prefix
yields every earlier checkpoint's state exactly** — which is what makes this
0.54 GB instead of 16 GB. Shard by `example_id` so a question's CoTs stay together.

**Do not extract per continuation.** All K continuations of a prefix share that
prefix, so with causal attention their checkpoint-token activations are
bit-identical. The continuations supply the *label*; the prefix supplies the
*feature*.

### 6. Probe

```bash
"$PY" src/probing/probe_checkpoints.py \
  --act-dir runs/goal2/activations_ckpt/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B \
  --run-dir runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B \
  --out-dir runs/goal2/probes_ckpt/gpqa_diamond --folds 5 --layers all
```

Per (checkpoint × layer × target), with **nested** regularisation selection for
every model *including the baselines*, so the comparison measures features rather
than tuning effort. Ridge is solved along the whole alpha grid from one SVD of the
training block (`RidgePath`) — that is what makes 28 layers × 5 checkpoints
affordable. `GroupKFold` grouped by `example_id`. Also fits **[text ++ activations]**,
because the research question is about information *added* to the prefix text,
which a head-to-head loss to TF-IDF does not settle.

Targets: `answer_entropy_bits_mm`, `rem_len_mean`, `rem_len_std`, `p_mode`,
`committed_by_here`, `will_commit`. `--k100-only` restricts to the 40 questions
with clean K=100 labels.

Then `src/probing/bootstrap_ci.py` for CIs on the gap, resampled over questions —
**report this, not the raw R² deltas.**

## Reproducibility

Each run writes `run_config.json` alongside its outputs with model, GPU, and
sampling config. Keep that habit for anything new: results whose sampling config
is not recorded next to them cannot be compared to these.
