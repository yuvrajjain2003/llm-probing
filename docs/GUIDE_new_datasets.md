# Guide: K-sufficiency test + activation extraction for MATH-500, MMLU-Pro and GlobalOpinionQA

_Written 2026-10-29 for an agent on a different server with no prior context. Read all of it
before submitting anything._

## 1. What this project is, in five lines

- A reasoning model (`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`) writes a chain of thought (CoT).
  We cut each CoT at 10/30/50/70/90% of its sentences, and also take a 0% cut (the prompt alone).
- From each cut we let the model **continue K times**. The K continuations give the labels:
  answer entropy, P(most common answer), mean and spread of remaining reasoning length, commitment.
- We also store the model's **hidden state at the last token of each cut** (all 28 layers). Later,
  probes predict the labels from those states. Probing is NOT part of this job.
- This was done for GPQA-Diamond. **Your job: do the same data collection for `math500`,
  `mmlu_pro` and/or `global_opinions` (whichever the user names), choosing K per dataset with a
  K-sufficiency test first.** The procedure (§5) is identical for all three; only the variables
  (§4) and the dataset notes (§6) differ.
- Everything you need is in this repo. Background, if you want it: `CLAUDE.md` ("Project notes"),
  `.claude/SETUP.md`, `.claude/GOTCHAS.md`.

## 2. Decisions to confirm with the user BEFORE any GPU time

Ask; do not guess. Suggested values in brackets.

1. **Questions per dataset (`N_EXAMPLES`).** MATH-500 has 500 [all 500]. Full MMLU-Pro has 12,032
   [a few hundred to ~1,000; questions are sampled at random with seed 42, so categories are mixed].
   GlobalOpinionQA has 2,556 [ask; sampled at random with seed 42, so option counts are mixed].
2. **CoTs per question (`N_COTS`).** GPQA used 25. Our later analysis found that, for a fixed budget,
   more questions with **2-5 CoTs each** gives better probes than few questions with 25 [ask].
3. **0% cut and K.** The 0% cut's labels are NOT made from continuations: continuing from an empty
   prefix is the same as sampling a fresh answer, so its labels come from the base CoTs, and **its
   K equals `N_COTS`**. With 2-5 CoTs the 0% labels rest on 2-5 samples.
   **Decided by the user (2026-10-29): accept noisy 0% labels for now.** Do not raise `N_COTS` or
   add a 0% sampling stage for this reason; just state `N_COTS` as the 0% K in your report.
4. **How K\* is chosen** (write it down before seeing results). Suggested: the largest of the
   Miller-Madow bias K (`k_bias_mm`) and the commitment-agreement K (`k_decision`) in
   `summary.csv`, checked at every cut point (`frac=...` rows), plus an eyeball check that the
   spread of mean remaining length (`rem_sd` in `curves.csv`) has flattened.
5. **Pilot K.** 100 by default. For MATH-500 (free-form answers, many distinct values) consider
   300; AIME needed more than 100 in the past. GlobalOpinionQA: 100 is likely enough for the 2-5
   option questions (~82% of the set), but the Miller-Madow bias grows with the number of distinct
   answers, so questions with 10-18 options may need more (see §6).

**Cost of the pilot** = (N_EXAMPLES / 5) × N_COTS × 5 cuts × K_pilot continuations, each up to
16,384 tokens. Example: 500 questions, 5 CoTs, K=100 → 50,000 continuations. Give the user this
number before submitting.

## 3. Setting up on the new server

Follow `.claude/SETUP.md`. In short (login node, needs internet):

```bash
git clone git@github.com:yuvrajjain2003/llm-probing.git && cd llm-probing
export WORK_DIR=/path/with/space          # envs, weights and runs go here (~100 GB+)
source env.sh                             # sets PY, PY_ES, RUNS_ROOT, HF caches; see its header
env/setup_goal1.sh                        # env 1: generation + analysis -> $PY
runs/goal2/setup_easysteer.sh             # env 2: activation capture   -> $PY_ES
env/fetch_models.sh                       # model weights into $HF_HUB_CACHE
"$PY" src/setup/download_dataset.py math500 mmlu_pro global_opinions   # -> data/<name>/ (jobs read these copies)
"$PY" -c "import math_verify"             # MATH-500 grading needs it; pip install math-verify if missing
```

- **Two Python envs are required and cannot be merged** (two incompatible vLLM versions).
  `$PY` = generation + analysis; `$PY_ES` = activation capture only.
- **Cluster settings.** The stage scripts in `runs/gpqa25/` carry Sycamore `#SBATCH` lines
  (`--partition=h100_all`, `--gres=gpu:h100:1`, `--partition=small` for CPU stages). Options given
  on the `sbatch` command line override them, so pass your cluster's values there, e.g.
  `sbatch -p <gpu_partition> --gres=gpu:1 -A <account> runs/gpqa25/04_continuations.sh`.
  Re-check the CUDA driver notes in `.claude/SETUP.md` §2 and §5.
- **Always submit from the repo root** after `source env.sh` (the scripts find the repo that way,
  and Slurm logs go to `runs/logs/slurm/`).

## 4. How the pipeline is driven

All stages are scripts in `runs/gpqa25/`. They read their settings from environment variables
(defaults in `runs/gpqa25/common.sh`). The ones you set:

| variable | meaning | math500 | mmlu_pro | global_opinions |
|---|---|---|---|---|
| `DATASET` | key in `DATASETS` (`src/generation/run_eval.py`) | `math500` | `mmlu_pro` | `global_opinions` |
| `RUN_TAG` | names all output folders; one per dataset | `math500_v1` | `mmlupro_v1` | `globalop_v1` |
| `N_EXAMPLES` | questions | from §2 | from §2 | from §2 |
| `N_COTS` | CoTs per question | from §2 | from §2 | from §2 |
| `K` | continuations per cut | pilot / K\* | pilot / K\* | pilot / K\* |
| `SUBSET_MOD`, `SUBSET_KEEP` | run only questions with `example_id % MOD` in `KEEP` | see §5 | see §5 | see §5 |

Outputs: `$RUNS_ROOT/goal1/$RUN_TAG/$DATASET/DeepSeek-R1-Distill-Qwen-7B/` (= `RUN_DIR`: `raw/`,
`labels/`, `k_sufficiency/`) and `$RUNS_ROOT/goal2/$RUN_TAG/` (activations). Progress:
`runs/gpqa25/status.sh` (with the same variables set).

GPU stages are "worker" jobs that pull shards from a queue in `$QUEUE_ROOT` (default
`$RUNS_ROOT/queue/$RUN_TAG`) until none are left. Submit several copies of a worker script to use
several GPUs. Everything resumes: a timed-out worker loses nothing; resubmit it.

Set the variables once per shell, e.g. for MATH-500:

```bash
source env.sh
export DATASET=math500 RUN_TAG=math500_v1 N_EXAMPLES=500 N_COTS=<from §2>
```

Do the whole procedure for one dataset, then repeat with the other dataset's variables
(a fresh shell, so nothing leaks between them).

## 5. Step by step (per dataset)

### Step 0. Smoke test (one GPU, ~1-2 h)
```bash
sbatch runs/gpqa25/00_smoke.sh      # uses $DATASET; 4 questions x 2 CoTs x K=3, RUN_TAG=smoke (wiped first)
```
Read `$RUNS_ROOT/logs/smoke/smoke_summary_job<ID>.txt`. It exercises every stage including
capture verification and extraction. Check the parse-failure rate in the base CoTs: for MATH-500
the prompt asks for `{"Final Answer": ...}` and falls back to the last `\boxed{}`.
**Do not continue if the smoke test fails.**

### Step 1. Base CoTs for all questions
```bash
sbatch runs/gpqa25/01_base.sh                      # submit 2-4 copies for more GPUs
# when all workers have finished:
sbatch runs/gpqa25/02_merge_base.sh                # merge + completeness check
```

### Step 2. Capture verification (one GPU, < 45 min; can run alongside step 3)
```bash
sbatch runs/gpqa25/03_verify_capture.sh
```
Checks the capture env against plain HF transformers on the same input. Extraction refuses to run
without its PASS (`$RUNS_ROOT/goal2/$RUN_TAG/_verify/cmp.txt`). On the original machine: min cosine
0.9969 over layers 0-26. Anything clearly lower means the capture conventions changed: stop.

### Step 3. Pilot: 20% of questions at a high K
```bash
K=100 SUBSET_MOD=5 SUBSET_KEEP=0 sbatch runs/gpqa25/04_continuations.sh    # K=300 if chosen in §2
```

### Step 4. K-sufficiency test (CPU only, no generation)
Do NOT use `05_merge_label.sh` here: its completeness check expects every question.
```bash
RUN_DIR=$RUNS_ROOT/goal1/$RUN_TAG/$DATASET/DeepSeek-R1-Distill-Qwen-7B
srun -p <cpu_partition> -c 4 --mem=32G -t 60 bash -c ". ./env.sh; \
  \$PY src/generation/merge_shards.py --run-dir $RUN_DIR --no-metrics && \
  \$PY src/analysis/k_sufficiency.py --run-dir $RUN_DIR --draws 200"     # add --kgrid extended if pilot K=300
```
It subsamples the pilot's continuations down to smaller K and asks how much the labels change.
Outputs in `$RUN_DIR/k_sufficiency/`:
- `summary.csv` — smallest K meeting each criterion, overall (`scope=ALL`) and per cut point:
  `k_bias` (plug-in entropy within 0.02 bits of its large-K value), `k_bias_mm` (same for
  Miller-Madow entropy), `k_precision` (entropy SD < 0.05 bits), `k_decision` (commitment label
  agrees with the full-K label ≥ 99%).
- `curves.csv` (includes `rem_sd`), `commitment_agreement.csv`, plots.

How to read it:
- If the log prints **"reference uncertainty ... EXCEEDS the tolerance"**, K\* cannot be trusted
  at any K tested: report to the user (a deeper pilot is needed).
- An empty (NaN) criterion means it was never met within the grid. On GPQA `k_precision` was never
  met at K ≤ 100. Report it; do not silently drop it.
- Apply the rule agreed in §2 and **report K\* and the full `summary.csv` to the user before
  step 5.**

### Step 5. The other 80% at K\*
```bash
K=<K*> SUBSET_MOD=5 SUBSET_KEEP=1,2,3,4 QUEUE_ROOT=$RUNS_ROOT/queue/${RUN_TAG}_pC \
  sbatch runs/gpqa25/04_continuations.sh
```
**The new `QUEUE_ROOT` is required.** The queue skips any shard marked done, and the pilot marked
every shard done, so in the old queue this step would finish instantly having done nothing. Do not
set `NSHARDS` (both phases must use the same shard count, default 64).

If K\* > pilot K, also top the pilot slice up (continuations resume by count, so only the
shortfall is generated): rerun step 3 with `K=<K*>` and a third `QUEUE_ROOT`.

### Step 6. Merge + labels
```bash
K=<K*> sbatch runs/gpqa25/05_merge_label.sh
```
Writes `$RUN_DIR/raw/checkpoints.jsonl`, `labels/checkpoint_labels.csv` and `labels/commitment.csv`.
**Its final check will FAIL if the pilot slice has more continuations than K\*** (pilot cuts have
100): the log prints `n_continuations per prefix: {K*: ..., 100: ...}` and
`FAIL: N prefixes do not have exactly K=...`. That is expected if N equals the number of pilot
cuts and nothing else is wrong (no missing questions). Check that, then continue manually (step 7
only needs `checkpoints.jsonl`). The labels are then two-tier (pilot questions use 100
continuations), as GPQA's were; tell the user.

### Step 7. Activation extraction, 10-90% cuts
```bash
sbatch runs/gpqa25/06_extract.sh                   # worker; submit several copies for more GPUs
# when all workers have finished:
sbatch runs/gpqa25/07_check_activations.sh         # CPU: shapes, finite, 1:1 with labels, etc.
```
Output: `$RUNS_ROOT/goal2/$RUN_TAG/activations_ckpt/$DATASET/DeepSeek-R1-Distill-Qwen-7B/`
(all 28 layers, fp16, one vector per (question, CoT, cut)).

### Step 8. The 0% cut (labels + activations)
```bash
srun -p <cpu_partition> -c 4 --mem=32G -t 60 bash -c ". ./env.sh; \
  \$PY src/analysis/labels_f0.py --run-dir $RUN_DIR"        # -> labels/checkpoint_labels_f0.csv
sbatch runs/gpqa25/06b_extract_f0.sh                         # one GPU, minutes
```
Labels: one row per question, `cot_id = -1`, from the `N_COTS` base CoTs (see §2.3).
Activations: `$RUNS_ROOT/goal2/$RUN_TAG/activations_ckpt_f0/$DATASET/...`, one vector per question
at the last prompt token. Do NOT add 0.0 to the cut list in `common.sh`: that would regenerate the
same full answers once per CoT, and extraction rejects an empty prefix.

## 6. Dataset-specific notes

**MATH-500** (`math500`, 500 problems, free-form LaTeX answers)
- Answer extraction exists: `parse_output` in `run_eval.py` reads `{"Final Answer": ...}` and falls
  back to the last `\boxed{...}`.
- Correctness uses symbolic equivalence (`math_verify`). Checked: `0.5`, `1/2`, `\frac{1}{2}`,
  `\dfrac12` are one answer; `(3, \frac{\pi}{2})` with or without `\left(\right)` are one answer.
- **Known weakness for entropy:** some equal answers get different canonical strings
  (`3\sqrt{13}` vs `\sqrt{117}` are graded equal but grouped as different), so entropy can be
  slightly too high. Mention it in your report.
- The free-form prompt was written for word problems ("just the number or short phrase"); watch the
  parse-failure rate in the smoke test and pilot.

**MMLU-Pro** (`mmlu_pro`, 12,032 questions, up to 10 options A-J, 14 subject categories)
- New entry (2026-10-29); `mmlu_pro_math` is the older math-only subset. Options are lettered
  A-J and the answer parser accepts A-J (checked).
- Each question's `category` is saved in `$RUN_DIR/raw/examples.csv` (join on `example_id`), so
  results can be split by subject.
- The multiple-choice prompt's example says "(e.g. A, B, C, D)"; letters up to J are still parsed.

**GlobalOpinionQA** (`global_opinions`, 2,556 survey questions from Pew Global Attitudes (2,203)
and the World Values Survey (353); added 2026-10-29)
- **No correct answer.** The labels that need one are blank or meaningless: `p_gold` is NaN and
  `is_correct` / `acc_*` / `pass_at_1` are always 0. Entropy, `p_mode`, remaining length and
  commitment all work (commitment uses only `p_mode` >= 0.8, not a gold answer).
- **2 to 18 options**, lettered A-R. Counts: 2 (440), 3 (416), 4 (709), 5 (524), 6 (181), 7 (141),
  8-18 (145). The "DK/Refused" option (in ~1,170 questions) is **kept on purpose** (user decision,
  2026-10-29); it may be excluded later.
- **Prompt.** The standard multiple-choice prompt, with this sentence added to the start of the
  question text: "This is a survey question about your own opinion. There is no correct answer.
  Choose the option that best matches your view." It is part of the question text, so generation,
  continuations and activation extraction all use the same prompt. The system prompt still says
  "the correct answer choice"; watch for refusals in the smoke test.
- **Parsing** (`grading="opinion"` in `run_eval.py`): the `{"Final Answer": ...}` value must be a
  bare letter A-R ("B", "(B)", "B) text", "B."). Anything else (option text, "I think B", a refusal)
  counts as a parse failure, NOT a guessed letter. The "ANSWER: X" fallback line also accepts A-R.
  So **the parse-failure rate doubles as the refusal rate**: report it, per cut.
- **Entropy columns.** `labels/checkpoint_labels.csv` (and `checkpoint_labels_f0.csv`) now also
  hold `n_options`, `answer_entropy_bits_norm` and `answer_entropy_bits_mm_norm` (entropy divided
  by log2(n_options), 0-1; Miller-Madow can go slightly above 1). Raw entropy's maximum depends on
  the option count (1 bit for 2 options, 4.17 bits for 18), so use the normalised column when
  comparing questions; keep both.
- **K-sufficiency.** Run §5 steps 3-4 unchanged. `k_sufficiency.py` judges K on raw entropy in
  bits, which is the stricter choice for many-option questions. Extra checks for this dataset:
  - If K\* comes out far above the GPQA value, see whether it is driven by the many-option
    questions: join `example_id` to `n_options` in `raw/examples.csv` and report K\* for
    <= 5 options vs > 5 if the user asks.
  - If most pilot cuts have entropy ~0 (the model always picks the same option), the dataset will
    not give useful entropy labels. **Report the share of cuts with `p_mode` = 1 before step 5.**
- **Extra checks after labels:**
  - Answers beyond the question's options (e.g. "Q" on a 4-option question) are not rejected by
    the parser. Count them:
    `"$PY" -c "import pandas as pd,json; e=pd.read_csv('$RUN_DIR/raw/examples.csv'); n=dict(zip(e.example_id,e.n_options)); c=pd.read_json('$RUN_DIR/raw/continuations.jsonl',lines=True); a=c.dropna(subset=['final_answer_norm']); print((a.final_answer_norm.map(ord)-64 > a.example_id.map(n)).sum(), 'out-of-range of', len(a))"`
  - Opinion CoTs may be short. With few sentences, several cuts land on the same prefix (e.g.
    10% and 30% of a 3-sentence CoT are both sentence 1). Report the median number of sentences
    per CoT and how many CoTs have fewer than 10.
- **Human answer distributions** (`selections`, per country) are in `data/global_opinions/`
  (join on `dataset_index` in `raw/examples.csv`). They are not used by the pipeline.

## 7. What to report back to the user, per dataset

1. Smoke-test summary and parse-failure rate (for `global_opinions` this is also the refusal rate).
2. Pilot size, `summary.csv`, the curves, and the K\* you chose with the rule it came from.
3. GPU hours per stage (from `sacct`).
4. Final counts: label rows, questions, CoTs, `n_continuations` distribution, activation check
   result, 0% rows.
5. Anything that failed or was skipped, with the log line.

## 8. Pitfalls already hit (from `learnings.md` and `.claude/GOTCHAS.md`)

- Pilot and main run in one queue folder → main run silently does nothing (§5 step 5).
- Two jobs importing vLLM at the same moment can clash on the outlines sqlite cache: stagger
  submissions by ~1 min.
- Tokenizer loads in offline jobs may need `HF_TOKEN=dummy`.
- `env.sh` picks paths from `$USER`: on a shared folder, check `echo $RUNS_ROOT $PY $PY_ES` before
  submitting.
- Never run heavy work on the login node.
