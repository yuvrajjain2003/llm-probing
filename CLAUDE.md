# CLAUDE.md

## Rules (always follow — these take priority)
- **Brief replies.** Plain language, little jargon, clear over clever.
- **No assumptions.** If something is unknown, check it or ask. Say "I don't know" rather than guess.
- **Verify claims.** Back every statement about code, data, or results with something you actually ran or read. Say what you checked.
- **Minimal code.** Write the simplest code that works. No extra features, layers, or "just in case" options.
- **Minimal changes.** When editing existing code, change only what's needed. No drive-by refactors or reformatting.
- **Don't run long jobs without asking.** Use `sbatch --test-only` to check a submission without running it.
- **Never `git push` without asking the user explicitly first.** Committing locally is fine when asked.
- **Read `learnings.md`** at the start of each session, and add to it when we learn something or make a mistake.
- **Keep this file current.** At the start of each session, re-check "Repo state" below and update it. Mark tasks done in "Current work" as they finish.

## Repo state
_Last updated: 2026-10-03_

- **Folder:** `/proj/ssrivalab/projects/llm-probing` (group-shared, owned by user `yuvraj`). Git repo: `git@github.com:yuvrajjain2003/llm-probing.git`, branch `main`, at commit `28f0b56` (2026-10-01).
- **Code:** `src/` (Python pipeline), `runs/gpqa25/*.sh` (current Slurm pipeline, numbered stages 00–09), `runs/goal1/full/`, `runs/goal2/` (older launchers), `env/` (setup), `env.sh` (paths).
- **Project docs** (from the repo — read when needed): `.claude/README.md` (start here), `STATE.md` (results as of 2026-09-18, old UNC box), `NEXT.md` (agreed plan), `PIPELINE.md`, `SETUP.md`, `GOTCHAS.md`. The repo's original instructions are kept below under "Project notes".
- **Not in git (local only):** `cache/` (~70 GB: HF model weights, pip, conda), `data/gpqa_diamond/`, `runs/**` outputs (~24 GB), `learnings.md`.
- **Results present here** (Sycamore runs, 2026-09-20 to 09-28, newer than `STATE.md`):
  - `runs/goal1/gpqa25/` — GPQA-Diamond, 198 questions × 25 CoTs, Qwen-7B. pass@1 0.487, majority-vote 0.551.
  - `runs/goal2/gpqa25/` — checkpoint activations, probes (`probes_ckpt/fast/`), ablation CSVs, capture check (PASSED).
  - `runs/*/smoke/` — small test runs.
- **Paths caveat:** `env.sh` puts envs/cache/runs under `/work/users/r/i/rishp`, which doesn't exist for us, so it falls back to `<repo>/.work`. Existing outputs and cache are in `runs/` and `cache/` here. Decide where things should live before running anything.
- **Python envs:** none set up for us yet (earlier ones were in yuvraj's `/work`, which we can't read).

## Slurm / cluster (Sycamore, UNC)
- Account: `rc_ssriva_pi`, QOS: `normal`. Login node has no GPU — don't run heavy work there.
- **GPUs:** use partition `h100_all` — 4 nodes × 4 H100 (16 total), 256 CPUs and ~1.5 TB RAM per node. Max 5 days. (`h100` partition is NOT allowed for us.) GPUs are heavily shared; on 2026-10-03 all 16 were busy with ~8-day estimated wait.
- **CPU partitions:**
  | partition | min CPUs | max time | notes |
  |---|---|---|---|
  | `small` | 2 | 5 days | up to 192 CPUs/user |
  | `batch` (default) | 48 | 5 days | |
  | `inter` | 1 | 8 h | interactive |
  | `debug` | — | 4 h | 2 nodes |
- Default time limit is 2 h if `-t` not set. Default memory is 512 MB/CPU — always set `--mem`.
- `--exclusive` is not allowed.
- No access: `h100`, `hipri`, `CMSlab`.
- Software via modules: `module load anaconda/2025.12-2`, `cuda/12.9` / `cuda/13.x`.
- Example GPU job (past runs used 1 GPU, 8–16 CPUs, 96–128 GB):
  ```bash
  sbatch -p h100_all --gres=gpu:1 -c 8 --mem=96G -t 02:00:00 job.sh
  ```
- Useful: `squeue -u $USER`, `sacct -j <id>`, `sbatch --test-only ...` (dry run), `sinfo -a` (shows hidden partitions).

## Current work
_User gives direction; track tasks here. Mark `[x]` when done._
- [x] Explore folder and cluster setup
- [x] Pull code from GitHub into this folder
- [ ] Set up our own Python environment (and decide where envs/cache/runs live)

## Next steps
- (waiting on user)

---

# Project notes (original repo CLAUDE.md, by yuvraj)

> **New session / new machine? Read [`.claude/README.md`](.claude/README.md) first.**
> This file is the authority on **scope and working practices**. `.claude/` is the
> authority on **current state and mechanics** -- what has actually been run
> ([`STATE.md`](.claude/STATE.md)), how to bootstrap a cluster
> ([`SETUP.md`](.claude/SETUP.md)), exact stage commands
> ([`PIPELINE.md`](.claude/PIPELINE.md)), silent-failure traps
> ([`GOTCHAS.md`](.claude/GOTCHAS.md)) and the agreed next steps
> ([`NEXT.md`](.claude/NEXT.md)). Where the two disagree about what has been run,
> trust `.claude/STATE.md`.
>
> **`runs/` is not in this repo.** All results described below and in `.claude/`
> live on the original UNC box; on a fresh clone they must be regenerated.

## Research question

Do internal representations at a chain-of-thought (CoT) prefix predict useful
properties of the **distribution over future continuations**, beyond what is
available from the prefix text itself?

Originally Goal 1 (behavioral labels only, no activations) was meant to be
validated before Goal 2 (activations/probing) started. **In practice a
collaborator (`yuvraj`) began Goal 2 on 2026-09-06/07 while Goal 1 was still
partial** — see "Goal 2 status" below. Treat the scope list as what was
originally agreed, not as a description of what has actually run; check
`runs/` before assuming a dataset/model/K combination hasn't been done.

## Scope (originally agreed — see actual state below before assuming gaps)

- **Models:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` and
  `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`. **Only Qwen-7B has been run, for
  everything, Goal 1 and Goal 2.** Llama-8B: zero runs anywhere.
- **Datasets:** AquaRAT, OpenBookQA, MMLU (college math) were the original set;
  GSM8K remains deferred. **Also actually run** (added by yuvraj without a
  CLAUDE.md update): GPQA-diamond, AIME2025, AIME 2022–2025, MATH500 — see
  `runs/goal1/full/`. GPQA-diamond is also the only dataset used for Goal 2.
- **Size:** 50–100 questions total was the target for the original 3 datasets;
  the added datasets vary in size (see `runs/goal1/full/*/raw/examples.csv`).
- **Base CoTs per question:** 3.
- **Checkpoints per CoT:** prefixes at 10 / 30 / 50 / 70 / 90% by sentence count.
- **Continuations per prefix (K):** originally 10. **AIME runs used K=100**
  (log records `K* = 300, ran at K = 100`) — K sufficiency findings from the
  original 3 datasets do not automatically apply; re-check per dataset.

### GPQA's K is two-tier, and the K=25 choice has a caveat (checked 2026-09-14)

`pipeline_gpqa.sh` ran phase A at K=100 on 20% of questions, measured K\* there,
then ran the rest at K\*. So the GPQA ground truth is **not** uniform:

| slice | questions | CoTs | K |
|---|---|---|---|
| `example_id % 5 == 0` | 40 | 120 | 100 |
| the rest | 155 | 455 | 25 |

`k_sufficiency/summary.csv` reports `k_bias=75`, `k_bias_mm=25`,
`k_precision=NaN` (the 0.05-bit SD tolerance was never met at K≤100),
`k_decision=100`, and the log states **`K* (max of the three criteria) = 100.0`**.
The pipeline then reads back `k_bias_mm` and runs at 25 — i.e. the K=25 choice is
justified only under the **Miller-Madow** bias correction, which `labels.py` was
not applying (`run_eval._entropy` is plug-in). Measured consequence: 66.5% of
K=25 rows were biased beyond the stated 0.02-bit tolerance, mean correction
0.022–0.053 bits depending on checkpoint.

**Fixed 2026-09-14:** `labels.py` now also emits `answer_entropy_bits_mm`
(`labels._entropy_mm`). Both estimators are stored so the gap stays auditable.
Prefer the `_mm` column as the entropy target; treat commitment labels on the
K=25 slice as ~82% stable (`k_sufficiency/commitment_agreement.csv`).

Any further expansion — more datasets/models, new Goal 2 stages beyond what's
described below — should still be flagged to the user before doing it.

## Go-forward design (decided 2026-09-14, NOT yet run)

Everything currently in `runs/` uses the original design: **3 CoTs per question,
5 checkpoints at 10/30/50/70/90%**. All Goal 2 results — `runs/goal2/probes_ckpt/`
included — describe that design.

New runs from here use instead:

- **1 CoT per question** (not 3)
- **6 checkpoints**, the first at **0%** — the rendered prompt before any
  reasoning has been emitted — then the existing five
- k rollouts from each checkpoint, k still set by the k-sufficiency analysis

Two consequences to keep in mind when comparing old and new numbers:

- **The 0% checkpoint removes the prefix-length confound.** Every 0% prefix has
  length zero, so the `prefix_len` baseline is constant and cannot predict
  anything. This matters because the strongest result so far — remaining
  reasoning length, R² 0.605 at the 10% checkpoint — is matched or beaten by
  `prefix_len` alone at the first three checkpoints, i.e. it is largely
  token-counting. At 0% that explanation is unavailable, so whatever a probe
  achieves there is clean. It also measures how much the model knows about where
  it is heading *before writing anything*.
- **Per-checkpoint training rows drop 575 → 195**, since each question now
  contributes one CoT rather than three. The learning curve in
  `runs/goal2/logs_ckpt/lcurve.log` shows entropy has already plateaued by ~97
  questions so it should be largely unaffected, but remaining-length was still
  improving with more data and should be expected to weaken.

Still open, parked rather than rejected: using the rollouts themselves as probe
training data. The activation at a checkpoint is **one vector shared by all k
rollouts** (causal attention: the prefix hidden states do not depend on what
follows), so repeating it per rollout adds rows but no information. The version
that would add real data is capturing activations at tokens *inside* each
continuation, where they genuinely diverge — that needs a new GPU pass and a
decision on what label a mid-continuation token carries.

## Current repo state (what already exists)

### Layout of `src/` (reorganized 2026-09-14)

Grouped by pipeline stage, not by goal. Every module is still runnable as
`python src/<package>/<module>.py` from the repo root; modules that import
across packages put `src/` on `sys.path` themselves, so `python -m
<package>.<module>` works too.

| package | what lives there |
|---|---|
| `src/generation/` | Everything that talks to the model (vLLM), plus maintenance of the raw records it writes: `run_eval`, `checkpoint`, `continuations`, `gpu_pool`, `merge_shards`, `reparse`. |
| `src/analysis/` | Pure CPU reductions over those records — no GPU, no model: `labels` (the Goal 1 behavioral labels), `robustness`, `k_sufficiency`, `year_effect`. |
| `src/probing/` | Goal 2: `verify_capture` → `extract_activations` → `probe_smoke` / `probe_linear` / `probe_goal1_labels`. |
| `src/viz/` | Figures, tables and PDFs: `aggregate_viz`, `plot_goal1`, `make_figures`, `report`, `make_timing_report`. |
| `src/setup/` | `download_dataset` — one-time fetch into `data/`. |

Cross-package imports are package-qualified (`from generation.run_eval import
...`); imports between siblings in the same package stay bare. Launcher scripts
under `runs/goal1/full/` and `runs/goal2/` were updated to the new paths.


- `src/generation/run_eval.py` — vLLM sampler. Per question, draws N **full** CoTs, parses
  the `{"Final Answer": ...}` JSON, normalizes MC/free-form answers, writes
  per-sample `sample_records.jsonl` + `example_metrics.csv`. Reuse its prompt
  rendering, `parse_output`, `normalize_mc` / `normalize_freeform`, entropy, and
  resume logic — do not reimplement these.
- `src/viz/report.py` — aggregate heatmaps / bars / PDF across a sweep.
- `src/setup/download_dataset.py` + `data/` — AquaRAT, OpenBookQA,
  MMLU college/pro math already downloaded locally.
- `archive/` — prior Qwen3 logit-lens / probing / activation-extraction code.
  **Reference only.** Note it hardcodes `/playpen-ssd/yuvraj/...` paths; this
  repo was copied from that original folder, so those paths are stale.

**The key gap:** existing code samples *whole* CoTs. Goal 1 needs
**prefix-conditioned continuation** — the main thing to build.

## Pipeline to build

New modules under `src/`, reusing `run_eval.py` helpers where possible.

1. **Base CoTs** — per question, generate 3 independent full CoTs (reuse
   `run_eval`).
2. **Checkpointing** — split each CoT's reasoning into sentences; take prefixes
   at 10/30/50/70/90% by sentence count. Store exact prefix text **and** its
   token offset. Use a **math-aware sentence splitter** — naive `.`-splitting
   breaks on decimals, equations, and abbreviations.
3. **Continuations** — from each prefix, generate K=10 continuations by feeding
   `prompt + prefix` and continuing generation (no chat-template re-application on
   the partial assistant turn).
4. **Parse & store** — raw output plus per-continuation answer / correctness /
   length, keyed by
   `(dataset, model, question_id, cot_id, checkpoint, continuation_id)`.

## Labels (computed per question × CoT × checkpoint)

1. **Answer distribution** — entropy over final answers; P(answer), P(gold),
   P(mode).
2. **Remaining reasoning length** — mean/median tokens to completion, and stdev
   across continuations (variability matters, not just central tendency).
3. **Value of added compute** — accuracy of a single continuation vs.
   majority-vote over 5 and 10 continuations from that prefix.
4. **Commitment juncture** (derived) — earliest checkpoint where one answer
   becomes sufficiently concentrated and *stays* concentrated at later
   checkpoints.

## Robustness analyses (the actual deliverable of Goal 1)

These are the questions that decide whether the labels are modelable:

- **K sufficiency** — subsample/bootstrap over K to find where entropy and length
  estimates stabilize. Report whether 10 is enough or K must grow.
- **Parser failure rate & patterns** — track parse-success per dataset/model;
  catalog failure modes, not just the rate.
- **Surface-form leakage** — how often the final answer is already literally
  present in the prefix text (a predictor could cheat off this).

## Storage & runs

- All outputs go **under this repo** (it was copied from the original
  `/playpen-ssd/yuvraj/...` tree — do not write back there). New Goal 1 outputs
  under `runs/goal1/...`.
- Raw records as JSONL; label tables as Parquet/CSV.
- Jobs must be **resumable/restartable** like `run_eval.py` (append-only JSONL +
  resume-by-count). Long GPU jobs should not lose work on interruption.
- Keep model / GPU / sampling config explicit and saved alongside each run
  (as `run_eval.py` does with `run_config.json`).

## Goal 2 status (started 2026-09-06/07, ahead of the original plan)

Baselines (mean-mode, surface-form-only, checkpoint-position/model-only
predictors) are still not built — that part of Goal 2 remains deferred.
Activation extraction and a first linear probe, however, already exist, on
GPQA-diamond / Qwen-7B only:

- `src/probing/verify_capture.py` — before trusting any captured activation, cross-checks
  EasySteer's (a forked vLLM) hidden-state capture against stock HF transformers
  on identical input, matching by cosine similarity (min 0.9969 over layers
  0–26). This pinned down two non-obvious conventions that are wrong by default:
  decoder layer L (1-indexed) → EasySteer index L−1, and token position t →
  EasySteer position t+1 (vLLM double-BOS). Also documents that vLLM's final
  layer is pre-RMSNorm, unlike HF's.
- `src/probing/extract_activations.py` — re-feeds Goal 1's already-generated CoT text
  through vLLM+EasySteer (`max_tokens=1`, prefix caching and chunked prefill
  both off — either silently truncates the captured sequence) to pull hidden
  states at layers 14/21/28, every 4th think-token. Output:
  `runs/goal2/activations/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B/` (5 shards,
  `.npy` activations + `.parquet` index).
- `src/probing/probe_linear.py` — mean-pools activations per (CoT, depth-bin) so each
  CoT contributes one vector per bin (not one per token — the first smoke
  attempt over-counted this way). Logistic regression, C swept over
  `[1e-4..1]`, `GroupKFold` grouped by `example_id` (a question's 3 CoTs share
  text, so ungrouped CV would leak). Includes a positive-control target
  (`depth`: early vs. late half of CoT) that must succeed for the setup itself
  to be trusted. Results: `runs/goal2/probes/gpqa_diamond/probe_results.csv`.
  Headline: `depth` control 0.81–0.82 acc (vs 0.50 base) — pipeline is sound;
  `correct` 0.69–0.72 (vs 0.53 base), flat across all 10 depth bins — real but
  doesn't sharpen with depth, unexplained; `answer` (which of 4 choices)
  0.25–0.34 (vs 0.264 base) — no reliable early signal.

- `src/probing/probe_goal1_labels.py` — the first attempt at the actual research
  question: predict the Goal 1 labels from prefix activations, within-checkpoint,
  GroupKFold by `example_id`, against a TF-IDF-on-prefix-text baseline. **It ran,
  and its result was undocumented until 2026-09-14.** Result in
  `runs/goal2/probes/gpqa_diamond/goal1_label_probes.csv`: **TF-IDF on the prefix
  text beats the activation probe on nearly every target and checkpoint** (e.g.
  entropy at 30%: text 0.263 vs best activation 0.105; `rem_len_mean` at 10%:
  text 0.475 vs 0.459). Only `committed_by_here` at the 10% checkpoint favours
  activations (0.736 vs 0.715). Taken at face value the answer to the research
  question is *no*.

  That comparison was, however, handicapped against activations in three ways,
  all fixed by the modules below: activations were read at the nearest captured
  token *at or before* the checkpoint (stride 4) while TF-IDF got the exact
  prefix; only 3 of 28 layers were stored; and regularisation was never swept
  (`Ridge(alpha=1000)` for activations vs `Ridge(alpha=1.0)` for text). It also
  scored 8 CoTs that have checkpoints but no labels as "not committed".

### Checkpoint-exact, all-layer probing (added 2026-09-14)

- `src/probing/extract_checkpoints.py` — captures **all 28 decoder layers** at
  the **exact checkpoint token** (last token of each prefix), one vector per
  `(example_id, cot_id, fraction)` — 1:1 with `checkpoint_labels.csv`. 2,915
  checkpoints over 583 CoTs, 0.54 GB fp16, versus 16 GB for the stride-4 store.
  Two facts that make this cheap and correct: EasySteer already returns all 28
  layers on every call (the old code discarded 25), and because the model is
  causal, one forward pass over each CoT's longest prefix yields every earlier
  checkpoint's state exactly. Output:
  `runs/goal2/activations_ckpt/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B/`.

  **Do not extract per continuation.** All K continuations of a prefix share that
  prefix, so with causal attention their checkpoint-token activations are
  bit-identical. The K continuations supply the *label*; the prefix supplies the
  *feature*.

- `src/probing/probe_checkpoints.py` — per (checkpoint × layer × target) probes
  with **nested** regularisation selection for every model including the
  baselines, so the comparison is about features rather than tuning effort.
  Ridge is solved along the whole alpha grid from one SVD of the training block
  (`RidgePath`), which is what makes 28 layers × 5 checkpoints affordable.
  Targets: `answer_entropy_bits_mm`, `rem_len_mean`, `rem_len_std`, `p_mode`,
  `committed_by_here`, `will_commit`. Also fits **[text ++ activations]** — the
  research question is about information *added* to the prefix text, which a
  head-to-head loss to TF-IDF does not settle. `--k100-only` restricts to the 40
  questions with K=100 labels.

Anything past this (more datasets/models, non-linear probes, the still-missing
baselines) should be flagged to the user before starting.

## Working practices

- **Do not assume.** Verify claims against the code and data before acting.
  Don't guess dataset schemas, column names, or model behavior — check.
- **Ask when a decision is genuinely the user's** (scope, model/dataset choices,
  anything hard to reverse). Pick sensible defaults for reversible details and
  say what you chose.
- **Report outcomes honestly.** If a job fails, parsing breaks, or a step was
  skipped, say so with the evidence. Don't paper over parser failures — they are
  a Goal 1 metric.
- **Reuse before rewriting.** Prefer existing `run_eval.py` helpers over new
  copies of prompt/parse/normalize logic.
- **Prefer resumable, config-logged jobs.** Assume any GPU run can be
  interrupted.
- **Under git as of 2026-09-18.** Commit before risky edits rather than making
  `.bak` copies (`archive/src_bak/` is the pre-git convention; don't grow it).
  Run outputs, datasets, venvs and the vendored EasySteer checkout are NOT
  tracked -- see `.gitignore` and `.claude/SETUP.md`.
