# Idea 1 spec — how redundant is our dataset, and does it taint the probes?

_Owner: agent 1. Written 2026-10-22 by the orchestrating session. Output folder:
`runs/goal2/gpqa25/sweep/_redundancy/`. Code: NEW files `src/analysis/redundancy.py` (parts A-C)
and `src/analysis/redundancy_curves.py` (part D). Figures: `docs/figures/redundancy/`._

## The question and why it matters

Each of our 198 questions contributed **25 independent chains of thought**. The user's concern:
if those 25 chains are near-duplicates, the probes may be scoring well by recognising *which
question this is and how hard it is*, never reading anything about the particular chain or the
position within it. That would make the headline result a question-difficulty classifier wearing
a probe costume.

Two things that normally sink such a result are ALREADY handled, and you must not re-report them
as findings:
- **Leakage:** folds are split by question, so all 25 chains of a question are always in the same
  fold. A probe cannot train on chain 7 of question 42 and be tested on chain 12 of question 42.
- **Inflated CIs:** the bootstrap resamples *questions*, not rows, so error bars already account
  for clustering.

The live concern is **label redundancy** and whether the probe resolves anything *within* a
question. Neither question-level folds nor the question bootstrap protect against that.

## Fact sheet (verify anything you rely on; do not trust this file blindly)

- Repo root `/proj/ssrivalab/projects/llm-probing`. `source env.sh`, then `$PY` is the analysis
  python (`envs/goal1`). Never compute on the login node: use `srun -p small -c <n> --mem=<m>`.
- Data: GPQA-Diamond, 198 questions. **182 questions have all 25 CoTs; 16 have 19-24.**
  `checkpoint_labels.csv` has 24,575 rows. Handle unequal group sizes explicitly.
- Checkpoints: fractions 0.1/0.3/0.5/0.7/0.9, plus a 0% checkpoint stored separately
  (`checkpoint_labels_f0.csv`, 198 rows, `cot_id = -1`). **0% has one row per question, so it has
  no within-question pairs — exclude it from parts A and B.**
- Labels: `runs/goal1/gpqa25/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B/labels/`.
- Activations: `runs/goal2/gpqa25/activations_ckpt/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B/`,
  32 shards, `.npy` + `index.shard*.parquet`, shape (rows, 28, 3584), fp16, 5.2 GB total.
- **Layer indexing: `sweep.py`'s `Act` is 1-indexed — `act_L21` reads stored index 20
  (`a[r, L-1, :]`). Get this wrong and every number is wrong.**
- **Reuse `sweep.py` helpers: `load_meta`, `Act`, `fixed_fold_of`, `_fixed_folds`.** `load_meta`
  does specific inner joins (labels + commitment + checkpoints.jsonl); rolling your own row set
  will silently disagree with `ref_fixed`. `src/probing/r3_rf_diagnosis.py` is the precedent for
  importing these in a one-off analysis.
- Targets: `answer_entropy_bits_mm`, `p_mode`, `rem_len_mean`, `rem_len_std` (scored by R²) and
  `committed_by_here` (accuracy).
- Folds: `fixed_fold_of(run_dir, 5, 0)` — by question, 40/40/40/39/39.
- Reference probe results and stored out-of-fold predictions:
  `runs/goal2/gpqa25/sweep/ref_fixed/{results.csv,preds*.csv}`. Features there are `act_L21`,
  `prefix_len`, `tfidf`; probes `ridge_act`, `ridge_len`, `ridge_text`, `logistic`.
- Reference numbers (test R², mean over 10-90%): entropy 0.289, p_mode 0.251, rem_len_mean 0.524,
  rem_len_std 0.298, committed acc 0.741. Per checkpoint for `act_L21`:

  | target | 10% | 30% | 50% | 70% | 90% |
  |---|---|---|---|---|---|
  | entropy | 0.278 | 0.295 | 0.298 | 0.302 | 0.270 |
  | rem_len_mean | 0.626 | 0.587 | 0.548 | 0.484 | 0.375 |
  | p_mode | 0.251 | 0.261 | 0.259 | 0.261 | 0.222 |

- Chosen ridge alpha is 1e4 or 3.16e4 in essentially every fold (grid `logspace(0,6,13)`).
  Train R² ~0.48 vs test ~0.28 on entropy: still overfitting, i.e. **variance-limited**.
- 9 label rows have all-NaN targets (no valid continuation). Drop them consistently.
- Known traps (`learnings.md`): `HF_TOKEN=dummy` needed for tokenizer loads; the session
  scratchpad under `/tmp` is NOT visible to compute nodes, so all outputs go under `runs/`;
  `fraction` floats differ between files — **round to 3 decimals before any merge**, 0.7 especially.

## Part A — cosine similarity (explicitly requested by the user's professor)

Layer 21 primary; also L14 and L28 for context. Checkpoints 10-90% only. **Compute per checkpoint
separately — never pool across checkpoints**, since depth shifts the state and contaminates both
distributions.

Pair populations, per checkpoint:
1. **within-question** — distinct CoTs of the same question. ~56k pairs/checkpoint; exhaustive.
2. **across-question** — pairs from different questions, sampled to match count.
3. **same CoT across depth** (bonus) — the same chain at 50% vs 90%, etc. Says whether depth or
   chain identity moves the state more.

Three preprocessings, **all three reported**:
- **raw** cosine on stored vectors. Expect ~0.9+ for everything including unrelated questions —
  LLM states share a large mean direction (the 0% activations are on record at ~0.96 correlated
  across questions purely from the shared prompt template). **Reporting raw alone would mislead.**
- **centered** — subtract the mean activation over all rows at that checkpoint, then cosine.
- **standardised** — center and divide by per-dimension SD. This matches what the probe sees.

Summary statistics per checkpoint x layer x preprocessing: mean/median/quantiles of both
distributions, plus:
- **separation AUC** — P(a random within-question pair is more similar than a random
  across-question pair). 0.5 = the 25 chains are no more alike than two unrelated chains;
  1.0 = perfect clustering by question. This is the headline number: one scale everyone reads
  the same way. A bare cosine has no "good value"; this does.
- Cohen's d between the two distributions.

Also the **text side**, which is cheaper and more intuitive: the same within/across split on
TF-IDF cosine of `prefix_text`, and the coefficient of variation of `prefix_tokens` across a
question's 25 chains.

Outputs: `cos_summary.csv` (one row per checkpoint x layer x preprocessing x population),
`cos_auc.csv`, and one figure per layer: within vs across histograms, panel per checkpoint.

## Part B — is the LABEL redundant

Per target x checkpoint, one-way variance decomposition with question as the grouping factor:
- naive eta^2 (between-question variance share)
- **unbiased ICC(1)** with unequal group sizes: (MSB - MSW) / (MSB + (m_bar - 1) * MSW)
- the implied **ceiling**: a perfect question-level predictor can reach R² = the between-question
  share. This is the number that says whether 0.289 is impressive or weak.

For `committed_by_here` also report the share of questions that are all-0 or all-1 across their
chains — constant-within-question is redundancy in its purest form.

Output: `icc.csv`.

## Part C — the taint test (no refitting, minutes)

From the stored out-of-fold predictions `ref_fixed/preds*.csv`, for features `act_L21`,
`prefix_len`, `tfidf`, per target x checkpoint:

- **pooled R²** — must reproduce `results.csv` to ~1e-6. This is your correctness check; if it
  does not match, stop and fix before computing anything else.
- **between-question R²** — collapse y and y_hat to question means, score over the ~198 points.
- **within-question R²** — subtract each question's mean from both y and y_hat, then
  R² = 1 - sum((y_dm - yhat_dm)^2) / sum(y_dm^2). May be negative; report as-is.
- **within-question Pearson and Spearman correlation**, and within-question R² after an optimal
  linear rescale. **This matters: ridge predictions are shrunk, so within-question R² can sit at
  or below zero even when the probe does rank chains correctly. Reporting only R² here would be a
  false negative.** Report all three and say which you are interpreting.
- **Question-bootstrap CIs** (2,000 draws, resample the 198 questions with replacement, all their
  rows come along, recompute on stored predictions, 2.5th-97.5th percentile) on within-question R²
  and within-question correlation. Is within-question signal significantly above zero?

Output: `decomposition.csv`, `decomposition_ci.csv`, one figure.

## Part D — does the redundancy actually matter (the only part that refits)

Fixed folds, same alpha grid `logspace(0,6,13)`, inner CV as in `sweep.py`.

**D1 — CoT learning curve.** Train on m in {1, 2, 3, 5, 10, 25} CoTs per question, 3 subsample
seeds, **always evaluating on the full held-out rows** so the test set is identical across m.
All 5 checkpoints, 4 ridge targets at full grid, `committed_by_here` on a reduced logistic grid
(logistic is ~13x slower than ridge on these features).
- Flat from m=1 to m=25 → the extra chains add no information; the data collection was wasteful
  but the result stands.
- Rising → the 25 samples carry independent signal, which is the claim the user wants to be able
  to make.

**D2 — question-vs-CoT exchange rate (keep; the user confirmed).** Training sets matched on
*total row count* but split differently — e.g. ~1,000 rows as 40 questions x 25 CoTs, vs 100 x 10,
vs 200 x 5. If rows from new questions buy more R² than rows from new chains, the ratio gives
"one new question is worth N new chains" — which answers whether this dataset is really 198 points
or 24,575.

Outputs: `d1_cot_curve.csv`, `d2_exchange.csv`, figures.

## What the outcomes mean (fixed in advance so nothing is retrofitted)

| Part A | Part C | conclusion |
|---|---|---|
| AUC ~0.5-0.6 | — | the 25 chains are genuinely diverse; concern dismissed with evidence |
| AUC > 0.9 | within R² ~0 | **the probe is a question-difficulty reader** — serious, reframes the paper |
| AUC > 0.9 | within R² > 0, significant | data is clustered but the probe resolves chains within a question; concern quantified and answered |

## Rules

- **Write only** into `runs/goal2/gpqa25/sweep/_redundancy/`, `docs/figures/redundancy/`, and the
  two new `src/analysis/` files named above.
- **Never** edit existing shared code (`sweep.py`, `env.sh`, anything already in `src/probing/`).
  Import from it; do not modify it. If you believe a change is needed, stop and ask.
- **Never** write into `runs/goal2/gpqa25/{probes_ckpt,ablation}` — another researcher's.
- **Do not touch `CLAUDE.md`.** The orchestrator owns the Results index.
- **Do not submit Slurm jobs.** Prepare the script, verify with `sbatch --test-only`, and report
  the exact command to the orchestrator, who submits and staggers.
- CPU only, partition `small` (min 2 CPUs), always set `--mem` and `-t`.
- Stuck? Append to `_redundancy/QUESTIONS.md` and **keep going on whatever else is unblocked.**
  Do not guess on anything that would invalidate a number.
- Report every result you produce back to the orchestrator: path, what made it, what it means.
