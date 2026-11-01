# Idea 2 spec — CLOSE the MLP/linear gap

_Owner: agent 2. Written 2026-10-22 by the orchestrating session. Output folder:
`runs/goal2/gpqa25/sweep/_mlpgap/`. Code: NEW files under `src/probing/` prefixed `mlpgap_`.
Figures: `docs/figures/mlpgap/`._

## The goal, stated as the user stated it

**The objective is to CLOSE THE GAP, not to explain it.** The user's position: a linear probe is a
special case of an MLP, so an MLP should perform at least as well as ridge, and plausibly better.
Every result so far says otherwise, and the user wants that fixed. Explanation is welcome as a
by-product; "we understand why the MLP loses" is NOT success. Success is an MLP that matches ridge,
and ideally beats it. **As many experiments as needed.**

## What we already know (do not re-derive)

- Ridge is the best probe. MLPs (incl. well-regularised), PCA->ridge, random forests, gradient
  boosting all score at or below it. Mean over 10-90%: ridge entropy 0.289, p_mode 0.251,
  rem_len_mean 0.524, rem_len_std 0.298, committed acc 0.741.
- Best MLP to date: h=256 with weight decay selected from {0,1,10,100}, entropy 0.283 vs ridge
  0.289 (`sweep/4a_mlp_wd_selected/`). Best-per-cell vs ridge: mean -0.003, 9/25 wins, max +0.004.
- **The regularisation history is the key clue.** The first MLP sweep found weight decay changed
  nothing — results identical to three decimals — because effective decay per step was
  lr x wd ~ 1e-5 (AdamW, decoupled). Pushing wd to {1,10,100} moved MLPs from clearly-below to
  approximately-equal, and wd=10 was chosen in 72% of folds. A model that swings that much on the
  regularisation scale is being **regularised wrong**, not starved of capacity.
- **The chosen ridge alpha is 1e4-3.16e4 in essentially every fold**, grid `logspace(0,6,13)`.
  On standardised 3,584-dim features with n~4,900 rows that is roughly twice the *mean* eigenvalue
  of the Gram matrix: most directions are shrunk by more than half.
- Ridge still overfits at that alpha (entropy train 0.48 vs test 0.28). So the regime is
  **variance-limited, not bias-limited.** Extra flexibility costs more than it buys. The effective
  sample size is ~198 independent questions, not 4,900 rows (that is idea 1's subject; these two
  ideas are the same question from two sides).
- R3 found ridge's weights dense over ~1,600 of 3,584 dims, and showed RF cannot fit even a
  noise-free linear target on this X. If the truth is linear-plus-noise, an MLP's best case is a
  tie, and tying requires SGD to find a heavily-shrunk linear solution it has no reason to find.
- **LoRA is already vindicated and is not the problem.** The frozen-feature control settles it:
  the same head and recipe on the *stored* layer-21 states gave entropy -0.254 where ridge on
  those identical features gives +0.278. The recipe cost ~0.5 R² of self-inflicted damage. When
  ridge was fitted on LoRA-adapted states instead, LoRA showed a small significant *gain*
  (+0.01 to +0.026 at 30-90%). Do not re-litigate LoRA; if you touch it at all it is GPU work
  (~12 GPU-h for R8) and needs the orchestrator's approval first.
- Step-4a settings that worked for SGD heads on these features: head lr 1e-4, batch 256, wd 10,
  patience 30, <=300 epochs.
- A linear head trained by AdamW with decoupled weight decay is **not** ridge: decoupled decay is
  not a CV-chosen L2 penalty per target, and one shared early-stop across 5 targets is not 5
  independently selected alphas.

## Fact sheet

- `source env.sh`; `$PY` = `envs/goal1` python. Never compute on the login node.
- Data: GPQA-Diamond, 198 questions, 182 with all 25 CoTs and 16 with 19-24; checkpoint fractions
  0.1/0.3/0.5/0.7/0.9; n~4,913 rows per checkpoint.
- Activations `runs/goal2/gpqa25/activations_ckpt/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B/`,
  (rows, 28, 3584) fp16 across 32 shards.
- **Layer indexing: `Act` in `sweep.py` is 1-indexed — `act_L21` reads stored index 20.**
- **Reuse `sweep.py`: `load_meta`, `Act`, `fixed_fold_of`, `_fixed_folds`, `Ridge`/`RidgePath`,
  `get_svd`.** `load_meta`'s joins define the row set; rolling your own will disagree with
  `ref_fixed`. `src/probing/r3_rf_diagnosis.py` is the precedent for importing these.
- Folds: `fixed_fold_of(run_dir, 5, 0)`, by question, 40/40/40/39/39. Inner 3-fold by question for
  hyperparameter selection, then refit on the whole training split. Standardise on training rows
  only. **Use these folds, unchanged.**
- Reference: `runs/goal2/gpqa25/sweep/ref_fixed/{results.csv,preds*.csv}`.
- Targets: `answer_entropy_bits_mm`, `p_mode`, `rem_len_mean`, `rem_len_std` (R²),
  `committed_by_here` (accuracy). 9 all-NaN rows to drop. Round `fraction` to 3 dp before merges.
- `learnings.md` traps: `HF_TOKEN=dummy` for tokenizer loads; `/tmp` scratchpad invisible to
  compute nodes; starting several `envs/goal1` jobs simultaneously can crash on an outlines/diskcache
  sqlite error, so submissions must be staggered (the orchestrator handles this).

## Run these two FIRST — they bound the whole effort

**B1 — correctness control.** Train a `hidden=0` (linear) head on `act_L21` with an explicit L2
penalty **in the loss** at ridge's chosen alpha, full-batch, a real optimiser (LBFGS, or Adam with
a long schedule and tight tolerance). It must reproduce ridge's closed-form solution. **Compare the
weight vectors directly — cosine similarity and relative norm difference — not just the scores.**
Target: cosine > 0.999 and test R² matching ridge to ~1e-3.
**If this control fails, stop and report immediately: it would mean our SGD-head machinery cannot
even find the linear optimum, which retroactively explains every MLP and LoRA-head result we have,
and it is a bigger finding than anything else in this spec.**

**B2 — the ceiling.** Kernel ridge regression, **closed-form, so zero optimisation confound**:
RBF (sweep bandwidth, e.g. median-heuristic scaled by {0.25,0.5,1,2,4}) and polynomial degree 2,
alpha on the same `logspace(0,6,13)`, same folds and inner CV. n~4,900 means a 4,900^2 kernel —
trivially affordable on CPU.
- If RBF and poly-2 never beat linear ridge, **there is no nonlinear signal in these features**,
  "improve the MLP" tops out at "tie ridge", and the honest deliverable becomes a tie plus a
  demonstration of why. Report this the moment you know it — it changes how much effort the rest
  deserves.
- If they do beat linear ridge, you have an upper target to chase and the gap is real and findable.

## Then the improvement ladder, roughly in order of expected payoff

1. **Skip connection from ridge.** `y = w_ridge . x + g(x)`, with `g` an MLP initialised to output
   exactly zero. This makes "MLP >= linear" true *by construction*: it starts at ridge's score and
   can only climb. Staged variant: fit the MLP on ridge's residuals. **This is the single most
   promising item — it removes both the optimisation and initialisation excuses.**
2. **L2 in the loss**, swept on ridge's own `logspace(0,6,13)` grid, replacing decoupled AdamW
   decay. This is the specific thing that was broken before.
3. **Seed ensembling / weight averaging.** In a variance-limited regime this is the most reliable
   gain available; seed std was ~0.01, so averaging should bank most of it. Cheap. Report both the
   single-seed and ensembled numbers so the comparison stays honest.
4. **Per-target heads and per-target early stopping**, instead of 5 outputs off a shared trunk with
   one shared stopping criterion.
5. **LR schedule + longer training** under per-target patience.
6. **Bagging over question bootstraps** — again attacks variance rather than bias.
7. **Effective degrees of freedom** of the chosen ridge fit: `sum(d_i^2 / (d_i^2 + alpha))` from the
   SVD. One line, and it gives the explanatory sentence ("ridge is effectively using N of 3,584
   directions"). Cheap context for whatever you find.
8. **Gap vs training size** — MLP minus ridge at 40/80/120/198 training questions. If the gap closes
   as n grows, we can say how much more data an MLP would need. Shares machinery with idea 1's part D.

## Reporting rules

- Every comparison against ridge must be on **the same rows and the same folds**, with
  **paired question-bootstrap CIs** (2,000 draws, resample the 198 questions). A raw score
  difference without a paired CI is not a result — the CI half-width for between-feature
  comparisons is about +/-0.05 R², and the effects here are ~0.01.
- Report what you tried **and failed**, with numbers. A ladder rung that did not help is a result.
- Never select on test. Any hyperparameter must be chosen on inner folds. If you report a
  best-of-many number, label it as such and give the honestly-selected number alongside.

## Rules

- **Write only** into `runs/goal2/gpqa25/sweep/_mlpgap/`, `docs/figures/mlpgap/`, and new
  `src/probing/mlpgap_*.py` files.
- **Never** edit existing shared code. Import from `sweep.py`; do not modify it. If you think a
  change is needed, stop and ask.
- **Never** write into `runs/goal2/gpqa25/{probes_ckpt,ablation}`.
- **Do not touch `CLAUDE.md`.**
- **Do not submit Slurm jobs.** Prepare, verify with `sbatch --test-only`, report the command.
- CPU only, partition `small`, always set `--mem` and `-t`. **GPU needs approval** — ask the
  orchestrator (partition would be `h100_all --gres=gpu:1`, and it is heavily contended).
- Stuck? Append to `_mlpgap/QUESTIONS.md` and keep going on whatever is unblocked.
