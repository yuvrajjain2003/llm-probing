# Plan — stronger baselines + probe sweeps

_Written 2026-10-04. Status of each step is tracked in `CLAUDE.md` → Current work._

## STATUS of round 1 (updated 2026-10-05) — everything below this table is the original plan
All results: `CLAUDE.md` → "Results index". Write-up: `docs/update_2026-10-05.md`.

| step | status | where / note |
|---|---|---|
| 0. 0% checkpoint fix + labels + activations | **DONE** | code fixed; `checkpoint_labels_f0.csv`; `activations_ckpt_f0/` (job 4672844). **Not yet probed** (→ 4d) |
| 1. Sweep pipeline (`sweep.py`) | **DONE** | reproduces old results 90/90; fixed per-question folds added (default) |
| 2. Pick embedders | **DONE** | `docs/embedders.md` |
| 3. Embeddings + baseline 1 | **DONE** (ridge/logistic) | Qwen3-Emb-8B, F2LLM-v2-8B, + other-LLM controls Qwen2.5-Math-7B (cut 4k) and Qwen2.5-7B-Instruct (L21 last token). Best non-linear probes NOT yet run on these |
| 4a. MLP | **DONE** | lr 1e-4 (not 1e-3), ≤300 epochs, patience 30 (from 4a_diag). wd grid rerun at {1,10,100} because ≤0.1 had no effect |
| 4b. PCA-k → ridge | **DONE** | |
| 4c. Random forest | **DONE** (shrunk grid) | dropped max_depth 16 everywhere and max_features 0.3 on raw (cost) |
| 4c'. Gradient boosting | **DONE** | fair score searched 6 of 24 configs (others had no effect) |
| 4d. Follow-ups | **NOT STARTED** | other layers; 0% probes; bootstrap CIs; best probes on embeddings. New idea: unrelated-LLM control (e.g. Llama-3.1-8B) |
| 5. LLM-predicts baselines | **NOT STARTED** | user to choose the LLM |
| 6. Write-up | **interim done** | `docs/update_2026-10-05.md` |

**Headline so far:** ridge on full layer-21 activations is the best probe (no MLP/RF/HGB/PCA beats it). Activations beat TF-IDF/embeddings, but other Qwen2.5-7B models' hidden states on the same text match them.

## ROUND 2 (user list, 2026-10-09) — current work; read this first
Status per item is also in `CLAUDE.md` → Current work. Order: cheap CPU items first (R6, R7, R5, R2, R4, R3), then GPU items **R1 before R8** (user, 2026-10-09).

**User decisions (2026-10-09):**
- R1: **do NOT shorten example prefixes** (full question + prefix text for every example). Test **two LLMs: one ~8B and one larger.**
- R1 before R8.
- R8: purpose = **compare with the existing probes → same settings** (layer 21, last prefix token, same 5 targets, same fixed folds, same metrics); **our model first** (control model later, if at all). OK to install `peft`.

**R1 consequence of full prefixes (checked):** input = question + prefix is median ~1.9k tokens, max ~14.9k (Qwen tokenizers; `runs/logs/gpqa25/embed_*.log`); mean ≈ 2.7k (from 8.96k mean chars, ~3.3 chars/token — estimate). So k examples need ≈ 2.7k·k tokens: k=10 ≈ 27k, k=30 ≈ 80k, k=100 ≈ 270k, k=300 ≈ 0.8M. **k=1000 (~2.7M tokens) fits no available model → max k is set by context: ~85 at 256k, ~350 at 1M.** (Superseded grid — see k grid below.)
**R1 k grid (user, final 2026-10-09): k ∈ {0, 10, 30, max} for ALL three models** (same k everywhere for easy comparison; max differs per model: ours ≈40, Gemma-4-31B ≈85, Qwen3.5-9B ≈350 with YaRN — estimates; set exactly by counting tokens of the chosen examples, leaving ~15k for the longest test prefix). Old note, superseded:  k=100 (~270k tokens, estimate) is just over a 256k/262k context → for Gemma-4-31B (256k) k=100 is skipped and its max is ~85; Qwen3.5-9B needs YaRN (→ ~1M) for k=100 and max (~350). Count tokens of the actual chosen examples to set "max" exactly (leave room for the longest test prefix, ~15k).
**R1 models (user, corrected 2026-10-09): THREE — our model (DeepSeek-R1-Distill-Qwen-7B), Qwen3.5-9B, Gemma-4-31B.** Our model = reference (same LLM that wrote the chains), the other two = "different LLM".
- Our model: config `max_position_embeddings` 131072, but rope_theta 10000 and its base (Qwen2.5-Math-7B) is a 4k model; chains were generated with max_tokens 16384 → quality on 100k+ inputs unverified. Max k ≈ (131k − 15k)/2.7k ≈ 40, so its grid is {0, 10, max≈40} (no 50/100). It is a reasoning model: it will think before answering → either let it think (cost, set a token cap) or prefill `</think>` to force a direct answer; decide at implementation, note which.
**R8 (user, corrected 2026-10-09): our model ONLY for now; expand to other models only if needed.**
**Final decisions (user, 2026-10-09):** R1 — subsample (~1,178 rows), Qwen3.5-9B native context (k max 92), GPU smoke tests first, thinking off AND on. R8 — one LoRA model per checkpoint (not pooled), GPU check + pilot approved. Notes kept for a later expansion:
- Comparison per model needs that model's FROZEN probe too (ridge on its own hidden state, same text/position) → extract with `embed_prefixes.py --layer L --no-instruct` (as for the `feat_*` controls), so "LoRA gain" = LoRA score − frozen-ridge score of the same model.
- Layer: 21 of 28 for our model. Others: same relative depth (75%) unless the user says otherwise — layer counts of Qwen3.5-9B / Gemma-4-31B not yet checked.
- Input text for other models = question + choices + prefix, no chat template (same as the `feat_*` controls); our model = exactly what was probed.
- Qwen3.5 is a hybrid (Gated DeltaNet + attention) and both are multimodal → check `peft` target modules and loading the text-only part. Gemma-31B LoRA on ~15k-token inputs likely needs 2 H100s; cost roughly 4× our model (estimate).
**R1 cost trick:** use ONE fixed example set per (fold, k) and put it first in the prompt, test prefix last → vLLM prefix caching computes the long example block once per fold; each test row only adds its own ~2–15k tokens. Nested example sets (k=10 ⊂ 30 ⊂ max), sampled from training-fold questions, balanced over checkpoints. Non-thinking mode, answer as JSON numbers.
**R1 model notes (Qwen3.5-9B + Gemma-4-31B confirmed by user; our model added — see below):**
- ~8B: `Qwen/Qwen3.5-9B` — 262k native context, extendable to ~1M (YaRN). Caveat: Qwen family, like our model.
- larger: `google/gemma-4-31B` (dense, 256k context) — different family from our model (cleaner "different LLM"), fits 1–2 H100. Alternative if more examples matter more than family: `Qwen/Qwen3.5-122B-A10B` (262k native, ~1M with YaRN) but needs ~4 H100 (bf16), hard to get.
- **To check before running:** neither is downloaded; vLLM support (our vLLMs: 0.5.5 in goal1 — too old; easysteer fork reports `0.1.dev14461+g21d7a9f61`, transformers 4.57.6 — support for these models unknown) → likely a new env with current vLLM; long-context quality at 256k+ (YaRN) is unverified for our task.

**Facts checked 2026-10-09 (don't re-verify):**
- Every sweep `results.csv` has a `train` column next to `test` (in-sample score of the same fits) → R6 needs no new runs.
- Out-of-fold predictions (`preds*.csv`) exist for every `*_selected`, `ref_fixed`, `4b_pca`, `emb_*`, `feat_*` sweep (not for `*_ablation` MLP runs) → R5 and R7 need no refitting.
- `src/probing/bootstrap_ci.py` resamples QUESTIONS from a wide `predictions.csv` (made by `sweep.py --export ... --pred F/P --base name=F/P`). Its defaults point at yuvraj's `runs/goal2/probes_ckpt/` → **always pass `--preds` and `--out-dir` inside our `sweep/` folder.** It was written for the 3-CoT data; check it still groups by question with 25 CoTs.
- "Existing subsampling" in the repo: only `src/analysis/k_sufficiency.py` (subsamples the K continuations, not questions). No question-subsampling / learning-curve script found in `src/`; old notes mention `runs/goal2/logs_ckpt/lcurve.log`, not present here. Ask the user what they meant if it matters.
- `peft` is NOT installed in either env (`envs/goal1`, `envs/easysteer`) → R8 needs an install (ask first; prefer `envs/goal1`, torch 2.4).
- Weights already in `cache/huggingface/hub`: R1-Distill-Qwen-7B, Qwen2.5-Math-7B, Qwen2.5-7B-Instruct, Qwen3-Embedding-8B, F2LLM-v2-8B. Plain Qwen2.5-7B (base) is NOT downloaded.
- Input lengths: question+prefix up to ~15k tokens (90% checkpoint median 4.3k). Embedder "trained length" was not found for Qwen3-Emb / F2LLM-v2 (`docs/embedders.md`).

| # | item | how | cost | status |
|---|---|---|---|---|
| R1 | **ICL baseline**: a DIFFERENT LLM sees k labeled examples (prefix → entropy / remaining length), predicts for a new prefix; k = 10 … 1000 (or max that fits) | Examples from training-fold questions only (fixed folds), never the test question. Ask for a number; parse it; score R² + MSE on the same rows/folds; also a linearly calibrated version (fit on train folds). k=0 = zero-shot (baseline 2). Problem: full prefixes are ~2–15k tokens, so even k=10 overflows a 32k context → examples must be shortened (e.g. last 256–512 tokens of each prefix) or only the test prefix is full. Score on a subsample of test rows (e.g. 500/checkpoint) to bound cost. | GPU, vLLM; hours | **PREPARED, no GPU yet** — decisions pending, see CLAUDE.md Next steps |
| R2 | **Check model bases** are what we claim (R1-Distill-Qwen-7B built on Qwen2.5-Math-7B; Instruct a sibling) | Compare `config.json` (layers, hidden, vocab, rope theta, max pos) and per-layer weight similarity (cosine / relative diff of each matrix) between the 3 cached models. Load on CPU, one layer at a time. | CPU, <1 h | **DONE** 2026-10-09 (see CLAUDE.md Results index) |
| R3 | **Why RF underperforms ridge** | (a) train vs test R² of RF (overfit vs underfit); (b) can RF even fit a linear signal here: fit RF to ridge's own out-of-fold predictions / to a synthetic linear target on the same X; (c) RF on a few supervised directions (ridge prediction as 1 feature, or PLS-k) — if that closes the gap, the problem is "signal spread over many dims", which single-feature splits handle badly. | CPU, few h on `small` | **DONE** 2026-10-09 (see CLAUDE.md Results index) |
| R4 | **Embedder training length vs our inputs** | Find trained/max length for Qwen3-Emb-8B and F2LLM-v2-8B (papers/configs). Our token counts per checkpoint with each tokenizer (logged in `runs/logs/gpqa25/*embed*`). If inputs exceed trained length: re-embed with the end truncated to that length and compare. | research; GPU only if re-embedding | **DONE** 2026-10-09 (see CLAUDE.md Results index) |
| R5 | **Error bars** | `sweep.py --export` → `bootstrap_ci.py` (question bootstrap, 2000 draws) for: ridge vs each baseline, ridge vs other-LLM controls, ridge vs best MLP/HGB. Also fold-to-fold std. Add CIs to the update doc tables/figures. | CPU, minutes | **DONE** 2026-10-09 (see CLAUDE.md Results index) |
| R6 | **Train vs test R² (overfitting)** | Read `train` vs `test` from existing `results.csv` for each probe; table + plot of the gap per probe/checkpoint. | CPU, minutes | **DONE** 2026-10-09 (see CLAUDE.md Results index) |
| R7 | **MSE next to R²** | Compute MSE (and RMSE in target units) from `preds*.csv`; add to tables/figures. Note: within one checkpoint×target the variance is the same for all probes, so MSE ranks probes the same as R²; it adds absolute scale. | CPU, minutes | **DONE** 2026-10-09 (see CLAUDE.md Results index) |
| R8 | **LoRA + MLP head probe** | Our model (bf16, frozen) + LoRA (rank 8–16, attention + MLP, layers ≤21) + MLP head on the layer-21 last-token state; one multi-task head for all 5 targets (MSE + BCE); same fixed folds (5 fine-tunes), early stopping on held-out training questions. Pilot first: 1 fold, inputs cut to last 4k tokens, to measure time. Overfitting risk (198 questions). Fairness: same LoRA on a control model (Qwen2.5-7B-Instruct). | GPU; est. 10–30 GPU-h (not measured) | **PREPARED, no GPU yet** — decisions pending, see CLAUDE.md Next steps |

## Goals (from user, 2026-10-04)

**Question:** do the model's activations at a CoT prefix tell us more about the rest of the reasoning than the prefix text does?

**A. Stronger baselines** (all use existing data — no new generation)
1. **Sentence embeddings.** Embed each prefix with the strongest available embedders (try several, keep the best), train probes on them, compare to activation probes.
2. **Another LLM predicts.** Ask a different LLM (not the one that wrote the CoTs) to predict the targets from the prefix text.
3. **Same, with in-context examples** (few labeled examples in the prompt).
4. **ON HOLD:** force the model to answer at each prefix, use that answer entropy as a predictor (incl. of remaining length). Needs new GPU generation.

**B. Probe architectures**
- Low-dimensional probes: hidden/bottleneck size ~5–15 (ablate).
- Each with/without dropout, with/without weight decay.
- Random forest probes, with hyperparameter ablation.
- Run the best probes on the sentence-embedding baseline too (fair comparison).

**C. Sweep pipeline.** One script + config that runs every (features × probe × hyperparameters × target × checkpoint) combination and writes one results table. A new probe type = one small class + one config entry. No manual code edits per run.

## Facts this plan is built on (checked 2026-10-04)

- **Data:** `runs/goal2/gpqa25/` — 198 GPQA questions × 25 CoTs × 5 checkpoints (10/30/50/70/90%) = 24,575 rows. Activations: all 28 layers × 3584 dims, fp16, 5.2 GB.
- **Prefix lengths (tokens), median / max:** 10%: 472 / 2.1k · 50%: 2.4k / 8.8k · 90%: 4.3k / 14.8k. → **Embedders need long context** (most classic ones stop at 512 tokens).
- **Current probes** (`probe_checkpoints.py`, `probe_ablation.py`):
  - Ridge (regression) / logistic (binary) on the full 3584-dim activation, regularization picked by nested CV.
  - RBF kernel ridge / SVM.
  - MLP: one hidden layer of **256**, L2 picked from {1e-4…100}, 50 iterations, **no dropout**; plus an unregularized copy.
  - Baselines: `prefix_len` (token count) and TF-IDF on prefix text.
  - Evaluation: separate model per checkpoint, 5-fold CV grouped by question, R² / accuracy, bootstrap CI vs text.
- **Current results (layer 21, best):** linear ≈ kernel > MLP-256 everywhere. Activations beat TF-IDF on all targets by +0.01–0.12 R². But on remaining length at 10–30%, `prefix_len` alone (0.66/0.62) beats activations (0.62/0.59).
- **0% checkpoint: broken.** `make_checkpoints(fraction=0.0)` returns the *first sentence*, not an empty prefix, and then drops the 10% checkpoint as a duplicate. Activation extraction would also reject an empty prefix (assertion). See step 0.

## Steps

### 0. Fix the 0% checkpoint (small, CPU)
- `checkpoint.py`: make fraction 0 give an empty prefix (1–2 line change). `extract_checkpoints.py`: allow an empty prefix (activation at the last prompt token).
- **Labels at 0% need no new generation:** continuing from an empty prefix = sampling fresh full answers, which we already have (25 per question in `runs/goal1/gpqa25/.../raw/sample_records*.jsonl`). Recompute entropy (Miller-Madow) and remaining length (tokens before `</think>`) from those. Note `run_eval`'s `reasoning_tokens` counts differently (includes the post-`</think>` summary) — don't reuse it.
- **0% activations need one short GPU pass** (198 prompts, minutes). Wait for GPUs; ask before submitting.
- Caveat: at 0% all 25 CoTs of a question share the prompt → **only 198 rows**, not ~4,900.

### 1. Sweep pipeline (build first — everything else plugs into it)
- New `src/probing/sweep.py` + a YAML config. Reuses `probe_checkpoints.load()`, the same grouped folds, nested hyperparameter selection, and output format usable by `bootstrap_ci.py`.
- **Features** = named providers: `act:L21`, `emb:<model>`, `prefix_len`, `tfidf`, `llm:<model>`.
- **Probes** = classes with `fit / predict`, hyperparameter grid in config.
- Output: one CSV row per (feature, probe, chosen hparams, checkpoint, target, test score, train score).
- **Sanity check:** reproduce the existing ridge numbers on layer 21 before trusting anything new.
- Runs on CPU (`small` partition, up to 192 CPUs) — GPUs not needed.

### 2. Pick sentence embedders (research, with justification)
Criteria, in order:
1. **Context length ≥ 8k tokens** (90% prefixes reach ~15k).
2. Strong on MTEB **classification / clustering / STS** (closest to "embedding → regression"), not just retrieval.
3. Handles math/LaTeX-heavy text.
4. Runnable here: open weights, fits one H100 (or CPU for small ones).
5. A spread of sizes, including one near 7B so model size isn't the confound.

Deliverable: a short ranked table with sources (leaderboard numbers, model cards), and the truncation rule for prefixes longer than the context (e.g. keep the *end* of the prefix).

### 3. Compute embeddings + run baseline 1
- Embed all 24,575 prefixes per chosen model, save as `runs/goal2/gpqa25/embeddings/<model>.npy` aligned to the activation index.
- Likely needs GPU for the large embedders.
- Run through the sweep: ridge first, then the best probes from step 4.

### 4. Probe architecture sweep — detailed plan (2026-10-04; 4a–4c' DONE, 4d not started — see STATUS)

**Setup (same for every probe, so scores are paired):** layer 21 activations (best for ridge so far), one model per checkpoint (10–90%), the same 5 outer folds grouped by question, hyperparameters picked by inner grouped CV on the training part only. Report test R² / accuracy **and** train score (to see over- vs under-fitting). Targets: `answer_entropy_bits_mm`, `rem_len_mean`, `rem_len_std`, `p_mode`, `committed_by_here`. Drop `will_commit` (never beat majority class). Reference = ridge (`runs/goal2/gpqa25/sweep/baseline`). ~3,900 training rows × 3,584 dims per fold.

**Two ways of reporting (keep separate):**
- *Ablation table:* outer-CV score of every grid point — to see how dimension / dropout / weight decay change things. Optimistic if you pick the max from it; for understanding only.
- *Fair score:* nested selection over the grid (what the pipeline does) — the only number compared against ridge and the baselines.

**4a. Small MLP** (`mlp` probe already in `sweep.py`): 3584 → h → ReLU → dropout → 1.
- hidden h ∈ {5, 8, 10, 12, 15}, plus 32 and 256 as references (256 = current probe).
- dropout ∈ {0, 0.1, 0.3, 0.5}; weight decay ∈ {0, 1e-3, 1e-2, 1e-1} (AdamW).
- Fixed: lr 1e-3, batch 256, up to 200 epochs, early stopping (patience 15) on 15% of training *questions*.
- 3 seeds per config; report the mean (small nets vary a lot between seeds).
- Add *input* dropout (on the 3584 inputs) ∈ {0, 0.2} only if 4a shows heavy overfitting.

**4b. Low-dimensional linear controls** — tells "small dimension" apart from "non-linearity":
- PCA to k ∈ {5, 8, 10, 12, 15, 64, 256} (fit on training rows only) → ridge. Needs a small `pca` option on the `act` feature (train-only fit).
- Note: a *linear* probe with a 5–15-dim bottleneck predicting one number is no different from ridge (it's rank 1 anyway). So the bottleneck only matters with a non-linearity (4a) or as unsupervised compression (PCA, here).
- Optional: one multi-task MLP predicting all 4 regression targets through a shared 5–15-dim bottleneck — there a bottleneck *does* constrain things. Only if 4a looks promising.

**4c. Random forest — detailed plan (DONE 2026-10-04, shrunk grid — see STATUS)**

*Inputs* (layer 21, standardised; PCA fit on each training split only, as in 4b):
- raw 3584 dims · PCA-16 · PCA-64 · PCA-256. Trees pick one feature per split, so thousands of weak, correlated dims are a known weak spot — PCA inputs test whether compressing first helps.

*Grid* (`rf` probe in `sweep.py`; sklearn RandomForestRegressor / Classifier):
- `n_estimators` 500, fixed (more trees only lower variance, they don't overfit).
- `max_depth` ∈ {None, 8, 16} — tree size.
- `min_samples_leaf` ∈ {1, 5, 20, 50} — the main regulariser for noisy targets (our labels come from 25 samples each, so they're noisy).
- `max_features` ∈ {sqrt, 0.1, 0.3} — dims tried per split (on raw: 60 / 358 / 1075).
- = 36 configs per input.

*Same protocol as 4a:* fixed per-question folds, one model per checkpoint (10–90%), 5 targets (`committed_by_here` as a classifier: accuracy + AUC).
- **Ablation:** every grid point scored directly, 1 seed (with 500 trees the seed matters little — check on 2 configs with 3 seeds).
- **Fair score:** nested selection on inner grouped folds. To keep cost down, select only over the hyperparameters the ablation shows matter (likely `min_samples_leaf` × `max_features`, ~12 points).
- Don't use RF's built-in out-of-bag score for selection: it's not grouped by question, so the 25 CoTs of a question would leak.
- Note: RF train scores are always near 1 (deep trees memorise), so train-vs-test gaps mean less here than for MLPs.

*Step 0 — timing test first (one short job):* fit one 500-tree forest per `max_features` value on raw and PCA-256, one fold, one target. Expected: `max_features` 0.3 on raw dims is ~18× slower than `sqrt`; if one fit > ~2 min, drop 0.3 on raw (or cap at 0.1).

*Rough cost (estimate, to be replaced by the timing test):* sqrt on raw ≈ 10–20 s per fit on 12 CPUs → ablation ≈ 4 inputs × 36 configs × 5 folds × 25 ≈ 18k fits ≈ 50–100 CPU-node-hours if every `max_features` were as cheap as sqrt; the 0.3 setting dominates. Shard over ~10 jobs.

*After:* feature importances of the best RF per target (which PCA components / dims it uses), via a small script that refits the chosen config on each training split. Best RF config → embeddings (4d).

**4c'. Gradient-boosted trees (user approved 2026-10-04)** — `hgb` probe in `sweep.py` (sklearn HistGradientBoosting).
- Same inputs as RF (raw 3584, PCA-16/64/256), same protocol (fixed folds, ablation + fair score).
- Early stopping on 15% of training *questions* (grouped; sklearn's own early stopping uses a random row split, which would leak), up to 1000 rounds, learning rate 0.05 fixed.
- Grid: `max_leaf_nodes` ∈ {7, 15, 31} × `min_samples_leaf` ∈ {20, 100} × `l2_regularization` ∈ {0, 1} × `max_features` ∈ {0.1, 0.3} = 24 configs.
- Usually much faster than RF on this size (binned features); include it in the RF timing test.

**4d. Follow-ups**
- Best 2–3 configs overall → layers {7, 14, 17, 21, 24, 28}.
- Best 2–3 configs → Qwen3-Embedding-8B features (baseline 1) for a like-for-like text-vs-activation comparison.
- 0% checkpoint (198 rows only): ridge + PCA-ridge only; MLP/RF would just overfit at that n.
- Bootstrap CI (`bootstrap_ci.py`) for best probe vs ridge, vs TF-IDF, vs embeddings.

**Cost (estimate, not measured):** MLP grid 7×4×4 = 112 configs × 3 seeds × 5 folds × 25 (checkpoint×target), with inner selection ×3 → ~10⁵ small CPU fits of seconds each ≈ 50–150 CPU-hours → a few hours on `small` (up to 192 CPUs). RF ≈ 4 inputs × 36 configs, similar order. GPU not needed.

**Order:** smoke test each probe on `--max-questions 20` → 4b (cheap) → 4a → 4c → 4d. Check MLP-256 against the old `probe_ablation` numbers first (sanity: should land near its R² ≈ 0.20–0.24 for entropy, though that run used sklearn's MLP with 50 iterations).

### 5. Baselines 2 & 3 — another LLM predicts the targets
- Prompt: question + prefix → predict remaining reasoning length, answer distribution over A–D (gives entropy, p_mode), commitment.
- Score on the same rows and folds. Also report a version with a linear calibration fit on training folds (LLM numbers may be badly scaled but still informative).
- **ICL version:** add k labeled examples from *training-fold questions only* (never the same question). Long prefixes limit k.

### 6. Write-up
Results table: every baseline vs best activation probe, per target and checkpoint, with bootstrap CIs.

## Open decisions (need user input)
1. **Which LLM for baselines 2–3?** Still open (user: decide later).
2. Scope: GPQA only so far (only dataset with activations) — not explicitly decided.
3. ~~0% activation GPU job~~ — approved and done (job 4672844).
