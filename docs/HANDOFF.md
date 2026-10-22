# Handoff — read this first

_Written 2026-10-15 for new Claude sessions starting experiments in this repo._

Read in this order:
1. **`CLAUDE.md`**: rules (they take priority), cluster notes, and the **Results index** (every result: path, job, numbers).
2. **This file**: where things stand, how the methods work, how to run things, open items.
3. **`learnings.md`**: pitfalls we already hit. It is local only, not in git.
4. If you need more: `docs/update_2026-10-05.md` (round 1) and `docs/update_2026-10-12.md` (round 2) are the write-ups sent to the team. `PLAN.md` has the original plans and the user's decisions.

## The project in five lines
- **Question:** does a reasoning model's hidden state at a point in its chain of thought predict how the rest of the chain will go, beyond what the text itself shows?
- **Model:** DeepSeek-R1-Distill-Qwen-7B ("our model"). **Data:** GPQA-Diamond, 198 questions × 25 CoTs, cut at 10/30/50/70/90% (plus a 0% checkpoint, not yet probed).
- **Labels:** from each cut point, the model was continued 25 times. The targets are answer entropy (Miller–Madow, `answer_entropy_bits_mm`), `p_mode`, `rem_len_mean`, `rem_len_std` (scored by R²) and `committed_by_here` (scored by accuracy).
- **Features:** layer-21 hidden state (3,584 dims) at the last token of the prefix.
- **Reference probe:** ridge (and logistic for committed) on those features: `runs/goal2/gpqa25/sweep/ref_fixed/`. Every comparison is against it.

## Headline results so far
- **Probe type:** ridge is the best probe. MLPs (even well regularised), PCA→ridge, random forests (default and strongly regularised) and gradient boosting all score at or below it. R3 explains why trees lose: the signal is a weak linear trend spread over ~1,600 dims.
- **Our activations vs text baselines:** our activations beat TF-IDF, prefix length, Qwen3-Embedding-8B and F2LLM-v2-8B. They are **not** better than the layer-21 states of Qwen2.5-7B-Instruct or Qwen2.5-Math-7B on the same text. So the signal is "a 7B LM reading the text", not something specific to the reasoning model. Caveat: those controls share pretraining with our model (R2: same architecture and tokenizer, weights differ by about 0.79). An unrelated model (e.g. Llama) has not been tested.
- **In-context LLMs (R1):** Gemma-4-31B, Qwen3.5-9B and our model were given k labelled examples and asked for the 5 numbers. Best: Gemma, k=0, thinking off. Its calibrated score is not significantly different from ridge on entropy, p_mode and committed, but much worse on remaining length. More examples and thinking did not help.
- **LoRA (R8):** LoRA's own trained heads were always below ridge; their training recipe was the bottleneck, as shown by frozen-feature controls. Ridge on LoRA-adapted layer-21 states gives a small, significant gain at 30–90% (+0.01 to +0.026 R²) and none at 10%.

## How the evaluation works (so new results stay comparable)
- **Folds:** 5 fixed folds by question (`sweep.fixed_fold_of`, seed 0; 40/40/40/39/39 questions). A question keeps its fold at every checkpoint and for every probe and feature. All 25 CoTs of a question stay together. **Reuse these folds for any new experiment.**
- **One model per checkpoint.** Inside each outer training split, hyperparameters are picked by an inner 3-fold split by question, then the model is refit on the whole training split.
- **Standardisation:** features are standardised using training rows only.
- **Grids:** ridge α ∈ logspace(0, 6, 13); the chosen α is ~3e3–3e4. Logistic C ∈ logspace(−4, 1, 6); the chosen C is mostly 1e-4, the edge of the grid (open item).
- **Test score:** pooled out-of-fold. Each row is predicted by the fold model that didn't see its question, then one R² (or accuracy) is computed over all rows: `1 − Σ(y−ŷ)² / Σ(y−ȳ)²`, where ȳ is the mean over all rows.
  - **"Mean over checkpoints"** is the average of the 5 per-checkpoint scores.
  - **`train` is different:** it is the mean over folds of each model's score on its own training rows.
- **Error bars:** question bootstrap, 2,000 draws. Each draw resamples 198 questions with replacement (all their rows come along), recomputes the pooled score on the stored predictions (no refit), and the CI is the 2.5th–97.5th percentile.
  - Comparisons use **paired** draws (same questions for both probes, difference per draw). These are much tighter than single-score CIs (about ±0.05 R²).
  - The bars cover question sampling only, not seeds or label noise.
- **ICL calibration:** LLM guesses are on the wrong scale, so the calibrated score fits `y ≈ a + b·guess` per target and checkpoint on the other 4 folds and applies it to the held-out fold. Raw R² is negative almost everywhere; the exception is entropy for Gemma k=0 (0.139 off, 0.010 on).
- **Rows:** 9 of 24,773 label rows have all-NaN targets. Drop them everywhere.

## How to run things
- **Setup:** `source env.sh` from the repo root (inside `sbatch --wrap` use `. ./env.sh` and `\$PY`). Python: `$PY` (envs/goal1: analysis, sweeps, LoRA), `$PY_ES` (activation capture), `$ENVS_DIR/icl/bin/python` (ICL, vLLM 0.31).
- **Never compute on the login node.** For CPU, use `srun -p small -c <n> --mem=<m> -t <min> bash -c '. ./env.sh; $PY ...'` or `sbatch -p small`. For GPU, use `-p h100_all --gres=gpu:1`. Ask the user before long jobs; check with `sbatch --test-only` first.
- **Probe sweeps:** write `configs/sweep_<name>.yaml` (format at the top of `src/probing/sweep.py`; `configs/sweep_ref.yaml` is the reference). Then run `$PY src/probing/sweep.py --config configs/sweep_<name>.yaml [--shard i/N]` and finish with `--merge`. Any precomputed feature (e.g. another LLM's hidden states) plugs in as `type: npy` with a keys file.
- **Hidden states of another LLM:** `runs/gpqa25/10_embed.sh` with `--layer 21 --no-instruct` (how the Qwen2.5 controls were made; see the Results index for exact flags). New models need downloading to `cache/`; `HF_TOKEN=dummy` is needed for tokenizer loads (see `learnings.md`).
- **LoRA:** `runs/gpqa25/12_lora.sh` with `LORA_FRACTIONS=...` and `LORA_EXTRA="..."`. Not `FRACTIONS`, because `common.sh` overwrites it. Use `--save-states` + `src/probing/lora_ridge.py` for ridge on adapted states, and `src/probing/lora_compare.py <fraction>` for paired CIs against ridge.
- **ICL:** `runs/gpqa25/11_icl.sh` (`ICL_MODEL`, `ICL_KS`, `ICL_EXTRA`), then `icl_predict.py score`, then `src/baselines/icl_compare.py`.
- **CIs for sweep results:** `src/viz/round2_metrics.py` (R5) or `src/probing/bootstrap_ci.py`. Always pass `--preds`/`--out-dir` under `runs/goal2/gpqa25/sweep/`, because the defaults point at yuvraj's folder.
- **Outputs:** everything goes under `runs/` (gitignored). Slurm stdout goes to `runs/logs/slurm/slurm-<id>.out` (set in every `runs/gpqa25/*.sh`; submit from the repo root). The session scratchpad in `/tmp` is not visible to compute nodes.
- **Log every new result** in the CLAUDE.md Results index as soon as it exists.

## Working with other sessions and people
- **Several Claude sessions** (e.g. exp-A, exp-B, "probing chat 1") may use this folder at once. Give each experiment its own output folder (e.g. `runs/goal2/gpqa25/sweep/<new name>/`). Don't overwrite or delete another session's outputs, and don't commit another session's files unless the user says so.
- **CLAUDE.md is shared.** Edit it with small, targeted changes (not whole-file rewrites) so concurrent edits don't clobber each other.
- **yuvraj:** never write into `runs/goal2/gpqa25/{probes_ckpt,ablation}`. Before any commit, `git fetch` and check `git log HEAD..origin/main`. Never force-push. Push only when the user explicitly asks.
- **Staggering:** stagger job submissions by ~1 min (simultaneous starts hit an sqlite cache error; see `learnings.md`).

## Git state (2026-10-15)
- Last push: `56e71b0` (2026-10-12) = origin/main.
- **Not yet committed:**
  - `docs/update_2026-10-12.md`, `src/viz/update_1012_figures.py`, `docs/figures/round2/fig{6,7,8}_*.png` (round-2 write-up);
  - `CLAUDE.md` edits;
  - `#SBATCH --output` lines in 18 `runs/gpqa25/*.sh`;
  - this file.
- `tmux-client-*.log` / `tmux-server-*.log` in the repo root are tmux debug logs, not project files. Don't commit them.
- The round-2 update is also a Claude doc: https://claude.ai/code/artifact/79a0ea60-86f7-46eb-b484-ab12cb218c14. It still has the old sentence "raw R² is < 0 for every setup"; the local markdown is corrected.

## Open items (the user picks)
1. **Unrelated-LLM control** (e.g. Llama-3.1-8B, layer ~2/3 deep, last token, same text) through the same ridge sweep and paired CIs. This decides whether the signal is generic to any LM.
2. **0% checkpoint probes:** labels and activations exist (`checkpoint_labels_f0.csv`, `activations_ckpt_f0/`); a sweep config is needed. Prefix length is constant there, so the result is free of the length confound.
3. **Other layers** (all 28 are stored).
4. **Logistic C grid below 1e-4** for `committed_by_here` on our activations.
5. **LoRA follow-ups:** seeds, other ranks or layers for the ridge-on-states result.
6. **Final write-up.**
