# Plan: rerun the interesting GPQA experiments on MMLU-Pro and GlobalOpinionQA

Written 2026-11-01. Deadline **Tuesday**. Priorities set by the user: **(1) frontier-model ICL** (the
professor's main interest), **(2) question difficulty**, (3) the probe reference everything compares to.
**Explicitly out of scope:** GEPA (rishp owns it), LoRA, MLP/kernel architecture work, embedders,
multilayer, the full 22-approach master table.

## The data (verified 2026-11-01, see CLAUDE.md Results index)

| | GPQA | MMLU-Pro (`mmlupro_v1`) | GlobalOpinionQA (`globalop_v1`) |
|---|---|---|---|
| questions | 198 | 2,500 | 2,555 |
| CoTs/question | 25 | 5 | 5 |
| checkpoint rows | 24,566 | 61,971 | 63,684 |
| 0% rows | 198 | 2,500 | 2,555 |
| K | 25 (+100 on 20%) | 40 (+100 on 20%) | 30 (+100 on 20%) |
| folds (fixed, 5) | 40/40/40/39/39 | 500 x 5 | 511 x 5 |

Paths: `runs/goal1/<tag>/<ds>/DeepSeek-R1-Distill-Qwen-7B` (= RUN_DIR) and
`runs/goal2/<tag>/{activations_ckpt,activations_ckpt_f0,_f0_input}`.

**Infrastructure already confirmed working on both:** `sweep.py` takes `act_dir`/`run_dir` from its
config (no GPQA hardcoding), `fixed_fold_of` gives even folds, `_f0_input/{checkpoints.jsonl,examples.csv}`
exist, activations are exactly 1:1 with labels and untruncated.

## Known differences that change the analysis

1. **`p_gold` is null on ALL GlobalOpinionQA rows.** No accuracy targets, and **the difficulty banding
   cannot be built the GPQA way** (bands come from `p_gold` at the 0% cut). A proxy is required - see P2.2.
   `committed_by_here` IS still valid there (commitment = concentration, not correctness).
2. **Two-tier K** on both (20% of rows at the pilot's K=100). Score with `n_continuations` recorded;
   decide empirically whether it matters (P1.3).
3. **`n_options` varies** (MMLU-Pro mostly 10, some 4; GlobalOpinionQA 2-15), so entropy ceilings differ
   per question. Use `answer_entropy_bits_mm` as the primary target for continuity with GPQA, and report
   `answer_entropy_bits_mm_norm` alongside. **`_norm` is NOT bounded by 1** (max 1.58) because the
   Miller-Madow correction can exceed `log2(n_options)`.
4. 5 CoTs/question, not 25, so within-question variance is far thinner. The between/within decomposition
   from round 4 will be weaker by construction; do not read a small within-question R2 as a new finding.

## Reporting conventions (carried over, non-negotiable)

- Score **within each cut point, then average. Never pool across depth.**
- **Always report BOTH answer entropy and remaining length.**
- `committed_by_here`: **AUC alongside accuracy**, plus the majority-class baseline.
- No em dashes in docs or slides.

## Phases

### P0. Enablement (blocks P3 only)
- P0.1 `icl_predict.py` has module-level `RUN_DIR`/`F0_INPUT`/`OUT` hardcoded to gpqa25, and
  `icl_openai.py` has `DOMAIN_CSV = data/gpqa_diamond/csv/train.csv` plus
  `S = runs/goal2/gpqa25/sweep/ref_fixed`. Make all of these env-overridable with the **current GPQA
  values as defaults**, then prove GPQA behaviour is unchanged.
- P0.2 Domain/subject mix for the subset matchers comes from `RUN_DIR/raw/examples.csv`:
  `category` (MMLU-Pro) or `source` (GlobalOpinionQA). GPQA keeps its `High-level domain` column.

### P1. The probe reference (CPU, unblocked, critical path)
Everything else compares against this, so it goes first.
- P1.1 Smoke a cut-down config to measure runtime, then shard. 62k rows is 2.5x GPQA, and ridge solves
  by SVD of the training block, so time and memory must be measured not guessed.
- P1.2 `ref_fixed` equivalent per dataset: ridge + `logistic_deep` (C to 1e-8) on **layer 21 last token**,
  plus the **TF-IDF-on-prefix** and **prefix_len** baselines. 5 targets x 5 cuts.
- P1.3 Two-tier K check: rescore one cut with the K=100 rows dropped. If the gap is inside the CI,
  keep all rows and note it; otherwise go uniform.
- P1.4 Question-bootstrap CIs, paired vs TF-IDF and prefix_len.

### P2. Question difficulty (CPU, unblocked, high interest)
- P2.1 MMLU-Pro bands: replicate `_difficulty/question_difficulty.csv` from the 0% cut's `p_gold`.
  NB the 0% cut has only **5** continuations per question here (vs 25 on GPQA), so `p_gold` is a
  5-point estimate: coarse, 6 possible values. Use `band_quintile` on `p_gold` with ties broken by
  the 0% entropy, and say so.
- P2.2 GlobalOpinionQA proxy: **decide and document**. Candidates, in preference order: 0%-cut
  `answer_entropy_bits_mm` (how spread the model's opinion is before reasoning), 0%-cut `p_mode`,
  or `n_options`-normalised entropy. This is a judgement call - write the reasoning down.
- P2.3 Probe-side band stratification (`icl_bands.py` method): score the probe inside each band,
  **offset-corrected** (`score_recentred`) as well as as-stored, plus the offset-free within-cut
  correlation. Needs P1 preds.
- P2.4 Probe-side partialling (`icl_decompose.py` part 2): remove a quadratic in the difficulty
  measure from target and prediction, refitted inside each bootstrap draw. Needs P1 preds.
  **Caveat to repeat: a label-derived difficulty over-removes, so partialled numbers are a LOWER bound.**

### P3. Frontier-model ICL (BLOCKED ON API CREDIT - see below)
- P3.1 Prep fully so it launches the moment credit exists: generalised code, row subsample, exact
  token counts, dry-run prompts, cost estimate per config.
- P3.2 Configs to run, in priority order (round 5/6 established k=0 is the best setting and that
  examples HURT, so do not repeat all 8): **`ckpt` k=0**, then **`ckpt` k=10** (confirms examples hurt
  on a new dataset), then **`blind` k=0** if budget allows (tests whether hiding depth costs anything
  where it is informative).
- P3.3 Score, CIs vs the P1 probe on identical rows, then the P2 partialling on the ICL predictions.
  **The GPQA headline to test: the frontier model ties our probe on entropy, and that tie is question
  difficulty.** Does it replicate off GPQA?

### P4. The 0% checkpoint probe (CPU, unblocked, much better powered now)
`sweep.py:88` hardcodes `labels/checkpoint_labels.csv`; the f0 labels are a separate file with
`cot_id=-1` and no commitment columns. ~30-line change plus configs. GPQA had 198 rows at this cut;
MMLU-Pro has 2,500 and GlobalOpinionQA 2,555, so for the first time this is well powered. Prefix length
is constant at 0%, so the length confound vanishes, and one row per question means the probe can only
read question difficulty - its R2 is a direct **measurement** of the difficulty floor.

## THE BLOCKER: no API credit

`gpt-6-luna` returns `429 insufficient_quota` / `credit_balance_exhausted` on the `yuvraj` key
(tested 2026-11-01). This is the same wall rishp's GEPA prep hit on his own key. **P3 cannot run until
credit is added**, and P3 is the professor's top interest.

Cost to budget: the GPQA full run was 1,178 rows x 8 configs = 9,424 calls for **$17.86**, i.e. about
$2.20 per 1,178-row config. The three configs above on both datasets is roughly **6 configs x ~$2.20
= $13**, and k=0 is the cheapest of them since it sends no examples. **Adding $25-50 covers the plan
with headroom.** Note CLAUDE.md already records an unexplained gap on the previous card: ~$33 tracked
spend against a $100 credit.
