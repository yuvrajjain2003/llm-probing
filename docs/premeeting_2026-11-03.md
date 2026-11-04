# Pre-read for the 2026-11-03 meeting

Private companion to the slide deck. Written 2026-11-02. Audience: Yuvraj only.
Every number below was read out of the CSV named beside it, not copied from `CLAUDE.md`.
Where a file and `CLAUDE.md` disagree, the file wins and the disagreement is listed in
section 11, "Discrepancies I found".

Skim the headings. Section 1 is the whole story in one page. Sections 2 to 8 are the
evidence. Section 9 is what we have not run (read this one before the meeting).
Section 10 is a crib sheet of likely questions.

---

## 0. The row-set key (read this first, it is the main hazard)

Four different row sets are in play. A score from one is not comparable to a score from
another, and the numbers differ enough to look like a result if you mix them.

| name | what it is | rows per cut | total |
|---|---|---|---|
| **FULL sweep** | all questions, all chains, cuts 10 to 90 percent. GPQA: `sweep/_master/`. New datasets: `sweep/_ref_ci/`. | GPQA 4,911 to 4,914; MMLU-Pro 12,385 to 12,407; GlobalOpinionQA 12,730 to 12,738 | GPQA 24,566; MMLU-Pro ~61,970; GlobalOpinionQA ~63,680 |
| **ICL subset** | the rows the frontier model was actually called on. `sweep/_icl/`. | GPQA 196; MMLU-Pro 495 to 496; GlobalOpinionQA 500 | GPQA 980 (plus 198 at the 0 percent cut = 1,178); MMLU-Pro 2,476 (plus 500 = 2,976); GlobalOpinionQA 2,500 (plus 500 = 3,000) |
| **0 percent cut** | the rendered prompt before any reasoning token. One row per question. `sweep/f0_probe/`. | GPQA 198; MMLU-Pro 2,499; GlobalOpinionQA 2,555 | same |
| **question level** | one row per question, used by the question-text baseline. `sweep/_qtext/`. | same as above | same |

Concrete illustration of why this matters: our probe's entropy R squared is **0.2885** on
GPQA's full sweep and **0.3094** on GPQA's ICL subset. Same probe, same model, same cuts.
The ICL subset happens to be a slightly easier draw. Quote one or the other, never both
in the same sentence without saying which.

Two further conventions, both load bearing:

- **Score within each cut point, then average.** Never pool across depth. Pooled, a single
  scalar (prefix length) matches a 3,584 dimensional probe on remaining length.
- **Never compare an R squared on `log1p(y)` to one on raw `y`.** Section 7 is entirely on
  the log scale and its numbers must not be set beside sections 3 or 4. This is a real
  temptation, because the log scale numbers are the flattering ones.

---

## 1. One page: what is true now that was not true a week ago

1. **We have two new datasets, 12.6 times the questions of GPQA, and both are verified.**
   MMLU-Pro (2,500 questions) and GlobalOpinionQA (2,555 questions), 5 chains each, all
   28 layers of activations, exactly 1:1 with labels, no truncation. Section 2.

2. **More questions did not widen the activations-minus-text gap. It made the gap
   measurable.** Ours minus TF-IDF on entropy is +0.052 (GPQA), +0.055 (MMLU-Pro), +0.036
   (GlobalOpinionQA), essentially unchanged, while our single-score confidence interval
   half-width shrinks from 0.056 to 0.018 to 0.015. Effects of 0.01 R squared are now
   measurable where before they were not. Section 3.

3. **The frontier model's tie with our probe does not replicate.** On GPQA, GPT-6 Luna
   zero-shot tied us on entropy (difference -0.015, not significant). On GlobalOpinionQA we
   beat it (-0.079, significant). On MMLU-Pro we beat it on all five targets, every one
   significant. Section 4.

4. **The difficulty-specific collapse does not replicate either, and the real effect is a
   depth effect.** Partial a quadratic in question difficulty out of both target and
   prediction: GPQA entropy falls 0.289 to 0.025 (not significant), but MMLU-Pro falls only
   0.410 to 0.200 (significant) and GlobalOpinionQA 0.155 to 0.112 (significant, normalised).
   Per cut, the partialled entropy **rises with depth on all three datasets**. Averaging over
   depth, which rounds 4 and 6 both did, hid this. Section 5.
   **Say "question-LEVEL reader" (replicates everywhere), never "question-DIFFICULTY reader"
   (does not replicate).**

5. **New and stronger: removing difficulty makes our advantage over text BIGGER on both
   large datasets.** Ours minus TF-IDF on entropy after partialling: MMLU-Pro +0.0553 goes
   to +0.0627, GlobalOpinionQA +0.0356 goes to +0.0493 (label measure) or +0.0583 (text
   measure), all significant. On GPQA alone it goes the other way, +0.0517 significant to
   +0.0325 not significant. GPQA is the outlier, not the rule. Section 5.

6. **The 0 percent cut plus the question-text baseline is the cleanest pair of results in
   the project.** At the prompt, before any reasoning token, MMLU-Pro activations score
   0.301 on entropy and 0.606 on remaining length, and they beat a tuned TF-IDF on the
   question text on **all five targets, all significant** (entropy 0.301 against 0.119).
   So the signal is question level but **not text recoverable**. Section 6.

7. **The "your length prediction is just token counting" objection is a raw-scale
   artefact.** On `log1p`, ours minus prefix length on remaining length goes from +0.039 to
   +0.164 (GPQA), +0.044 to +0.254 (MMLU-Pro), and -0.013 not significant to +0.521
   (GlobalOpinionQA). All six log-scale gaps significant. Section 7.

8. **GEPA made the prompt worse.** Validation score rose in all 10 runs, test R squared
   roughly halved. Same mechanism as "examples hurt": the learned prompt supplies the scale
   and destroys the ranking. Section 8.

---

## 2. The two new datasets

### 2.1 Why we collected them

Round 4 (idea 1, `sweep/_redundancy/`) measured the question-versus-chain exchange rate on
GPQA directly. Verified from `_redundancy/d2_f*.csv`, mean over cuts, entropy and remaining
length:

| allocation | rows | entropy R2 | rem_len_mean R2 |
|---|---|---|---|
| 12 questions x 25 chains | 300 | 0.063 | 0.266 |
| 30 questions x 10 chains | 300 | 0.168 | 0.355 |
| 60 questions x 5 chains | 300 | 0.182 | 0.386 |
| **150 questions x 2 chains** | **300** | **0.202** | **0.407** |
| 30 questions x 25 chains | 744 | 0.196 | 0.406 |
| 75 questions x 10 chains | 750 | 0.227 | 0.451 |
| 150 questions x 5 chains | 750 | 0.243 | 0.463 |

Read the two bold-adjacent rows: **150 questions x 2 chains (300 rows) matches 30 questions
x 25 chains (744 rows)**, the same quality for 2.5 times fewer rollouts. A new question is
worth roughly 3.5 times a new chain, per row. Separately the learning curve in chains is
still rising at m=25 with no plateau (`_redundancy/d1_f*.csv`, 50 percent cut, entropy
0.152 at m=1 rising monotonically to 0.298 at m=25), so the chains are not redundant, they
are just badly allocated.

Both new datasets are 2,500-plus questions at 5 chains each, which is exactly what that
result recommends.

### 2.2 Why one objective and one subjective

MMLU-Pro is the objective control: a harder, broader multiple-choice benchmark with a gold
answer, so every target GPQA has is available.

GlobalOpinionQA has **no gold answer**. `p_gold` is null on all 63,684 label rows,
`acc_maj5` and `acc_maj10` are null, `acc_single` is 0.0 throughout. That is deliberate: it
tests whether the probe needs **correctness** to work at all. The answer is that it does
not. Answer entropy, p_mode, remaining length and `committed_by_here` are all still
well defined, because they measure concentration of the answer distribution rather than
agreement with a key, and `commitment.csv` is valid there (`committed` mean 0.784).

The cost is that the round-6 difficulty banding, which derives from `p_gold`, cannot be
computed on GlobalOpinionQA. See section 5.4.

### 2.3 Verified ground truth

From `runs/goal1/<tag>/<subset>/DeepSeek-R1-Distill-Qwen-7B/overall_summary.json`, read
directly:

| | GPQA-Diamond | MMLU-Pro | GlobalOpinionQA |
|---|---|---|---|
| questions | 198 | 2,500 | 2,555 |
| samples per question | 25 | 5 | 5 |
| total samples | 4,950 | 12,500 | 12,775 |
| pass@1 | 0.4873 | 0.5579 | 0.0 (no gold answer, meaningless) |
| majority vote accuracy | 0.5505 | 0.5952 | 0.0 (same) |
| mean answer entropy (bits) | 1.0070 | 0.6506 | 0.6937 |
| mean reasoning tokens | 5,696 | 2,335 | 630 |
| parse success | 98.38% | 98.93% | 99.63% |

Three things to notice. MMLU-Pro is **easier** than GPQA for this model (pass@1 0.558
against 0.487) and its chains are **2.4 times shorter**. GlobalOpinionQA's chains are
**9 times shorter** than GPQA's (630 tokens), which matters for section 7: a short mean with
a few 12,000-token outliers is what makes its raw length targets tail dominated.
And parse success is better on both new datasets, so parsing was never the risk we feared.

Label counts, read from `labels/checkpoint_labels.csv`: MMLU-Pro 61,971 rows over 2,499
questions with checkpoints (plus 2,500 at the 0 percent cut); GlobalOpinionQA 63,684 rows
over 2,555 questions (plus 2,555).

### 2.4 Data-quality verification that was actually done

- **Activations exactly 1:1 with labels**, zero mismatches in either direction, on both
  datasets. 64 shards each, fp16, all finite.
- **No truncation.** `max_model_len` 24,576; longest prompt-plus-prefix 15,469 tokens
  (MMLU-Pro) and 14,370 (GlobalOpinionQA).
- **Capture check** (EasySteer hidden states against stock HuggingFace on the same input),
  from `runs/goal2/<tag>/_verify/cmp.txt`:
  - MMLU-Pro: layers 0 to 26 min cosine **0.997160**, mean 0.998697. Clean PASS.
  - GlobalOpinionQA: min cosine **0.989806**. The automatic gate (threshold 0.99) **FAILED**.
    A Claude session on 2026-10-29 then re-checked all 247 positions x 27 layers and found
    median 0.9982, 1st percentile 0.9882, with the low values coming from only 2 token
    positions, and the **last token (the only position `extract_checkpoints.py` reads) at
    0.9982 or better at every layer**. The verdict line in the file now reads
    "PASS (manual override, see above)".

**Be straight about that override if he asks.** The defensible statement is: the automatic
gate failed by 0.0002, the failure is confined to two interior token positions, the position
we actually use passes comfortably, and the layer-and-position alignment convention is
unchanged. It is not a clean pass, and the file says so.

### 2.5 The two known caveats

**(a) Two-tier K.** Both datasets ran a 20 percent pilot at K=100 and the rest at the chosen
K, exactly as GPQA did. Verified from the label files: MMLU-Pro 12,390 rows at K=100 and
49,581 at K=40; GlobalOpinionQA 12,735 at K=100 and 50,949 at K=30. The pilot slice is
exactly `example_id % 5 == 0`.

Verdict from `sweep/_ktier/` (50 percent cut): **keep all rows.** K=100 rows score slightly
higher but inside both intervals (MMLU-Pro entropy 0.4306 against 0.4028). Going uniform
**costs** R squared: `b_paired.csv` gives MMLU-Pro -0.0060 entropy, -0.0046 p_mode, -0.0043
rem_len_mean, -0.0030 rem_len_std (all significant), commitment not significant.

The one exception is a trap: on GlobalOpinionQA, dropping the K=100 training rows appears to
**gain** +0.1325 on `rem_len_mean`. That is not a K effect. It is the heavy tail (section 7):
10 rows of 12,734 carry 55.6 percent of that target's total sum of squares at the 50 percent
cut. Do not quote GlobalOpinionQA raw-scale length numbers without that caveat.

**(b) `n_options` varies.** MMLU-Pro is mostly 10 options but has 3,400 rows at 4 (mean
9.452). GlobalOpinionQA ranges 2 to 15 (mean 4.421). So cross-question work needs the
normalised entropy column, `answer_entropy_bits_mm_norm`. Note `_norm` is **not bounded by
1** (max 1.225 MMLU-Pro, 1.583 GlobalOpinionQA) because the Miller-Madow correction can
exceed `log2(n_options)`.

Minor defects, for completeness: MMLU-Pro has 1 row with `n_valid=0`, `example_id` 1758 has a
0 percent row but no checkpoint rows, and 70 of 2,499 questions have fewer than 5 chains.
GlobalOpinionQA has 24 rows with `n_valid=0` or null entropy, 18 parse failures at the 0
percent cut, and 36 of 2,555 questions with fewer than 5 chains.

---

## 3. The probe reference on all three datasets

**Row set: FULL sweep.** Mean over the five cuts (10, 30, 50, 70, 90 percent), scored within
each cut then averaged. R squared for the four continuous targets, **accuracy** for
`committed_by_here`. 2,000-draw question bootstrap for intervals.
Sources: `runs/goal2/gpqa25/sweep/_master/ci_scores.csv` and
`runs/goal2/{mmlupro_v1,globalop_v1}/sweep/_ref_ci/ci_scores.csv`.

| | entropy | p_mode | rem_len_mean | rem_len_std | committed (acc) |
|---|---|---|---|---|---|
| **GPQA, ours L21 ridge** | **0.2885** | 0.2508 | 0.5241 | 0.2978 | 0.7409 |
| GPQA, TF-IDF on prefix | 0.2369 | 0.2117 | 0.3910 | 0.2114 | 0.7285 |
| GPQA, prefix length | 0.2137 | 0.1876 | 0.4853 | 0.1330 | 0.7267 |
| **MMLU-Pro, ours** | **0.4103** | 0.3437 | 0.7062 | 0.5400 | 0.7959 |
| MMLU-Pro, TF-IDF | 0.3551 | 0.2995 | 0.6774 | 0.4924 | 0.7898 |
| MMLU-Pro, prefix length | 0.2063 | 0.1595 | 0.6623 | 0.3840 | 0.7282 |
| **GlobalOpinionQA, ours** | **0.2193** | 0.1389 | 0.3595 | 0.1497 | 0.6897 |
| GlobalOpinionQA, TF-IDF | 0.1837 | 0.1226 | 0.2518 | 0.1227 | 0.6928 |
| GlobalOpinionQA, prefix length | 0.0237 | 0.0152 | 0.3720 | 0.0869 | 0.6642 |

Majority-class baselines for commitment: GPQA 0.6275, MMLU-Pro 0.6785, GlobalOpinionQA
0.6548.

**MMLU-Pro beats GPQA on all five targets. GlobalOpinionQA is below GPQA on all five on the
raw scale** (section 7 softens the two length rows considerably).

### 3.1 The framing point: measurability, not magnitude

This is the sentence to say out loud, because the intuitive expectation (more data, bigger
effect) is wrong and he may expect it.

Ours minus TF-IDF on entropy, paired question bootstrap
(`_master/ci_paired.csv`, `_ref_ci/ci_paired.csv`):

| dataset | ours minus TF-IDF, entropy | our single-score CI half-width |
|---|---|---|
| GPQA (198 q) | **+0.0517** [+0.0213, +0.0831] | 0.0558 |
| MMLU-Pro (2,500 q) | **+0.0553** [+0.0439, +0.0678] | 0.0177 |
| GlobalOpinionQA (2,555 q) | **+0.0356** [+0.0242, +0.0469] | 0.0150 |

The effect size barely moved. The interval shrank by a factor of about 3.1 to 3.7, which is
what sqrt(12.6) = 3.5 predicts. So: **more questions did not make the effect bigger, they
made 0.01-scale effects measurable.** That is the argument for the new datasets, and it is
also the argument for not reading GPQA's non-significant results as nulls.

### 3.2 The other paired gaps, for completeness

MMLU-Pro, ours minus TF-IDF, all significant: p_mode +0.0442, rem_len_mean +0.0288,
rem_len_std +0.0476, commitment accuracy +0.0061. Ours minus prefix length: entropy +0.2041,
p_mode +0.1842, rem_len_mean +0.0439, rem_len_std +0.1560, commitment +0.0677, all
significant.

GlobalOpinionQA, ours minus TF-IDF: entropy +0.0356, p_mode +0.0163, rem_len_mean +0.1077,
rem_len_std +0.0270 all significant; **commitment accuracy -0.0031, not significant** (TF-IDF
is nominally ahead). Ours minus prefix length: entropy +0.1955, p_mode +0.1238, rem_len_std
+0.0627, commitment +0.0255 significant; **rem_len_mean -0.0126 [-0.1496, +0.2123], not
significant**. That last one is the awkward raw-scale result; section 7 resolves it.

### 3.3 Normalised entropy (use this for GlobalOpinionQA)

From `_ref_ci/entropy_norm.csv`. MMLU-Pro ours 0.4042 against TF-IDF 0.3484, gap +0.0559
significant. GlobalOpinionQA ours 0.1553 against TF-IDF 0.1216, gap +0.0338 significant.

**Report GlobalOpinionQA entropy normalised, not raw.** Section 6.3 explains why: about two
thirds of the raw text-entropy signal on that dataset is option counting.

### 3.4 Commitment: accuracy understates, AUC is the honest metric, and the grid matters

Computed by me from `sweep/{ref_fixed,ref_logistic_deep}/preds*.csv` (mean AUC over the five
cuts):

| | ours AUC | TF-IDF AUC | prefix length AUC | ours acc |
|---|---|---|---|---|
| GPQA, shallow grid (`ref_fixed`) | 0.7840 | 0.7581 | 0.7562 | 0.7409 |
| GPQA, deep grid (`ref_logistic_deep`) | 0.7612 | 0.7581 | 0.7170 | 0.7403 |
| MMLU-Pro, either grid | 0.8303 | 0.8216 | 0.7090 | 0.7959 |
| GlobalOpinionQA, shallow | 0.6830 | 0.6741 | 0.5820 | 0.6897 |
| GlobalOpinionQA, deep | 0.6708 | 0.6741 | 0.5725 | 0.6897 |

Two warnings in here, both his-question shaped.

**(a) AUC reverses the sign on GlobalOpinionQA.** On accuracy TF-IDF is ahead (0.6928 against
0.6897, not significant). On AUC we are ahead (0.6830 against 0.6741, +0.0089, significant,
`_ref_ci/committed_auc.csv`). Report both or the sign flips.

**(b) Which logistic grid produced an AUC is not recorded in the files, and it changes the
conclusion on two of three datasets.** On GPQA, ours beats TF-IDF by +0.026 AUC under the
shallow grid and by only +0.003 under the deep one. On GlobalOpinionQA the sign **flips**:
+0.0089 significant under the shallow grid, -0.0033 not significant under the deep one. The
accuracy numbers are stable (identical to 4 dp), which is why the 2026-10-26 grid fix
concluded "the published number stands". It stands **for accuracy**. It does not stand for
AUC. See section 9 and section 11.

### 3.5 Depth shape differs by dataset

Entropy per cut, our probe (`per_checkpoint.csv` in each reference directory):

- MMLU-Pro: 0.4671, 0.4291, 0.4086, 0.3874, 0.3594. Best early, decays.
- GPQA: 0.2777, 0.2952, 0.2977, 0.3023, 0.2697. Flat.
- GlobalOpinionQA (normalised): 0.1084, 0.1022, 0.1193, 0.1858, 0.2610. **Rising**, more than
  doubling from the 10 percent cut to the 90 percent cut.

That GlobalOpinionQA shape is worth having ready: it is the one dataset where the probe gets
clearly **better** the deeper into the chain you read, which is the shape the "tracking where
this chain is heading" story predicts.

---

## 4. The frontier model comparison (his explicit ask)

**Row set: ICL subset.** 500 questions per new dataset, 196 questions on GPQA; 2,476 to
2,500 rows over the five cuts. Predictions are **calibrated** (a linear map fitted per fold
on the other folds' rows). Source: `runs/goal2/<tag>/sweep/_icl/{ci_scores,ci_paired}.csv`.

Model: GPT-6 Luna, reasoning high, answer forced to a compact JSON regex. Three configs on
the new datasets: `ckpt` k=0 (zero-shot, depth label shown), `ckpt` k=10, and `blind` k=0
(depth label hidden on examples and on the query). GPQA additionally has k=50 and the
`nockpt` variants, plus the local Gemma-4-31B and Qwen3.5-9B runs.

### 4.1 Scores, mean over the five cuts

| | entropy | p_mode | rem_len_mean | rem_len_std | committed (acc) |
|---|---|---|---|---|---|
| **GPQA** ours (ICL rows) | **0.3094** | 0.2614 | 0.5626 | 0.3281 | 0.7265 |
| GPQA TF-IDF | 0.2674 | 0.2337 | 0.4148 | 0.2213 | 0.7337 |
| GPQA prefix length | 0.2467 | 0.2114 | 0.5516 | 0.1873 | 0.7367 |
| GPQA Luna k=0 | 0.2945 | 0.2334 | 0.4553 | 0.1220 | 0.7184 |
| GPQA Luna k=10 | 0.2159 | 0.1871 | 0.1924 | 0.0668 | 0.7214 |
| GPQA Luna k=50 | 0.2555 | 0.2186 | 0.2724 | 0.0977 | 0.7204 |
| GPQA Luna k=0 blind | 0.2972 | 0.2139 | 0.3365 | 0.0630 | 0.7204 |
| GPQA Gemma-4-31B k=0 | 0.2623 | 0.2370 | 0.2832 | 0.0687 | 0.7418 |
| **MMLU-Pro** ours (ICL rows) | **0.3949** | 0.3384 | 0.7133 | 0.5501 | 0.7985 |
| MMLU-Pro TF-IDF | 0.3371 | 0.2945 | 0.6796 | 0.4903 | 0.7904 |
| MMLU-Pro prefix length | 0.1515 | 0.1219 | 0.6521 | 0.3710 | 0.7270 |
| MMLU-Pro Luna k=0 | 0.3469 | 0.2768 | 0.5724 | 0.3204 | 0.7710 |
| MMLU-Pro Luna k=10 | 0.3206 | 0.2700 | 0.4217 | 0.1753 | 0.7630 |
| MMLU-Pro Luna k=0 blind | 0.3419 | 0.2672 | 0.4912 | 0.2430 | 0.7702 |
| **GlobalOpinionQA** ours (ICL rows) | **0.2359** | 0.1525 | 0.4520 | 0.2192 | 0.6764 |
| GlobalOpinionQA TF-IDF | 0.2212 | 0.1513 | 0.3025 | 0.2185 | 0.6896 |
| GlobalOpinionQA prefix length | 0.0401 | 0.0304 | 0.2654 | 0.1737 | 0.6576 |
| GlobalOpinionQA Luna k=0 | 0.1572 | 0.0995 | 0.1996 | 0.1489 | 0.6756 |
| GlobalOpinionQA Luna k=10 | 0.1181 | 0.0925 | 0.0223 | -0.0365 | 0.6736 |
| GlobalOpinionQA Luna k=0 blind | 0.1445 | 0.0986 | 0.1741 | 0.0808 | 0.6668 |

95 percent intervals on the two headline targets, our probe and Luna k=0:

| | ours, entropy | Luna k=0, entropy | ours, rem_len_mean | Luna k=0, rem_len_mean |
|---|---|---|---|---|
| GPQA | 0.3094 [0.2292, 0.3745] | 0.2945 [0.2077, 0.3635] | 0.5626 [0.5109, 0.6068] | 0.4553 [0.3745, 0.5253] |
| MMLU-Pro | 0.3949 [0.3348, 0.4460] | 0.3469 [0.2905, 0.3979] | 0.7133 [0.6632, 0.7580] | 0.5724 [0.5070, 0.6348] |
| GlobalOpinionQA | 0.2359 [0.1882, 0.2725] | 0.1572 [0.1148, 0.1929] | 0.4520 [0.3418, 0.5256] | 0.1996 [0.0762, 0.2802] |

### 4.2 Paired differences. READ THE SIGN CAREFULLY

In `ci_paired.csv` the `diff` column is **`label` minus `minus`**. The rows below have
label = the Luna config and minus = our probe, so **a negative number means our probe is
ahead**. The same file also stores the mirrored rows (label = ours, minus = Luna), where the
sign is positive for the same fact. Do not read one set as contradicting the other.

Luna minus our probe (negative = we win), all three datasets:

| dataset | config | entropy | p_mode | rem_len_mean | rem_len_std | committed |
|---|---|---|---|---|---|---|
| GPQA | k=0 | **-0.0150 [-0.0786, +0.0481] n.s.** | -0.0281 n.s. | -0.1073 sig | -0.2060 sig | -0.0082 n.s. |
| GPQA | k=10 | -0.0936 sig | -0.0743 sig | -0.3702 sig | -0.2613 sig | -0.0051 n.s. |
| GPQA | k=0 blind | -0.0123 n.s. | -0.0475 n.s. | -0.2262 sig | -0.2650 sig | -0.0061 n.s. |
| MMLU-Pro | k=0 | **-0.0480 [-0.0917, -0.0012] sig** | -0.0616 sig | -0.1409 sig | -0.2297 sig | -0.0275 sig |
| MMLU-Pro | k=10 | -0.0743 sig | -0.0684 sig | -0.2916 sig | -0.3748 sig | -0.0355 sig |
| MMLU-Pro | k=0 blind | -0.0530 sig | -0.0712 sig | -0.2221 sig | -0.3071 sig | -0.0283 sig |
| GlobalOpinionQA | k=0 | **-0.0788 [-0.1204, -0.0372] sig** | -0.0529 sig | -0.2524 sig | -0.0703 sig | -0.0008 n.s. |
| GlobalOpinionQA | k=10 | -0.1178 sig | -0.0600 sig | -0.4297 sig | -0.2558 sig | -0.0028 n.s. |
| GlobalOpinionQA | k=0 blind | -0.0914 sig | -0.0539 sig | -0.2779 sig | -0.1384 sig | -0.0096 n.s. |

**The headline.** On GPQA the frontier model ties us on entropy. On GlobalOpinionQA we beat
it by 0.079, significant. On MMLU-Pro we beat it on **all five targets with every one
significant**, which is the first dataset where that is true.

**Be honest about the margin on MMLU-Pro.** The entropy interval is
[-0.0917, **-0.0012**]. The upper edge is 0.0012 from zero. It is significant by the stated
rule and I would report it as significant, but I would not lead with it, and if he presses
on it the right answer is "marginal, and the robust statement is that it is significant on
all five targets jointly and comfortably significant on four of them". The other two
datasets carry the claim: GlobalOpinionQA's entropy margin is nowhere near the boundary, and
on remaining length the margins are enormous on every dataset.

**Commitment accuracy ties on GPQA and GlobalOpinionQA and loses on MMLU-Pro.** Round 6
established on GPQA that the accuracy tie is an artefact: under AUC our probe beats **all
nine** GPQA ICL configs by 0.096 to 0.122, every one significant
(`_iclsplit/p3_committed_auc_ci.csv`: ours 0.7732, Luna k=0 0.6658, Gemma k=0 0.6634).
**That AUC check has not been run for the new datasets** (section 9), so on MMLU-Pro and
GlobalOpinionQA we can only quote accuracy, and we should say so.

### 4.3 Two internal checks that reproduce, and why the second one matters

**Check 1: "examples hurt", k=0 beats k>0.** Now in **four independent
(model, dataset) pairs**:

| pair | k=0 entropy | best k>0 entropy |
|---|---|---|
| Gemma-4-31B on GPQA | 0.2623 | 0.1761 (k=10); k=30 0.1445, k=93 0.1583 |
| Luna on GPQA | 0.2945 | 0.2555 (k=50); k=10 0.2159 |
| Luna on MMLU-Pro | 0.3469 | 0.3206 (k=10) |
| Luna on GlobalOpinionQA | 0.1572 | 0.1181 (k=10) |

On GlobalOpinionQA k=10 is worse than k=0 on **every single target** (rem_len_mean 0.0223
against 0.1996; rem_len_std actually goes negative at -0.0365). Two unrelated model families
and three datasets agree, so this is the best-supported in-context finding we have.

The mechanism (round 6, `sweep/_iclwhy/`) is measured, not guessed: across 28 comparisons
with a same-model k=0 control, **within-cut correlation falls in 28 of 28** (mean -0.131),
while the spread ratio sd(prediction)/sd(truth) moves **toward** 1 in 19 of 28 and the raw
uncalibrated R squared **improves** in 15 of 28. So the examples supply the output **scale**
and damage the **ranking**. Because our published comparison is calibrated, the scale gain is
given away for free and only the ranking damage shows up. Verbatim copying of example values
is real but modest and entropy-only (+0.08 to +0.29 over a chance baseline, about zero on
both length targets).

**Check 2: hiding the depth label costs only the length targets.** Now in three pairs. The
`blind` config hides which cut point the prefix came from, on the examples and on the query.

| dataset | entropy cost | p_mode cost | rem_len_mean cost | rem_len_std cost | committed cost |
|---|---|---|---|---|---|
| GPQA (k=0) | +0.003 (no cost) | -0.020 | **-0.119** | **-0.059** | +0.002 |
| MMLU-Pro (k=0) | -0.005 | -0.010 | **-0.081** | **-0.077** | -0.001 |
| GlobalOpinionQA (k=0) | -0.013 | -0.001 | **-0.026** | **-0.068** | -0.009 |

**Why this is the single best sanity check on the whole in-context setup.** Depth is
genuinely informative about remaining length (if you are 90 percent through a chain there is
less of it left) and genuinely much less informative about the answer distribution. A model
that was pattern-matching, bluffing, or leaking the answer through some other route would not
show that asymmetry. The fact that removing exactly one token of information degrades
exactly the targets that token is informative about, and leaves the others alone, on three
datasets, says the model is really reading the prompt we think it is reading and the
pipeline is wired correctly. It is also the reason we can trust the k=0 numbers.

### 4.4 Cost and operational facts, in case he asks

GlobalOpinionQA: 9,000 calls, **0 parse failures**, all 9,000 completed, **$5.85**.
MMLU-Pro and GlobalOpinionQA together were planned at $21.61. The GPQA full run (8 configs,
1,178 rows, 9,424 calls) cost $17.86 with 0 parse failures. Prompt caching hit 91.8k of 92k
input tokens on k=50.

One correction worth having ready: an earlier estimate of "$13 for the full set" was wrong by
an order of magnitude. `--full` on the new datasets is about 15,000 rows and about $118,
because GPQA was cheap only by virtue of having 198 questions.

---

## 5. Isolating question difficulty as a confounder

### 5.1 The method, in plain words

The worry (his, from the 2026-10-27 meeting) was that both our probe and the frontier model
might just be reading **how hard the question is**. A hard question produces spread-out
answers; an easy one produces concentrated answers; so a predictor that reads nothing but
difficulty would score well on entropy without knowing anything about the chain.

The test: pick a per-question difficulty measure, fit a **quadratic** in it to the target and
to the prediction separately, take the residuals of both, and score the residual prediction
against the residual target. Whatever R squared survives is the part of the probe's skill that
is **not** a function of that one scalar. The quadratic is refitted **inside each of 2,000
question bootstrap draws**, per cut, so the intervals account for the partialling itself.

Code: `src/analysis/newds_difficulty_scoring.py`, which imports the published round-6
modules (`icl_decompose`, `icl_bands`, `icl_bands_refit`, `icl_bands_spread`) and repoints
their constants at runtime without editing them. It was validated three ways: the four
published scripts rerun unchanged with only the output directory redirected give all 12 CSVs
identical (max difference 0.000e+00); the wrapper on GPQA returns the published 0.2885 /
0.5241 / 0.251 / 0.298 / 0.741; and the band-refit table matches the published GPQA file at
max difference 0.0 over 36 cells.

Ignore the `r2_partialled_rescaled` metric in those files. It is a different quantity
(rescaled by the residual variance) and mixing it with `r2_partialled` will give nonsense.

### 5.2 Why TWO difficulty measures, and the bracket

**Label-derived.** On GPQA and MMLU-Pro this is `p_gold`, the fraction of the question's
rollouts that are correct. It is computed from the **same rollouts that produced the target**,
so it shares sampling noise with the target and **over-removes**. The partialled value under
this measure is therefore a **lower bound** on what survives.

**Text-derived.** `qtext_pgold`, an out-of-fold TF-IDF prediction of `p_gold` from the
question text alone. For a held-out question it uses none of that question's own rollouts, so
it is **non-circular**. But it is an imperfect prediction, so it removes less than the true
difficulty does and **attenuates**: the partialled value is an **upper bound**.

Report both. The truth is between them. This bracket is the single most important
methodological point in the section, because a single number here can be made to say anything.

### 5.3 GPQA and MMLU-Pro

**Row set: FULL sweep, both.** This matters a lot, see 5.5.
Sources: `runs/goal2/_newds_reports/_p6_gpqa_ref/{rollout,qtext}/p2_ci_scores.*.csv` and
`runs/goal2/mmlupro_v1/sweep/_iclsplit/p2_ci_scores.{rollout,qtext}.csv`.

Our probe, mean over the five cuts:

| | published | label-derived (lower bound) | text-derived (upper bound) |
|---|---|---|---|
| **GPQA entropy** | 0.2885 [0.2260, 0.3377] | **0.0248 [-0.0453, +0.0858] not sig.** | 0.2013 [0.1460, 0.2459] sig. |
| **MMLU-Pro entropy** | 0.4103 [0.3922, 0.4276] | **0.2004 [0.1740, 0.2263] sig.** | 0.3676 [0.3461, 0.3884] sig. |
| GPQA rem_len_mean | 0.5241 [0.4814, 0.5571] | 0.4784 [0.4233, 0.5169] | 0.5048 [0.4597, 0.5403] |
| MMLU-Pro rem_len_mean | 0.7062 [0.6919, 0.7196] | 0.6966 [0.6821, 0.7103] | 0.7035 [0.6896, 0.7171] |

**The finding: the entropy collapse does not replicate.** On GPQA, against the most
aggressive (over-removing) measure, nothing survives. On MMLU-Pro, **half** of the entropy
R squared survives against that same aggressive measure and **90 percent** survives against
the honest one, and the intervals do not overlap GPQA's. Remaining length loses 1.3 percent
on MMLU-Pro and 8.7 percent on GPQA, that is, it is essentially flat on both.

The two measures bracket consistently in all six comparisons: wide on entropy, negligible on
length. That is itself informative: **the question-level part of remaining length is not
`p_gold`-shaped.** Difficulty alone never explains more than 0.053 of remaining length on
MMLU-Pro, against up to 0.518 of entropy.

### 5.4 GlobalOpinionQA (finished 2026-11-02, newer than anything in CLAUDE.md)

Jobs 5003235 and 5003236 ran today. Nothing about this is in the Results index yet.
Source: `runs/goal2/globalop_v1/sweep/_iclsplit/p2_{ci_scores,ci_paired,per_checkpoint}.{rollout,qtext}.csv`.
**Row set: FULL sweep.**

GlobalOpinionQA has no gold answer, so **neither measure is `p_gold`**:

- label-derived = **`ent0_norm`**, the normalised answer entropy at the 0 percent cut,
  computed from that question's own rollouts at the prompt;
- text-derived = **`qtext_ent_norm_mean`**, the out-of-fold text prediction of the question's
  mean normalised entropy (chosen in `_qtext/candidates.csv` because it is the only candidate
  that is both predictive and uncorrelated with `n_options`: correlation -0.163 against +0.642
  for a raw-entropy text prediction, and its quintiles come out exactly 511 each with zero
  arbitrary tie-breaks).

Our probe, mean over the five cuts:

| target | published | label-derived (`ent0_norm`) | text-derived (`qtext_ent_norm_mean`) |
|---|---|---|---|
| raw entropy | 0.2193 [0.2041, 0.2341] | 0.1839 [0.1683, 0.1985] sig. | 0.2032 [0.1880, 0.2178] sig. |
| **normalised entropy (use this one)** | 0.1553 [0.1436, 0.1668] | **0.1119 [0.0999, 0.1236] sig.** | **0.0884 [0.0773, 0.1001] sig.** |
| p_mode | 0.1389 | 0.1047 sig. | 0.1224 sig. |
| rem_len_mean | 0.3595 | 0.3580 | 0.3573 |
| rem_len_std | 0.1497 | 0.1475 | 0.1482 |

Three things to understand before you present this.

**(a) The bracket direction does not hold here.** On GPQA and MMLU-Pro the label measure
reliably over-removes, so label is the floor and text the ceiling. On GlobalOpinionQA the
ordering **empirically flips between targets**: on raw entropy the label measure removes more
(0.035 against 0.016), on normalised entropy the **text** measure removes more (0.067 against
0.043). So present them as **a range of two measures, not a directed bracket**, and say
plainly that neither is `p_gold`, so the label-derived column is **not strictly comparable**
across the three datasets.

**(b) The label-derived measure here is close to a self-partial.** `ent0_norm` is literally
the target (normalised entropy) measured at the prompt. Partialling it out of the 10 to 90
percent targets removes most of what any question-level predictor could know, which makes it
an unusually severe test. That 0.1119 surviving, significantly above zero, is therefore a
stronger result than the number looks. It is also why we did **not** band on this measure:
40.6 percent of its quintile assignments are arbitrary tie-breaks (`_difficulty/candidates.csv`,
`quintile_arbitrary_frac` 0.4055), which would make band sizes meaningless.

**(c) Report GlobalOpinionQA entropy normalised.** Option counts run 2 to 15, and P5
established that a quadratic in `log2(n_options)` alone explains 0.145 of raw entropy on this
dataset (against 0.004 on MMLU-Pro), so roughly two thirds of the raw text-entropy signal
there is just counting options. Quote 0.1553 and the 0.1119 / 0.0884 pair, not 0.2193.

One aside, in case he asks: GlobalOpinionQA also ships **actual human response
distributions**, and `human_ent` / `human_ent_norm` were evaluated as difficulty candidates.
`human_ent` was rejected because it correlates +0.788 with `n_options`. That is a lead worth
mentioning (a human-disagreement-based difficulty measure is genuinely non-circular with
respect to the model) but nothing has been run with it.

### 5.5 ROW-SET WARNING: there are two legitimate GPQA partialling results

This is exactly the kind of thing that could embarrass you in the room, so it is worth 30
seconds of preparation.

| row set | our probe, published entropy | partialled (label-derived) | file |
|---|---|---|---|
| **ICL subset** (1,178 rows, one chain per question per cut) | 0.3094 | **0.0921 [-0.0102, +0.1801]** | `gpqa25/sweep/_iclsplit/p2_ci_scores.csv` |
| **FULL sweep** (24,566 rows, all 25 chains) | 0.2885 | **0.0248 [-0.0453, +0.0858]** | `_newds_reports/_p6_gpqa_ref/rollout/p2_ci_scores.rollout.csv` |

Both are correct for their own row set. **Round 6's write-ups and
`docs/update_2026-10-29.md` quote the ICL-subset version. P6 quotes the full-sweep version.**
MMLU-Pro's and GlobalOpinionQA's partialling runs are on the FULL sweep, so the
GPQA-against-MMLU-Pro comparison in 5.3 uses the full-sweep GPQA numbers. Comparing 0.309
against 0.410 would mix row sets.

If he says "I thought it fell to 0.09", he is remembering the ICL-subset number and he is
right about that row set.

### 5.6 The depth effect: the cleanest part of this section, now on all three datasets

Round 6 averaged over depth and so was dominated by the shallow cuts, where the difficulty
story is most true. Per cut, with label-derived difficulty removed
(`p2_per_checkpoint.*.csv`):

| cut | MMLU-Pro published | MMLU-Pro difficulty alone | MMLU-Pro partialled | GPQA partialled | GlobalOpinionQA partialled (normalised) |
|---|---|---|---|---|---|
| 10% | 0.4671 | 0.5182 | **0.0867** | **-0.1847** | **0.0316** |
| 30% | 0.4291 | 0.4049 | 0.1589 | -0.0222 | 0.0451 |
| 50% | 0.4086 | 0.2981 | 0.2232 | 0.0571 | 0.0817 |
| 70% | 0.3874 | 0.2086 | 0.2611 | 0.1321 | 0.1606 |
| 90% | 0.3594 | 0.1370 | **0.2720** | **0.1415** | **0.2405** |

Read the MMLU-Pro columns together: at the shallowest cut only about 19 percent of the
entropy signal is not difficulty (0.0867 of 0.4671); at the deepest cut about 76 percent is
(0.2720 of 0.3594). The difficulty-alone column falls monotonically with depth while the
published column falls much more slowly, which is the whole effect.

**This replicates on all three datasets and has never been stated anywhere.** It is the most
defensible single new claim in the section, because it does not depend on which difficulty
measure you choose: the same monotone rise appears under the text-derived measure too
(MMLU-Pro 0.3743, 0.3677, 0.3752, 0.3701, 0.3508, that is, essentially flat and high, and
GlobalOpinionQA -0.0003, 0.0272, 0.0582, 0.1310, 0.2260).

The framing: **a probe read at the prompt is mostly reading the question; a probe read deep
in the chain is mostly not.** That is the shape the research question predicted.

### 5.7 And the activations-beat-text claim survives partialling on both large datasets

From `p2_ci_paired.*.csv`, rows with `label` = "Our model L21 (ridge)" and
`minus` = "TF-IDF on prefix". Entropy:

| dataset | published | after removing label-derived difficulty | after removing text-derived difficulty |
|---|---|---|---|
| **MMLU-Pro** | +0.0553 [+0.0439, +0.0678] sig. | **+0.0627 [+0.0477, +0.0784] sig.** | +0.0611 [+0.0489, +0.0748] sig. |
| **GlobalOpinionQA** (raw) | +0.0356 [+0.0242, +0.0469] sig. | **+0.0493 [+0.0375, +0.0612] sig.** | **+0.0583 [+0.0476, +0.0691] sig.** |
| GlobalOpinionQA (normalised) | +0.0338 [+0.0251, +0.0433] sig. | +0.0405 [+0.0308, +0.0506] sig. | +0.0558 [+0.0466, +0.0655] sig. |
| **GPQA** | +0.0517 [+0.0213, +0.0831] sig. | **+0.0325 [-0.0104, +0.0753] NOT sig.** | +0.0586 [+0.0241, +0.0926] sig. |

**The gap WIDENS under difficulty removal on both large datasets.** GPQA is the only one of
the three where removing difficulty erases our advantage, and even there it erases it only
under the over-removing measure, and the interval still reaches +0.075.

On MMLU-Pro the probe also beats TF-IDF on entropy **inside every one of the five difficulty
bands, all five significant**, and the offset-free within-cut correlation gap is significant
in all five too. On GPQA's bands it is not significant in three of five, which is a power
statement (39 questions per band) rather than a null.

**SIGN WARNING for the P6 report.** `runs/goal2/_newds_reports/p6_difficulty_scoring.md`
prints these four numbers **with inverted signs** in its headline section (lines 59 to 61:
"GPQA -0.032 ... MMLU-Pro -0.063 ... before partialling both are significant (-0.052 and
-0.055)"). It read the `TF-IDF minus ours` rows, which the same file also stores. Its prose
conclusion is right but the signs are backwards. Do not be thrown if you open that report.

### 5.8 Band stratification, for completeness

An independent, offset-free route to the same question: score inside a fixed difficulty band,
so difficulty is held roughly constant by construction rather than regressed out.

MMLU-Pro, label-derived `band_int` bands (sizes 648, 276, 342, 255, 978), offset-corrected
entropy R squared, hardest to easiest: **0.312, 0.261, 0.169, 0.195, -0.247**, against 0.410
for all questions. Falls, but does not collapse to GPQA's level, and the hardest band holds
0.312. Remaining length: 0.779, 0.739, 0.709, 0.625, 0.584, essentially flat.

MMLU-Pro, text-derived `band_quintile` bands (exactly 500 each, zero arbitrary tie-breaks):
entropy **0.199, 0.360, 0.420, 0.439, 0.403**, that is, **essentially no band collapse at all**.

GPQA, full sweep, same label-derived quintile bands (40 questions each), offset-corrected
entropy: 0.134, 0.089, 0.041, 0.060, **-0.651**. Remaining length 0.566, 0.484, 0.434, 0.496,
0.350.

Two notes on the negative values. First, **offset correction matters**: the uncorrected GPQA
`v_easy` value is -2.25 and the uncorrected MMLU-Pro `v_easy` is -0.589, because the stored
predictions come from a fit on all questions and are never recentred per band, so an
uncorrected band score is charged for an offset the full-set fit never carried. Recentring
costs one degree of freedom and leaves the slope alone. Both columns are kept in the files.
Second, part of the `v_easy` negativity is **real**: in that band the truth is nearly
constant while the predictions go on varying as much as ever, so they add error. MMLU-Pro's
`v_easy` entropy standard deviation is 0.43 of the overall figure (0.2994 against 0.6940)
while the predicted spread ratio sd(pred)/sd(true) stays at 1.12
(`p6_band_refit_spread.rollout.band_int.csv`).

There is also a best-case check: refit an intercept **and** a slope inside the band, which is
the most forgiving thing you can do without refitting the probe's weights. MMLU-Pro entropy
then reads 0.318, 0.278, 0.205, 0.232, 0.198 across the five bands. Even at best, entropy
inside a band sits at 0.20 to 0.32 against 0.41 for all questions, while remaining length
stays at 0.58 to 0.78. So the band drop on entropy is partly an offset artefact and partly
real, and the length result is genuinely flat.

Two further decompositions worth having in your pocket
(`p8_band_decomposition.*`, `p9_question_decomposition.*`):

- **Within a difficulty band**, with no partialling and no recentring, MMLU-Pro retains
  **0.2124** of entropy against GPQA's **0.0499**, a factor of 4 (both on `band_quintile`,
  the matched scheme). Note `CLAUDE.md` quotes 0.204 for MMLU-Pro, which is the `band_int`
  value; the matched-scheme comparison is the one above.
- **Within a single question**, both datasets are **negative on every target** (MMLU-Pro
  entropy -0.117, remaining length -0.456; GPQA -0.074 and -0.388). Between questions they
  are strong (MMLU-Pro entropy 0.593, remaining length 0.863).

That last pair is the honest limit of the whole project and you should say it before he finds
it: **the probe is a question-level reader. It does not discriminate between chains of the
same question.** The new datasets replicate the question-level half and refute the
difficulty-specific reading; they do not rescue the within-question half.

### 5.9 Partialling against the 0 percent cut: two measurements, same ordering

These two methods answer the same question differently and should be reported together, not
substituted.

- The **0 percent cut** (section 6) says 70 to 92 percent of the probe's performance is
  available before any reasoning token exists. That is a 3,584-dimensional read of the
  prompt.
- **Partialling** removes a two-parameter function of one scalar, and leaves half of the
  MMLU-Pro entropy signal standing.

They agree on ordering and differ on magnitude, which is exactly what the caveats predict.
If he asks which to believe: the 0 percent cut is the more conservative measurement of "how
much is question level", and partialling is the measurement of "how much is **difficulty**".
They are not the same quantity, and conflating them is how round 6 ended up with a
conclusion that did not replicate.

---

## 6. The 0 percent cut and the question-text baseline

### 6.1 What the 0 percent cut is, and why it is clean

The 0 percent cut is the **rendered prompt before a single reasoning token exists**. One row
per question. Two properties make it the most interpretable experiment we have:

- **Prefix length is constant**, so the prefix-length confound vanishes entirely. Verified:
  the prefix-length control scores -0.014 to -0.019 on GPQA and -0.0005 to -0.003 on the new
  datasets (`f0_probe/results.csv`). Exactly the small negative a useless predictor should
  give under cross-validation.
- **One row per question**, so the probe can **only** read question-level information. Its
  R squared is therefore a direct **measurement** of the question-level floor, not an
  inference from a decomposition.

TF-IDF on the prefix is not runnable here (the prefixes are empty), which is why the
companion baseline is TF-IDF on the **question text**.

### 6.2 Scores at the 0 percent cut, and the share of the 10 to 90 percent average

**Row set: 0 percent cut.** Source `runs/goal2/<tag>/sweep/f0_probe/results.csv`. Share is
the 0 percent score divided by the FULL-sweep 10 to 90 percent mean from section 3.

| target | GPQA 0% | share | MMLU-Pro 0% | share | GlobalOpinionQA 0% | share |
|---|---|---|---|---|---|---|
| entropy | 0.2337 | 81% | **0.3010** | 73% | 0.1545 | 70% |
| normalised entropy | not applicable (4 options) | | 0.2939 | 73% | 0.0448 | 29% |
| p_mode | 0.2091 | 83% | 0.2902 | 84% | 0.1227 | 88% |
| rem_len_mean | 0.4798 | 92% | 0.6058 | 86% | 0.2391 | 67% |
| rem_len_std | 0.2049 | 69% | 0.3779 | 70% | 0.0363 | 24% |

**Three readings.**

1. **67 to 92 percent of the probe's performance is available before a single reasoning token
   exists**, on every target on every dataset. And MMLU-Pro's 0 percent entropy (0.3010)
   **exceeds GPQA's entire 10 to 90 percent average** (0.2885). That is the strongest direct
   evidence that the probe is largely a question-level reader. State that ratio **within one
   dataset**; the cross-dataset version is a different and weaker claim.

2. **These are lower bounds.** GPQA's 0 percent labels average 25 continuations; the new
   datasets' average 5. Label noise attenuates R squared, so a 25-draw MMLU-Pro 0 percent
   probe would score above 0.3010. GPQA's better-measured 0.2337 is still the smaller number.

3. **`rem_len_std` has the lowest share everywhere** (69, 70, 24 percent), so it is the target
   that most needs the prefix. That matters because `rem_len_std` is also the sharpest
   novelty claim in `RESEARCH.md` (nobody else predicts answer-distribution spread), and on
   GlobalOpinionQA 76 percent of it needs the prefix.

### 6.3 The head-to-head at the 0 percent cut: activations against the words

**Row set: 0 percent cut, identical rows for both sides.** Source
`runs/goal2/<tag>/sweep/_qtext/cmp_f0.csv`. The text baseline is TF-IDF on question plus
choices (`qc`, the headline version, because `n_options` varies). Diff is text minus
activations, so negative means activations win.

**MMLU-Pro, activations ahead on all five targets, every one significant:**

| target | activations @0% | question text @0% | diff | 95% CI | text's share |
|---|---|---|---|---|---|
| entropy | 0.3010 | 0.1193 | -0.1817 | [-0.2088, -0.1552] | 40% |
| normalised entropy | 0.2939 | 0.1243 | -0.1696 | [-0.1976, -0.1434] | 42% |
| p_mode | 0.2902 | 0.1201 | -0.1701 | [-0.1977, -0.1438] | 41% |
| rem_len_mean | 0.6058 | 0.3919 | -0.2139 | [-0.2449, -0.1822] | 65% |
| rem_len_std | 0.3779 | 0.2247 | -0.1532 | [-0.1820, -0.1253] | 59% |

**This is the cleanest result in the project.** Before emitting a single token, layer 21
encodes something about how its own rollouts will spread that a tuned linear model on 2,499
questions cannot read off the question text. The activations keep 1.5 to 2.5 times the text's
R squared.

**GPQA points the same way but is underpowered.** Activations ahead on all five, but only
`rem_len_mean` clears significance (0.4798 against 0.1907, diff -0.2891
[-0.3812, -0.1928]). Entropy is 0.2337 against 0.1671, diff -0.0665 [-0.1666, +0.0255], not
significant at 198 questions with a CI half-width near 0.10. **Do not read GPQA's
non-significant gaps as nulls.**

**GlobalOpinionQA inverts it, partially.** Activations are significantly ahead on raw entropy
(0.1545 against 0.1219, diff -0.0327, significant) and p_mode (0.1227 against 0.0965,
significant), but **tie** on normalised entropy (0.0448 against 0.0420, diff -0.0028, not
significant), `rem_len_mean` (0.2391 against 0.2284, not significant) and `rem_len_std`
(0.0363 against 0.0333, not significant). So on three of five targets, including the one we
should be quoting (normalised entropy), the question text is 92 to 96 percent of the way to
the activations at the prompt. There, at the prompt, the probe really is a question-text
reader.

### 6.4 And at 10 to 90 percent, does question text alone recover the probe?

**Row set: FULL sweep.** Source `_qtext/cmp_paired.csv`. Question text (plus choices) minus
our probe, negative means our probe is ahead.

- **GPQA: no, on every target.** Entropy -0.1722 [-0.2298, -0.1126], p_mode -0.1438,
  rem_len_mean -0.4144, rem_len_std -0.2462, commitment accuracy -0.0533, all significant.
  Question text recovers 40 percent of the probe's entropy and 21 percent of its remaining
  length.
- **MMLU-Pro: no, on every target.** Entropy -0.2741, p_mode -0.2249, rem_len_mean -0.3873,
  rem_len_std -0.2738, commitment -0.0605, all significant. Text recovers 33 percent of
  entropy and 45 percent of remaining length.
- **GlobalOpinionQA: yes for the answer-distribution targets, no otherwise.** Raw entropy
  +0.0045 [-0.0129, +0.0217] not significant (text nominally ahead), p_mode +0.0027 not
  significant, commitment accuracy +0.0050 not significant. But **normalised entropy -0.0405
  [-0.0561, -0.0262] significant** in our favour, `rem_len_mean` -0.2287 significant,
  `rem_len_std` -0.0646 significant.

**The critical control, and have this one ready.** About two thirds of GlobalOpinionQA's
text-entropy signal is **option counting**: a quadratic in `log2(n_options)` alone explains
0.145 of raw entropy there, against 0.004 on MMLU-Pro. So the "text ties us on
GlobalOpinionQA entropy" result is largely an artefact of the raw target, and on the
normalised target we are significantly ahead. **That is why GlobalOpinionQA entropy must be
reported normalised.**

### 6.5 Why 6.2 plus 6.3 is the strongest pair in the project

Put together:

- 6.2 says most of the signal is **question level** (available at the prompt, where no chain
  exists).
- 6.3 says that question-level signal is **not text recoverable** (MMLU-Pro activations keep
  1.5 to 2.5 times the text's R squared on identical rows, with 2,499 questions of power).

This **refines** round 4 rather than confirming it. Round 4's sentence was "the probe is
mostly a question-difficulty reader", which invites "so just read the question". The
measured answer is that reading the question, with a tuned linear model and 2,500 training
examples, gets you 33 to 42 percent of the way on MMLU-Pro. **So "the probe is just a
question-difficulty reader" is too weak a statement, and it is also the wrong word: say
question-LEVEL.**

Also from this work, a side product that unblocks GlobalOpinionQA: `qtext_ent_norm_mean` is a
**non-circular difficulty measure** for that dataset (out-of-fold text prediction of a
question's mean normalised entropy, using none of that question's own rollouts). 58 percent
of `ent0_norm`'s power, correlation with `n_options` only -0.163, quintiles exactly 511 each
with zero arbitrary tie-breaks. It is what section 5.4's text-derived column uses.

One method note that made this affordable: the question-level fits are weighted by row counts,
which is exactly equivalent to the row-level fit (verified, max difference 4.1e-4 of a target
standard deviation over 9 checks).

Finally, the between-question ceiling for **any** question-level predictor, which is the
right yardstick for sections 5 and 6: entropy 0.618 (GPQA) / 0.695 (MMLU-Pro) / 0.615
(GlobalOpinionQA); `rem_len_mean` 0.713 / 0.841 / 0.411. Question text reaches 15 to 38
percent of it.

---

## 7. The length targets on the raw against the log scale

**Row set: FULL sweep, all three datasets.** Source `runs/goal2/<tag>/sweep/_loglen/ci.csv`.
Three transforms of the same target, refitted from scratch each time with the same grids and
folds: `raw`, `log1p`, `winsor99` (99th-percentile winsorising).

**The raw refits reproduce `ref_fixed` to 4 decimal places, which is the control that says
the pipeline is the same one.**

### 7.1 The headline: the token-counting objection is a raw-scale artefact

Ours minus prefix length on `rem_len_mean`:

| dataset | raw | log1p | winsor99 |
|---|---|---|---|
| GPQA | **+0.0387** [+0.0108, +0.0690] sig. | **+0.1643** [+0.1361, +0.1934] sig. | +0.0413 sig. |
| MMLU-Pro | **+0.0439** [+0.0347, +0.0531] sig. | **+0.2538** [+0.2433, +0.2646] sig. | +0.0481 sig. |
| GlobalOpinionQA | **-0.0126** [-0.1496, +0.2123] NOT sig. | **+0.5214** [+0.4723, +0.5908] sig. | +0.4772 sig. |

All six log1p gaps are significant. The long-standing objection, that our remaining-length
prediction is largely token counting because prefix length nearly matches it, **holds only on
the raw scale**, and it is driven by a heavy tail that both predictors are being scored
against.

Ours minus TF-IDF on `rem_len_mean`, for completeness: GPQA +0.1331 raw and +0.1757 log1p;
MMLU-Pro +0.0288 and +0.0936; GlobalOpinionQA +0.1077 and +0.2165. All significant.

`rem_len_std` behaves the same way. Ours minus prefix length: GPQA +0.1648 raw to +0.1873
log1p; MMLU-Pro +0.1560 to +0.3274; GlobalOpinionQA +0.0627 to +0.3664. Ours minus TF-IDF:
GPQA +0.0864 to +0.1532; MMLU-Pro +0.0476 to +0.1301; GlobalOpinionQA +0.0270 to +0.1185.
All significant.

### 7.2 Absolute scores on log1p

| | rem_len_mean | rem_len_std |
|---|---|---|
| GPQA ours | 0.6517 [0.6058, 0.6864] | 0.4068 [0.3238, 0.4638] |
| MMLU-Pro ours | **0.7828** [0.7721, 0.7926] | 0.6496 [0.6347, 0.6636] |
| GlobalOpinionQA ours | 0.6284 [0.6170, 0.6395] | 0.4208 [0.4042, 0.4359] |

**GlobalOpinionQA is roughly level with GPQA once the tail is handled** (0.628 against 0.652
on `rem_len_mean`; 0.421 against 0.407 on `rem_len_std`, where it is actually ahead). That
materially softens the earlier claim that GlobalOpinionQA is "worse on all five targets",
which is a raw-scale statement.

**DO NOT COMPARE THESE TO SECTION 3.** GlobalOpinionQA's 0.628 against GPQA's raw 0.524 is
scale mixing, not a result. Every number in this section is on `log1p(y)` and only comparable
to other numbers in this section. This is the trap the slide deck must also avoid.

### 7.3 Why: the tail, measured

From `_loglen/tail_mass.csv`, share of the target's total sum of squares carried by the top 10
rows, at the 50 percent cut:

| | GPQA | MMLU-Pro | GlobalOpinionQA |
|---|---|---|---|
| rem_len_mean, raw | 2.7% | 2.8% | **55.6%** |
| rem_len_mean, log1p | 2.6% | 0.9% | 3.5% |
| rem_len_std, raw | 4.5% | 2.8% | 30.0% |
| rem_len_std, log1p | 5.3% | 1.1% | 2.8% |

On GlobalOpinionQA the raw figure runs 43 to 56 percent at **every** cut for `rem_len_mean`,
and `rem_len_std` hits **57.5 percent** at the 90 percent cut. `log1p` cuts it to 2 to 4
percent, winsorising to 1.5 to 3 percent. The cause is visible in section 2.3: a 630-token
mean with outliers up to 12,269 tokens.

`log1p` and `winsor99` agree, which says it is **the tail and not the transform**.

One provenance note, in case he has seen the P6 report. That report carried only three cuts
for GlobalOpinionQA because the CI job appeared to time out. It had not: all 90 fits had been
saved and the crash came from `loglen_refit.py` filtering on `P.transform == tr`, where
`transform` resolves to a pandas DataFrame **method**, so the comparison was the scalar
`False`, every filter returned zero rows and the bootstrap sampler got a zero-length
probability vector. The fix is to write `P["transform"]`. The intervals were then recovered
from the saved predictions without refitting, and all five cuts are in `ci.csv` now. The
full-five-cut numbers are **stronger** than the three-cut version P6 reported. Worth knowing
because it is the second time this project has lost a job to that exact mistake.

### 7.4 And the depth shape changes sign

`rem_len_mean`, our probe, per cut:

| | 10% | 30% | 50% | 70% | 90% |
|---|---|---|---|---|---|
| GPQA raw | 0.626 | 0.587 | 0.548 | 0.484 | 0.375 |
| GPQA log1p | 0.698 | 0.659 | 0.648 | 0.636 | 0.619 |
| MMLU-Pro raw | 0.785 | 0.774 | 0.740 | 0.678 | 0.556 |
| MMLU-Pro log1p | 0.800 | 0.782 | 0.771 | 0.777 | 0.784 |
| GlobalOpinionQA raw | 0.348 | 0.341 | 0.377 | 0.354 | 0.377 |
| GlobalOpinionQA log1p | **0.569** | 0.606 | 0.624 | 0.637 | **0.707** |

On the raw scale the probe looks like it decays badly with depth (GPQA 0.63 to 0.38,
MMLU-Pro 0.79 to 0.56). On `log1p` it is essentially flat on MMLU-Pro, mildly declining on
GPQA, and **clearly rising** on GlobalOpinionQA. Prefix length does the opposite: on
GlobalOpinionQA it goes from 0.167 at the 10 percent cut to 0.094 at the 90 percent cut on
`log1p`, that is, it is useless there at every depth.

The implication for the paper: the apparent "the probe gets worse the deeper you read" effect
on length is a property of the raw-token scale, not of the probe.

### 7.5 One P1 result this overturns

P1 reported the awkward finding that on GlobalOpinionQA our probe does **not** beat prefix
length on `rem_len_mean` (-0.0126, not significant). That reproduces on the raw scale, and on
the raw per-cut numbers prefix length actually wins at the 30 and 50 percent cuts (0.387 and
0.464 against our 0.341 and 0.377). On `log1p` our probe is at 0.606 and 0.624 against prefix
length's 0.076 and 0.039. **The awkward result is an outlier artefact.**

---

## 8. GEPA (status only: a colleague owns this)

Round 5, ask 4. Code `src/baselines/gepa_luna.py`, launcher `runs/gepa/run_luna.sh`, env
`envs/dspy` (dspy 3.4.1, gepa 0.1.4). Design: per outer fold, GEPA optimises the **zero-shot
instruction** on the other folds' questions (70/30 split by question) and is tested on that
fold's rows, so it is scored on exactly the rows the Luna runs and the probes were scored on.
Optimising the k=0 instruction is the right call given section 4.3: examples hurt, so example
selection would optimise toward k=0 anyway.

**It ran on 2026-11-01, about $28, zero parse failures, and it made the prompt WORSE on
test.** Validation score rose in all 10 runs (for example 0.52 to 0.60). Mean 10 to 90
percent calibrated R squared, DSPy seed prompt to GEPA prompt, verified from
`runs/goal2/gpqa25/icl/results_{dspyseed,gepa}_luna_{ckpt,blind}_k0.csv` (`icl_cal` rows):

| config | target | seed | GEPA |
|---|---|---|---|
| `ckpt` | entropy | 0.2955 | **0.1394** |
| `ckpt` | p_mode | 0.2277 | 0.1095 |
| `ckpt` | rem_len_mean | 0.4114 | 0.1045 |
| `ckpt` | rem_len_std | 0.1280 | 0.0012 |
| `blind` | entropy | 0.2733 | 0.1243 |
| `blind` | p_mode | 0.2198 | 0.1131 |
| `blind` | rem_len_mean | 0.3528 | -0.0603 |
| `blind` | rem_len_std | 0.0836 | 0.0219 |

Sanity check: the DSPy seed prompt reproduces the old Luna k=0 run (entropy 0.2955 against
0.2945), so the harness is not the problem.

**Mechanism: the same one as "examples hurt".** The learned prompts are long (about 430 to
735 words against 196) and they **paste per-checkpoint training medians and ranges** into the
instruction. So the model's predictions acquire the right **scale** and lose the **ranking**.
Measured: within-cut correlation falls (`ckpt` entropy 0.55 to 0.42, `rem_len_mean` 0.65 to
0.44) while the **raw** uncalibrated `rem_len_mean` R squared **improves** (-0.1381 to
+0.1337). Our metric is raw R squared, which rewards scale, so GEPA's search climbed the
metric and destroyed the thing we care about.

**Two retries, both stopped by a rule agreed before looking at the data.**

1. Percentile metric (`--pct`): predict the within-checkpoint percentile, score on
   percentiles. Stop rule: stop if in 2 or more folds a candidate beats the seed on
   validation while its within-cut rank correlation on entropy or `rem_len_mean` drops more
   than 0.05. Triggered (`ckpt` fold 2: validation 0.649 to 0.671 while entropy correlation
   0.645 to 0.578). Cause: squared error on percentiles still rewards shrinking toward 50
   when the correlation is moderate. Killed after about 25 minutes, about $7.
2. Batch rank score (`--rank`): score each batch by mean Spearman over the four targets.
   Triggered again (`ckpt` fold 3: rank validation 0.727 to 0.731 while entropy correlation
   0.569 to 0.476). Validation gains were tiny (0.000 to 0.008) and 4 of 7 finished runs kept
   the seed prompt. Stopped after about 50 minutes, about $25. No full 5-fold score.

**The honest summary for him: prompt optimisation did not help, and we understand why. The
scorer is a scale-sensitive metric and every optimiser we tried bought scale by selling
ranking. A pure ranking objective is the obvious next design, and the two attempts at one
were not pure enough.** 7 of the 10 rank runs finished before the budget ran out, so the
remaining 3 would be cheap to complete if he wants the full number.

---

## 9. What we have NOT run, and known weaknesses

Read this section before the meeting. Everything here is a question he could reasonably ask.

### 9.1 Gaps in the commitment story

- **Commitment was never partialled for difficulty, on any dataset.** The published round-6
  `part3` does not do it, and it is not a trivial extension: the residual of a binary target
  is not a probability, so an honest version needs new code and a decision about what metric
  to report. This is the single largest hole in section 5.
- **Commitment AUC is not computed for the ICL configs on the new datasets.** We have it on
  GPQA (our probe 0.7732 against 0.6511 to 0.6774 for all nine configs, every difference
  significant), and round 6 showed the accuracy tie there was an artefact. On MMLU-Pro and
  GlobalOpinionQA we can only quote accuracy, where Luna ties us on GlobalOpinionQA. **We
  should not claim the AUC result generalises until it is run.** It is cheap, all the
  predictions exist.
- **The logistic-grid convention is unresolved and is recorded nowhere in the files.** AUC
  differs by which grid produced the fit (section 3.4). On GPQA ours beats TF-IDF by +0.026
  AUC under the shallow grid and +0.003 under the deep one; on GlobalOpinionQA the sign
  flips. Nothing in `ci_scores.csv`, `committed_auc.csv` or the reports marks which grid a
  given AUC came from, and the two values coexist in the repo (`_ref_ci/` uses the shallow
  grid, `_qtext/` uses the deep one). This needs a decision and a labelled rerun.
- **`will_commit` has never been run on either new dataset.** Verified: its target is absent
  from both `ref_fixed/results.csv` files. It exists on GPQA.

### 9.2 Gaps in the frontier-model comparison

- **The ICL length comparison is on raw-scale targets**, which section 7 establishes are
  tail dominated on GlobalOpinionQA (10 rows carry 55.6 percent of the sum of squares). Every
  approach is scored on identical rows so the **ranking** is sound, but the **magnitudes** on
  those two rows are driven by a handful of extreme rows.
- **No log-scale ICL calibration.** A log-scale ICL comparison needs the per-fold calibration
  refitted on `log1p`, which has not been done. So we cannot currently say how much of Luna's
  length deficit survives the transform.
- **The difficulty partialling of the ICL predictions has not been run on either new
  dataset** (only on GPQA, round 6, ICL subset). So we cannot yet say whether Luna's
  remaining skill on MMLU-Pro is also non-difficulty.
- **The round-4 between/within decomposition is degenerate on ICL rows**, which is a finding
  rather than a gap but should be said: there is 1 chain per question per cut, so the
  within-question sum of squares is exactly 0.000 at every cut and every cut's R squared is
  100 percent between-question by construction. The published Luna-against-probe comparison
  was therefore **never** a test of within-question skill. (On the full `ref_fixed` set the
  within-question share is 0.183, 0.291, 0.394, 0.473, 0.572 across the five cuts.)
- **Only three ICL configs on the new datasets** (`ckpt` k=0, `ckpt` k=10, `blind` k=0),
  against eight on GPQA. No k=50, no `nockpt`, no second model family. "Examples hurt" rests
  on k=10 alone on the new datasets.
- **500 questions per new dataset, not all 2,500.** Cost, not principle. The full set would
  be about 15,000 rows and about $118.

### 9.3 Gaps in the difficulty work

- **GlobalOpinionQA difficulty BANDS were deliberately not computed**, only the partialling.
  Both P2 and P5 advise against banding there: the label-derived measure (`ent0_norm`) is
  close to circular with the target, and 40.6 percent of its quintile assignments are
  arbitrary tie-breaks.
- **Probe weights were never refitted inside a difficulty band.** On GPQA that was
  impossible (39 questions against 3,584 dimensions). On MMLU-Pro the bands hold 255 to 978
  questions, so it is now feasible and has not been done. Only the **output map** (intercept
  and slope) has been refitted.
- **GlobalOpinionQA has no `p_gold`-equivalent**, so its label-derived column is not strictly
  comparable to the other two datasets'. Section 5.4(a).
- **The cross-dataset length comparison on a common scale is not finished.** Comparing
  GlobalOpinionQA's `log1p` 0.628 to GPQA's raw 0.524 is invalid; the matched `log1p` refits
  on all three datasets now exist (section 7), but they postdate the P6 report, which still
  carries the three-cut partial version.

### 9.4 Standing gaps from earlier rounds

- **24 of the 28 stored layers have never been probed on GPQA**, and on the new datasets
  **only layer 21 has been probed at all** (27 of 28 untouched). The one multi-layer result we
  have is that layers 20 to 22 concatenated beat L21 alone by a small but significant margin
  (+0.005 entropy), so there is probably a little more on the table and we have never looked.
- **No non-linear probe has been run on the other-LLM or embedder features.** Lower value now
  that closed-form kernels found nothing on 3 of 5 targets for our own features.
- **The black-box monitor with other representations** (round 5, ask 8) still needs his
  design call: question only, last-N sentences, or trivial scalars. The middle one is cheap
  (`--ex-sents` exists).
- **GPQA's capture check is clean; GlobalOpinionQA's is a manual override.** Section 2.4.
- **Unverified security item**, worth closing: earlier in this project the frontier API key
  was briefly written to `temp.txt` in this group-shared repo at mode 644. The file is now 0
  bytes but **key rotation was recommended and never confirmed**.

### 9.5 The three claims I would most expect to be attacked

My honest ranking, so you are not surprised.

1. **"We beat the frontier model on MMLU-Pro entropy."** The interval upper edge is -0.0012.
   It is significant by the stated rule and I would not retract it, but it is one bootstrap
   away from not being. The defensible version is "significant on all five targets, and
   comfortably so on four". GlobalOpinionQA (-0.079) is the dataset to lead with for the
   entropy claim.
2. **"The probe predicts properties of the distribution over future continuations."** Section
   5.8: **within a single question the probe is negative on every target, on both datasets.**
   It is a between-question predictor. Everything else in the story is consistent and
   replicated; this one limit is not fixed by the new data and it is the most likely place he
   lands. The honest framing, which is also the one that survives: activations beat text at
   judging **which questions** produce uncertain rollouts, not **which chain** of a given
   question will. Nobody else in this literature decomposes their R squared this way, which
   is both the risk and the contribution.
3. **The `committed_by_here` story.** Accuracy and AUC disagree in sign on GlobalOpinionQA;
   AUC depends on a logistic grid choice that is not recorded; the ICL AUC check is missing on
   the new datasets; and at GPQA's 90 percent cut accuracy is just the base rate (majority
   0.7609 against our 0.7583). If he probes commitment, concede that it is the least settled
   of the five targets and that we are mid-cleanup.

Two smaller ones worth pre-empting: the GlobalOpinionQA capture override (section 2.4), and
the fact that **ICL predictions are calibrated while the probe's are not** (raw Luna R squared
is far below zero). The latter is defensible, since at k=0 the model was never told the output
scale, but it must be stated in any write-up, and section 4.3 shows the calibration is exactly
what makes "examples hurt" visible.

---

## 10. Crib sheet: likely questions and the one-line answer

- **"Did more data make the effect bigger?"** No, and it was not supposed to. It made
  0.01-scale effects measurable: the interval shrank 3.1 to 3.7 times while the effect stayed
  at about 0.04 to 0.06. Section 3.1.
- **"So it is just reading question difficulty?"** Not on the larger datasets. That
  conclusion came from GPQA, averaged over depth. On MMLU-Pro half the entropy signal
  survives the most aggressive difficulty removal, and partialled entropy rises with depth on
  all three datasets. Say question-**level**, not question-**difficulty**. Section 5.
- **"Can you not just read the question text?"** On MMLU-Pro, at the prompt, with identical
  rows, question text gets 40 percent of the probe's entropy and 65 percent of its remaining
  length, and loses significantly on all five targets. Section 6.3.
- **"Is the length result just token counting?"** On the raw scale that objection has force.
  On `log1p` the probe beats prefix length by +0.164 / +0.254 / +0.521 across the three
  datasets, all significant, and the raw-scale tie on GlobalOpinionQA comes from 10 rows
  carrying 55.6 percent of the sum of squares. Section 7.
- **"Does the frontier model still tie you?"** Only on GPQA, and only on entropy. On
  GlobalOpinionQA we beat it by 0.079 and on MMLU-Pro on all five targets. Section 4.2.
- **"Why does it get worse with more examples?"** Measured, not guessed: within-cut
  correlation falls in 28 of 28 comparisons while the spread ratio moves toward 1. Examples
  supply the scale and break the ranking, and our metric is calibrated so the scale gain is
  free. Section 4.3.
- **"Does it work without a correct answer?"** Yes. GlobalOpinionQA has no gold answer at all
  and the probe still scores 0.155 normalised entropy, 0.628 `log1p` remaining length, and
  beats both text baselines. Section 3 and section 7.
- **"What is the within-question result?"** Negative on every target on every dataset. That is
  the honest limit. Section 5.8.
- **"Did GEPA help?"** No. It raised validation and halved test, by pasting training medians
  into the prompt. Section 8.

---

## 11. Discrepancies I found

Everything I checked against its source file, with the disagreements listed. Where no
disagreement is listed, the CSV matched.

**Verified exactly, no disagreement.** All three `overall_summary.json` ground-truth figures
(section 2.3); all 15 reference scores and all intervals in section 3; the three CI
half-widths 0.0558 / 0.0177 / 0.0150; the three ours-minus-TF-IDF entropy gaps 0.0517 /
0.0553 / 0.0356; every ICL score and paired difference in section 4; the four
difficulty-partialling headline values 0.2885 to 0.0248 and 0.2013, 0.4103 to 0.2004 and
0.3676; the four paired partialled gaps +0.0553 / +0.0627 and +0.0517 / +0.0325; the depth
effect per cut on all three datasets; all 0 percent cut scores; all six `log1p` versus raw
length gaps and the three `log1p` absolute scores 0.6517 / 0.7828 / 0.6284; all eight GEPA
seed-against-GEPA numbers; the two-tier K row counts and verdict; both capture-check cosines;
the round-4 allocation and learning-curve numbers.

**1. The P6 report prints four signs backwards.**
`runs/goal2/_newds_reports/p6_difficulty_scoring.md`, lines 59 to 61, gives the
ours-against-TF-IDF entropy gaps as "GPQA -0.032", "MMLU-Pro -0.063", "before partialling
both are significant (-0.052 and -0.055)". The CSVs give **+0.0325, +0.0627, +0.0517,
+0.0553** for ours minus TF-IDF. The report read the mirrored `TF-IDF minus ours` rows, which
the same file also stores. `CLAUDE.md` already flags this as a SIGN WARNING, and the flag is
correct. The real result is that the gap **widens** on MMLU-Pro.

**2. The within-band retention comparison in `CLAUDE.md` mixes band schemes.** It states
"MMLU-Pro retains 0.204 of entropy vs GPQA's 0.050 (4x)". The 0.204 is MMLU-Pro's
**`band_int`** value (`p8_band_decomposition.rollout.band_int.csv`) while the 0.050 is GPQA's
**`band_quintile`** value. The matched-scheme comparison is MMLU-Pro **0.2124** against GPQA
**0.0499**, which happens to give the same factor of 4, so the conclusion stands but the
numbers as paired do not come from the same scheme.

**3. `CLAUDE.md`'s P1 row overstates the GlobalOpinionQA 0 percent text-recoverability
result.** It says "activations add essentially nothing (92-96% text-recoverable, n.s., at
2,555 questions)". That is true for `rem_len_mean` (94.7 to 95.5 percent), `rem_len_std` (91.7
percent) and normalised entropy (93.7 percent), all not significant. But on **raw entropy**
(78.9 percent, diff -0.0327) and **p_mode** (78.7 percent, diff -0.0262) activations are
**significantly ahead** (`_qtext/cmp_f0.csv`). So "nothing on three of five targets" is the
accurate statement, not "nothing".

**4. `CLAUDE.md`'s K-tier cost range is slightly wide.** It says going uniform "COSTS 0.003
to 0.010 R2". `mmlupro_v1/sweep/_ktier/b_paired.csv` gives 0.0030 to 0.0060 on MMLU-Pro. The
0.010 appears to come from GlobalOpinionQA's entropy (-0.0102), which is a different dataset.
The direction and order of magnitude are right.

**5. The commitment AUC convention is genuinely ambiguous in the repo, and no file records
which grid produced which number.** This is not a `CLAUDE.md` error, it is an unresolved
convention, but it reads like a discrepancy when you open two files. Computed by me from the
stored predictions: GPQA ours 0.7840 (shallow `ref_fixed` grid) against 0.7612 (deep
`ref_logistic_deep` grid); GlobalOpinionQA ours 0.6830 against 0.6708. `_ref_ci/committed_auc.csv`
uses the shallow grid; `_qtext/cmp_scores.csv` uses the deep one. On GlobalOpinionQA the
**sign of our advantage over TF-IDF flips between them** (+0.0089 significant against -0.0033
not significant). The 2026-10-26 grid-fix conclusion that "the published number stands" is
correct for **accuracy** (identical to 4 decimal places) and does **not** hold for AUC. Also,
contrary to that row's note that TF-IDF and prefix length are "identical to 4 dp", prefix
length's AUC moves a lot between grids on GPQA (0.7562 to 0.7170); it is the **accuracy** that
is identical.

**6. One number I could not verify as stated.** `CLAUDE.md` quotes GPQA commitment AUC as
0.784 in some places and 0.761 in others without saying which grid. Both are real (see item
5), so this is not an error in either value, but there is no single file that fixes "the" GPQA
commitment AUC. I report both everywhere above.

### Numbers you gave me that I could not reproduce

None. Every figure in the brief matched its source file, with two wording refinements:

- The "roughly 67% to 92%" range for the 0 percent share is right for the four targets
  reported on all three datasets, but GlobalOpinionQA's `rem_len_std` share is **24 percent**
  and its normalised-entropy share is **29 percent**, both below the stated range. I have
  reported the full table rather than the range.
- "log1p is flat or rising" with depth is right for MMLU-Pro (0.800 to 0.784) and
  GlobalOpinionQA (0.569 to 0.707) but GPQA's `log1p` does decline mildly (0.698 to 0.619).
  I have written it as "flat on MMLU-Pro, mildly declining on GPQA, clearly rising on
  GlobalOpinionQA".
