# Round 7: to-do list from the 2026-11-03 professor meeting

Written 2026-11-03 by Claude from the user's notes straight after the meeting. This supersedes
`NEXT_STEPS.md` (the 2026-10-27 list, now fully closed) as the live to-do list.

**Read in this order before starting:** this file, then `CLAUDE.md` ("Rules", "Results index",
"Reporting conventions to keep"), then `learnings.md`. `docs/premeeting_2026-11-03.md` is the
detail pre-read behind what was presented. The deck that was presented is
https://claude.ai/artifact/YVgRxdoe54dLU28GZVCHSA (19 slides, version `1793713759-6232`).

Nothing below has been started. No jobs are running. The working tree has many uncommitted new
files (see `git status`); commit only when the user says so.

---

## The eight items, in the order to do them

Cost tiers: items 1 to 4 need **no new compute** (they are plots over predictions already on disk).
Item 5 is a few CPU-hours. Item 6 is a colleague's. Items 7 and 8 are the expensive ones.

| # | ask | cost | blocking on |
|---|---|---|---|
| 1 | MSE per difficulty band, not R2 | free, ~1 h | nothing |
| 2 | Accuracy vs probe error, accuracy vs monitor error (continuous, no buckets) | free, ~2 h | nothing |
| 3 | GlobalOpinionQA: x axis = consistency, not difficulty | free, ~1 h | nothing |
| 4 | Remaining-length R2 against checkpoint | free, ~1 h | nothing |
| 5 | A probe explicitly given the checkpoint as well as the activation | ~4 CPU-h | nothing |
| 6 | GEPA: smaller dataset, bigger batch | ~$0 (local Gemma) | coordinate with Rish |
| 7 | Attention probes with weighted pooling (Reasoning Theater) | ~165 GB disk + ~3 GPU-h per dataset | user's go-ahead on disk |
| 8 | Two more models end to end | ~100 to 600 GPU-h | user picks the models |

---

## Item 1. MSE for each band of difficulty

**The ask, and why it is the right one.** R2 inside a band is `1 - SS_err / SS_band`, and the
denominator changes from band to band. That is exactly why the band figures have two absurd
cells: GPQA `v_easy` at **-1.090** and MMLU-Pro `easy` at **-1.049**. In both the truth is nearly
constant inside the band (GPQA `v_easy` entropy sd 0.27 vs 0.64 overall; MMLU-Pro `easy` 0.279 vs
0.693), so a small absolute error divided by a tiny variance explodes. **MSE has the same units in
every band and does not do this.** The professor's ask fixes a presentation problem we had already
documented but worked around with offset correction.

**What to produce.** For each dataset, target, band and label (ours, Luna k=0, TF-IDF, prefix_len):
mean squared error, with a 95% question-bootstrap interval, plus **two reference lines so MSE is
readable**:
- `MSE of predicting that band's own mean` (the within-band variance). Ours below this line means
  the probe beats the band mean; this is the same information R2 carried, in absolute units.
- `MSE of predicting the full-set mean`, so the reader sees how much of the work the band
  membership alone does.
Report RMSE in the target's own units alongside MSE (bits for entropy, tokens for length); the
professor reads "0.52 bits of error on a 0.64 bit spread" far faster than an R2.

**Where the code is.** `src/analysis/icl_bands.py` (GPQA) and `src/analysis/newds_bands_icl.py`
(the new datasets, which imports and repoints the former; written 2026-11-03, see its Results index
row). `icl_bands.score_rows` already builds the bootstrap weight matrix `W` and computes
`dc.wr2`, `wr2_recentred` and `wcorr`. Add a `wmse(y, p, W) = (W * (y - p) ** 2).sum(1) / W.sum(1)`
next to them and emit `mse`, `mse_ci_lo`, `mse_ci_hi`, `mse_band_mean`, `mse_global_mean`,
`rmse` into `p5_bands_scores.csv`. Do not remove the existing columns: the recentred R2 is what the
current slides quote.

**Then re-make the figures.** `src/viz/meeting_bands.py` makes `m9a/m9b/m9c` (2x2: R2 and
within-cut correlation). Switch the top row to MSE with the two reference lines and keep the
correlation row. Same palette slots, no captions under the figure (the user removed those from all
figures on 2026-11-03; do not reintroduce `caption()` in `src/viz/meeting_figures.py`).

**Say this when reporting it.** The two degenerate bands stop being degenerate under MSE, and the
honest reading flips: on easy questions the probe has the *lowest* absolute error of any band and
is still beaten by the band mean. Both facts are true and the second is the interesting one.

## Item 2. Accuracy vs probe error, accuracy vs black-box-monitor error

**The ask.** As question accuracy falls, prediction error should rise. Plot it continuously and
**move away from difficulty buckets** (the professor said this explicitly). This replaces the band
figures as the primary difficulty evidence; keep the band figures for item 1 but demote them.

**This is free.** Everything needed is on disk:
- probe predictions: `runs/goal2/<tag>/sweep/ref_fixed/preds*.csv` (`y_true`, `y_pred`, keyed
  `example_id, cot_id, fraction`, out of fold)
- monitor predictions: `runs/goal2/<tag>/icl/preds_gpt6_luna_high_k0.csv`, filtered
  `fraction > 0 & probe == "icl_cal"` (calibrated, as published)
- accuracy per question: `p_gold` in `runs/goal2/<tag>/sweep/_difficulty/question_difficulty.csv`
  (GPQA and MMLU-Pro), the fraction of that question's rollouts that are correct.

**Design.** x = question accuracy (`p_gold`, 0 to 1). y = per-question mean squared error of the
prediction. One line per label (ours, Luna, TF-IDF, prefix_len), one panel per target, one figure
per dataset, scored **within each cut and then averaged** (never pooled across depth). Overlay the
raw per-question points lightly behind the binned means, and draw the band of a
question-bootstrap interval around each line. Fit nothing; a LOWESS or 10-bin running mean is
enough and does not invite a reader to read a slope off noise.

**Three traps.**
1. `p_gold` on MMLU-Pro takes **only 11 values and 65% of questions sit at exactly 0 or 1**
   (5 rollouts per question). So the x axis is lumpy there; use the value itself with jitter plus
   binned means, and say n per bin on the figure. On GPQA (25 rollouts) it is genuinely continuous.
2. The monitor is calibrated per fold and our probe is not. That is defensible (at k=0 Luna was
   never told the scale) but it must be said, because MSE is scale sensitive in a way R2 is not.
   Report both the calibrated monitor and, as a second line, the raw one.
3. Pick **one** row set and name it. Luna only ran on 500 questions per new dataset
   (`runs/goal2/<tag>/icl/`), so a figure that includes the monitor must inner-join to those rows.
   A probe-only version can use the full sweep. Two legitimate numbers for the same quantity is
   the project's most repeated trap (see "ROW-SET TRAP" in the Results index).

**Expected result, so a surprise is recognisable.** We already know difficulty explains a large
share of the entropy signal at shallow cuts and little at deep ones (partialled entropy runs
0.087 to 0.272 over the five MMLU-Pro cuts), and that both the probe and Luna over-predict
uncertainty on easy questions by about 4x (GPQA `v_easy` truth 0.10 bits vs ~0.40 predicted). So
expect error to be **highest on easy questions**, which is the opposite of the professor's stated
expectation, because the dominant failure is the over-prediction, not noise on hard questions.
If that is what the figure shows, lead with it: it is a cleaner statement of the shrinkage finding
than anything currently on a slide.

## Item 3. GlobalOpinionQA: switch the x axis to consistency

GlobalOpinionQA has **no gold answer** (`p_gold` null on all 63,684 rows), so item 2's accuracy
axis does not exist there. Use consistency instead. Three measures are already computed in
`runs/goal2/globalop_v1/sweep/_difficulty/question_difficulty.csv`; all were checked this session:

| column | what it is | distinct values | use it as |
|---|---|---|---|
| `p_mode0` | share of the question's 5 base rollouts that gave the modal answer, at the 0% cut. The model's self-consistency. | **9** | the **primary** x axis |
| `ent0_norm` | normalised answer entropy at the 0% cut | 93 | already the published label-derived difficulty measure; keep for continuity |
| `human_disagree` | disagreement in the **human** survey responses that ship with GlobalOpinionQA | **1,702** | a second, externally derived panel |

**Recommendation: `p_mode0` as the primary axis**, because it is the direct analogue of accuracy
("how often does the model land on the same answer") and reads as "consistency" on a slide without
explanation. Two things to handle:
- It has only 9 values, so **do not band it** (its quintiles are 82.6% arbitrary tie-breaks). Plot
  it continuously exactly as in item 2. This is also what the professor asked for.
- It is label-derived from the same rollouts as the entropy targets, so it is partly circular.
  Say so, and add the non-circular companion: `qtext_ent_norm_mean` from
  `runs/goal2/globalop_v1/sweep/_qtext/`, the out-of-fold prediction of the question's mean
  normalised entropy from the question words only.

**`human_disagree` is the find worth showing him.** It is external to the model entirely, nearly
continuous, and (measured this session) correlates **-0.054** with `p_mode0` and **+0.042** with
`human_ent_norm` vs model consistency. It explains almost none of our target variance
(r2 0.009 on raw entropy). So: **the model's uncertainty on an opinion question has essentially
nothing to do with whether humans disagree about it.** That is a short, quotable, genuinely
new result from a dataset column nobody has used yet, and it is free. Caveat to state:
`human_disagree` correlates +0.370 with `n_options`, so control for option count or use the
normalised target.

Report GlobalOpinionQA entropy **normalised** (`answer_entropy_bits_mm_norm`): about two thirds of
the raw text-entropy signal there is option counting (`n_options` 2 to 18, a quadratic in
`log2(n_options)` alone gets 0.1432 of raw entropy). And for its length targets quote **both** raw
and `log1p` (`sweep/_loglen/`); raw is tail dominated, 10 rows of 12,736 carrying 55.6% of
`rem_len_mean`'s sum of squares.

## Item 4. Remaining-length R2 against checkpoint

**Free, and it is the easiest win on the list.** Per-cut scores already exist:
- raw: `runs/goal2/<tag>/sweep/ref_fixed/results.csv` (`fraction`, `test`) and the tidy
  `sweep/_ref_ci/per_checkpoint.csv` (`label, target, fraction, score, n`)
- log1p and winsor99: `runs/goal2/<tag>/sweep/_loglen/results.csv` (`transform` column)

**What is missing is per-cut intervals.** `_ref_ci/per_checkpoint.csv` has no CI columns: the
stored bootstrap covers only the mean over cuts (this is already flagged on figure `m6`). Add them
by resampling questions from the per-cut `preds.csv`; `src/probing/loglen_ci.py` already does
exactly this for the length targets and imports `boot`/`r2` from `loglen_refit.py`, so reuse it
rather than writing a new bootstrap.

**Design.** x = cut point (10 to 90%), y = R2, one line per label, two panels (raw and log1p),
one figure per dataset or one figure with three columns. Put the three datasets side by side: the
depth shapes genuinely differ and that is the content.

**The result is already known and it is a good slide.** Raw length R2 decays with depth on all
three (GPQA 0.626 to 0.375) but on `log1p` it is **flat on GPQA (0.698 to 0.619), flat on MMLU-Pro
(0.800 to 0.784) and RISES on GlobalOpinionQA (0.569 to 0.707)**. So "length gets harder to predict
deeper into the chain" is partly a tail artefact. Also put `prefix_len` on the same axes: raw has
it ahead of us at the early cuts on GPQA and GlobalOpinionQA, log1p has us ahead at every cut on
all three. **Never put a raw R2 and a log1p R2 on the same axis.**

## Item 5. A probe explicitly given the checkpoint as well as the activation

**Read this before coding, it is the whole difficulty.** `src/probing/sweep.py` fits **one probe
per cut point** (`for frac in fractions:` at line ~544). Inside a cut, `fraction` is a constant, so
handing it to the probe as a feature does literally nothing. The professor's ask therefore
necessarily means **a single probe pooled across all five cuts, with the checkpoint as an input.**

**And pooling is the thing CLAUDE.md warns about most.** Pooled across depth, `prefix_len` scores
0.718 on remaining length against our 0.716: the depth variable alone carries most of the pooled
variance, so a pooled R2 is not comparable to any number in any of our write-ups. Both halves must
be reported:

1. **Pooled R2**, three models on identical pooled rows: activations only, `fraction` only
   (one-hot, 5 columns, a floor), and activations + `fraction`. Say loudly that these are pooled
   numbers and give the `fraction`-only floor right next to them.
2. **Within-cut R2 from the pooled model**: take the pooled model's out-of-fold predictions, split
   them by cut, score inside each cut, average. This is directly comparable to every published
   number (0.289 GPQA, 0.410 MMLU-Pro, 0.219 GlobalOpinionQA) and is the honest answer to "does
   telling the probe where it is help?".

**Implementation, minimal.** Add a feature class to `sweep.py` beside `PrefixLen` (line ~170):

```python
@register_feature("fraction")
class Fraction(Feature):
    """One-hot of the cut point. Only meaningful in a pooled fit (fractions: [null] / all cuts
    in one fit); inside a single cut it is constant and contributes nothing."""
    def __init__(self, M, cfg):
        f = M.fraction.to_numpy(float)
        self.raw = np.column_stack([(np.isclose(f, v)).astype(np.float32)
                                    for v in sorted(np.unique(f))])
```

and a `concat` feature that stacks two named features column-wise (there is no existing way to
combine two feature providers; `Act` already concatenates *layers*, which is not the same thing).
Then add a `pooled: true` config key that skips the per-fraction loop and fits once on all rows.
Keep the fixed folds (`fixed_fold_of`) so questions stay in the same fold as everywhere else.
**Standardisation note:** `Feature._prep` standardises columns on train rows; a one-hot survives
that fine, but check the `sd < 1e-8` guard does not zero a cut that is absent from a fold.

**Three controls worth running at the same time, all cheap:**
- `fraction` as a scalar instead of one-hot (is the relationship monotone in depth?)
- `prefix_len` + `fraction` (the professor's ask applied to the trivial-scalar baseline; if the
  gain is the same size there, the gain is about depth and not about activations)
- the 0% cut included vs excluded. It has no commitment labels and a constant prefix length, so
  including it changes what "pooled" means. Default: exclude, and say so.

**Honest prior.** Depth is already implicit in the activations (a probe reads prefix length well),
so expect the within-cut gain to be near zero and the pooled number to jump a lot. If that is the
result, it is a clean negative and worth one slide, because it tells him the per-cut protocol was
not throwing information away.

## Item 6. GEPA with a smaller dataset and a bigger batch

**Coordinate first: this is Rish's thread, not a fresh start.** `runs/gepa/run_luna.sh`,
`src/baselines/gepa_luna.py`, `src/baselines/gepa_monitor.py` and `envs/dspy` are his, and he has
already built the next step: `runs/gepa/gemma_job.sh` (untracked, written 2026-11-02) serves
Gemma-4-31B-it locally on 2 H100s with TP=2 and runs GEPA against it for free, and
`runs/goal2/gpqa25/gepa/rows_all.jsonl` (492 MB, all 25 chains per question) was prepped on
2026-11-02. **Ask the user to loop Rish in before touching any of it.** Write-up of the three
failed tries: `docs/gepa_luna_2026-11-01.md`.

**What the ask maps to in the code.** Two separate knobs:
- "reduce dataset size" = the trainset handed to `GEPA.compile`. Today, per outer fold, GEPA gets
  the other folds' questions split 70/30 by question, so about 656 train rows over 111 questions.
- "increase batch size" = `reflection_minibatch_size`, already exposed as `--minibatch`
  (default 30 in `--rank` mode, which is DSPy's default of 3 raised tenfold). `--budget` sets
  `max_metric_calls`.

**Why it is a sensible fix and what to watch.** Try 3 failed because the signal was noise: over
7 finished runs the best candidate beat the seed by **0.000 to 0.008** on the `(rho+1)/2` scale and
4 of 7 kept the seed outright. Fewer rows times a bigger minibatch means each candidate is scored
on a larger share of the trainset, so the ranking between candidates is less noisy per dollar.
Suggested grid on Gemma (free, so run several): trainset 111 questions (as now) and 40 questions,
crossed with minibatch 30 and 100, `--rank`, one fold first.

**Keep the pre-registered stop rule.** It is in `docs/gepa_luna_2026-11-01.md`: stop if in two or
more folds a candidate beats the seed on validation while its within-cut rank correlation on
entropy or remaining length drops by more than 0.05. It fired twice already and it is the reason
we did not report a bad result. `gepa_monitor.py` watches `metric_log.jsonl` for it.

**State the prior honestly to the professor.** Round 6 established that in-context examples supply
the output scale and break the ranking, in 28 of 28 comparisons across two model families, and
GEPA's three failures are the same mechanism. A smaller, better-sampled trainset addresses the
*noise* in candidate selection but not that mechanism. Worth one free run; not worth more paid API.

## Item 7. Attention probes with weighted pooling (Reasoning Theater)

**The paper.** "Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought",
arXiv **2603.05488** (v4 is current; the method text below is from the v3 HTML). Code:
https://github.com/AskSid/disentangling-computation-from-cot, probe implementation in
`src/analysis/run_probing.py` (858 lines). Read both before writing anything; the summary below
was taken from them this session and is reliable for the architecture but **the hyperparameters
come from a model-extracted summary of Appendix B and must be re-checked against the paper and
against `experiments/*.yaml` + `src/analysis/experiment_config.py`, which were not read.**

**Their probe, exactly (paper Eq. 1).** `z = W_v H^(l) . softmax(W_q H^(l))` with `W_q` in
`R^(1 x d)` and `W_v` in `R^(C x d)`, where `H^(l)` is the `d x T` matrix of layer-l hidden states
over the first `T` response tokens. In code (`class AttentionProbe`):
- `self.q = nn.Linear(d, 1)` (a learned query vector, with bias); attention logits are `q(x)`,
  masked to `-inf` past the valid length, then `softmax` over the **token** axis.
- `self.v = nn.Linear(d, C)`, or an MLP variant `Linear(d, 32) -> ReLU -> Linear(32, C)` that the
  paper does not mention. Pooled output is `sum_t attn_t * v(x_t)`. Single head, no temperature,
  no dropout, parameters in bf16.
- Training: each sample is randomly truncated to `end_idx = randint(1, seq_len)`, redrawn every
  epoch, **including for the validation loader**, so their val loss is not deterministic.
- Evaluation: causal prefix pooling in closed form, `cumsum` of `exp(logits)` and of
  `exp(logits) * v(x)`, so the prediction at token t uses only tokens 1..t. Softmax in fp32.
- One probe per layer, best layer reported. Normalisation, if enabled, is **one scalar mean and
  one scalar std per layer** over all tokens of all training samples, not per dimension.
- Loss cross-entropy (4-way answer) or BCE (binary); Adam with weight decay (not AdamW); no
  patience, just keep the best-val-loss epoch. Split by question. Single seed, no ensembling.
- Their headline: attention probe **87.98%** macro vs a last-token **linear probe 31.85%** on the
  same hyperparameters, and 28.24% on randomised labels. Their target is the model's final answer
  (classification), not a distribution property.

**What has to change for us, and be explicit about each deviation.**
1. **Regression, not classification.** `C = 1` per target (or 4 outputs, one per regression
   target), MSE loss. Their 56-point gain over a linear probe is on a classification task whose
   last-token probe was near chance; ours is a regression whose last-token probe is at R2 0.29 to
   0.41. **Do not expect a 56-point analogue.** Say this to the professor up front.
2. **We need a baseline they did not run.** They compare attention pooling only against
   last-token. Mean pooling and max pooling are the controls that tell you whether the *learned
   weighting* matters or merely *having more than one token* does. Run all four: last token
   (= our published probe), mean pool, max pool, attention pool. This is the single most
   important design decision in the item.
3. **Ridge is our reference, not a trained linear head.** `src/probing/mlpgap_*.py` established
   (round 4, idea 2) that the step-4a SGD recipe reproduces ridge's closed form at weight cosine
   only 0.865 and costs 0.008 to 0.025 R2. So an attention probe trained with Adam must be
   compared against **ridge on the last token**, and a mean-pool *ridge* should be run too, which
   is closed form and free of optimiser confounds. Expect to be asked "is the gain the pooling or
   the optimiser"; answer it by construction.
4. **Fixed folds.** Use `sweep.fixed_fold_of(run_dir, 5, 0)` so questions land in the same folds
   as all 22 published approaches. Group by `example_id`; a question's chains share text.
5. **Seed ensembling.** Round 4 found a single seed was significant at 0 of 5 checkpoints while a
   5-seed ensemble was significant at 4 of 5. Run 5 seeds.

**The blocker: our activation store is last-token only.** `extract_checkpoints.py` writes exactly
one vector per `(example_id, cot_id, fraction)`, all 28 layers, 4.6 GB for GPQA. Weighted pooling
needs every token. Measured this session, for **one** layer, fp16, 3,584 dims, reasoning tokens
only (prompt excluded):

| dataset | CoTs | reasoning tokens at the 90% cut (mean / max / sum) | stride 1 | stride 4 | stride 8 |
|---|---|---|---|---|---|
| gpqa25 | 4,915 | 4,669 / 14,798 / 2.295e7 | **164.5 GB** | 41.1 GB | 20.6 GB |
| mmlupro_v1 | 12,408 | 1,912 / 14,978 / 2.373e7 | **170.1 GB** | 42.5 GB | 21.3 GB |
| globalop_v1 | 12,738 | 515 / 14,167 / 6.558e6 | **47.0 GB** | 11.8 GB | 5.9 GB |

(`/proj` showed 3.0 PB free, so disk is available, but **ask the user before writing 165 GB** into
a group-shared folder, and check whether the project has its own quota: `quota -s` only reports the
home filesystem.)

GPU cost is small: capture is one forward pass per CoT over its longest prefix (causal, so all five
cuts come out of one pass), and the original GPQA capture ran about 3 minutes per 145-CoT shard on
one H100, i.e. roughly **2 GPU-hours per dataset**. The cost is disk and I/O, not compute.

**Recommended path, cheapest first.**
- **Step A, pilot, 33 GB.** GPQA, layer 21 only, stride 1, **5 of the 25 chains per question**
  (all 198 questions). Enough to fit an attention probe and to run all four pooling baselines, and
  it answers "does learned weighting beat mean pooling at all" before committing 165 GB. If the
  answer is no, the item stops here and that is a publishable negative.
- **Step B, if A is positive.** Full GPQA at layer 21 (164 GB), then GlobalOpinionQA (47 GB, the
  cheapest full dataset and the one with the shortest chains). MMLU-Pro last.
- **Decide stride deliberately.** The paper pools every token. A stride-4 store (41 GB) is 4x
  cheaper and the old `extract_activations.py` already used stride 4, but it is a deviation from
  the paper and the user asked explicitly for consistency with it. **Default to stride 1 on the
  pilot** so the deviation question never arises, and only discuss stride if the full run is
  refused on disk grounds.
- **Which layer.** L21 for comparability with everything published. Note 24 of the 28 layers have
  never been probed and the paper picks the best layer per model, so a later layer sweep is a
  natural follow-up (and 28 layers x 165 GB is not affordable, which is itself a reason to do the
  pilot at one layer).

**Code to write.**
- `src/probing/extract_tokens.py`: a sibling of `extract_checkpoints.py`, same EasySteer
  conventions (decoder layer L maps to ES index L-1; token position t maps to ES position t+1,
  both asserted at runtime), `--layers 21`, `--stride`, writing per-shard `.npy` plus a parquet
  index giving each `(example_id, cot_id)` its row range and each cut's token offset. Reuse
  `build_cases`. Keep it resumable per shard like the original. **Verify against the existing
  store**: the last token of a CoT's longest prefix must equal the stored `ref_fixed` vector for
  that row, bitwise or at cosine > 0.9999. That check is the whole guarantee that the new store is
  the same quantity, so write it first.
- `src/probing/attn_probe.py`: dataset with per-sample token ranges read by `np.load(mmap_mode="r")`
  (do **not** copy the paper's approach of loading everything into a Python list; 164 GB will not
  fit in RAM), the four poolings, MSE loss, fixed folds, 5 seeds, best-val-epoch selection, and
  out-of-fold predictions written in the same `preds.csv` schema the rest of the project uses
  (`feature, probe, target, example_id, cot_id, fraction, y_true, y_pred`) so
  `src/viz/master_table.py` and the existing bootstrap can score it with no new scoring code.

**What the user actually wants to know:** "the effect that pooling will have on entropy". So the
headline deliverable is one number with an interval: attention pooling minus last token on
`answer_entropy_bits_mm`, per cut and averaged, paired question bootstrap, against mean pooling as
the control.

## Item 8. Two more models

**Recommended pair, and the reason for the pair.** One varies **scale** holding family and training
recipe fixed; the other varies **family** holding size fixed. That way a difference is attributable.

### Pick 1, bigger Qwen: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
- **Why this and not a newer Qwen.** It is the *same distillation of the same R1 teacher onto the
  same Qwen2.5 base* as our 7B, so it is the only choice that isolates scale. `Qwen/Qwen3-32B` or
  `Qwen/Qwen3.5-27B` would change the recipe and the architecture at the same time as the size, and
  Qwen3.5 is a hybrid (GatedDeltaNet) model whose layers are not all attention layers, which
  complicates layer-wise probing.
- 4.6x the parameters, **64 decoder layers vs 28** and 5,120 hidden vs 3,584. The extra depth is a
  bonus: our layer story rests on one layer of 28.
- **Architecture is `Qwen2ForCausalLM`, so it runs unchanged in both envs**, including the
  generation env (`envs/goal1`, vLLM 0.5.5) and the capture env (`envs/easysteer`). Nothing in
  `run_eval.py`, `continuations.py` or `extract_checkpoints.py` needs porting. The think-tag
  convention, the `{"Final Answer": ...}` parser and the `</think>` prefill all carry over.
- Needs TP=2 on H100s (64 GB of bf16 weights leaves too little KV cache for 15k-token contexts at
  K=25 on one card).

### Pick 2, different family: `allenai/Olmo-3-7B-Think`
- **Why.** Apache 2.0, **fully open training data and code**, a genuine long-CoT reasoning model
  that emits think tags, and **7B, so family is not confounded with scale** against our model.
- **We already have the matching control.** Round 3 probed `allenai/Olmo-3-1025-7B` (the base) as
  an other-family activation control and found it **ties our model on all five targets** (entropy
  0.282 vs 0.289, all n.s.). So we already know its *representations* carry the same information
  about our model's rollouts; running its own rollouts through the whole pipeline asks the much
  better question of whether the finding is a property of the model or of the task. The base
  weights are already in `cache/huggingface/hub`.
- Shape: 32 layers, 4,096 hidden, `Olmo3ForCausalLM`, `sliding_window` 4,096 (note this: attention
  is windowed on most layers, so a hidden state late in a 10k-token chain does not attend to the
  whole chain, which is a real and interesting difference from Qwen and must be said rather than
  discovered).
- **Supported in the capture env** (`Olmo3ForCausalLM` is in the EasySteer vLLM registry, mapped to
  `olmo2`) but **not in vLLM 0.5.5**, so generation must move to `envs/icl` (vLLM 0.31). The port
  looks cheap: `run_eval.py` and `continuations.py` use only `LLM(...)`, `SamplingParams(...)` and
  `llm.generate(...)`, all stable across those versions. Budget half a day and a smoke test, not a
  rewrite.

**The alternative to pick 2, if he wants a stronger non-Qwen model:** `openai/gpt-oss-20b`
(Apache 2.0, MoE, 21B total and 3.6B active, native reasoning effort levels, `GptOssForCausalLM` is
in the EasySteer registry). Its advantage is a **direct tie to the Reasoning Theater paper**, which
ran GPT-OSS-120B, so item 7's attention-probe numbers would be comparable to a published result on
the same family. Its disadvantages are that MoE changes what a "layer activation" means, the
harmony channel format changes prefix construction and parsing, and 20B total vs our 7B reopens the
scale confound. Recommend it as the second-round addition, not the first.

**Cost, and this is the part to settle before anything is launched.** Measured from this repo:
GPQA-Diamond at 198 questions x 25 chains x 5 cuts x K=25 took about **40 hours per GPU worker
over 4 workers** for the continuation stage with the 7B (`runs/logs/gpqa25/cont_worker_*.log`,
2026-09-20 to 09-26), i.e. roughly 130 GPU-hours, generating about 1.75e9 tokens at around
3,700 tokens/s/H100. A 32B at the same workload is roughly 4.6x the FLOPs, so **about 600
GPU-hours**, which is not affordable on a shared 16-GPU partition.

**So do not repeat the GPQA design.** Round 4's allocation result says to spend rollouts on
questions, not chains: `150 questions x 2 chains` matched `30 questions x 25 chains`, and a new
question is worth about 3.5x a new chain per row. Concretely, for each new model:
- GPQA-Diamond, all 198 questions, **5 chains** (not 25), K=25: about **120 GPU-hours** for the
  32B, about 26 for a 7B.
- plus an MMLU-Pro subsample of 500 questions x 5 chains if the first lands.
That keeps a new 7B model inside a day of cluster time and the 32B inside a week, and it matches
the 5-chain design the new datasets already use, so the comparison is clean.

**Checklist before launching either model** (all of these have bitten this project):
1. `03_verify_capture.sh` must PASS for the new model before any capture is trusted. The EasySteer
   position and layer conventions were verified for Qwen only. OLMo's tokenizer **adds no BOS**,
   which is exactly the kind of thing the `ES_POS_OFFSET = 1` assertion exists to catch.
2. Run the k-sufficiency stage rather than assuming K=25. It is per model and per dataset.
3. Check the parser on 40 samples first. Our own model produces degenerate ICL output and
   GlobalOpinionQA needed `n_options` handling; a new model's `{"Final Answer": ...}` compliance is
   not a given.
4. Sentence splitting for the 10 to 90% cuts uses `generation.checkpoint.split_sentences`. Check it
   on the new model's output; OLMo and gpt-oss format think blocks differently.
5. Write outputs to `runs/goal1/<newtag>/` and `runs/goal2/<newtag>/` and add a Results index row
   the moment the first file exists.

---

## Facts measured during this session, so nobody re-measures them

- `sweep.py` fits one probe per cut point (`for frac in fractions:`), so item 5 requires a pooled
  fit; `fraction` is constant within a cut.
- Per-dataset reasoning-token counts and the resulting all-token store sizes: the table in item 7.
- `/proj` has 3.0 PB free; `quota -s` reports only the home filesystem (32.9 GB of 71.5 GB used).
- Capture throughput: about 3 minutes per 145-CoT shard on one H100 (`extract_worker_*.log`).
- Continuation stage for GPQA with the 7B: about 40 h x 4 GPU workers.
- `envs/goal1` has vLLM **0.5.5** (generation), `envs/easysteer` a vLLM dev build
  (`0.1.dev14461+g21d7a9f61`) whose registry includes `Qwen3`, `Qwen3Moe`, `Gemma3`, `GptOss`,
  `Olmo3`, `Glm4`, `Nemotron`, `DeepseekV3` and a `qwen3_5.py` module, `envs/icl` vLLM **0.31.0**.
- `allenai/Olmo-3-1025-7B` (cached): 32 layers, 4,096 hidden, `Olmo3ForCausalLM`,
  `sliding_window` 4,096, `max_position_embeddings` 65,536.
- GlobalOpinionQA consistency columns and their properties: the table in item 3, including that
  `human_disagree` exists for 2,554 of 2,555 questions with 1,702 distinct values and correlates
  -0.054 with `p_mode0`.
- `runs/goal2/gpqa25/gepa/rows_all.jsonl` (492 MB) and `runs/gepa/gemma_job.sh` already exist and
  belong to Rish.

## Traps that apply to every item here

- **Row sets.** Three legitimate and mutually incomparable sets: the full sweep
  (`ref_fixed`, `_ref_ci`), the ICL subset (500 questions on the new datasets, 1,178 rows on
  GPQA), and the 0% cut (`f0_probe`). GPQA has two published partialling results for this reason.
  Name the row set on every figure.
- **Scales.** R2 on `log1p(y)` is not R2 on raw `y`, and a question-mean R2 is not a row-level R2.
  Both mistakes have been made in this project and caught.
- **Pandas column-vs-method collisions.** `d.transform`, `d.score`, `d.mean`, `d.minus` can all
  resolve to DataFrame methods instead of columns, which silently returns a scalar `False` in a
  filter. One such bug cost a crashed job. Print `list(d.columns)` and use `d["transform"]`.
- **Commitment.** Quote accuracy (unambiguous) and name the logistic grid whenever quoting an AUC;
  the grid flips the sign of our GlobalOpinionQA advantage. Commitment has never been partialled.
- **Figures.** No em dashes, white background, `#111111` text, Arial, no fine print under a figure
  (the user removed all of it on 2026-11-03), significance by filled vs hollow markers and not by
  colour. Arial is not installed here; `meeting_figures.py` falls back to Liberation Sans.
- **Never `git push` without asking. Commit only when told.** Do not commit other sessions' files
  (`src/baselines/gepa_*`, `runs/gepa/*` are Rish's).
- **Unresolved security item:** the frontier API key was once written to `temp.txt` at mode 644 in
  this group-shared repo. The file is empty now but **rotation was never confirmed**. Ask.
