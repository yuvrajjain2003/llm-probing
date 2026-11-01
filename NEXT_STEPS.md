# NEXT_STEPS.md

_Written 2026-10-29 from the user's notes after the 2026-10-27 meeting; updated 2026-11-01._
_**Jump to "Where to actually start" near the bottom if you are a new session.** Actions 1 and 2 and_
_both concerns are DONE; the rest of this file is the record of how and why._

_This is the live to-do list.
`CLAUDE.md` holds the rules, the cluster notes and the Results index (every result produced so far).
Read `CLAUDE.md` first, then this file._

**Status, updated 2026-10-29 (later).** **Actions 1 and 2 are DONE, concern 2 is SETTLED and concern 1
is ANSWERED, all for zero compute and zero dollars.** Write-up: `docs/update_2026-10-29.md`. Outputs:
`runs/goal2/gpqa25/sweep/_iclsplit/` and `runs/goal2/gpqa25/sweep/_iclwhy/`. **What remains in round 6 is
actions 3 and 4 (getting off GPQA), which need the user's decisions, not compute.** The action 1 and 2
sections below are kept as written because the predictions in them were recorded before the work ran and
both were confirmed; each now carries a RESULT block at its top.

Three things changed from the plan below and are worth knowing before reading it:

1. **The requested between/within decomposition of Luna's predictions is degenerate.** The ICL runs use
   1 CoT per question per cut, so within-question SS is exactly zero and every cut's R2 is 100%
   between-question by construction. The published Luna-vs-ridge comparison was therefore never a test
   of within-question skill. The substitute used instead: partial a quadratic in `p_gold` out of both
   the target and each prediction, on all 196 questions.
2. **Action 2 needed no money.** A query prompt does not depend on the difficulty band, and
   `example_order` draws examples from all training-fold questions regardless of band, so subsetting the
   stored predictions IS the band rerun. All five bands were scored, not one, and the 3-question-cell
   calibration worry disappeared (`icl_cal` was fitted per fold on the full other-fold rows).
3. **A published conclusion changed.** On `committed_by_here` the in-context models tie us on accuracy
   and lose by 0.10 to 0.12 on AUC, significant in all nine configurations.

---

## The one-paragraph version

The meeting produced four actions and two concerns, but they are **not six pieces of work**:

- **Actions 1 + 2 ARE the investigation of concern 2.** Stratifying by difficulty and rerunning Luna
  inside one band is exactly how you find out whether the frontier model's zero-shot tie with our probe
  is really both of them reading question difficulty. Concern 2 is the *why*; actions 1-2 are the *how*.
  They are written up together below, under actions 1 and 2. The concern-2 section only holds the one
  residual piece that stratification does **not** settle (the calibration asymmetry).
- **Concern 1 is genuinely separate.** Why examples hurt has nothing to do with difficulty, and it is
  testable for free from predictions already on disk.
- **Actions 3 + 4 are about getting off GPQA** and are independent of the above.

So: one investigation (1+2, answering concern 2), one free diagnostic (concern 1), and one data-collection
track (3+4).

---

# ACTION 1 — stratify GPQA questions by difficulty

> **RESULT (2026-10-29): DONE. `band_quintile` was used** (equal ~39-question sizes, so a between-band
> difference is not confounded with sample size). `band_int` is kept for the interpretable extremes.
> Nothing further is needed here.

**Goal.** Split the 198 questions into difficulty bands, measured by how often the model gets the
question right.

**[groundwork done]** The measurement already exists and needs no new compute. The 0% cut's labels
come from the 25 base CoTs per question, so `p_gold` there is exactly "what fraction of 25 attempts
were correct":

`runs/goal1/gpqa25/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B/labels/checkpoint_labels_f0.csv`
→ columns `example_id`, `p_gold`, `n_continuations` (always 25).

Spread is wide and usable: mean 0.494, median 0.468, and both ends populated (17 questions at 0/25,
28 questions at 25/25).

I wrote the banded table to **`runs/goal2/gpqa25/sweep/_difficulty/question_difficulty.csv`**
(198 rows): `example_id`, `p_gold`, `n_correct` (0-25), `band_int`, `band_quintile`, `fold`.
The `fold` column was produced by `sweep.fixed_fold_of(run_dir, 5, 0)`, so it matches the fixed folds
every other result uses.

Two banding schemes are in that file. Pick one and say which:

| band | `band_int` (integer cutoffs) | `band_quintile` (equal sizes) |
|---|---|---|
| v_hard | 0 correct → 17 q | 0-3 correct → 40 q |
| hard | 1-5 → 49 q | 3-7 → 39 q |
| medium | 6-15 → 56 q | 8-15 → 40 q |
| easy | 16-24 → 48 q | 15-23 → 39 q |
| v_easy | 25 → 28 q | 23-25 → 40 q |

**Recommendation: use `band_quintile`** for any experiment that compares bands, because equal sizes
mean a score difference between bands is not confounded with sample size. Use `band_int` when you want
the interpretable extremes (never right / always right).

**The thing worth knowing before designing anything:** the relationship between difficulty and entropy
is **not monotonic**. Mean entropy by `band_int` over the 10-90% rows:

| band | entropy | rem. length |
|---|---|---|
| v_easy | 0.050 | 1,564 |
| easy | 0.508 | 2,588 |
| medium | 1.061 | 3,336 |
| hard | 1.028 | 3,525 |
| **v_hard** | **0.642** | 2,552 |

Very hard questions have **low** entropy, below medium and hard. The model is confidently wrong on
them. So "harder means more uncertain" is false at the top end, and any analysis that assumes a
monotonic difficulty-entropy relationship will mislead.

**How large is the confound?** In-sample R² of predicting each target from that one scalar `p_gold`
alone (quadratic, to allow the non-monotonic shape):

| cut | entropy | p_mode | rem_len_mean | rem_len_std |
|---|---|---|---|---|
| 10% | 0.586 | 0.531 | 0.194 | 0.084 |
| 30% | 0.461 | 0.405 | 0.177 | 0.068 |
| 50% | 0.358 | 0.314 | 0.149 | 0.064 |
| 70% | 0.262 | 0.230 | 0.105 | 0.052 |
| 90% | 0.186 | 0.152 | 0.068 | 0.047 |

Our probe, for comparison, gets 0.289 / 0.251 / 0.524 / 0.298 (mean over 10-90%).

**Read this carefully, and do not quote it as a baseline.** `p_gold` is computed from rollouts, so it
is *label-derived*, not a feature available at prediction time. These numbers measure **shared
variance between difficulty and the targets**, which is what we want for the confound question. They
are not a legitimate predictor and must never be reported as "a scalar baseline beats our probe".
They are also in-sample, so they are an upper bound. If you want a quotable baseline, you need a
difficulty estimate built **only from the question text** (see action 2's companion below).

What the table does say, and this is the finding: **difficulty and entropy share more variance at
shallow cuts than our probe captures in total, and the overlap decays with depth.** That is the same
story as round 4's between/within decomposition, measured a second independent way.

---

# ACTION 2 — rerun the frontier model inside one difficulty band

> **RESULT (2026-10-29): DONE, $0 spent, and all five bands rather than one.** No API calls were needed
> (see point 2 at the top of this file). **The prediction table below was confirmed on both of its
> rows.** Entropy R2 (all questions / v_hard / hard / medium / easy / v_easy): ours
> +0.309/+0.131/+0.057/-0.143/+0.096/-2.71, Luna k=0 +0.294/+0.008/+0.085/-0.263/+0.093/-1.70, so the
> entropy tie collapses inside every band. rem_len_mean: ours +0.563/+0.582/+0.548/+0.511/+0.484/+0.323,
> flat, and Luna is significantly below us in `medium` (-0.168) and `easy` (-0.225). CI half-widths are
> 0.21 to 0.46 for entropy in a band, exactly as risk 3 below predicted, so no single band is decisive
> alone; the agreement across bands and with the partialling result is what carries it. Two reading
> traps are recorded in the `_iclsplit/p5_*` Results index row: the large negative v_easy entropy R2 is
> mostly an uncorrected prediction offset, and the offset-free within-cut correlation is the companion
> view. Files: `_iclsplit/p5_bands_{scores,paired}.csv`, script `src/analysis/icl_bands.py`.
>
> **The cheap companion at the bottom of this section was the main event and is also done**, via the
> partialling substitute rather than the between/within split. See `docs/update_2026-10-29.md`.

**Goal.** Find out whether GPT-6 Luna's zero-shot tie with our probe survives once question difficulty
is held roughly constant. If both scores collapse inside a band, difficulty was carrying both, and the
"frontier model matches our probe" result is really "both read difficulty".

**This is the test for concern 2.** Do not treat that concern as separate work; it is the reason this
action exists.

**The predictions, so the result is falsifiable before you run it.** Write down which you expect:

| | if difficulty was carrying the tie | if it was not |
|---|---|---|
| our probe's entropy R2 inside the band | **collapses** toward 0 | stays near 0.29 |
| Luna's entropy R2 inside the band | **collapses** too | stays near 0.29 |
| the gap between them on entropy | stays small (both near 0) | stays small (both still high) |
| our advantage on remaining length | **survives** | survives |

That last row is the sharp part, and it comes out of the action-1 groundwork rather than from
hand-waving. Difficulty shares far less variance with the length targets than with entropy
(rem_len_mean 0.194 falling to 0.068, against entropy's 0.586 falling to 0.186). So if the story is
right, **stratifying should kill the entropy tie and leave our length advantage intact.** If instead
Luna still ties us on entropy inside a single band, difficulty was *not* the explanation and concern 2
is wide open again, which would be the more interesting outcome.

**Model: GPT-6 Luna** (`gpt-6-luna`, reasoning high), per the user. Not Astra: the Astra credit ran
out and the spend never reconciled (see the Results index).

**How to run it.** `src/baselines/icl_openai.py` already does everything. Prior full run:
`icl_openai.py run --model gpt-6-luna --full`, from the login node, key in `~/.frontier_api_key`.
Costs for calibration: the full 1,178-row run of **all 8 configurations** was **$17.86 / 9,424 calls**,
so a single configuration on ~40 questions is a couple of dollars at most.

**Design problems to settle before spending anything.**

1. **Which configuration?** The cheapest informative one is `ckpt` at **k=0**, because k=0 was the best
   configuration in the full run (entropy 0.294 vs our 0.309) and it is the one the concern is about.
   Running k=50 as well would cost little and tells you whether "examples hurt" also holds within a
   band.
2. **Which band?** `medium` is the natural first choice: it has the highest entropy (1.061) so there is
   the most variance to predict, and it avoids the degenerate ends. If you only run one band, run
   medium. If you run two, add `v_easy`, where entropy is 0.050 and both predictors should collapse to
   nothing. That contrast is the actual test.
3. **n is small, and this is the real risk.** A quintile band is ~40 questions. The ICL subsample uses
   1 chain per question × 6 cuts, so one band is roughly **40 rows per cut, 240 rows total**. The full
   1,178-row run already had CIs of about ±0.15; at 40 rows per cut expect roughly **±0.3 or worse**,
   which may be too wide to distinguish 0.29 from 0.00.
4. **Calibration will be fragile.** The ICL scorer fits `actual = a + b × predicted` on the other four
   folds. Within one band the smallest (band, fold) cell has **3 questions** (see the crosstab in
   `question_difficulty.csv` groundwork). Fitting a 2-parameter line on 4 folds of a ~40-question band
   is possible but noisy. **Options: pool to 2 or 3 folds for this experiment, or report raw and
   calibrated side by side, and say which you did.** Do not silently keep 5 folds.
5. **Score our probe on exactly the same rows.** The comparison is only meaningful paired. Reuse
   `src/viz/icl_table.py`, which already scores the probe on an arbitrary ICL row set by the
   master-table method.

**Cheap companion worth doing FIRST, before any API spend.** Luna's predictions for all 1,178 rows are
already on disk (`runs/goal2/gpqa25/icl/preds_gpt6_luna_high*.csv`). Apply the round-4
between-question / within-question decomposition (`src/analysis/redundancy.py` has the method) to
**Luna's predictions** rather than to the probe's. If Luna's R² is also almost entirely
between-question, that answers concern 2 with **zero new compute and zero dollars**, on the full row
set rather than a 40-question band. Do this first. It may make action 2 unnecessary, or it may tell you
exactly which band to spend on.

---

# ACTION 3 — find a subjective dataset

**Goal.** A dataset whose questions have **no single objective answer**, so "entropy over final
answers" measures genuine spread of opinion rather than correctness.

**Nothing has been done on this.** It is a research-and-propose task, and it needs the user's sign-off
before any collection. Open questions the next session should answer with a written proposal:

1. **What becomes the label?** Our five targets assume a discrete answer per rollout so we can compute
   entropy, p_mode and commitment. With no objective answer, `p_gold` is undefined. Candidates: entropy
   over a model-extracted stance or category; spread of a numeric rating the model is asked to give;
   embedding-space dispersion of the rollouts. **The remaining-length targets still work unchanged**
   and may be the cleanest starting point.
2. **How is an answer parsed?** The whole pipeline depends on `run_eval.parse_output` pulling a
   `{"Final Answer": ...}` JSON and normalising it. A subjective dataset needs a defined answer space
   or this breaks silently. Do not reimplement the parser; extend it.
3. **Does commitment still mean anything?** `committed_by_here` is defined via concentration on one
   answer. With a continuous or open answer space that definition needs replacing.
4. **Candidate datasets to research:** moral-dilemma and opinion sets, argument-quality corpora,
   creative-writing or open-ended QA with human preference labels, and forecasting questions (which
   have the nice property of being subjective *now* and objective later). Report licences and sizes.

This is the action most likely to change what the paper is about, so write the proposal before
spending GPU time.

---

# ACTION 4 — get Scott to run two new datasets: MMLU-Pro and a subjective dataset

**Partly prepared already, by another session.** `docs/GUIDE_new_datasets.md` (committed `c8c301c`,
amended `043218f`) is a standalone guide written for an agent on a different server with no context.
It covers environment setup, the K-sufficiency test, dataset/subset selection and activation
extraction.

**Two gaps to close before handing it over:**

1. **The guide targets `math500` and `mmlu_pro`.** The user now wants **MMLU-Pro and a subjective
   dataset**. MMLU-Pro is ready; the subjective half does not exist yet and depends on action 3. Either
   send the guide for MMLU-Pro now and follow up, or wait for action 3. **Ask the user which.**
2. **The guide asks the recipient to confirm decisions with "the user".** If Scott is running it, say
   explicitly who decides what: questions per dataset, CoTs per question, pilot K. The guide's own
   section 2 lists them.

One decision is already recorded, so do not reopen it: **noisy 0% labels are accepted** (the 0% cut's
K equals `N_COTS`, so with 2-5 CoTs the 0% labels rest on 2-5 samples; commit `043218f`).

Relevant prior finding to pass on: round 4 showed that at a fixed number of rows, **many questions with
2-5 CoTs each beats few questions with 25** (150q × 2 ≈ 30q × 25, for 2.5× fewer rollouts). That argues
for breadth in the new collections.

---

# CONCERN 1 — zero-shot beats in-context for frontier models

> **ANSWERED 2026-10-29, free. Examples supply the output SCALE and degrade the RANKING.** Within-cut
> correlation falls versus k=0 in **28 of 28** comparisons across two model families (mean -0.131, worst
> -0.335), while the prediction spread moves closer to the truth's spread in 19 of 28 and raw
> uncalibrated R2 improves in 15 of 28. Because the published comparison calibrates, the scale gain is
> handed over for free and only the ranking damage counts, so k=0 wins.
> On the hypotheses below: **H1 anchoring in the shrink-to-the-example-mean sense is NOT happening**
> (spread widens, distance to the example mean stays 0.6 to 1.0 target sd); **verbatim copying is real
> but modest and entropy-only** (+0.08 to +0.29 over a chance baseline, about zero on both length
> targets, and the chance baseline is essential because p_mode is shown to two decimals); **H2 (noisy
> example labels) is untestable with these runs**, because `example_order` depends only on (fold, k) so
> every row in a fold saw an identical example set. Files: `_iclwhy/{anchoring,vs_zero_shot}.csv`,
> script `src/analysis/icl_anchoring.py`.
> **Verdict for ask 4: GEPA example selection is not worth running. Instruction optimisation is
> untouched by this and remains open.**

**What we actually observed** (not speculation; `runs/goal2/gpqa25/sweep/_icl/ci_scores.csv`):

| | entropy |
|---|---|
| Luna k=0 | 0.294 |
| Luna k=50 | 0.255 |
| Luna k=10 | 0.216 |
| Gemma k=0 | 0.262 |
| Gemma k=93 | 0.158 |

k=0 is best for **both** models, and they are unrelated families, so it is not a quirk of one model.
Shortening the examples (ask 1) helped slightly and did not change the ordering. More examples at k=400
were the worst of all.

**Hypotheses, cheapest first. The first three need no new API calls** — the raw predictions are all on
disk in `runs/goal2/gpqa25/icl/raw/` and `preds_*.csv`.

1. **Anchoring on the example values.** We already *saw* this failure mode with our own model: at k=10
   it often copied an example's five numbers verbatim, and at k=40 it repeated one number in every
   field. Test: does the spread of the model's predictions shrink as k grows, and do predictions cluster
   toward the mean of the examples shown? If yes, the examples are acting as an anchor rather than as
   training data.
2. **The example labels are noisy.** Each example's label is a mean over 25 rollouts, so it carries real
   sampling error. Few-shot learning from noisy targets can be worse than using a good prior. Test:
   correlate per-row error against the noise level of the examples that row saw.
3. **Attention dilution.** k=50 at full prefix length is a very long prompt. But note k=10 is *worse*
   than k=50, which dilution alone does not explain, and shortening examples by 91% barely helped. So
   dilution is probably not the main cause. Worth stating as ruled-down.
4. **The prior is simply good and the examples are redundant.** At k=0 the model reasons about the
   question from scratch; the examples may add nothing it did not already infer.

Hypotheses 1 and 2 are the promising ones, and both are free to test.

**Why it matters beyond curiosity:** it decides whether ask 4 (GEPA prompt optimisation) is worth
anything. If examples actively hurt, example selection optimises toward k=0 and only instruction
optimisation is left. That is already noted in `docs/meeting_2026-10-27.md`.

---

# CONCERN 2 — the frontier model ties our probe on entropy, zero-shot

> **SETTLED 2026-10-29: the tie IS question difficulty, and the length advantage is not.** Partial a
> quadratic in `p_gold` out of both target and prediction, within each cut, and entropy R2 falls
> 0.309 -> 0.092 [-0.010, 0.180] for our probe and 0.294 -> 0.037 [-0.072, 0.140] for Luna k=0. Neither
> is significantly above zero afterwards; TF-IDF goes 0.267 -> 0.066 and only prefix length stays
> significant (0.247 -> 0.121). Remaining length barely moves: ours 0.563 -> 0.528 [0.469, 0.587], Luna
> 0.455 -> 0.397, and our edge over Luna (+0.131 [0.070, 0.195]) and over TF-IDF (+0.144 [0.092, 0.194])
> survives partialling. Band stratification gives the same answer independently. **Caveat that must
> travel with these numbers: `p_gold` is label-derived, so partialling over-removes and the residual
> figures are a lower bound.** The residual below about calibration still stands and is the one piece
> none of this settles.

> **This is answered by actions 1 and 2, not by separate work.** This section exists to record why the
> result is surprising, what the likely explanation is, and the one piece stratification does not settle.

**What we observed.** Luna k=0, calibrated, mean over 10-90%: entropy 0.294 against our probe's 0.309,
difference −0.015, not significant. Same for p_mode and commitment. It loses badly on both length
targets (0.455 vs 0.563 on rem_len_mean; 0.122 vs 0.328 on rem_len_std).

**Why it is surprising.** The model sees only the prompt text. It has no activations, and at k=0 no
labelled examples at all, so it cannot even know the scale of the answers.

**The likely explanation is already in our own results, and it is uncomfortable.** Round 4 found the
probe is mostly a **question-difficulty reader** (pooled R² is almost entirely between-question;
within-question R² is near zero). Question difficulty is plainly readable from the question text by any
competent model. The difficulty numbers in action 1 support this independently: a single scalar per
question shares 0.586 of entropy variance at the 10% cut.

So the honest reading is: **the frontier model ties us on entropy because entropy at the question level
is mostly difficulty, and difficulty is legible from the text.** It loses on length because remaining
length depends on where *this particular chain* is, which the text of a prefix conveys less well and
which our activations genuinely do capture better.

**That interpretation is testable by actions 1 and 2**, which is why the user paired them with this
concern. The two tests, in cost order, are both written up above:

1. **Free, no API calls:** decompose Luna's existing predictions between/within question (action 2's
   companion). Prediction: Luna's R² is even more purely between-question than ours.
2. **Within one difficulty band:** action 2 proper, with the prediction table given there.

If both hold, this stops being a concern and becomes a **result**: direct evidence for the
characterisation the paper should adopt, arrived at by a completely different method than round 4's
variance decomposition. Two independent routes to the same conclusion is much stronger than one.

**The one residual that stratification does NOT settle.** The calibration step hands the in-context
model the scale for free
(it fits `actual = a + b × predicted` on held-out folds). Raw, uncalibrated, Luna is far below zero.
That is defensible — at k=0 it was never told the scale, so scoring it on scale would penalise it for
information we withheld — but any write-up must say that the ICL numbers are calibrated and our probe's
are not, because the probe never needed calibrating.

---

# Where to actually start (written 2026-11-01)

Everything above is the record of the 2026-10-27 meeting and what came of it. If you are a new session,
this is the short version.

**Done, do not redo:** actions 1 and 2, and both concerns. All of it cost zero compute and zero dollars
and is written up in `docs/update_2026-10-29.md`, with figures in `docs/figures/slides/fig{9,10,11,12}_*.png`
and numbers in `runs/goal2/gpqa25/sweep/_iclsplit/` and `_iclwhy/`.

**The one-line result:** the frontier model's zero-shot tie with our probe on answer entropy was question
difficulty. Remove difficulty and entropy R2 falls 0.31 to 0.09 for the probe and 0.29 to 0.04 for Luna,
neither above zero. Remaining length barely moves (0.56 to 0.53) and stays significantly ahead of every
in-context model. Said the other way: inside a difficulty band the probe explains about 11% of what is
left on entropy and about 48% on remaining length.

**Nothing is running.** `squeue` was empty on 2026-11-01.

**The next three things, in order:**

1. **The 0% checkpoint probes.** Still the highest-value single experiment, and round 6 made it more
   valuable, not less: at 0% there is one row per question, so the probe can only read question
   difficulty and its R2 becomes a direct measurement of the floor that round 6 inferred two indirect
   ways. Data has been ready since 2026-10-04. Needs about 30 lines at `src/probing/sweep.py:88`, which
   hardcodes `labels/checkpoint_labels.csv`, plus a config. A few CPU-hours. Report R2 and Spearman.
2. **The question-text-only baseline.** The legitimate version of the `p_gold` diagnostic, which is
   label-derived and must never be quoted as a baseline. If TF-IDF on the question alone recovers most
   of the entropy R2, the question-level story is closed.
3. **Actions 3 and 4 below (getting off GPQA).** These need the user's decisions, not compute. Note a
   different session has already added a GlobalOpinionQA option and normalised entropy (commit
   `fe985b1`), which is groundwork for the subjective dataset, so check with them first.

**Also sitting ready and blocked:** GEPA prep (round 5 ask 4) was built by another session on 2026-10-31
as user `rishp`: `src/baselines/gepa_luna.py`, `envs/dspy`, `runs/gepa/run_luna.sh`, and `prep` has run.
It is **blocked because that account's API key has no credit**; the 2026-10-26 Luna runs went through the
`yuvraj` key. It optimises the zero-shot instruction, which is the right call given round 6 found that
examples hurt. **Ask the user before spending anything on it.**

---

# Suggested order of work

1. ~~The between/within decomposition of Luna's existing predictions.~~ **DONE 2026-10-29** (via the
   difficulty-partialling substitute, because the literal decomposition is degenerate).
2. ~~The anchoring and noisy-label checks (concern 1).~~ **DONE 2026-10-29.**
3. ~~Decide the banding and run action 2.~~ **DONE 2026-10-29, for $0 and on all five bands.**
4. **The 0% checkpoint probes.** Now the highest-value remaining experiment, and more valuable than it
   was: one row per question means the probe can *only* read question difficulty, so its R2 is a direct
   **measurement** of the floor that round 6 has been inferring two indirect ways. Data ready since
   2026-10-04; needs about 30 lines at `src/probing/sweep.py:88`. Report R2 **and** Spearman.
5. **The question-text-only baseline.** The legitimate version of the `p_gold` diagnostic, which is
   label-derived and therefore not quotable as a baseline. If TF-IDF on the question alone recovers most
   of our entropy R2, the question-level story is closed beyond argument.
6. **Write the action-3 proposal** on subjective datasets and get the user's sign-off. (A different
   session was working on GlobalOpinionQA as of 2026-10-29; check with it before duplicating.)
7. **Hand `docs/GUIDE_new_datasets.md` to Scott** for MMLU-Pro, once the user says whether to wait for
   the subjective dataset.

Still open from before the meeting, and not superseded by any of this: the **0% checkpoint probes**
(needs a ~30-line change at `src/probing/sweep.py:88`, which hardcodes `labels/checkpoint_labels.csv`)
and the **question-text-only difficulty baseline**. Both are described in `CLAUDE.md` → "Next steps",
and both bear directly on concern 2. The question-text-only baseline is the legitimate version of the
`p_gold` diagnostic in action 1, so it is now more valuable than it was.

---

# Decisions waiting on the user (not work items)

None of these are blocking the actions above, but they are the things a new session cannot settle alone.

1. **Nothing is committed.** Last push `c8c301c` (2026-10-29, by another session). Uncommitted here:
   `CLAUDE.md`, `NEXT_STEPS.md`, `learnings.md`, `RESEARCH.md`, all of `docs/update_2026-10-2[37].md`,
   `docs/briefing_2026-10-27.md`, `docs/meeting_2026-10-27.md`, `docs/deck_2026-10-27/`,
   `docs/plan_*.md`, `docs/figures/{redundancy,mlpgap,slides}/`, `configs/sweep_ref_logistic_deep.yaml`,
   `src/analysis/redundancy*.py`, `src/probing/mlpgap_*.py`,
   `src/viz/{master_table,icl_table,slide_figures}.py`, and edits to `src/baselines/icl_{openai,predict}.py`.
   `git fetch` and check `git log HEAD..origin/main` first - another session and yuvraj both push here.
2. **Unverified security item.** The frontier API key was at one point written to `temp.txt` in this
   group-shared repo at mode 644. The file is now 0 bytes, but **rotation was recommended and never
   confirmed.** Worth a direct question. The key lives in `~/.frontier_api_key` (mode 600) and
   `src/baselines/icl_openai.py` reads it from there.
3. **Both artifacts are still PRIVATE** - the deck
   (https://claude.ai/artifact/JLTVCFvt2Go7XeU7qE9Dso) and the dashboard
   (https://claude.ai/artifact/NCW8FuLt4GLE9gxTcCk46V). Sharing is done from each page's Share menu,
   not from here. The deck's sources are now in `docs/deck_2026-10-27/project/`; **the dashboard has no
   local copy of its page** (it was edited in place on the artifact), so recover it with
   `Artifact action:"read"` on its URL if it ever needs changing.
4. **Repo-root junk**, deletable on the user's say: `temp.txt` (empty) and two `tmux-*.log` files
   (~19 MB, owned by `rishp`).

# Conventions that apply to everything above

- Score **within each cut point, then average. Never pool across depth.**
- **Always report both answer entropy and remaining length.** They diverge constantly: in-context
  models tie us on entropy and lose on length.
- For `committed_by_here`, report **AUC alongside accuracy**, with the majority-class baseline (0.6275
  mean over 10-90%).
- Units: R² for entropy, p_mode, rem_len_mean, rem_len_std; **accuracy** for committed.
- Reuse the 5 fixed folds by question so numbers stay comparable to `sweep/ref_fixed/`.
- Write each experiment to its **own** folder under `runs/goal2/gpqa25/sweep/`. Never write into
  `probes_ckpt/` or `ablation/` (those are yuvraj's).
- No em dashes in slides or docs; plain bullet-only slides.
