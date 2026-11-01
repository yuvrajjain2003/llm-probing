# Idea 3 spec — what does the existing literature actually tell us?

_Owner: agent 3. Written 2026-10-22 by the orchestrating session. Output folder:
`runs/goal2/gpqa25/sweep/_litreview/`. Deliverables are CSV + markdown there. This agent writes
NO code and runs NO jobs._

## The goal, stated as the user stated it

Find out **what people have already done** in this area and judge whether it is useful for us,
against our actual hypothesis. The user was explicit on two points:

1. **This is NOT a port-these-methods task.** Re-implementing someone's method is one possible
   outcome, not the objective. Do not produce a porting plan as the main deliverable.
2. **An isolated literature summary is a FAILURE.** The user's words: "I'm scared about just an
   isolated summary which doesn't really connect to our work. I want it intertwined." Every finding
   must be stated as a consequence for *our* project, or explicitly marked as having none.

## Our hypothesis, which every judgement is made against

Do internal representations at a chain-of-thought prefix carry information about the
**distribution over future continuations**, beyond what is available from the prefix text itself?

Note what is distinctive: our targets are properties of a *distribution* over 25 independent
continuations from the same prefix (answer entropy, p_mode, mean and std of remaining length,
commitment), not properties of one sampled trajectory.

## Our setup and numbers (cite these files when you reference our side)

- Model `DeepSeek-R1-Distill-Qwen-7B`. Data GPQA-Diamond, 198 questions x 25 CoTs, cut at
  10/30/50/70/90% by sentence count, plus a 0% cut (prompt before any reasoning).
- Labels: from each cut the model was continued **25 times**; targets are
  `answer_entropy_bits_mm` (Miller-Madow), `p_mode`, `rem_len_mean`, `rem_len_std` (all R²) and
  `committed_by_here` (accuracy). `rem_len_*` count tokens before `</think>`, **raw, not logged.**
- Features: layer-21 hidden state (3,584 dims) at the **last token of the prefix**. All 28 layers
  are stored. Layer 21 is **fixed, not selected** on performance.
- Probe: ridge, alpha from `logspace(0,6,13)`, chosen by inner CV (lands on 1e4-3.16e4);
  logistic for commitment. 5 fixed folds **split by question**, so no question appears in both
  train and test.
- Headline (mean over 10-90%), from `runs/goal2/gpqa25/sweep/ref_fixed/results.csv`:
  entropy 0.289, p_mode 0.251, rem_len_mean 0.524, rem_len_std 0.298, committed acc 0.741.
  Per checkpoint, rem_len_mean: 0.626 / 0.587 / 0.548 / 0.484 / 0.375 at 10/30/50/70/90%.
- Error bars: question bootstrap, 2,000 draws; between-feature CI half-width about +/-0.05 R².
- Baselines we already beat: TF-IDF on the prefix (entropy +0.052 [+0.021,+0.083]), prefix length,
  and several sentence embedders. Source `runs/goal2/gpqa25/sweep/_round3/ci_paired.csv`.

### Our four known soft spots — a paper that speaks to any of these is high value

1. **Prefix-length confound.** Prefix length *alone* gets rem_len_mean 0.652 at the 10% checkpoint,
   beating our activations (0.626). Our length result is substantially token-counting at early
   checkpoints; we only win at 70-90%.
2. **The other-model tie.** OLMo-3-7B's layer-24 states on the same text match ours on all five
   targets (all n.s., |diff| <= 0.008). So the signal is "a competent 7B transformer read this
   text", not something specific to a reasoning-trained model.
3. **R² is flat across depth** (entropy 0.278 -> 0.302 -> 0.270). If the probe were reading an
   evolving chain-specific state you would expect it to sharpen as the chain commits.
4. **Raw vs log targets** — see below. This may be the single biggest comparability issue.

## The papers (from the user; treat as a hypothesis to verify, not as fact)

1. **DyCon: Dynamic Reasoning Control via Evolving Difficulty Modeling** (claimed ICML 2026) —
   600 MATH problems, 4 models 4B-32B; hidden state at each `\n\n` reasoning-step boundary from a
   *selected* layer; ridge predicts remaining tokens. **Claimed R² 0.642-0.802 on log-transformed,
   normalized remaining length.**
2. **On Reasoning Strength Planning in Large Reasoning Models** (claimed NeurIPS 2025) — MATH,
   90/10 split, R1-Distill 1.5B-32B and QwQ-32B; activation at the opening `<think>` position,
   one layer at a time; Lasso predicts total reasoning tokens. **Reports correlation ~0.84, not R².**
3. **How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in CoT Reasoning** (claimed ICML
   2026) — 12 tasks incl. GPQA; "Tele-Lens": per-layer states through a low-rank GeLU adapter plus
   a regression head, predicting total thinking length. **No numeric R² claimed.**
4. **Steering LLM Thinking with Budget Guidance** (claimed Findings ACL 2026) — OpenR1-Math-220k;
   concatenates last-token states from **all** layers; a BERT-base predictor outputs Gamma
   parameters for remaining length. **Likelihood objective, no R².**
5. **SpecExit: Accelerating Large Reasoning Model via Speculative Exit** (2025) — linear auxiliary
   heads on the last accepted token predict remaining length, progress, confidence; MSLE loss.
   **No R²; and its target is an early-exit point, not the trajectory's natural end.**
6. **Emergent Response Planning in LLMs** (2025) — GSM8K/MATH plus UltraChat/AlpacaEval, 8 LLMs;
   representation at the end of the prompt; one-hidden-layer ReLU MLP predicts **number of
   reasoning steps** (not token count). **Reports Pearson/Spearman/Kendall only.**

**Verify each paper exists with that title and venue, and that each claimed number is actually in
it.** Several postdate the orchestrator's knowledge cutoff and the table may itself have been
assembled by a model, so citations or figures may not survive checking. **"I could not find this
number in the paper" is a valuable finding — report it plainly rather than producing a tidy table.**
Prefer the arXiv PDF/HTML over secondary sources. Record the arXiv id and version you read.

## Why most of these numbers are NOT comparable to ours

Establish this properly rather than assuming it; it is the core analytical content.

- **Log transform.** DyCon's 0.642-0.802 is on log-transformed, normalized remaining length; ours
  is on raw token counts. Length distributions are heavy-tailed to the right and the tail holds
  most of the variance, so logging compresses exactly the hard part and systematically flatters
  R². We may already match or beat DyCon. Find out exactly what transform and what normalization.
- **Single trajectory vs distribution.** Papers 1, 2, 3, 6 predict properties of **one sampled
  trajectory**. Our `rem_len_mean` is the mean over 25 continuations — an expectation with the
  sampling noise already averaged out, which makes it an intrinsically *easier* target. Both
  effects must be separated before any comparison means anything.
- **Correlation is not R².** r = 0.84 implies r^2 ~ 0.71, but out-of-sample R² is <= r^2 and can be
  far lower under miscalibration. Rows reporting only correlation cannot be placed in an R² column.
- **Layer selection.** "A selected layer" or "one layer at a time" reporting the best is optimistic
  if selected on test. We fix layer 21. Note which papers do which.
- **Split granularity.** A split by *trajectory* rather than by *problem* leaks, because a problem's
  trajectories share text. Check each paper. Any paper that leaks has an inflated number relative
  to ours, and that is worth saying.
- **Effective sample size.** DyCon takes a state at *every* `\n\n` across 600 problems: many highly
  correlated rows per trajectory pooled into one R². That is the same critique idea 1 is applying
  to our own data, so apply it to theirs.

## Deliverable 1 — the verified comparison table (`papers.csv`)

One row per paper, fixed schema so rows are mergeable:

`paper` · `venue_claimed` · `venue_verified` · `arxiv_id_version` · `models` · `dataset` ·
`n_problems` · `n_trajectories` · `target_exact` (single trajectory vs distribution; total vs
remaining; tokens vs steps; raw vs log vs normalized) · `feature` (layer, token position, pooling) ·
`layer_selection_method` · `predictor` · `metric_reported` (R² / r / rho) · `in_sample_or_heldout` ·
`split_granularity` (problem / trajectory / unclear) · `number_as_reported` ·
`comparable_to_ours` (yes / no / only-after-X) · `what_would_make_it_comparable` ·
`relevance_to_our_hypothesis` · `action`

**Every row's `action` must be one of:** an experiment we could run, a sentence we could put in
the paper, or an explicit "no implication for us". **Delete any row that ends in none of those.**

## Deliverable 2 — answers to OUR questions (`findings.md`)

This is the part that makes it intertwined, and it matters more than the table. Answer each
explicitly, with citations, and say plainly when the answer is "nobody does this":

1. Does **anyone** predict a *distributional* property — entropy over sampled answers, agreement
   across samples, p_mode — rather than one trajectory's length? If nobody does, that is our
   novelty and should be stated as such.
2. Does anyone run an **other-model control** (features from a model that did not generate the
   text)? If nobody does, our OLMo tie is a *contribution*, not an embarrassment — it would be the
   first evidence that this signal is not reasoning-model-specific.
3. Does anyone report a **prefix-length or trivial-scalar baseline**? If not, that is a hole in
   the literature we are positioned to occupy — and it also means their reported R² may carry the
   same token-counting confound as ours, unexamined.
4. How does each paper **split train/test**, and does anyone leak?
5. Does anyone address **effective sample size** or within-problem variance?
6. In this literature, what counts as a **good** R² — separately for raw and for log targets? This
   is the calibration we most need: is our 0.524 strong or weak?
7. Does anyone predict at the **pre-generation position** (prompt end / `<think>`), and what do
   they get? We have that data ready and unprobed.
8. Is anyone's reported result **flat across reasoning depth**, like ours is?

## Deliverable 3 — relevance triage (`triage.md`)

For each paper, one short paragraph: what it would give us, in which of these shapes —
**calibration** (tells us whether our numbers are good), **positioning** (tells us what is novel),
**convention** (how this literature reports things, e.g. log targets, so our numbers are readable
by its readers), **a gap we can occupy**, **a method worth trying** (real but the weakest and
least likely), or **nothing** (stated outright, with the reason).

## Rules

- **Write only** into `runs/goal2/gpqa25/sweep/_litreview/`. No code, no jobs, no repo edits.
- **Do not touch `CLAUDE.md`.**
- Per-paper prose is **secondary and short** — an appendix at most. The table and `findings.md`
  are the deliverables.
- Any claim about *our* results must name the file it came from (`sweep/ref_fixed/results.csv`,
  `sweep/_round3/ci_paired.csv`). This forces you to open our CSVs rather than paraphrase this
  spec — and it will catch the spec if it is wrong.
- **Default to "not comparable" unless you can show otherwise.**
- Distinguish throughout: what the paper *claims*, what you *verified*, and what you *inferred*.
- Stuck or a paper is inaccessible? Note it in `_litreview/QUESTIONS.md` and keep going.
