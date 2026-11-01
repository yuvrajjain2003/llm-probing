# RESEARCH.md — related work, and how our numbers compare

_Literature review for this project. Written 2026-10-26 from the round-4 review (2026-10-22/23).
Primary artefacts: `runs/goal2/gpqa25/sweep/_litreview/{papers.csv,findings.md,triage.md,QUESTIONS.md}`.
Spec that produced it: `docs/plan_literature.md`. Everything below was verified against the papers
themselves and, for two items, against the authors' released code; what could not be verified is
marked as such._

## Our setup, for reference

- Model `DeepSeek-R1-Distill-Qwen-7B`. Data GPQA-Diamond, **198 questions x 25 chains**, cut at
  10/30/50/70/90% by sentence count, plus a 0% cut (prompt before any reasoning, not yet probed).
- Features: layer-21 hidden state (3,584 dims) at the **last token of the prefix**. Layer 21 is
  **fixed, not selected**. All 28 layers are stored.
- Targets: `answer_entropy_bits_mm`, `p_mode`, `rem_len_mean`, `rem_len_std` (R²) and
  `committed_by_here` (accuracy). `rem_len_*` count tokens before `</think>`, **raw, not logged**.
- Probe: ridge, alpha from `logspace(0,6,13)` by inner CV (lands on 1e4-3.16e4); 5 fixed folds
  **split by question**. Error bars: question bootstrap, 2,000 draws.
- Headline (mean over 10-90%, `sweep/ref_fixed/results.csv`): entropy 0.289, p_mode 0.251,
  rem_len_mean 0.524, rem_len_std 0.298, committed acc 0.741.

## The seven papers

Full schema (20 columns per paper) in `_litreview/papers.csv`. **Nothing in the list was fabricated:**
all exist with substantially the claimed titles.

| paper | venue | what it predicts | metric reported | comparable to us? |
|---|---|---|---|---|
| **DyCon: Dynamic Reasoning Control via Evolving Difficulty Modeling** (arXiv 2606.07108v2) | ICML 2026 (verified, poster 61464) | remaining reasoning tokens, single chain, log1p+min-max | R² | **only one that is**, see below |
| **On Reasoning Strength Planning in LRMs** (2506.08390v2) | NeurIPS 2025 (verified, poster 118916) | total reasoning tokens, **mean over 8 rollouts**, raw | **Spearman** (confirmed from code) | no, rank correlation bounds nothing about R² |
| **How Far Ahead Do LLMs Plan?** (Tele-Lens, 2602.02103v2) | ICML 2026 (verified, poster 61496) | total thinking length | none numeric | no number exists |
| **Steering LLM Thinking with Budget Guidance** | Findings ACL 2026 (verified) | remaining length, as Gamma parameters | likelihood only | no |
| **SpecExit: Speculative Exit** | 2025 | earliest correct exit point | none | no, and not fixable: different physical quantity |
| **Emergent Response Planning in LLMs** (2502.06258v3) | ICML 2025 | number of reasoning **steps** | Pearson/Spearman/Kendall, **figures only** | no, no extractable value |
| **Latent Planning Emerges with Scale** | ICLR 2026 | (found while verifying; critique paper) | — | no, different kind of paper |

## DyCon in detail: the only directly comparable number

**Their setup.** 600 instances from the **MATH** train split, four reasoning models from 4B to 32B,
one of which is our exact model. They take the hidden state at every `\n\n` paragraph break (a
reasoning-step boundary), select a single layer on a validation split (layer 27 of 29 tried for our
model, refit on train+val, reported on test), and fit ridge regression to predict remaining
reasoning tokens.

**Table 8 publishes three scales for the same fits.** For R1-Distill-Qwen-7B:

| scale | R² |
|---|---|
| log1p + min-max | **0.642** (their headline) |
| min-max only | 0.522 |
| **raw** | **0.501** |

So the **log transform is worth +0.14, measured in their own table** rather than argued from theory.
Min-max is affine and cannot change R² at all (`y_min 0.0`, `y_max 9.7029` in their metadata), so
"log-transformed, normalized" overstates it: only the log matters. **The comparable number for our
model is 0.501 against our 0.524.**

**Their split leaks.** From `train_remain_length_reg_auto_layer.py` L712-720, `train_test_split` runs
over `idx_all = np.arange(N)` where `N = len(events)` and an event is one step boundary. There is
**no `groups=` argument and no grouped splitter anywhere**. Shipped metadata confirms event counts:
`N_events 83342`, split 50004 / 16669 / 16669. That is ~139 step boundaries per trajectory from 600
single-chain trajectories, scattered randomly across train, val and test, so near-identical prefixes
from the same chain sit on both sides. **Their 0.501 is therefore an upper bound**, and the comparison
moves in our favour (our folds keep all 25 chains of a question together). The +0.14 log measurement
is unaffected, since raw and log rows share the same split.

**Effective sample size: ~600 trajectories against 83,342 nominal rows**, a ~139x gap with no
clustering correction. This is exactly the critique we applied to ourselves in round 4, and corrected
for; they did not.

**They report no baseline for this regression.** No step index, no prefix length, no token count.
Their ablations are control-policy ablations, not R² comparisons. **Why:** DyCon's regression is
*instrumental*, not the object of study. The paper is about a reasoning-control policy that decides
how much to think, and the length predictor is a component feeding it. They needed it to work well
enough to steer with. We are studying the predictor itself, so the bar is different. Worth saying if
anyone asks how an ICML paper has no baseline: they were not making the claim we are making.

**Provenance wrinkle, not tidied away.** Their metadata records `val r2 0.6424`,
`test-during-search 0.6442`, `test-after-refit 0.6506`. Table 8's entry is (R² 0.642, MAE 0.063): the
R² matches the **validation** value while the MAE matches the **post-refit test** value. So the
published pair is not exactly reproducible from the repo and may be a different run or seed. All
candidates sit in 0.642-0.651, so the magnitude is confirmed; the provenance is not.

### Method: same family, different design

| | DyCon | us |
|---|---|---|
| dataset | MATH, 600 problems (free-form answers) | GPQA-Diamond, 198 questions (4-way multiple choice) |
| where they cut | every paragraph break, ~139 per chain | 5 fixed fractions, 10-90% |
| chains per problem | 1 | 25 |
| layer | selected on validation (27 of 29) | fixed at 21 |
| target scale | log1p then min-max | raw tokens |
| split | random at step-event level, no grouping (**leaks**) | by question |
| targets | remaining length only | five, including answer entropy and spread |
| role of the regression | component of a control policy | the object of study |

The core method is the same: read the generating model's residual stream at a point in the chain,
one layer, ridge regression, out-of-sample R² on remaining tokens.

**On the dataset difference.** MATH is free-form (answer graded by equivalence); GPQA-Diamond is
4-way multiple choice. This does **not** affect the remaining-length comparison, which is a token
count and format independent. It **does** affect our other targets: answer entropy and p_mode are
cheap to measure only because our answer space is four options (entropy bounded at 2 bits, modes
trivial to count). On MATH you would first need an answer-equivalence step. That is probably part of
why nobody predicts answer-distribution properties, so our novelty there is partly enabled by the
dataset choice rather than being purely a better idea. It also means "ours is harder data" is a claim
about GPQA-Diamond being PhD-level science, not a controlled comparison. **DyCon is the closest
published setup, not an apples-to-apples one, and should not be oversold.**

## What nobody in this literature does

- **No other-model control.** 0 of 6. All attribute their results to reasoning-specific internal
  planning. **We ran it: OLMo-3-7B matches our model on all five targets** (round-3 CIs), so the
  signal is not specific to a reasoning-trained model. Our OLMo tie is a contribution, not an
  embarrassment. It is the empirical form of the critique in *Latent Planning Emerges with Scale*,
  that decodability does not entail use in model processing.
- **No trivial-scalar baseline.** 0 of 6. Verified absences: DyCon ("no baseline regressor such as
  step index, prefix length or token count"), paper 2, paper 4, paper 5, paper 6. Tele-Lens is a
  partial exception (random guessing, direct answering, perplexity/entropy/self-certainty) but none
  of those is a position or length scalar for the length target.
- **No confidence intervals on any prediction metric**, in any of the seven, and no correction for
  effective sample size. Our question bootstrap is ahead of the field.

### The warning that comes with that

Scored **pooled across depth**, the way these papers pool, `rem_len_mean` gives **act_L21 0.716
against prefix_len alone 0.718** (verified independently from `sweep/ref_fixed/preds*.csv`). A single
scalar matches our 3,584-dim probe. **Never pool across depth**, always score within checkpoint. And
their headline numbers are exposed to the same confound, unexamined.

## Novelty: the claim, stated carefully

The strong form, "every paper predicts a single trajectory", is **FALSE and must not be used.**
Paper 2's target is `avg_tokens_reasoning` = `sum(num_tokens_reasoning) / len(...)` over
`vote_num=8` rollouts at temperature 0.6 (confirmed from `AnalyzeOneModel.py` L213, `utils.py` L237,
`scripts/generate.sh`, `generator.py`). That is a mean over 8 samples, i.e. weakly distributional.

The surviving form still holds absolutely:

> Nobody predicts answer-distribution properties: entropy over sampled answers, `p_mode`, or the
> spread of the continuation distribution. One paper (Sheng et al., NeurIPS 2025) uses a multi-sample
> **length** target, a mean over 8 rollouts; no paper uses a distributional target of any other kind.

`rem_len_std` is the sharpest case and survives untouched: their mean-of-8 discards exactly the
spread we predict, and the quantity is not expressible for a single-trajectory label at all. Our
K=25 at five depths against their K=8 at the prompt only is a difference of **degree**, and should be
stated as such.

## Calibration: is 0.524 good?

**The single-trajectory sampling ceiling** (computed from our own labels and continuations) is
0.731 falling to 0.538 raw across the 10-90% checkpoints, versus **~0.97-0.99 for our 25-chain mean**
(independently reproduced by the redundancy analysis: 0.985 to 0.966 for `rem_len_mean`). So:

- DyCon's 0.642 is **near-optimal for its target**, not mediocre.
- Our 0.524 is ~53% of what is achievable for ours.
- The honest framing is not "we predict better" but "we predict a different target under stricter
  scoring". Our target is intrinsically *easier* (an expectation with sampling noise averaged out),
  which must be stated alongside any comparison.

**Conventions to adopt so our numbers are readable by this literature:** report raw and log scales
for length, score within checkpoint and say so, and report Spearman alongside R² for anything we
compare to paper 2.

## Other findings worth knowing

- **Tele-Lens makes flatness across depth its headline finding** ("the precise final-answer only
  emerges one step before the reasoning completion"; for Parity, probing "hovers around random
  guessing as 50%" until the final position). So **our flat R² across checkpoints replicates a
  published result rather than indicating a defect.** It also reports its length correlations as
  "unstable and often low".
- **Paper 2 probes the prompt end**, confirmed from `embed.py` (`hidden_states[:, -1, :]` after
  `add_generation_prompt=True`), which resolves its own Sec 3.1 / App B.1 contradiction. **That is our
  0% checkpoint exactly**, with a clean per-question split and a raw target. Our unrun 0% probe would
  be directly comparable to a published NeurIPS 2025 result. Report R² **and Spearman** when we run it.
- **Paper 2 had R² in hand and published Spearman** (`utils.py` L405/L413 computes `r2_score`).
- **Paper 6 is the selection-leak risk**, not DyCon: it grids over MLP hidden sizes *and*
  representation layers and reports "test scores for the best hyperparameters", with a 60:20:20 split
  but no statement that selection used the validation portion. DyCon's layer selection is clean.
- Flatness across *layers* also appears: paper 2's Table 7 has layers 25-30 within 0.006 of each other.

## Still unresolved

Four items remain open in `_litreview/QUESTIONS.md`:

- **DyCon's total step-level N per model** beyond ours: never reported, so no error bar is computable
  for their figures.
- **Paper 6's numeric regression values** exist only in Figures 2, 3 and App B.1 hexbins. No number
  is extractable from the text, which is itself worth reporting.
- **Paper 3's decoding setup** (greedy or sampled, how many samples per prefix) is not stated.
- **Paper 5's training data and split** for its auxiliary heads are not stated.
