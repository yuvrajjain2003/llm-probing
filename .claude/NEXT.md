# Go-forward plan

Decided **2026-09-14**, **not yet run**. Everything currently in `runs/` uses the
old design (3 CoTs per question, 5 checkpoints at 10/30/50/70/90%), including all
Goal 2 results.

## The design change

New runs use instead:

- **1 CoT per question** (not 3)
- **6 checkpoints, the first at 0%** — the rendered prompt *before any reasoning
  has been emitted* — then the existing five
- k rollouts per checkpoint, k still set by the k-sufficiency analysis

### Why the 0% checkpoint is the point

It **removes the prefix-length confound**. Every 0% prefix has length zero, so
`prefix_len` is constant and can predict nothing. This matters because the
strongest result so far — remaining reasoning length, R²=0.605 at the 10%
checkpoint — is matched or beaten by `prefix_len` alone at the first three
checkpoints, i.e. it is largely token-counting. At 0% that explanation is
unavailable, so **whatever a probe achieves there is clean**.

It also answers a sharper question than the original one: how much does the model
know about where it is heading *before writing anything at all*?

### The cost, stated honestly

**Per-checkpoint training rows drop 575 → 195**, since each question now
contributes one CoT instead of three. The learning curve in
`runs/goal2/logs_ckpt/lcurve.log` shows entropy had already plateaued by ~97
questions, so entropy targets should be largely unaffected. **`rem_len_mean` was
still improving with more data and should be expected to weaken.** Do not read a
drop in the length numbers as a failure of the 0% design without checking that
curve first.

Note the interaction with `bootstrap_ci.py`'s `questions_needed` column: several
currently-non-significant gaps need hundreds to thousands of questions to
resolve. Fewer rows per question makes that harder, not easier. If the goal is to
settle the entropy question, more *questions* — not more CoTs per question — is
the lever.

## Implementation notes

- `continuations.py --fractions` already takes an arbitrary list, so `0.0 0.1 0.3
  0.5 0.7 0.9` needs no new flag — but **verify `checkpoint.py` handles a 0.0
  fraction**, i.e. that it yields an empty prefix rather than the first sentence
  or an index error. This has not been tested and is the first thing to check.
- A 0% prefix means the activation is captured at the last token of the *rendered
  prompt*. `extract_checkpoints.py` captures at the last token of each prefix;
  confirm it does the right thing when the prefix is empty.
- `--samples-per-example 1` on `run_eval.py`.
- Old and new numbers are **not directly comparable** (different row counts,
  different CoT sampling). Keep them in separate run directories and say which
  design produced any figure.

## Parked, not rejected

**Using the rollouts themselves as probe training data.** The activation at a
checkpoint is **one vector shared by all k rollouts** — causal attention means
the prefix hidden states do not depend on what follows — so repeating it per
rollout adds rows but zero information. The version that would add real data is
capturing activations at tokens *inside* each continuation, where they genuinely
diverge. That needs a new GPU pass and, first, a decision on what label a
mid-continuation token should carry. Raise it with the user before building it.

## The standing gaps, in the order I would do them

1. **Verify the 0% checkpoint works** (cheap, CPU, blocks everything else above).
2. **Llama-8B.** Zero runs anywhere, in scope since day one, and the only way to
   know whether any of this is model-specific rather than a fact about Qwen-7B.
   The biggest single hole in the current evidence.
3. **The missing Goal 2 baselines** — mean-mode, surface-form-only, and
   checkpoint-position/model-only predictors. `prefix_len` and TF-IDF cover part
   of this but were never the agreed set.
4. **Goal 2 on math500 / AIME.** Labels already exist; only the GPU capture pass
   is missing, so this is the cheapest expansion of dataset coverage. It would
   also test whether the length result survives outside GPQA.
5. GSM8K remains deliberately deferred.

Items 2–4 expand scope. Per the standing instruction in `../CLAUDE.md`,
**flag them to the user before starting**, with a GPU-time estimate.

## What would actually answer the research question

Worth keeping in view, since it is easy to drift into producing more numbers that
do not settle it. The claim to establish is that activations add information
about the *future continuation distribution* beyond the prefix text. That needs:

- a clean checkpoint where text and length baselines are degenerate (**0%**),
- a target that is genuinely about the distribution and not about the prefix
  (`rem_len_std` and entropy qualify; `rem_len_mean` is contaminated by
  `prefix_len`),
- the **[text ++ activations]** combination beating text alone, not just
  activations beating text head-to-head, and
- a bootstrap CI over questions, not a raw R² delta.

`probe_checkpoints.py` already fits the combination and `bootstrap_ci.py` already
produces the CI. The infrastructure is in place; what is missing is the 0% run
and a second model.
