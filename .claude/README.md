# Handoff context — start here

This folder is the project's memory. It was written on **2026-09-18**, on the
machine where all existing results were produced (a UNC box, 8× RTX A6000 48GB),
immediately before the repo was moved to a larger cluster.

**Nothing in `runs/` was pushed.** The repo carries code, configuration and this
context only. Every result described here lives on the original box; on the
cluster it must be regenerated. That is the expected workflow, not a loss.

## Read in this order

| File | What it answers |
|---|---|
| [`SETUP.md`](SETUP.md) | I just cloned this on a new cluster. How do I get to a working state? |
| [`STATE.md`](STATE.md) | What has actually been run, and what did it show? |
| [`PIPELINE.md`](PIPELINE.md) | What is the exact command for each stage? |
| [`GOTCHAS.md`](GOTCHAS.md) | What will silently give me wrong numbers if I don't know it? |
| [`NEXT.md`](NEXT.md) | What was agreed to do next, and why? |

`../CLAUDE.md` in the repo root is the standing project instruction file and is
loaded automatically. It is the authority on **scope and working practices**.
This folder is the authority on **current state and mechanics**. Where the two
disagree about what has been run, trust `STATE.md` — `CLAUDE.md` has repeatedly
lagged behind what was actually executed.

## The one-paragraph version

The research question is whether internal representations at a chain-of-thought
prefix predict properties of the *distribution over future continuations* beyond
what the prefix text already tells you. Goal 1 (generate prefixes, sample K
continuations from each, reduce to behavioural labels) is built and has been run
on five datasets for DeepSeek-R1-Distill-Qwen-7B. Goal 2 (capture activations at
those prefixes, probe them against the labels, compare to a TF-IDF-on-text
baseline) is built and has been run on GPQA-diamond only. The current answer is
**mixed and target-dependent**: activations beat text significantly on remaining
reasoning *length* and its *variability*, but not on answer entropy or P(mode).
The strongest length result is confounded by prefix token count, which is the
specific reason for the go-forward design change in [`NEXT.md`](NEXT.md).

## Standing constraints carried over

- **Do not assume.** Verify against code and data before acting. Don't guess
  dataset schemas, column names, or model behaviour.
- **Ask when a decision is genuinely the user's** — scope, model/dataset choice,
  anything hard to reverse or expensive in GPU time.
- **Report outcomes honestly.** Parser failures are a Goal 1 *metric*, not an
  embarrassment to paper over.
- **Reuse before rewriting.** `run_eval.py`'s prompt rendering, `parse_output`,
  `normalize_mc`/`normalize_freeform` and resume logic are the canonical
  implementations. Do not write second copies.
- **Expanding scope** — new datasets, new models, new Goal 2 stages beyond what
  `NEXT.md` describes — gets flagged to the user first.
- The repo **is** under version control now (it was not before). Commit before
  destructive edits rather than making `.bak` copies; `archive/src_bak/` is the
  pre-git convention and should not grow.
