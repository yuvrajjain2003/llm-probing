# LLM CoT Probing — Goal 1 Status Report
_Generated 2026-08-17_

## TL;DR

The gap flagged in `CLAUDE.md` — "prefix-conditioned continuation sampling doesn't
exist yet" — has been closed. Your collaborator **rishp** built the rest of the
Goal 1 pipeline (checkpointing → continuations → labels → robustness → plots) on
top of your existing `run_eval.py`, and it has been **run end-to-end once**,
successfully, for **DeepSeek-R1-Distill-Qwen-7B** across all three in-scope
datasets. Llama-8B has not been run yet.

---

## New code (all added by rishp, reusing your `run_eval.py` helpers)

| File | Role |
|---|---|
| [`src/generation/checkpoint.py`](../src/generation/checkpoint.py) | Splits a CoT into sentences with a math-aware splitter (masks decimals, `$...$` math spans, abbreviations, single-letter choice labels like "A." before splitting on `.!?`), then cuts prefixes at 10/30/50/70/90% by sentence count. **Key design choice:** splits only the pre-`</think>` span, because R1-Distill models often write a plain-language answer summary right after `</think>` — splitting the full reasoning field would leak the answer into "late" prefixes and fake the surface-leakage signal Goal 1 is trying to measure. |
| [`src/generation/continuations.py`](../src/generation/continuations.py) | The core new piece. Reads a `run_eval.py` output dir, cuts each CoT into checkpoints, and for every prefix samples K continuations by feeding `prompt + prefix_text` raw to vLLM (no re-applying the chat template). Uses `enable_prefix_caching` so shared prefixes hit the KV cache instead of recomputing. Writes `checkpoints.jsonl` and `continuations.jsonl`, both resumable by count like your original script. |
| [`src/analysis/labels.py`](../src/analysis/labels.py) | Pure reduction over `continuations.jsonl` → `checkpoint_labels.{csv,parquet}` (entropy, P(mode), P(gold), remaining-reasoning length, accuracy at single / maj@5 / maj@10) and `commitment.{csv,parquet}` (derived "commitment juncture": earliest checkpoint whose modal answer stays modal and ≥80% concentrated through every later checkpoint). |
| [`src/analysis/robustness.py`](../src/analysis/robustness.py) | The three robustness analyses CLAUDE.md calls "the actual deliverable": **K-sufficiency** (bootstrap SE of entropy/length estimates at K′=2,4,6,8,10), a **parser-failure catalog** (truncation vs. parse-fail, with error patterns tallied), and **surface-form leakage** (splits into "answer letter explicitly declared" vs. "choice text merely present" — two distinct cheat signals, reported separately). |
| [`src/viz/plot_goal1.py`](../src/viz/plot_goal1.py) | Two static headline figures: entropy-vs-checkpoint, and a 3-panel of remaining-length / accuracy / leakage. |

Your `run_eval.py` itself wasn't restructured — rishp imports its prompt
rendering, `parse_output`, `normalize_mc`/`normalize_freeform`, `_entropy`,
`_append_jsonl`/`_load_jsonl` directly, per your CLAUDE.md instruction not to
reimplement.

---

## What's actually been run

`runs/goal1/base/` — one full pipeline pass, **Qwen-7B only**, on all three
in-scope datasets, 33 questions each (99 total, close to the 50–100 target),
3 base CoTs/question, checkpoints at the spec'd 5 fractions, K=10
continuations/prefix. All jobs exited cleanly (`EXIT_*=0` in the logs,
dated 2026-07-26).

### Headline numbers

| Dataset | pass@1 | maj-vote | mean entropy | base-CoT truncation | cont. parse-fail |
|---|---|---|---|---|---|
| aqua_rat | 79.8% | 84.8% | 0.167 bits | 6/99 | 3.5% |
| mmlu_college_math | 98.0% | 100% | 0.056 bits | 0/99 | — |
| openbookqa | 72.7% | 75.8% | 0.354 bits | 0/99 | — |

### Robustness signal (aqua_rat, representative)

- **K-sufficiency:** entropy bootstrap-SE keeps dropping out to K=10
  (0.023 → 0.011 bits at the 10% checkpoint) — i.e. **K=10 may not be fully
  saturated yet**. Worth flagging against the "is 10 enough" question in
  CLAUDE.md.
- **Surface-form leakage** rises sharply with prefix length: choice-text-present
  goes from 15% (10% checkpoint) to 68% (90% checkpoint); letter-explicitly-declared
  goes from 1% to 28%.

Two static figures already exist at `runs/goal1/figures/`
(`entropy_by_checkpoint.png`, `other_labels_by_checkpoint.png`).

**Loose end:** `plot_goal1.py` reads from a `goal1_viz.json` that **isn't
anywhere in the repo**. Whoever produced those figures must have hand-built
that aggregation file once and not committed it. There's no script yet that
rolls the per-dataset `checkpoint_labels.csv` + `robustness/*` outputs into
that combined JSON, so the figures currently aren't reproducible from what's
checked in.

`runs/goal1/pilot/` and `runs/goal1/smoke/` are smaller earlier test runs
(pilot = base CoTs only, no continuations; smoke = full pipeline dry run) —
superseded by `base/`, safe to ignore or clean up later.

---

## Still open relative to CLAUDE.md's scope

- **Llama-8B**: zero runs — everything so far is Qwen-7B only.
- **Missing aggregation script** for `goal1_viz.json` (see above).
- GSM8K correctly untouched (still deferred per your note).
- Activation extraction / probing: correctly not started (Goal 2, out of scope).
- `src/gsm8k_test.ipynb` is an old notebook (yours, June 30) — looks like
  pre-CLAUDE.md scratch work, probably safe to ignore/archive.
  (Since archived to `archive/notebooks/gsm8k_test.ipynb`, 2026-09-14.)

---

## Possible next steps

- Run the Llama-8B leg of the sweep to match Qwen-7B coverage.
- Write the missing aggregation step so `plot_goal1.py`'s figures are
  reproducible from checked-in code.
- Decide whether K needs to grow past 10 given the K-sufficiency curves
  haven't fully flattened.
- Once labels are validated as stable across both models, Goal 1 is
  essentially done and Goal 2 (activations/probing) can start.
