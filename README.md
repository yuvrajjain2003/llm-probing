# LLM chain-of-thought probing

Research code from work done at **UNC Chapel Hill**. MIT licensed — see [`LICENSE`](LICENSE).

Do internal representations at a chain-of-thought prefix predict properties of
the **distribution over future continuations**, beyond what the prefix text
already tells you?

The method, in one line: sample base CoTs → cut prefixes at fixed fractions →
sample K continuations from each prefix → reduce those to behavioural labels
(answer entropy, remaining reasoning length, commitment) → probe the model's
activations *at the prefix* against those labels, versus a TF-IDF-on-prefix-text
baseline.

## Orientation

| If you want | Go to |
|---|---|
| To pick up the work (agent or human) | **[`.claude/README.md`](.claude/README.md)** |
| To get this running on a new cluster | [`.claude/SETUP.md`](.claude/SETUP.md) |
| What's been run and what it showed | [`.claude/STATE.md`](.claude/STATE.md) |
| The exact command for each stage | [`.claude/PIPELINE.md`](.claude/PIPELINE.md) |
| Traps that silently corrupt results | [`.claude/GOTCHAS.md`](.claude/GOTCHAS.md) |
| Scope and working practices | [`CLAUDE.md`](CLAUDE.md) |

## Layout

| path | contents |
|---|---|
| `src/generation/` | Everything that talks to the model (vLLM), plus maintenance of the raw records it writes |
| `src/analysis/` | Pure CPU reductions over those records — no GPU, no model |
| `src/probing/` | Activation capture and the probes |
| `src/viz/` | Figures, tables, PDFs |
| `src/setup/` | One-time dataset fetch into `data/` |
| `runs/` | Run outputs — **not tracked**, only the launcher scripts are |
| `archive/` | Prior Qwen3 logit-lens work, reference only; paths inside are stale |
| `env/` | pip freezes for both environments |

## Quick start

```bash
source env.sh                                   # see .claude/SETUP.md for overrides
"$PY" src/setup/download_dataset.py             # data/ is not tracked
"$PY" src/probing/verify_capture.py --stage hf  # verify before trusting anything
```

Two separate python environments are required and cannot be merged — vLLM 0.5.5
for generation and analysis, an EasySteer fork of vLLM 0.17.1 for activation
capture. [`.claude/SETUP.md`](.claude/SETUP.md) explains why and how to build both.

## Status

Goal 1 (labels) is built and run on five datasets for DeepSeek-R1-Distill-Qwen-7B.
Goal 2 (probing) is built and run on GPQA-diamond only. Llama-8B has not been run.
Current finding: activations beat a text baseline on remaining reasoning **length**
and its variability, but not on answer entropy or P(mode) — and the strongest
length result is confounded by prefix token count. See
[`.claude/STATE.md`](.claude/STATE.md) and [`.claude/NEXT.md`](.claude/NEXT.md).
