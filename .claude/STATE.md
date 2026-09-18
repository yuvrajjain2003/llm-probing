# What has actually been run, and what it showed

Snapshot date **2026-09-18**. All of it is **DeepSeek-R1-Distill-Qwen-7B**.
**Llama-8B has zero runs anywhere**, despite being in scope since the start.

None of these outputs were pushed. This file is the record of them.

## Goal 1 — generation and labels

### `runs/goal1/full/` — the real runs

| dataset | questions | base CoTs | continuations | K | labels |
|---|---|---|---|---|---|
| math500 | 1482 | 1500 | 691,900 | 50 | yes |
| aime_2022_2025 | 165 | 360 | 134,000 | 100 | yes |
| gpqa_diamond | 656 | 594 | 116,875 | **100 / 25, split** | yes |
| aime2025 | 30 | 90 | 46,500 | 100 | yes |

Design throughout: **3 base CoTs per question**, checkpoints at **10/30/50/70/90%**
by sentence count. Note `CLAUDE.md`'s original scope (AquaRAT / OpenBookQA / MMLU
college math, 50–100 questions) describes neither the datasets nor the sizes that
were actually run — those three survive only in `runs/goal1/base/`.

### `runs/goal1/base/` — earlier full-pipeline pass, 33 questions/dataset, K=10

aqua_rat, openbookqa, mmlu_college_math, gpqa_diamond, math500, aime2025.
Superseded by `full/` but the only place the three originally-scoped datasets
were ever run. Headline: aqua_rat 79.8% pass@1 / 84.8% maj-vote / 0.167 bits
entropy; mmlu_college_math 98.0% / 100% / 0.056 bits; openbookqa 72.7% / 75.8% /
0.354 bits. `pilot/` and `smoke/` are older and superseded.

### GPQA's K is two-tier — the single most important caveat here

`pipeline_gpqa.sh` ran phase A at K=100 on 20% of questions, measured K\* there,
then ran the remainder at that K\*. So the GPQA ground truth is **not uniform**:

| slice | questions | CoTs | K |
|---|---|---|---|
| `example_id % 5 == 0` | 40 | 120 | 100 |
| the rest | 155 | 455 | 25 |

Measured K\* by criterion: `k_bias=75`, `k_bias_mm=25`, `k_precision=NaN` (the
0.05-bit SD tolerance was **never met at K≤100**), `k_decision=100`. The log
states `K* (max of the three criteria) = 100.0` — but the pipeline then read back
`k_bias_mm` and ran at **25**. The K=25 choice is therefore justified *only*
under the Miller–Madow bias correction, which `labels.py` was not applying at the
time (`run_eval._entropy` is plug-in). Consequence, measured: **66.5% of K=25
rows were biased beyond the stated 0.02-bit tolerance**, mean correction
0.022–0.053 bits depending on checkpoint.

**Fixed 2026-09-14:** `labels.py` now also emits `answer_entropy_bits_mm`
(`labels._entropy_mm`). Both estimators are stored so the gap stays auditable.
**Use `answer_entropy_bits_mm` as the entropy target.** Treat commitment labels
on the K=25 slice as ~82% stable (`k_sufficiency/commitment_agreement.csv`).
`probe_checkpoints.py --k100-only` restricts to the clean 40-question slice.

### K-sufficiency by dataset (`k_sufficiency/summary.csv`)

| dataset | k_bias | k_bias_mm | k_precision | k_decision |
|---|---|---|---|---|
| gpqa_diamond | 75 | 25 | NaN | 100 |
| math500 | 100 | 50 | NaN | 100 |
| aime_2022_2025 | — | 100 | — | 100 |
| aime2025 | — | 300 | — | 15 |

`k_precision` is NaN everywhere: the 0.05-bit SD tolerance was never reached at
any K tried. The original scope's "K=10" is **not** supported by any of this —
the `base/` run already showed entropy bootstrap-SE still falling at K=10
(0.023 → 0.011 bits at the 10% checkpoint). Re-check K per dataset; do not carry
a K across datasets.

### Robustness, GPQA (`robustness/`)

- Base CoT truncation **11/594 = 1.85%**.
- Continuation parse failure **926/116,875 = 0.79%**, every one the same pattern
  (`no_json_found`); 295 non-terminated; boxed-fallback 0.19%, latex-literal 0.15%.
- **Surface-form leakage rises steeply with prefix length** — the reason a
  text baseline is a serious competitor, not a formality:

  | checkpoint | mode letter declared | mode choice text present |
  |---|---|---|
  | 10% | 0.0% | 10.0% |
  | 30% | 0.4% | 21.1% |
  | 50% | 4.4% | 32.3% |
  | 70% | 9.8% | 39.7% |
  | 90% | 28.6% | 46.7% |

## Goal 2 — activations and probes

GPQA-diamond / Qwen-7B only.

- `activations/` (15 GB) — stride-4 think-token capture, layers 14/21/28 only.
  **Superseded** by the checkpoint-exact store below; kept for the `probe_linear`
  depth results. Do not build new work on it.
- `activations_ckpt/` (0.54 GB) — **all 28 layers at the exact checkpoint token**,
  one vector per `(example_id, cot_id, fraction)`, 1:1 with
  `checkpoint_labels.csv`. 2,915 checkpoints over 583 CoTs, fp16. This is the
  current store.

### `probe_linear.py` (older, stride-4 store)

`depth` positive control 0.81–0.82 acc vs 0.50 base — **the pipeline is sound**.
`correct` 0.69–0.72 vs 0.53 base, flat across all 10 depth bins (real but does
not sharpen with depth; still unexplained). `answer` (which of 4) 0.25–0.34 vs
0.264 base — no reliable early signal.

### `probe_goal1_labels.py` (older, and handicapped — do not cite)

Reported TF-IDF beating activations nearly everywhere. That comparison was unfair
to activations in three ways, all since fixed: activations were read at the
nearest captured token *at or before* the checkpoint (stride 4) while TF-IDF got
the exact prefix; only 3 of 28 layers were stored; and regularisation was never
swept (`Ridge(alpha=1000)` for activations vs `alpha=1.0` for text). It also
scored 8 CoTs that have checkpoints but no labels as "not committed".
**Superseded by `probe_checkpoints.py`.**

### `probe_checkpoints.py` — the current result

Checkpoint-exact, all 28 layers, nested regularisation selection for *every*
model including the baselines, so the comparison is about features and not tuning
effort. n=574–575 rows (195 questions × 3 CoTs). R² for regression targets,
accuracy for binary; `prefix_len` and TF-IDF-on-`text` are the baselines.

| checkpoint | target | prefix_len | text | act_best (layer) | text+act | gain over text |
|---|---|---|---|---|---|---|
| 0.1 | entropy_mm | 0.161 | 0.262 | **0.316** (17) | 0.316 | +0.054 |
| 0.1 | rem_len_mean | **0.605** | 0.514 | 0.605 (20) | 0.598 | +0.084 |
| 0.1 | rem_len_std | 0.113 | 0.185 | **0.282** (14) | 0.287 | +0.102 |
| 0.1 | p_mode | 0.143 | 0.246 | 0.261 (17) | 0.278 | +0.032 |
| 0.1 | committed_by_here | 0.724 | 0.765 | 0.793 (11) | 0.795 | +0.030 |
| 0.3 | entropy_mm | 0.171 | **0.274** | 0.218 (20) | 0.258 | −0.015 |
| 0.3 | rem_len_std | 0.160 | 0.218 | **0.332** (22) | 0.320 | +0.102 |
| 0.7 | rem_len_mean | 0.321 | 0.225 | **0.377** (24) | 0.345 | +0.120 |
| 0.9 | rem_len_mean | 0.186 | 0.170 | 0.248 (27) | 0.258 | +0.088 |

`will_commit` is at or barely above its 0.7565 majority-class base everywhere —
**treat it as unlearned**, not as a weak positive.

### Significance (`bootstrap_ci_vs_text.csv`) — this is what to believe

Bootstrap CI on the activation-minus-text gap, resampled over the 195 questions:

- **`rem_len_mean` — significant for activations at 0.1, 0.3, 0.7, 0.9** (gaps
  +0.091, +0.074, +0.162, +0.081).
- **`rem_len_std` — significant for activations at 0.1 and 0.3** (+0.076, +0.123).
- **`answer_entropy_bits_mm` — significant for *text* at 0.3** (−0.073); every
  other checkpoint is a non-significant wash.
- **`p_mode` — significant for *text* at 0.3** (−0.107); elsewhere a wash.

The file also reports `questions_needed` to resolve each non-significant gap at
the current effect size. Several are in the hundreds (e.g. entropy at 0.1 needs
~1240 questions, p_mode at 0.1 ~9053) — worth reading before proposing "just run
more questions" as a fix.

### The honest summary, and the confound that motivates everything in NEXT.md

Activations carry information about **how much further the model will reason, and
how variable that is**, beyond the prefix text. They do **not** reliably carry
extra information about **which answer** the distribution will land on — text
matches or beats them on entropy and P(mode).

But the strongest result is confounded. At the 0.1 checkpoint `rem_len_mean` is
R²=0.605 from activations and R²=0.605 from `prefix_len` **alone** — a single
scalar, the token count of the prefix. `prefix_len` matches or beats TF-IDF at
the first three checkpoints. So the headline length result is substantially
**token-counting**, not reasoning-state read-out. Resolving that is the entire
point of the 0% checkpoint in [`NEXT.md`](NEXT.md).

## Known loose ends

- **Llama-8B: nothing.** The single largest gap against agreed scope.
- **Goal 2 baselines never built** — the mean-mode, surface-form-only, and
  checkpoint-position/model-only predictors are still missing. `prefix_len` and
  TF-IDF partially cover this but were not the agreed set.
- **GSM8K** deferred deliberately; still untouched.
- `plot_goal1.py` reads a `goal1_viz.json` produced by `aggregate_viz.py`;
  confirm they agree before trusting any regenerated figure.
- Only GPQA has Goal 2 activations. math500 and the AIME sets have Goal 1 labels
  and no activations — they are the cheapest available expansion, since the
  labels already exist and only the GPU capture pass is missing.
