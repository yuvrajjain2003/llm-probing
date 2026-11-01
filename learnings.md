# Learnings

Things we've learned and mistakes not to repeat. Add new entries at the bottom with a date.

- 2026-10-03: `sinfo`/`scontrol` hide GPU partitions by default. Use `sinfo -a` / `scontrol --all show partition h100_all`.
- 2026-10-03: `batch` partition needs ≥48 CPUs and `small` needs ≥2, or the job is rejected (`QOSMinCpuNotSatisfied`).
- 2026-10-03: Use `h100_all`, not `h100` (the latter gives "Invalid qos specification").
- 2026-10-03: From `runs/goal2/*/_verify/cmp.txt`: EasySteer activations are shifted vs HuggingFace — decoder layer L → ES index L-1, token position t → ES position t+1. Final layer differs (pre- vs post-norm).
- 2026-10-03: The project folder is owned by `yuvraj`, so git needed `git config --global --add safe.directory /proj/ssrivalab/projects/llm-probing` (done).
- 2026-10-03: `env.sh` used to fall back to a hidden `<repo>/.work` when `/work/users/r/i/rishp` is missing. Fixed: it now falls back to this folder (`cache/`, `envs/`, `runs/`).
- 2026-10-03: The repo's `.claude/STATE.md` describes the old UNC box (2026-09-18). The Sycamore `gpqa25` runs in `runs/` are newer and not described there.
- 2026-10-04: `make_checkpoints(fraction=0.0)` returns the first sentence (not empty) and silently drops 10% as a duplicate. `extract_checkpoints.py` asserts on empty prefixes. Fix both before any 0% run.
- 2026-10-04: `run_eval`'s `reasoning_tokens` ≠ `rem_len_*`. The former includes the summary after `</think>`; the latter counts only tokens before `</think>`.
- 2026-10-04: Checkpoint prefixes are long (90%: median 4.3k, max 14.8k tokens). Any text model (embedder/LLM baseline) must handle this or truncate explicitly.
- 2026-10-04: Current MTEB leaderboard data is available as JSON from `https://mteb-leaderboard-backend.hf.space/v1/benchmarks/<name>/scores` (e.g. `MTEB(eng, v2)`, `LongEmbed`); the Space page itself is client-rendered and WebFetch sees nothing. Results in `docs/embedders.md`. goal1 env has no sentence-transformers/flash_attn/mteb; vllm 0.5.5 is too old for these embedders, use plain transformers.
- 2026-10-04: `make_checkpoints(fraction=0.0)` used to give the first sentence (and drop 0.1 on short CoTs); now `k=0` for f==0 gives an empty prefix, other fractions unchanged (verified byte-identical on 600 real CoTs).
- 2026-10-04: 0% rows have no CoT: all 25 CoTs of a question share the empty prefix, so f0 labels/activations use cot_id=-1 (198 rows); a 0.0 entry also changes `labels._commitment` (earliest checkpoint now may be 0.0) if merged into checkpoint_labels.
- 2026-10-04: `cache/huggingface/token` is unreadable for us (mode 0620, owner yuvraj), so AutoTokenizer.from_pretrained fails with PermissionError even with HF_HUB_OFFLINE=1; set `HF_TOKEN=dummy` (done in 06b_extract_f0.sh).
- 2026-10-04: base samples and continuations use identical prompt, temperature 0.7, top_p 0.95, max_tokens 16384 (and no seed), so 0% labels can be built from the 25 base samples; use tokens before `</think>`, not run_eval's reasoning_tokens.
- 2026-10-04: `sbatch --wrap` runs under `/bin/sh`: `source env.sh` fails ("file not found" via PATH lookup); use `. ./env.sh` and escape `\$PY` so it expands on the node.
- 2026-10-04: The session scratchpad under `/tmp` is login-node-local; Slurm jobs cannot see it. Put job configs/outputs under `runs/`.
- 2026-10-04: `sweep.py` skips combos by (feature, probe name, fraction, target); if you change a probe's grid, rename the probe (or delete its rows) or the old rows are silently kept.
- 2026-10-04: `sweep.py` shard balance: units are (fraction, row-mask, feature) in config order; pick N so that N does not divide the number of features per group (3 features -> 5 shards, not 6), else all heavy `act` units land on the same shards.
- 2026-10-04: `sweep.py` tiny-subset smoke test (30 questions, 8 CPUs): torch MLP and RF run fine; logistic on dense features is the slow step (~40 s there), not ridge (~3 s).
- 2026-10-04: sweep.py bug fixed: unit numbers used to skip already-done units, so shards started at different times disagreed on ownership (duplicated and missed combos). Units are now numbered before the done-filter.
- 2026-10-04: Qwen3-Embedding-8B (question+prefix, last-token, generic instruction) is a WEAKER text baseline than TF-IDF on all targets here; logistic on it picks C at the grid edge (1e-4) → use a grid down to 1e-6 for embedding features.
- 2026-10-04: MLP probe weight decay (AdamW, wd ≤ 0.1, lr 1e-4, early stopping) changes nothing — results identical to 3 dp. Effective decay per step is lr×wd ≈ 1e-5. Use much larger wd or an L2 loss term to actually test regularisation.
- 2026-10-04: yuvraj shares this repo folder but his runs go to his /work (env.sh picks /work when it exists). Collision risk is git, not run outputs: fetch + check origin/main before committing; never force-push.
- 2026-10-04: Hidden states of OTHER 7B LLMs (Qwen2.5-Math-7B, Qwen2.5-7B-Instruct; layer 21, last token, same text) predict the targets as well as our reasoning model's activations. Always include an other-LLM control before claiming model-specific signal.
- 2026-10-04: RF on raw 3584 dims with max_features 0.1 costs ~110 s per 5-fold combo (500 trees, 12 CPUs); nested selection (4 points) ~670 s per combo, i.e. ~3 h per shard for 25 combos. In RF on PCA-16, max_features sqrt and 0.3 both give 4 dims (identical results). HGB/RF configs and generators: configs/sweep_4c_*.yaml, runs/sweep_tmp/gen_*.py.
- 2026-10-09: `bootstrap_ci.py` defaults (`--preds`, `--out-dir`) point at yuvraj's `runs/goal2/probes_ckpt/`; always override them to a folder under `runs/goal2/gpqa25/sweep/`. `peft` is not installed in either env. Sweep `results.csv` already stores in-sample `train` scores; MLP `*_ablation` sweeps saved no predictions.
- 2026-10-09: sweep `results.csv` `test` = POOLED out-of-fold score (1 − MSE/Var over all rows reproduces it exactly); `train` = MEAN over folds of in-sample scores — not averaged the same way. Per-fold-averaged test R² would be ~0.012 lower.
- 2026-10-09: Mean MSE over checkpoints is dominated by early checkpoints (largest target variance); compare MSE per checkpoint, not averaged.
- 2026-10-09: Fold-to-fold std (~0.09 R²) is ~3× the question-bootstrap SD; paired differences (same bootstrap draws) are much tighter than single-score CIs — always report paired CIs for comparisons.
- 2026-10-09: `fraction` floats differ between checkpoints jsonl and preds CSVs (0.7 especially): merging on `fraction` silently drops rows — round to 3 decimals first.
- 2026-10-09: Qwen3-Embedding-8B and F2LLM-v2-8B share the Qwen3 tokenizer (identical token counts). Qwen3-Emb's training length is unpublished; F2LLM's released training code uses 1–2k tokens.
- 2026-10-09: Truncated HF trunk (layers dropped) still applies the final norm to its last state → set `model.norm = Identity` to read hidden_states[L]. Causal model: one forward over a CoT's longest prefix gives all 5 checkpoint states (LoRA training tokens 24.5M vs 71.8M).
- 2026-10-09: peft 0.21.2 installs into envs/goal1 with `--no-deps` without touching torch/transformers. Long `pip install` exceeds the 2-min tool timeout — run in background.
- 2026-10-09: float32 torch.cosine_similarity on huge tensors gives cosines >1 (up to 1.2); use float64 sums. Qwen2.5-7B base now cached (models--Qwen--Qwen2.5-7B).
- 2026-10-09: r3_rf_diagnosis.py imports sweep.py helpers (load_meta, Act, fixed_fold_of, _fixed_folds) — reuse that pattern for one-off analyses instead of editing sweep.py.
- 2026-10-09: Only `envs/icl` (vLLM 0.31.0, transformers 5.17) can serve Qwen3.5 / Gemma 4; build with `env/setup_icl.sh` (~20 min). Qwen3.5 thinks by default → `enable_thinking=False`; Gemma 4 doesn't; R1-Distill needs a manual `</think>\n\n` prefill.
- 2026-10-09: 9 of 24,773 label rows (ckpt + 0%) have all-NaN targets (no valid continuation); drop them consistently in every comparison.
- 2026-10-09: pandas concat then `.join` on a non-unique index duplicates rows silently — use `ignore_index=True` and check row counts.
- 2026-10-09: Per-checkpoint LoRA cost scales with fraction (0.9 ≈ 6× 0.1 tokens); split main R8 jobs by FRACTIONS.
- 2026-10-09: vLLM 0.31 (envs/icl) JIT-compiles FlashInfer kernels on the GPU node and needs nvcc: 11_icl.sh sets CUDA_HOME=/nas/sycamore/apps/cuda/13.2 + PATH (without it: 'Could not find nvcc').
- 2026-10-09: R1-Distill-Qwen-7B does NOT follow a long ICL instruction (0/40 JSON, thinking off or on): it solves the embedded GPQA question. Repeat the task after the query and force JSON (vLLM StructuredOutputsParams; with thinking, pass reasoning_parser so JSON is forced only after </think>).
- 2026-10-09: vLLM structured outputs with a JSON *schema* still let R1-Distill emit runaway whitespace/digits (disable_any_whitespace had no visible effect); a strict regex for the compact JSON fixes it. Gemma 4 + reasoning_parser='gemma4' + forced output skipped thinking entirely — use two-pass (think, then append end tag '<channel|>' and force the answer) instead.
- 2026-10-10: R1 cost estimate (~55 GPU-h) is a straight scale-up of 6–8-row smoke batches — NOT a saturated throughput measurement (e.g. ours thinking-on decoded only ~65 tok/s with 6 rows in flight). Only measured saturated number in the repo: our model's continuation generation, ~2,186 continuations/GPU-h before tuning, then 1.18× and 1.42× speed-ups (`.claude/GOTCHAS.md`; old vLLM/EasySteer, not envs/icl).
- 2026-10-11: The scratchpad (/tmp/...) is node-local: a Slurm job's --out-dir there writes to the compute node's /tmp, invisible from the login node. Put job outputs under runs/.
- 2026-10-11: runs/gpqa25/common.sh sets FRACTIONS=(0.1 ... 0.9) as an array, so a FRACTIONS env var passed to a script that sources it is overwritten and ${FRACTIONS:-x} gives "0.1". The R8 single-ckpt jobs (4790496–505) ran 0.1 instead of 0.5 because of this. 12_lora.sh now uses LORA_FRACTIONS. Check a job log for the fraction actually run.
- 2026-10-11: R8 LoRA head recipe (head lr 1e-3, wd 0.01, ≤3 epochs, patience 3 evals) is far below ridge even WITHOUT LoRA on the same stored features (linear head entropy R² −0.25 vs ridge 0.28). Always run the frozen-feature control with the same recipe before attributing a fine-tuning result to the adapter.
- 2026-10-11: Starting several `envs/goal1` jobs at the same moment can crash with `sqlite3.OperationalError: disk I/O error` (outlines' diskcache, imported via vllm from run_eval). Stagger submissions by ~1 min. Also: lora_probe.py needs `HF_TOKEN=dummy` even with --frozen (tokenizer load).
- 2026-10-11: R8 frozen heads with step-4a settings (lr 1e-4, batch 256, wd 10, patience 30) jump from entropy −0.25/0.12 (linear/MLP) to 0.15/0.22, but still < ridge 0.278 at 10%. A linear head trained by AdamW + weight decay is NOT ridge (decoupled decay ≠ CV-chosen L2 per target; one shared early stop for 5 targets).
- 2026-10-11: R8 conclusion so far: fit ridge on the LoRA-adapted states instead of judging LoRA by its SGD head. Ridge on LoRA states ≈ ridge on stored states (≤0.01 R²), so the earlier LoRA < ridge gaps were the head, not the features.
- 2026-10-12: R8 LoRA GPU time grows with prefix length: rank-16 layer-21 LoRA, 5 folds, per checkpoint ≈ 0.8/1.6/2.4/3.1/4.2 h at 10/30/50/70/90% (incl. saving states). Peak memory 11.7→16.5 GiB.
- 2026-10-15: Mistral-7B-v0.3's tokenizer has no pad token → batched tokenisation fails; embed_prefixes.py now sets pad_token = eos_token when missing (left padding + attention mask, so the last real token is unaffected). OLMo-3 tokenizer adds no BOS. OLMo-3's config warns 'beta_fast must be a float' — harmless, YaRN is still applied.
- 2026-10-15: BGE-M3 ships only pytorch_model.bin; transformers 4.57 refuses torch.load with torch 2.4 (CVE check, needs ≥2.6) → run it with envs/icl. UltraBERT-Embed needs transformers ≥5 (envs/icl) + trust_remote_code. Export PY=<python> before sbatch to override env.sh's choice (env.sh keeps an exported PY).
- 2026-10-15: Granite-embedding-multilingual-R2 (ModernBERT, transformers 4.57, sdpa, bf16, right-padded GPU batches) returned NaN CLS vectors for 22/24,773 rows — mostly the shortest texts (heavily padded last batch); the same rows are finite one at a time. Likely NaN at masked pad positions leaking through attention (0 × NaN). Fix used: --tokens-per-batch 1 (no padding). embed_prefixes.py's final isfinite assert caught it.

## 2026-10-16: OpenAI prompt caching (GPT-6 models)
- With one user message, the implicit cache point is at the END of the message, so a prompt with a changing query at the end never hits the cache (3 calls, 0 cached tokens). Fix: put the shared examples in their own `input_text` block with `"prompt_cache_breakpoint": {"mode": "explicit"}` and send `prompt_cache_options: {"mode": "explicit"}` (via `extra_body`). Then all later calls in a fold cached ~97% of input.
- Cache writes cost 1.25x the input rate; `usage` does not report them separately, so a cost tracker from `usage` under-counts (~15% here).
- New accounts may have zero prepaid credit: a valid key lists models fine but every call returns 429 `credit_balance_exhausted`.

## 2026-10-21: shared-folder permissions
- rishp's umask was 0022, so every file we made was writable only by rishp (yuvraj, same group `rc_ssrivalab_psx`, couldn't edit), and `.git` files got group `users`. Fixed: group set + g+w + setgid on all rishp-owned files here, `git config core.sharedRepository group`, `umask 002` in ~/.bashrc (applies to new shells / jobs submitted from them).

## 2026-10-23: round 4 (redundancy, MLP gap, literature)
- `env.sh` resolves `PY` by user: as `yuvraj` it points at `/work/users/y/u/yuvraj/envs/goal1/bin/python`, a DIFFERENT env from the project's. That env exists, so jobs silently run on the wrong interpreter instead of failing. Always use the explicit `./envs/goal1/bin/python` in submissions, and set `OMP_NUM_THREADS` to match `-c`. Analysis scripts use relative `runs/...` paths, so submit from the repo root.
- Comparing probe *scores* hides a badly wrong fit. The step-4a SGD recipe scores within 0.006 of ridge while its weight vector sits at cosine 0.865 (worst 0.630) from the closed-form solution. **Always compare weight vectors, not just scores, when checking whether a trained linear head equals ridge.** A properly optimised one hits cosine 1.000000.
- The 1-SE rule is NOT a safe default here: it loses in 20/20 cells, mean −0.013, worst −0.038, against a per-fold-CV alpha instability of only +0.0004 mean. A stabiliser that costs more than the instability is just a worse estimator.
- Per-fold inner-CV alpha selection is stable in 14/20 cells (all folds agree, cost exactly 0) and swings ±0.007 in the 6 where they disagree. Cause is the half-decade grid spacing. Bootstrap CIs hold fitted models fixed, so they do NOT model selection noise: treat +0.01-scale paired effects with extra caution.
- **Never pool R² across checkpoints.** Pooled, `rem_len_mean` gives act_L21 0.716 vs prefix_len alone 0.718: a single scalar matches a 3,584-dim probe, because pooling lets any depth proxy explain the between-checkpoint variance. Published papers in this area pool like this.
- A degree-2 kernel's structure is NOT in the leading PCA directions: pairwise products of the top-128 components give +0.0002 where the full poly2 kernel gives +0.0046. And there was never a reason to approximate the kernel at all, since n≈4,900 makes the exact kernel matrix trivial. Kernel ridge is itself a deployable probe.
- Closed-form kernel ridge is the right tool for "is there any nonlinear signal", because it removes the optimiser as an explanation. Result here: entropy, p_mode and `committed_by_here` are linear-ceilinged; only `rem_len_mean` (and `rem_len_std` at 90%) has exploitable curvature.
- An MLP result that ties ridge to many decimals can be *selection*, not a bug: the inner CV picked the pure-linear fallback in 26% of folds and trained 0 epochs in another 44%, so the MLP contributed nothing in 70%. Seed-ensembling turns that on/off decision into a partial blend, which is why 5 seeds are significant where 1 seed is not.
- When citing another paper's R², check their split granularity in their released code. DyCon splits at the step-event level over `np.arange(N_events)` with no `groups=`, putting ~139 near-identical prefixes per trajectory on both sides. Also check whether the number is on a transformed target: their log1p is worth +0.14 R², and their min-max is affine so it changes nothing.
- Agent orchestration: give each agent a written spec file rather than relying on the prompt; a fresh subagent inherits none of the parent's context. Reserve `CLAUDE.md` edits and all `sbatch` submissions for the orchestrator.
- 2026-10-26: Running as user `yuvraj` on Sycamore, `git fetch`/`push` to origin fails (`git@github.com: Permission denied (publickey)`) — no SSH key for GitHub in this account/session. Local commits work; pushing needs a key (or do it from the account that pushed `6637937`). `git` also needs `safe.directory` set per user (the folder is owned by yuvraj but check as whoever you are).
- 2026-10-26: `icl_openai.py`'s cache-miss guard (`misses >= 2` -> `os._exit(1)`) kills k=0 runs: with no examples the prompt is ~500 tokens, below OpenAI's 1024-token prompt-cache floor, so `cached_tokens` is always 0 and every call counts as a miss. It exits the whole invocation, so later versions/k values in the same command never run (silently — the log just stops). Guard now exempts k=0. Check per-tag jsonl line counts, not just the log tail, to confirm a multi-config run actually did every config.

## 2026-10-26 - GPQA biology questions trip OpenAI's content filter, and one bad call killed five configs

`icl_openai.py run` crashed with `openai.BadRequestError ... 'code': 'bio_policy'` ("flagged for
possible biological risk") partway through the full GPT-6 Luna run. GPQA-Diamond is about a third
biology, so this will happen again on any OpenAI frontier run over this dataset.

Two lessons:
- **The exception escaped `pool.map`, so the whole invocation died.** It had finished 817/1,178 rows
  of `ckpt k0`; the five configs queued behind it never started. Fixed: `call()` now catches
  `BadRequestError` and records the row as `status="refused"` with empty output. `icl_predict.parse`
  already returns NaN for unparseable text and the scorer drops those rows as parse failures, so a
  refusal costs one row, not a run.
- **My launcher printed `=== ALL 8 CONFIGS DONE ===` unconditionally**, so the log claimed success
  over a crash. The row counts (`wc -l` per raw jsonl) were the only honest signal. Always echo
  `EXIT CODE $?`, and check per-config row counts rather than the log tail.

## 2026-10-26 - a global "stop after 2 cache misses" guard cannot survive a long run

The second Luna resume stopped at 3,833 of 5,751 calls: `stop: spent $8.73 (budget $30.0), cache
misses 2`. The cache was working fine. `misses` was a single counter for the whole invocation with a
threshold of 2, so over thousands of calls two unlucky transient misses end the run - a near
certainty at that length.

Of the 1,453 "cached 0" calls in that log, all but ~2 were expected: every k=0 call (prompt under
OpenAI's 1,024-token cache floor) and exactly one per fold (the deliberate cache-warming first call).
Both were already exempt from the counter.

Fix: reset `misses` at the top of each fold. The guard exists to catch a broken cache *configuration*,
which misses every call in a fold; isolated transients now cannot accumulate across 5,751 calls.

Also: `echo "... $(date) EXIT CODE $?"` always prints 0 - the command substitution runs first and
overwrites `$?`. Capture `rc=$?` on its own line before building the message.

## 2026-10-26 - running as `yuvraj`, env.sh points the envs and model cache at /work (which is empty)

`env.sh` resolves `WORK_DIR` from the running user, so as `yuvraj` it sets
`ENVS_DIR=/work/users/y/u/yuvraj/envs` and `CACHE_ROOT=/work/.../cache`. Neither holds this
project's work: the Python envs and the Gemma / Qwen3.5 weights are in the repo
(`envs/`, `cache/huggingface`). A GPU job submitted without overrides dies instantly with
`ERROR: interpreter not found: /work/users/y/u/yuvraj/envs/icl/bin/python`.

Override both before `sbatch` (env.sh uses `${VAR:-default}`, so exported values win):

    export HF_HOME="$PWD/cache/huggingface" PY_ICL="$PWD/envs/icl/bin/python"

Outputs are safe either way: `icl_predict.py` uses paths relative to the submit directory
(`runs/...`), so results still land in the repo. Only the interpreter and the weights are affected.
`sbatch --test-only`'s start estimate was 4 days out while a node sat idle - it assumes every queued
job runs first. The real job started in seconds; don't take that estimate as a reason not to submit.

## 2026-10-27 - "drops 0.12 to 0.24" reads as a rising range

I wrote the ask-3 result as "remaining length drops 0.12 to 0.24" on a slide and in two docs. The user
read it as a value going UP from 0.12 to 0.24. It meant "the size of the drop is between 0.119 and
0.239", quoting the smallest and largest of three separate drops (k = 0/10/50).

When a span of numbers is a set of separate measurements rather than a range, list them with their
condition: "R2 falls by 0.12, 0.24, 0.17 at k = 0, 10, 50". Reserve "X to Y" for an actual interval.

Related trap in the same tables: four targets are R2 and `committed_by_here` is **accuracy**, and they
sit in adjacent columns. State the units when a number could be either, and give the majority-class
baseline (0.6275 mean over 10-90%) next to any commitment accuracy.

## 2026-10-27 - two prompt versions with similar names did different things

`nockpt` (round 3) hid the checkpoint label on the EXAMPLES only; the query still carried it. `blind`
(round 5) hides it everywhere. The user reasonably believed the professor's "remove checkpoint
location" ask had already been done, because `nockpt` existed and sounded like it.

The lesson is naming, not code: `nockpt` should have been `nockpt_examples`. When two variants differ
only in scope, put the scope in the name. The results differ a lot - `nockpt` scores HIGHER than
showing the label everywhere, while `blind` costs 0.12-0.24 R2 on remaining length.

## 2026-10-28 - "Nobody runs an other-model control" was only true of our 7 papers
The round-4 lit review (`_litreview/`, 2026-10-23) said 0 of 6 papers run an other-model control. A
wider search found Zur et al. 2025 (arXiv 2511.04527, "Are language models aware of the road not
taken?"): they probe Llama-3.2-3B states for the future answer distribution and use Gemma-2-2B on the
same text as a control (KL 0.11 vs 0.19). Also Bulut 2026 (arXiv 2609.03436) and Yuan et al. 2026
(arXiv 2605.09502) run difficulty / within-problem controls. Scope novelty claims to the papers
actually checked.

## 2026-10-29 - Pilot slice + rest in one RUN_TAG: give the second phase its own QUEUE_ROOT
`run_queue` (runs/gpqa25/common.sh) skips any shard with a `done.<s>` marker in `$QUEUE_ROOT/<stage>`.
A K-sufficiency pilot (SUBSET_KEEP=0) marks all its shards done, so the 80% phase in the same
queue would do nothing. Use e.g. `QUEUE_ROOT=$RUNS_ROOT/queue/<tag>_pC` and the SAME NSHARDS
(sharding is example_id % NSHARDS). Also: the 0% cut is not a continuation fraction - its labels
are the base samples (`labels_f0.py`), so its K = N_COTS.

## 2026-10-29 - the 0% cut's p_gold is a ready-made question-difficulty measure

`labels/checkpoint_labels_f0.csv` has one row per question with `p_gold` over `n_continuations`=25,
and those 25 are the base CoTs, so `p_gold` IS "what fraction of 25 attempts were correct". Any
difficulty stratification can be built from that file with no new compute. Banded table:
`runs/goal2/gpqa25/sweep/_difficulty/question_difficulty.csv`.

Two things that will bite anyone using it:

- **Difficulty vs entropy is NOT monotonic.** Mean entropy by band: v_easy 0.050, easy 0.508,
  medium 1.061, hard 1.028, **v_hard 0.642**. The model is confidently wrong on the hardest
  questions, so they look *certain*. Any analysis assuming "harder = more uncertain" is wrong at the
  top end.
- **`p_gold` is label-derived, so it is NOT a baseline.** It is computed from rollouts, the same kind
  of data the targets come from. Regressing a target on it measures shared variance, which is what
  you want for a confound question, but reporting it as "a scalar baseline beats our probe" would be
  wrong. The legitimate version is a difficulty estimate built only from the question text.

Also: if you subset to one difficulty band (~40 questions) and keep the 5 fixed folds, the smallest
(band, fold) cell is **3 questions**. Anything that fits per-fold parameters - the ICL scorer's
`actual = a + b * predicted` calibration - becomes unreliable. Pool folds or report raw too.

## 2026-10-29: HF token file unreadable for rishp
- `cache/huggingface/token` is owned by yuvraj and not readable by us, so any `load_dataset` from the Hub
  dies with PermissionError. Workaround: `HF_TOKEN_PATH=<nonexistent path>` (public datasets need no token).
- GlobalOpinionQA stores `options` (and `selections`) as Python-literal STRINGS, options can be floats
  (0.0..10.0), and up to 18 options per question -> needed A-R letters, not A-J.

## 2026-10-29 - the ICL row set cannot carry a between/within-question decomposition

The round-6 ask was to apply round 4's between-question / within-question split
(`src/analysis/redundancy.py` part C) to GPT-6 Luna's stored predictions. It is **degenerate there**.
`icl_predict.test_rows(d, 1)` keeps ONE CoT per question, so at every cut the ICL row set has exactly
196 rows and 196 questions: within-question SS is exactly 0.000, and every cut's R2 is 100%
between-question by construction. (Full `ref_fixed` set, for contrast: within share 0.183 / 0.291 /
0.394 / 0.473 / 0.572 at 10-90%.)

Consequence worth remembering when quoting round 5: **the published "Luna k=0 ties our probe on
entropy" comparison was never a test of within-question skill.** It only ever compared between-question
skill. Check rows-per-group before reaching for a variance decomposition.

The substitute that does work on those rows: partial a quadratic in the 0% cut's `p_gold` out of BOTH
the target and the prediction, within each cut, refitting inside each bootstrap draw. It works on all
196 questions instead of a 40-question band, so it has far more power than stratifying.

## 2026-10-29 - a band-restricted ICL rerun costs nothing, because the prompt does not depend on the band

Action 2 was budgeted at a few dollars to rerun Luna inside one difficulty band. It needed zero API
calls. A query prompt in `icl_openai.py` depends on the query row only, and `example_order(d, fold, k)`
draws examples from all training-fold questions with no reference to difficulty, so **restricting the
stored predictions to a band is exactly the same measurement as rerunning on that band's rows.** Check
whether a planned rerun would actually change any prompt before paying for it.

That also disposed of the calibration worry: `icl_cal` was fitted per fold on the FULL other-fold rows,
so subsetting afterwards refits nothing and the 3-question (band, fold) cell never arises. The only
version that would need new calls is drawing the EXAMPLES from the query's own band, which is a
different question.

## 2026-10-29 - R2 inside a narrow band punishes an offset the full-set fit never had to fix

Scoring the stored out-of-fold predictions inside the `v_easy` difficulty band gave entropy R2 of
-2.71 (ours), -3.16 (TF-IDF), -3.75 (prefix_len), -1.70 (Luna). That is not "the probe fails there".
The predictions come from a fit over all questions and are never recentred per band, and v_easy has
almost no entropy variance left (sd 0.265 against 0.642 overall), so a small prediction bias dominates
R2. Report the **offset-free within-cut Pearson correlation** alongside R2 whenever you subset to a
narrow slice: it showed entropy correlation falling about a third inside every band (ours 0.559 ->
0.23-0.42) while rem_len_mean correlation stayed flat (ours 0.753 -> 0.72-0.78).

## 2026-10-29 - verbatim-copy rates need a chance baseline or they are meaningless

Testing whether in-context models copy the example values, the raw exact-match rate looked damning:
Gemma's p_mode predictions matched one of the shown example values 94% of the time at k=93. The chance
rate is **95.5%**. `label_json` rounds p_mode and entropy to 2 decimals on a short range, so with 50-93
examples almost every representable value is present. The right null is the same model's k=0
predictions matched against the example values that fold would have been shown. With it: entropy has a
real excess of +0.08 to +0.29 in both families, p_mode's excess is about zero or negative, and the
length targets (integers in the thousands) are about zero.

Related: the dominant mechanism behind "examples hurt" is not copying. Within-cut correlation falls
versus k=0 in 28 of 28 comparisons while the prediction spread moves TOWARD the truth's spread in 19 of
28 and raw R2 improves in 15 of 28. **Examples supply the scale and break the ranking.** Since our
published comparison is calibrated, the scale gain is free and only the damage shows.

## 2026-10-29 - `committed_by_here` accuracy has almost no resolving power; always give AUC

Our probe and all nine in-context configurations sit within 0.03 of each other on commitment accuracy
(0.698-0.742 against our 0.727, majority baseline 0.634), which reads as "commitment ties", and that is
what round 5 reported. Under AUC our probe is at 0.773 and every ICL configuration is at 0.651-0.677,
a gap of 0.096 to 0.122 that is **significant in all nine**. Second time the AUC convention has flipped
a conclusion on this target.

Minor pandas trap hit while adding it: naming a column `corr` breaks `groupby(...).corr.mean()`,
because `DataFrameGroupBy.corr` is the correlation method. Renamed to `wcorr`.

## 2026-10-29 - negative R2, and why band-restricted intervals are so wide

Both came up reading the round-6 figures and both are real, not bugs.

**Negative R2.** R2 = 1 - SSE / SS_total, where SS_total is the variance of the target *inside the
subset being scored*. Negative simply means the prediction beats nothing: you would do better
predicting that subset's own mean. It happens here for two different reasons, and they should not be
confused:
- *Within-question scoring* (fig7's orange curve, round 4). Both truth and prediction are demeaned per
  question, so R2 = 0 is "predict the question's average". The probe's chain-to-chain wiggle has real
  magnitude but is nearly uncorrelated with the truth, so SSE = SS_y + SS_pred - 2*cov exceeds SS_y.
  The wiggle actively hurts. Entropy at 10%: -0.265.
- *Band-restricted scoring* (fig9's very-easy column). The predictions come from a fit over all 198
  questions and are never recentred inside the band. The very-easy band has almost no entropy variance
  left (sd 0.265 against 0.642 overall), so a small prediction offset dominates a tiny denominator:
  -2.7 for our probe, -3.2 for TF-IDF, -1.7 for Luna. This is mostly an offset artefact, NOT a
  statement that the probe is terrible there. Always pair it with the offset-free within-cut
  correlation, which for the same cell is a perfectly ordinary 0.234.

**Why the band intervals are so wide.** Measured, entropy, our probe: all questions n=196, half-width
0.072; inside a band n~39, half-width 0.21 to 0.33. Pure 1/sqrt(n) predicts only 0.16, so there is an
extra 1.3x to 2.0x on top. Freezing one half of the ratio at a time says where it comes from
(medium band, half-width 0.330): prediction error varying with the band's target variance frozen gives
0.251; the band's target variance varying with prediction error frozen gives **0.377**. So the
DENOMINATOR is the bigger driver, not the prediction. Note both parts exceed the full width, because
they move together (a resample with more spread also has more error), so the ratio is steadier than
either piece. On all 196 questions the same split is 0.076 / 0.068 against a full 0.072, i.e. nothing
is unstable there.

Two practical consequences: no single band is decisive on its own, and a metric with a FIXED
denominator (within-cut correlation) is the better companion at small n.

Related scale note: the ICL row set has 1 chain per question, so n is the question count exactly
(196 rows/cut, half-width 0.072). The full ref_fixed set has 198 questions x ~25 chains (~4,913
rows/cut) and only gets to 0.056, because the bootstrap resamples QUESTIONS either way. Extra chains
per question buy much less than extra questions.

## 2026-10-29 - positive pooled R2 with near-zero R2 in every difficulty band is not a contradiction

Asked why the band figure shows about zero everywhere while the headline is 0.309. Split the spread in
the truth into between-band and within-band parts and score each (196 questions, pooled over the five
cuts):

| | entropy | remaining length |
|---|---|---|
| share of spread BETWEEN the five bands | 33.0% | 16.5% |
| share WITHIN bands | 67.0% | 83.5% |
| probe R2 on the between part | +0.702 | +0.826 |
| probe R2 on the within part | +0.116 | +0.568 |
| pooled R2 | **+0.310** | **+0.610** |

Pooled R2 is exactly the variance-weighted sum: 0.330x0.702 + 0.670x0.116 = 0.310, and
0.165x0.826 + 0.835x0.568 = 0.610. On entropy the probe is excellent on the third of the spread that
is difficulty and near useless on the rest; on length the between part is only a sixth and the probe
still gets 0.568 on the rest, which is why stratifying does not hurt it.

Two cautions. The per-band R2 values are lower than the 0.116 and often negative because each band is
scored against its own small variance with predictions that were never recentred per band; the pooled
within-band number with a common recentring is the fairer summary. And this is the same arithmetic as
round 4's between/within-question decomposition, just with "difficulty band" as the group instead of
"question" - if a pooled score looks good, always ask which grouping is carrying it.

## 2026-10-29 - two checks any "it fails on this subgroup" claim needs

Asked whether the band figure retrains the probe (it does not: `ref_fixed/preds*.csv` are out-of-fold
predictions from a fit on all 198 questions, and a band is a row filter) and whether both predictors
are simply emitting the mean. Both questions have cheap, decisive answers, and both should be run
before presenting any subgroup breakdown.

**1. Refit the output map inside the subgroup.** Predictions fitted on the full population carry a
population intercept and scale that nobody recentred, so a subgroup score can look catastrophic for
that reason alone. Best probe, entropy, as-stored / +intercept / +intercept and slope: v_easy
-2.250 / -0.651 / **0.091**. So the -2.7 headline was almost entirely offset. But the conclusion
survives: even a best-case affine refit leaves entropy at 0.09-0.15 in every band against 0.289 over
all questions, while remaining length stays 0.42-0.57 against 0.524. Script
`src/analysis/icl_bands_refit.py`. (Refitting the 3,584 WEIGHTS inside a 39-question band was not
done and would be noise-dominated.)

**2. Look at the predictions themselves, not only the score.** "R2 is about zero" is consistent with
both "predicts a constant" and "predicts well but shrunk". Here it is the second: inside a band
sd(pred)/sd(true) is 0.47-0.55 for entropy, not 0, and the within-cut correlation is 0.19-0.43. The
actual failure is in the MEAN. True entropy by band (v_hard to v_easy) 0.807 / 1.086 / 1.086 / 0.627 /
**0.100**; the probe says 0.869 / 0.905 / 0.861 / 0.667 / **0.404** and Luna 0.947 / 0.941 / 0.910 /
0.656 / **0.365**. Both compress a 0.99-bit true range into about 0.50 and are four times too high on
the easiest questions. Remaining length behaves the opposite way, which is the whole result in one
table. Script `src/analysis/icl_bands_spread.py`, figure `fig12_predicting_the_mean.png`.

Report a predictor that is calibrated per fold (the ICL `icl_cal` rows) in BOTH raw and calibrated
form here, because the calibration partly sets the spread you are measuring.

## 2026-10-29 - fix for the subgroup offset artefact: recentre, do not drop to correlation

Following the band-offset finding above, `icl_bands.py` now emits `score_recentred` (+
`recentred_ci_lo/hi`) and fig9 plots THAT, not the raw subset R2. The correction matches the
prediction's mean to the truth's inside each band: one degree of freedom, slope untouched, so an
under-confident predictor is still penalised. Going all the way to correlation (two degrees of
freedom) would have hidden the under-confidence, which is the actual finding.

Effect on entropy, best probe (all / v_hard / hard / medium / easy / v_easy):
uncorrected +0.309 / +0.131 / +0.057 / -0.143 / +0.096 / **-2.71**
corrected   +0.311 / +0.148 / +0.174 / +0.108 / +0.110 / **-1.09**
The full-set number barely moves (0.309 -> 0.311, the out-of-fold predictions are nearly unbiased
there), which is the check that the correction is not doing anything sneaky. The conclusion is
unchanged: about 0.30 over all questions, 0.11 to 0.17 inside any band.

**The very-easy band stays negative, and that is a real result.** sd(prediction) there is 1.15x
sd(truth) while the correlation is only 0.30, and recentred R2 = 2*r*(sp/sy) - (sp/sy)^2, which is
negative whenever the predictor keeps emitting spread into a target that has none. Do not "fix" that
one by dropping the scale as well; report it.

Both columns are kept in `p5_bands_scores.csv` so the uncorrected values stay auditable.

## 2026-10-31: GEPA / DSPy setup
- `~/.frontier_api_key` for **rishp** has NO credit ("You have no credits remaining", 2026-10-31). The Luna
  runs of 2026-10-26 were made under the `yuvraj` account, presumably with its key.
- `pd.read_json` on rows.jsonl turns 0.3 into 0.30000000000000004: round `fraction` before using it as a key.
- DSPy's GEPA `auto` budget = 4V + 380 metric calls for 1 predictor at "light" (V = valset size; checked in
  `dspy/teleprompt/gepa/gepa.py:auto_budget`, minibatch term uses 35 although actual minibatches are 3).
- Offline check with `DummyLM`: the reflection dummy must use `adapter=dspy.JSONAdapter()`.

## 2026-11-01 - handoff check: `git fetch` fails as `yuvraj`, so "up to date" is not trustworthy

Confirmed again on 2026-11-01: as user `yuvraj`, `git fetch` dies with "Please make sure you have the
correct access rights" (no GitHub SSH key on this account). The consequence to remember is not the
failure itself but that **`git log HEAD..origin/main` then prints nothing and looks like "up to date"**,
because it compares against a stale local `origin/main` ref. Do not read that as confirmation before
committing. Re-check from an account that can actually fetch.

Also: several sessions genuinely do work here at once. On 2026-11-01 the working tree held this
session's round-6 analysis, three earlier sessions' uncommitted work, and a fourth session's GEPA prep
owned by `rishp`. `CLAUDE.md`'s "Not committed yet" is now grouped by owner for that reason. Do not
commit another session's files without asking.
