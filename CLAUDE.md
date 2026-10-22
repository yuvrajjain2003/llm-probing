# CLAUDE.md

## Rules (always follow — these take priority)
- **Brief replies.** Plain language, little jargon, clear over clever.
- **No assumptions.** If something is unknown, check it or ask. Say "I don't know" rather than guess.
- **Verify claims.** Back every statement about code, data, or results with something you actually ran or read. Say what you checked.
- **Minimal code.** Write the simplest code that works. No extra features, layers, or "just in case" options.
- **Minimal changes.** When editing existing code, change only what's needed. No drive-by refactors or reformatting.
- **Don't run long jobs without asking.** Use `sbatch --test-only` to check a submission without running it.
- **Never `git push` without asking the user explicitly first.** Commit only when asked — usually near the end of a session, when the user says so.
- **Read `learnings.md`** at the start of each session, and add to it when we learn something or make a mistake.
- **Log every result** in "Results index" (path, what made it, status) as soon as it exists.
- **Keep this file current.** At the start of each session, re-check "Repo state" below and update it. Mark tasks done in "Current work" as they finish.

## Repo state
_Last updated: 2026-10-15_

**New session? Read `docs/HANDOFF.md` first** (where things stand, how the methods work, how to run things, open items). Then `learnings.md` (pitfalls).

- **Folder:** `/proj/ssrivalab/projects/llm-probing` (group-shared, owned by user `yuvraj`). Git repo: `git@github.com:yuvrajjain2003/llm-probing.git`, branch `main`, last pushed commit `56e71b0` (2026-10-12, = origin/main as of 2026-10-15). Uncommitted local work is listed in `docs/HANDOFF.md`.
- **Code:** `src/` (Python pipeline), `runs/gpqa25/*.sh` (current Slurm pipeline, stages 00–12; 10 embed, 11 ICL, 12 LoRA), `configs/sweep_*.yaml` (probe sweeps for `src/probing/sweep.py`), `src/baselines/` (ICL), `runs/goal1/full/`, `runs/goal2/` (older launchers), `env/` (setup), `env.sh` (paths).
- **Project docs** (from the repo — read when needed): `.claude/README.md` (start here), `STATE.md` (results as of 2026-09-18, old UNC box), `NEXT.md` (agreed plan), `PIPELINE.md`, `SETUP.md`, `GOTCHAS.md`. The repo's original instructions are kept below under "Project notes".
- **Not in git (local only):** `cache/` (~70 GB: HF model weights, pip, conda), `data/gpqa_diamond/`, `runs/**` outputs (~24 GB), `learnings.md`.
- **Results present here** (Sycamore runs, 2026-09-20 to 09-28, newer than `STATE.md`):
  - `runs/goal1/gpqa25/` — GPQA-Diamond, 198 questions × 25 CoTs, Qwen-7B. pass@1 0.487, majority-vote 0.551.
  - `runs/goal2/gpqa25/` — checkpoint activations, probes (`probes_ckpt/fast/`), ablation CSVs, capture check (PASSED).
  - `runs/*/smoke/` — small test runs.
- **Paths:** `source env.sh` sets everything. We have no `/work/users/r/i/rishp`, so `env.sh` keeps all files in this folder: `cache/` (Qwen-7B weights already there), `envs/` (Python envs), `runs/` (outputs). `cache/` and `envs/` are gitignored. (Yuvraj has a `/work` dir, so his paths are unchanged.)
- **Hardcoded paths:** none left in live code (checked `src/`, `runs/`, `env/`, `env.sh`). `archive/` still has old `/playpen-ssd/yuvraj` paths — it's reference-only code from an older machine, left as is.
- **Python envs** (can't be merged — see `.claude/SETUP.md`): `envs/goal1` (`$PY`: generation + all analysis, + peft for LoRA) and `envs/easysteer` (`$PY_ES`: activation capture only); `envs/icl` (vLLM 0.31, for Qwen3.5/Gemma 4 ICL; NOT set by env.sh — use `$ENVS_DIR/icl/bin/python`). Build logs: `runs/logs/setup/`.

## Slurm / cluster (Sycamore, UNC)
- Account: `rc_ssriva_pi`, QOS: `normal`. Login node has no GPU — don't run heavy work there.
- **GPUs:** use partition `h100_all` — 4 nodes × 4 H100 (16 total), 256 CPUs and ~1.5 TB RAM per node. Max 5 days. (`h100` partition is NOT allowed for us.) GPUs are heavily shared; on 2026-10-03 all 16 were busy with ~8-day estimated wait.
- **CPU partitions:**
  | partition | min CPUs | max time | notes |
  |---|---|---|---|
  | `small` | 2 | 5 days | up to 192 CPUs/user |
  | `batch` (default) | 48 | 5 days | |
  | `inter` | 1 | 8 h | interactive |
  | `debug` | — | 4 h | 2 nodes |
- Default time limit is 2 h if `-t` not set. Default memory is 512 MB/CPU — always set `--mem`.
- `--exclusive` is not allowed.
- No access: `h100`, `hipri`, `CMSlab`.
- Software via modules: `module load anaconda/2025.12-2`, `cuda/12.9` / `cuda/13.x`.
- Example GPU job (past runs used 1 GPU, 8–16 CPUs, 96–128 GB):
  ```bash
  sbatch -p h100_all --gres=gpu:1 -c 8 --mem=96G -t 02:00:00 job.sh
  ```
- Useful: `squeue -u $USER`, `sacct -j <id>`, `sbatch --test-only ...` (dry run), `sinfo -a` (shows hidden partitions).

## Working alongside yuvraj (checked 2026-10-04)
- His runs write to his own `/work/users/y/u/yuvraj/llm-probing/runs` (his `env.sh` resolves there — verified by sourcing as his user). Ours write only to `runs/` in this folder. No yuvraj jobs on this cluster or files in `runs/` since 2026-10-01, so his current RF runs are elsewhere.
- Our sweep outputs live only in `runs/goal2/gpqa25/sweep/<name>/` (names new to us: `ref_fixed`, `4a_*`, `4b_*`, `4c_*`, `feat_*`, `emb_*`). Never write into `runs/goal2/gpqa25/{probes_ckpt,ablation}` (his).
- **Git:** before any commit/push, `git fetch` and check `git log HEAD..origin/main`; if he pushed, rebase/merge and look at conflicts with the user. Never force-push. Files we changed that he may also edit: `src/generation/checkpoint.py`, `src/probing/extract_checkpoints.py`, `env.sh`, `.gitignore`, `CLAUDE.md`.

## Results index
_Where every output lives. Add a line whenever a new result is produced. All paths relative to repo root; all under `runs/` (gitignored, local only)._

| what | path | produced by | status |
|---|---|---|---|
| Labels, 10–90% checkpoints | `runs/goal1/gpqa25/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B/labels/checkpoint_labels.csv` | yuvraj, Sep 2026 | done |
| Labels, 0% checkpoint (198 rows, cot_id −1) | same dir, `checkpoint_labels_f0.csv` | `src/analysis/labels_f0.py` | done 2026-10-04 |
| Activations, 10–90% (all 28 layers) | `runs/goal2/gpqa25/activations_ckpt/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B/` | yuvraj, Sep 2026 | done |
| Activations, 0% (198, 28, 3584) | `runs/goal2/gpqa25/activations_ckpt_f0/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B/` | `runs/gpqa25/06b_extract_f0.sh`, job 4672844 | done 2026-10-04 |
| Old probe results (old per-checkpoint folds) | `runs/goal2/gpqa25/probes_ckpt/fast/`, `runs/goal2/gpqa25/ablation/` | yuvraj | reference only |
| Sweep sanity check (old folds; = old results) | `runs/goal2/gpqa25/sweep/baseline/results.csv` | `configs/sweep_baseline.yaml` | done 2026-10-04 |
| **Reference: ridge/logistic on L21, prefix_len, TF-IDF (fixed folds)** | `runs/goal2/gpqa25/sweep/ref_fixed/results.csv` | `configs/sweep_ref.yaml`, jobs 4672913–17 | done 2026-10-04 |
| Qwen3-Embedding-8B embeddings (24,773 × 4096, L2-normalised, incl. 0% rows) | `runs/goal2/gpqa25/embeddings/qwen3_emb_8b.npy` + `.keys.csv` | `runs/gpqa25/10_embed.sh`, job 4672850 (82 min, 1 H100) | done 2026-10-04 |
| Baseline 1: ridge/logistic on Qwen3-8B embeddings | `runs/goal2/gpqa25/sweep/emb_qwen3_8b/results.csv` | `configs/sweep_emb.yaml`, jobs 4672979–83 | done 2026-10-04: activations beat embeddings on every target/checkpoint (mean R² gap: entropy +0.11, p_mode +0.08, rem_len_mean +0.17, rem_len_std +0.12); embeddings also below TF-IDF. Caveat: logistic C at grid edge (1e-4) in 46/50 folds → extend grid to 1e-6 before trusting `committed_by_here` |
| Step 4 smoke test (all probe types, 20 questions) | `runs/goal2/gpqa25/sweep/smoke_all/` | `configs/sweep_smoke_all.yaml`, job 4672942 | done 2026-10-04 (all run; MLP on PCA-8 underfits: train R² 0.01) |
| Step 4b: PCA-k → ridge/logistic | `runs/goal2/gpqa25/sweep/4b_pca/results.csv` | `configs/sweep_4b_pca.yaml`, jobs 4672945–49 | done 2026-10-04: every k loses to full 3584; k=5–15 keeps ~2/3 of entropy R² (0.20 vs 0.29) |
| Embedder research | `docs/embedders.md` | step 2 | done 2026-10-04 |
| Step 4a diagnostic: MLP training settings (h=8, 50% ckpt, entropy) | `runs/goal2/gpqa25/sweep/4a_diag/` | `configs/sweep_4a_diag.yaml`, job 4672951 | done: needs early stopping (no-stop → test R² 0.00); lr 1e-4 best (0.274 vs ridge 0.298) |
| Step 4a MLP ablation (7 hidden × 4 dropout × 4 wd × 3 seeds, no selection) | `runs/goal2/gpqa25/sweep/4a_mlp_ablation/results.csv` (8400 rows) | `configs/sweep_4a_mlp_ablation.yaml`, jobs 4672956–65 | done 2026-10-04: hidden size barely matters (h=5 ≈ h=256); dropout ±0.005; **weight decay had NO effect** (identical to 3 dp — AdamW wd×lr 1e-4 too small with early stopping; grid needs ≥1 or an L2 loss term); seed std ~0.01 |
| Step 4a MLP fair score (per hidden size; dropout×wd picked on inner folds) | `runs/goal2/gpqa25/sweep/4a_mlp_selected/results.csv` | `configs/sweep_4a_mlp_selected.yaml`, jobs 4672966–75 | done 2026-10-04: **no MLP beats ridge** (best hidden size vs ridge: mean −0.015, 3/25 tiny wins). Entropy 0.255–0.266 vs ridge 0.289. Supervised 5-dim MLP ≫ PCA-5 (0.256 vs 0.095) |
| Step 4a weight-decay rerun: ablation (7 h × wd {1,10,100} × dropout {0,0.3} × 3 seeds) | `runs/goal2/gpqa25/sweep/4a_mlp_wd_ablation/results.csv` (3150 rows) | `configs/sweep_4a_mlp_wd_ablation.yaml`, jobs 4677644–48 | done 2026-10-04: wd now matters. wd=10 best for small h (entropy h=5: 0.253→0.271); wd=100 too strong for h≤32 but best for h=256 (0.281) |
| Step 4a weight-decay rerun: fair score (dropout × wd {0,1,10,100} selected) | `runs/goal2/gpqa25/sweep/4a_mlp_wd_selected/results.csv` | `configs/sweep_4a_mlp_wd_selected.yaml`, jobs 4677649–53 | done 2026-10-04: MLPs now ≈ ridge, still not better (best per cell vs ridge: mean −0.003, 9/25 wins, max +0.004). wd=10 picked 72% of folds. Best: h=256 entropy 0.283 vs ridge 0.289 |
| Step 4c random forest: ablation (2550 rows) + fair score (100 rows) | `runs/goal2/gpqa25/sweep/4c_rf_{ablation,selected}/results.csv` | `configs/sweep_4c_rf_*.yaml` (generators in `runs/sweep_tmp/`), jobs 4677654 (timing), 4677693–99, 4677989–91 | done 2026-10-04: **below ridge everywhere**. Best RF (raw or PCA-64): entropy 0.235, rem_len_mean 0.404 (ridge 0.289 / 0.524). max_features is the main knob; small leaves best; seed spread ≤0.003. Not run: raw max_features 0.3, max_depth 16 |
| Step 4c' gradient boosting (HGB): ablation (2400 rows) + fair score (100 rows) | `runs/goal2/gpqa25/sweep/4c_hgb_{ablation,selected}/results.csv` | `configs/sweep_4c_hgb_*.yaml`, jobs 4677835 (cancelled, rows kept), 4677900–02, 4678022–24 | done 2026-10-04: **best tree model, still below ridge**. HGB raw: entropy 0.258, rem_len_mean 0.474, committed 0.734. Nearly insensitive to all hps (≤0.006); PCA inputs worse than raw |
| F2LLM-v2-8B embeddings (2nd embedder, same method as Qwen3) | `runs/goal2/gpqa25/embeddings/f2llm_v2_8b.npy` | `10_embed.sh`, job 4677668 (85 min) | done 2026-10-04 |
| Other-LLM control: Qwen2.5-Math-7B (our model's base) layer 21, last token, inputs cut to last 4096 tokens (26/44/55% of 50/70/90% inputs cut) | `runs/goal2/gpqa25/embeddings/qwen25_math_7b_L21.npy` | `10_embed.sh --layer 21 --no-instruct --max-len 4096`, job 4677669 (38 min) | done 2026-10-04 |
| Other-LLM control: Qwen2.5-7B-Instruct layer 21, last token, full inputs | `runs/goal2/gpqa25/embeddings/qwen25_7b_instruct_L21.npy` | `10_embed.sh --layer 21 --no-instruct`, job 4677670 (66 min) | done 2026-10-04 |
| Probes on the 3 features above (ridge + logistic C 1e-6..10), + Qwen3-emb logistic with the wider grid | `runs/goal2/gpqa25/sweep/feat_<name>/results.csv` | `configs/sweep_feat_<name>.yaml`, jobs 4677671–90 | done 2026-10-04: **other-LLM controls ≈ our model's activations** (mean over ckpts — entropy: ours 0.289, Qwen2.5-Math 0.283, Qwen2.5-Instruct 0.287; rem_len_mean 0.524 / 0.523 / 0.518; ours wins only 10–11/20 cells, mean gap ≤0.002). F2LLM-v2-8B worse than Qwen3-emb (entropy 0.157). Wider logistic grid didn't help Qwen3-emb (0.704 vs 0.707); C now mostly inside the grid |
| **Interim update doc + 5 figures** | `docs/update_2026-10-05.md`, `docs/figures/fig{1..5}_*.png` | `src/viz/update_figures.py` (reads sweep CSVs) | done 2026-10-05 |
| **R5/R6/R7: error bars (question bootstrap, 2000 draws), train vs test, MSE** — all from existing results/preds, no refitting | `runs/goal2/gpqa25/sweep/_round2/{chosen_probes.csv,r5_ci/,r6_overfit/,r7_mse/}`, figs `docs/figures/round2/r{5_ci_forest,6_train_vs_test,7_r2_vs_mse}.png` | `src/viz/round2_metrics.py`, job 4776925 (rerun 4776941 for legend) | done 2026-10-09: ours sig. > TF-IDF/prefix_len/Qwen3-emb/F2LLM on all 5 targets (entropy +0.05/+0.07/+0.11/+0.13); NOT sig. different from Qwen2.5-Instruct/Math controls (±0.006) or strong-wd MLP; sig. > HGB. Prefix_len ≈ ours for rem. length at 10–50% (n.s.), ours wins at 70/90%. CI half-width ≈ ±0.05 R². Overfitting: RF worst (entropy train 0.82 vs test 0.24), TF-IDF next; ridge gap 0.2 but best test. MSE ranks = R² ranks except mean-over-ckpt rem. length (prefix_len slightly lower MSE — early ckpts dominate MSE) |
| R4: embedder trained length vs our input lengths | `runs/goal2/gpqa25/sweep/_round2/r4_embed_len/` (`length_stats.csv`, `length_bins.csv`), section appended to `docs/embedders.md` | job 4776919 | done 2026-10-09: nothing > 32k (no truncation). F2LLM training code runs at 1–2k tokens (repo config 1024; v2-8B's own config not seen); Qwen3-Emb training length unpublished. Our inputs >2048 tokens: 39/60/69/75% at 30/50/70/90%. No clear evidence embeddings degrade with length (emb/act MSE ratio U-shaped over length terciles). Optional: re-embed F2LLM truncated to 2048 (~45 min GPU, est.) |
| R8 prep: LoRA+MLP-head probe code + CPU smoke test (tiny random model) | `src/probing/lora_probe.py`, `runs/gpqa25/12_lora.sh` (MODE=check / pilot / main), smoke → `runs/goal2/gpqa25/lora/smoke/` | job 4776948 | prep done 2026-10-09, NOT run on GPU. peft 0.21.2 in goal1 (--no-deps; torch/transformers unchanged). Design: ONE pooled model per fold (all checkpoints; scored per checkpoint) — deviates from per-checkpoint probes, user to confirm. Est. pilot 1–1.5 GPU-h, full 14–20 GPU-h (assumed throughput). Run `check` first (single- vs double-BOS match to stored activations unverified) |
| R2: model bases (configs, tokenizers, weight similarity of R1-Distill, Qwen2.5-Math-7B, Qwen2.5-7B-Instruct, Qwen2.5-7B base) | `runs/goal2/gpqa25/sweep/_round2/r2_model_bases/{summary.csv,tensor_similarity.csv,config_tokenizer.txt}`, fig `docs/figures/round2/r2_weight_similarity.png` | `src/analysis/compare_model_weights.py`, jobs 4776915 (download Qwen2.5-7B) + 4776974 | done 2026-10-09: same architecture + tokenizer (ids) in all 4. Weights: R1-Distill vs Math rel. diff 0.14 (fine-tune of Math ✓); Instruct vs Base 0.013 (light post-training); Math vs Base 0.79 (far — heavy continued pretraining, cause not verified); R1-Distill vs Instruct 0.79. So the other-LLM controls share lineage/tokenizer but NOT near-identical weights. rope_theta 1e4 (R1-Distill, Math) vs 1e6 (Instruct, Base) |
| R3: why RF < ridge (entropy + rem_len_mean, all ckpts) | `runs/goal2/gpqa25/sweep/_round2/r3_rf/r3_f*.csv`, `w_f*.csv`; figs `docs/figures/round2/r3_fig{1,2,3}_*.png` | `src/probing/r3_rf_diagnosis.py`, `src/viz/r3_figures.py`; jobs 4776944–57 (~20 CPU-h) | done 2026-10-09: **structural**. Ridge weights are dense (~1,600 of 3,584 dims hold 90% of weight energy). RF can't fit a known linear function on this X (noise-free synthetic: RF 0.68 vs ridge 0.92). Not overfitting-limited (leaf size 1→20 cuts train R² 0.90→0.58, test flat). Giving RF the ridge direction as 1 feature closes ~50% (entropy) / ~90% (rem_len) of the gap but never beats ridge; RF on PLS-5 = linear PLS-5. Not run: other targets, CIs |
| R1 prep: ICL env + models + prompts + plans (no GPU yet) | env `envs/icl` (build: `env/setup_icl.sh`; vLLM 0.31.0, torch 2.13+cu130, transformers 5.17); models `Qwen/Qwen3.5-9B`, `google/gemma-4-31B-it` in cache; code `src/baselines/icl_predict.py` (plan/dry/run/score), GPU script `runs/gpqa25/11_icl.sh`; plans/token counts `runs/goal2/gpqa25/icl/`; dry-run prompts `runs/logs/round2/icl_plan_*.log` | R1 prep agent; `--test-only` 4776995–97 | prep done 2026-10-09. Max k (exact token counts, min over folds): ours 40, Qwen3.5-9B 92 (native 262k; 370 with YaRN), Gemma-4-31B-it 93. Thinking off by default (R1: prefill `</think>`). Est. cost: subsample (1 CoT/question × 6 ckpts ≈ 1,178 rows) ≈ 4–5 GPU-h total; full set ≈ 75 GPU-h. Unverified: prefix caching on hybrid/sliding-window models (main cost risk), loading, JSON quality |
| R8 GPU check: LoRA model (no training) layer-21 state vs stored activations | `runs/logs/round2/lora_check_job4777564.log` | `12_lora.sh MODE=check`, job 4777564 (6 min) | done 2026-10-09: PASS, cosine min 0.9989 / mean 0.9996 (6 CoTs × 5 ckpts; single BOS OK). Pilot job 4777635 (0.5 ckpt, fold 0, 1 epoch) submitted after it |
| R8 pilot: LoRA+MLP head, 50% ckpt, fold 0, 1 epoch | `runs/goal2/gpqa25/lora/pilot/{results,preds}.csv`, log `runs/logs/round2/lora_pilot_job4777635.log` | `12_lora.sh MODE=pilot`, job 4777635 (37 min, peak 15.7 GiB) | done 2026-10-09: ≈ ridge on the SAME 993 test rows (LoRA vs ridge: entropy 0.383 vs 0.367, p_mode 0.324/0.320, rem_len_mean 0.646/0.627, rem_len_std 0.386/0.407, committed acc 0.744/0.758). Early stopping not triggered (best val at last step) → more epochs may help. Fold 0 is easier than average (ridge pooled 0.298 entropy) |
| R1 GPU smoke tests v1 (20 rows, k=10 + max, thinking off/on) | `runs/goal2/gpqa25/icl/raw/smoke_v1/`, logs `runs/logs/round2/icl_*_job47776*.log`, `*4777694*`, `*477774[2-6]*` | jobs 4777694, 4777742–46 (first try 4777558–63 failed: nvcc missing → fixed in 11_icl.sh) | done 2026-10-09. JSON answers: Qwen3.5 off 40/40, Gemma off 40/40, Gemma on 40/40; Qwen3.5 on 19/40 (hit 4k think cap); OUR MODEL 0/40 off and on (it solves the GPQA question instead). Prefix caching works (k=92: 8 rows in 26 s). Fix → v2: instruction repeated after the query, JSON forced via vLLM structured outputs (after thinking, via reasoning parser), think cap 16k (shrunk to fit ctx) |
| R1 GPU smoke tests v2 (instruction repeated + forced JSON via reasoning parser) | `runs/goal2/gpqa25/icl/raw/smoke_v2/` | jobs 4778425–30 | done: off — Qwen/Gemma 40/40, ours 35/40 (k=10 copies an example's values; k=40 some runaway whitespace / nonsense values). On — Gemma's reasoning parser skipped thinking entirely (JSON only); Qwen k=92 7/20 and ours k=40 1/20 (thinking runs to the cap). → v3 |
| R1 GPU smoke tests v3/v4 (FINAL setup): instruction repeated after query; answer forced to a compact-JSON regex (vLLM structured outputs); thinking on = two passes (think until end tag or 16k cap, then append end tag + forced answer) | `runs/goal2/gpqa25/icl/raw/*_smoke_fold*.csv` (older: `raw/smoke_v1..v3/`) | jobs 4778786–91 (Qwen, Gemma), 4779284–85 (ours, regex) | done 2026-10-09: 100% answers parse for all 3 models × off/on. Thinking: Gemma ~1.3k tokens, never capped; Qwen k=92 mostly hits 16k cap; ours k=40 hits cap. OUR MODEL's answers are degenerate (k=10 often copies an example's values; k=40 repeats one number in every field, e.g. all 12 or 1017) — a real result, not a bug. Smoke timings (6–8 rows, max k): ours off 8 s, Qwen off 26 s, Gemma off ~100 s; ours on ~250 s, Qwen on ~380 s, Gemma on ~320 s |
| R1 FULL runs, thinking off (subsample 1,178 rows, k {0,10,30,max}, prompt without model name) | raw `runs/goal2/gpqa25/icl/raw/<model>_off_k<k>_fold<f>.csv`; scored `icl/{preds,results}_<model>_off_k<k>.csv`; comparison `src/baselines/icl_compare.py` | jobs 4787229 (ours, 31 min), 4787230 (Qwen, 36 min), 4787231 (Gemma, 2 GPUs, 79 min); `icl_predict.py score` | done 2026-10-10. 0 parse failures. Same rows, 10–90% mean, calibrated (linear map fit on other folds): **Gemma k=0 best ICL** — entropy 0.262, p_mode 0.237, rem_len_mean 0.283, rem_len_std 0.069, committed acc 0.742; ridge on our activations 0.309/0.261/0.563/0.328, logistic 0.727; TF-IDF 0.267/0.234/0.415/0.221/0.734; majority acc 0.634. More examples HURT Gemma at 10–90% (k=93 entropy 0.158) but help a bit at 0% (0.065→0.187). Qwen3.5-9B ≈ 0 R² at every k. Our model: degenerate/huge outputs (e.g. 420230), R² ≪ 0. Raw (uncalibrated) R² < 0 everywhere except entropy for Gemma k=0 (off 0.139, on 0.010); raw committed acc up to 0.735 (Gemma off k0) — checked 2026-10-13 from icl/results_*.csv. CIs (question bootstrap, 2000 draws, paired vs ridge/logistic; `icl/ci_off_vs_probes.csv`, job 4787402): Gemma k=0 vs ridge n.s. for entropy (−0.047 [−0.126, 0.032]), p_mode, committed (+0.015 [−0.018, 0.049]); sig. worse for rem_len mean/std (−0.28/−0.26). All other ICL configs sig. worse than ridge on most targets. Caveat: only 196 rows/ckpt here, so TF-IDF vs ridge is also n.s. on entropy (was sig. in R5 with all rows) |
| R1 FULL runs, thinking on (same rows/k; --max-num-seqs ours 32, Qwen 48, Gemma 16; think cap 16k) | raw `runs/goal2/gpqa25/icl/raw/<model>_on_k<k>_fold<f>.csv`; scored `icl/{preds,results}_<model>_on_k<k>.csv`; CIs for ALL ICL configs (off+on) `icl/ci_vs_probes.csv` (`src/baselines/icl_compare.py`, job 4790490) | jobs 4787398 (ours, 13.0 h), 4787399 (Qwen, 18.1 h), 4787400 (Gemma, 2 GPUs, 15.2 h) | done 2026-10-11. Parse failures ≤0.1%. Median thinking / % capped at 16k — ours k0 940/8%, k40 16k/81%; Qwen k0 4.4k/6%, k92 16k/84%; Gemma ~1.3–1.6k/≤4%. **Thinking does not help**: best thinking-on config = Gemma k=30 (entropy 0.200, p_mode 0.180, committed 0.729) < Gemma thinking-off k=0 (0.262/0.237/0.742). All thinking-on configs sig. worse than ridge on entropy and rem. length; committed ties. Qwen on ≤0.105 entropy; ours on still ≪0 at k≤10, ≈0 at k30/40 |
| R8 single-checkpoint test (user 2026-10-11): LoRA + LINEAR head (ridge-like, `--hidden 0`, probe `lora_linear`) vs LoRA + MLP head (h=256), 50% ckpt, 5 folds, 3 epochs | `runs/goal2/gpqa25/lora/f05_{linear,mlp}/` | `12_lora.sh` with `FRACTIONS=0.5` (ignored — ran 0.1) `LORA_EXTRA="--folds-run <f> [--hidden 0] --out-dir ..."`, jobs 4790496–505 (one per head × fold) | done — see next row |
| Regularised RF (user 2026-10-11: train ≈ test): raw L21, max_features 0.1, 300 trees, min_samples_leaf {100,200,400} × max_depth {None,6,3}, every point scored (no selection), all 5 targets × ckpts | `runs/goal2/gpqa25/sweep/rf_reg/` | `configs/sweep_rf_reg.yaml`, jobs 4790514–18 (`small`, 12 CPUs; first try 4790507–11 failed: config lacked `runs:`) | done 2026-10-11 (~23 min/shard): **regularising closes the gap only by lowering test**. Entropy (train/test, mean over ckpts): leaf 100 0.32/0.21, leaf 200 0.25/0.19, leaf 400 0.19/0.15 (ridge 0.50/0.29; 4c leaf 5 0.83/0.24). rem_len_mean: 0.43/0.32 → 0.26/0.22 (ridge 0.70/0.52). Committed acc: 0.78/0.72 → 0.72/0.70 (logistic 0.81/0.74). max_depth 6 ≈ None; depth 3 slightly lower. Context: 4c leaf 1→50 already cut train 0.90→0.41 with test flat — test is bias-limited (R3) |
| R8 single-checkpoint result (fraction **0.1**; common.sh overwrote FRACTIONS → fixed with LORA_FRACTIONS): LoRA + linear head vs LoRA + MLP head vs ridge/logistic, same rows | `runs/goal2/gpqa25/lora/f01_{linear,mlp}/` (renamed from f05_*), CIs `lora/ci_f0.1.csv`, script `src/probing/lora_compare.py` | jobs 4790496–505 (13–30 min each) | done 2026-10-11. **Both LoRA heads WORSE than ridge** (pooled test; ridge 0.278/0.251/0.626/0.304/0.767): MLP head 0.189/0.151/0.591/0.286/0.755 (entropy −0.09 [−0.14,−0.04], p_mode −0.10, rem_len_mean −0.035 sig.; rem_len_std, committed n.s.); linear head 0.069/0.083/0.558/0.237/0.744 (sig. worse on 4/5). Overfits (linear train entropy 0.64 vs test 0.07) and very fold-unstable (linear entropy −0.09…0.25 across folds). Early stopping triggered (200–480 steps) |
| R8 single-LAYER LoRA (user 2026-10-11): adapters on decoder layer 21 only (`--lora-layers 21`, new flag → peft layers_to_transform [20]; checked: only layer index 20 adapted), rank {4,16,64} × head {linear, MLP}, fraction 0.1, 5 folds | `runs/goal2/gpqa25/lora/f01_L21_r<rank>_<head>/` | `LORA_FRACTIONS=0.1 LORA_EXTRA="--lora-layers 21 --rank <r> [--hidden 0] --out-dir ..." sbatch -t 06:00:00 runs/gpqa25/12_lora.sh`, jobs 4790662–67 | done 2026-10-11 (~25 min/job, 5 folds; ~5 min/fold). CIs `lora/ci_f0.1.csv` (all f01_* runs). **None beats ridge** (0.278/0.251/0.626/0.304/0.767). MLP head: r4 0.182/0.141/0.528/0.235/0.757, r16 0.206/0.195/0.561/0.272/0.772, r64 0.220/0.192/0.520/0.207/0.768 — sig. below ridge on entropy, p_mode, rem_len_mean; committed ties. LINEAR head collapses: entropy −0.11…−0.13, p_mode < 0, rem_len_mean 0.31–0.38 (all ranks) — worse than all-layer linear (0.069). Linear head's val loss 1.45–1.55 vs MLP 1.24–1.28 → suspect the SGD-trained linear head (lr 1e-3, wd 0.01, no ridge-strength L2), not LoRA. Rank barely matters for MLP |
| R8 no-LoRA CONTROL: same heads + same training recipe (lr 1e-3 head, wd 0.01, 16/step, ≤3 epochs, early stop) on the STORED layer-21 states, fraction 0.1, 5 folds (`lora_probe.py --frozen`, CPU) | `runs/goal2/gpqa25/lora/f01_frozen_{linear,mlp}/`, CIs in `lora/ci_f0.1.csv` | jobs 4790705–06 (`small`, ~3 min each) | done 2026-10-11. **The training recipe is the bottleneck, not LoRA**: frozen linear head entropy −0.254, p_mode −0.242, rem_len_mean 0.316, committed 0.709 (ridge on the SAME features 0.278/0.251/0.626/0.767); frozen MLP 0.123/0.069/0.534/0.750. LoRA improves on its own frozen control (linear −0.25 → −0.11…+0.07; MLP 0.12 → 0.18–0.22) but the recipe starts far below ridge. Step 4a already found good settings for SGD heads on these features (lr 1e-4, wd 10, patience 30, ≤300 epochs → MLP ≈ ridge) |
| R8 head-recipe fix on frozen L21 (step-4a settings: head lr 1e-4, batch 256, eval every epoch, patience 30, ≤300 epochs, wd {1,10}), linear + MLP head, fraction 0.1 | `runs/goal2/gpqa25/lora/f01_frozen_4a_{linear,mlp}_wd{1,10}/`, CIs in `lora/ci_f0.1.csv` | `lora_probe.py --frozen --head-lr 1e-4 --wd <wd> --accum 256 --eval-every 13 --patience 30 --epochs 300 [--hidden 0]`, jobs 4790854–56, 4790858 (`small`, 4–8 min; first tries 4790848–53 failed: missing HF_TOKEN, then outlines sqlite cache clash when started together) | done 2026-10-11. **Much better than the old recipe, still below ridge** (ridge 0.278/0.251/0.626/0.304/0.767). MLP wd10 0.220/0.191/0.602/0.289/0.763 (= best LoRA run; entropy −0.058 [−0.081,−0.037], p_mode −0.060, rem_len_mean −0.024 sig.; rem_len_std, committed n.s.); MLP wd1 0.212/0.188/0.593/0.283/0.763; linear wd10 0.154/0.125/0.584/0.221/0.740; linear wd1 0.121/0.086/0.555/0.193/0.738. Early stop at epoch 35–55 (MLP train loss ~0.4 vs val ~1.29). So frozen MLP with this recipe already matches every LoRA run so far |
| R8 RIDGE ON LoRA FEATURES (user 2026-10-11): layer-21 LoRA rank 16 + MLP head (old recipe) retrained with `--save-states` (adapted L21 state of every row, per fold), then ridge/logistic with the ref_fixed grids + inner CV on fold i's states (from the LoRA trained on fold i's train questions only). Check: same pipeline on the STORED states reproduces ref ridge exactly (diff 0.000, all 5 targets; `lora/check_frozen_ridge/`) | states `runs/goal2/gpqa25/lora/L21_r16_mlp_states/states.f0.1.fold*.npz`; preds `lora/f01_L21_r16_ridge/preds.csv`; CIs `lora/ci_f0.1.csv` | `lora_probe.py --save-states` (new flag) via `12_lora.sh`, job 4790871 (48 min GPU); `src/probing/lora_ridge.py` (new), job 4790872 (11 min CPU); check job 4790863 | done 2026-10-11. **LoRA features ≈ stored features under ridge**: 0.281/0.253/0.635/0.294/0.765 vs ridge 0.278/0.251/0.626/0.304/0.767. Paired diff: entropy +0.003 [−0.002,0.009], p_mode +0.001, rem_len_mean +0.009 [0.004,0.014] (sig. but tiny), rem_len_std −0.010 n.s., committed −0.002 n.s. So LoRA (as trained here) does not make layer 21 meaningfully more predictive; earlier LoRA deficits were the head |
| R8 ridge on LoRA features at **50%** (same setup as the 10% row above) | states `runs/goal2/gpqa25/lora/L21_r16_mlp_states_f05/`; ridge preds `lora/f05_L21_r16_ridge/preds.csv`; LoRA head preds linked as `lora/f05_L21_r16_mlp/`; CIs `lora/ci_f0.5.csv` (`lora_compare.py 0.5`; glob now `f{FR*10:02d}_*`) | jobs 4793627 (GPU, 2 h 23 min; 24–33 min/fold), 4793628 (CPU, 9 min) | done 2026-10-12. **Small but significant gain on all 5 targets**: ridge on LoRA states 0.309/0.269/0.574/0.321/0.729 vs ridge 0.298/0.259/0.548/0.310/0.720. Paired diff: entropy +0.012 [0.005,0.019], p_mode +0.010 [0.003,0.017], rem_len_mean +0.026 [0.020,0.033], rem_len_std +0.011 [0.005,0.017], committed +0.010 [0.003,0.016]; ≥ ridge in nearly every fold. LoRA's own MLP head again well below (entropy 0.224, sig. worse on 4/5). Gains ~3–5% relative; compare CI half-width ±0.05 for between-feature comparisons in R5 |
| R8 ridge on LoRA features at **30% and 70%** (same setup) | states `lora/L21_r16_mlp_states_f0{3,7}/`; ridge preds `lora/f0{3,7}_L21_r16_ridge/`; head preds linked `lora/f0{3,7}_L21_r16_mlp/`; CIs `lora/ci_f0.{3,7}.csv` | jobs 4793785/86 (30%: GPU 1 h 34 min, CPU 12 min), 4793787/88 (70%: GPU 3 h 06 min, CPU 15 min) | done 2026-10-12. 30%: ridge-on-LoRA 0.307/0.272/0.603/0.322/0.752 vs ridge 0.295/0.261/0.587/0.315/0.744; diff +0.012 [0.003,0.022], +0.012, +0.016 [0.010,0.022], +0.006 (n.s.), +0.008 (sig.). 70%: 0.312/0.268/0.503/0.304/0.706 vs 0.302/0.261/0.484/0.288/0.711; diff +0.009 [0.003,0.016], +0.007 (edge), +0.019 [0.014,0.023], +0.015 [0.009,0.021], −0.006 (n.s.). LoRA MLP head sig. below ridge on 4–5/5 at both. 90% (job 4793782) still running |
| R8 ridge on LoRA features at **90%** (same setup) — completes R8 over 10–90% | states `lora/L21_r16_mlp_states_f09/`; ridge preds `lora/f09_L21_r16_ridge/`; head preds linked `lora/f09_L21_r16_mlp/`; CIs `lora/ci_f0.9.csv` | jobs 4793782 (GPU, 4 h 15 min), 4793783 (CPU, 11 min) | done 2026-10-12. Ridge-on-LoRA 0.291/0.239/0.399/0.294/0.771 vs ridge 0.270/0.222/0.375/0.271/0.761; diff entropy +0.021 [0.013,0.029], p_mode +0.018 [0.010,0.025], rem_len_mean +0.024 [0.014,0.035], rem_len_std +0.023 [0.016,0.030], committed +0.010 (n.s.). **R8 summary: LoRA-adapted layer-21 states give a small, significant ridge gain at 30–90% (≈+0.01–0.026 R²; largest at 90%), none at 10%; LoRA's own MLP head is always below ridge.** Total GPU ≈ 12 h (10% 0.8, 30% 1.6, 50% 2.4, 70% 3.1, 90% 4.2) |
| **Round-2 update doc + Figs. 6–8** (everything since 2026-10-05) | `docs/update_2026-10-12.md`, `docs/figures/round2/fig{6_icl,7_rf_reg,8_lora_ridge}.png` | `src/viz/update_1012_figures.py` (reads icl/ci_vs_probes.csv, sweep/rf_reg, lora/ci_f0.*.csv) | done 2026-10-12. Also mirrored as a Claude doc |
| Research: long-trained embedders + other-family LLM candidates | `docs/embedders.md` → sections 2026-10-15 and 2026-10-15 (2) | web research (papers, HF cards/configs, LongEmbed leaderboard) | done 2026-10-15; user to pick models |
| Other-LLM control (other family): OLMo-3-1025-7B layer 24/32, last token, full inputs, no instruction (base model; tokenizer adds no BOS) | `runs/goal2/gpqa25/embeddings/olmo3_7b_L24.npy` | `EMB_EXTRA="--layer 24 --no-instruct" 10_embed.sh`, job 4811576; weights download job 4811572 | done 2026-10-15 (78 min) |
| Other-LLM control (other family): Mistral-7B-v0.3 layer 24/32, same setup | `runs/goal2/gpqa25/embeddings/mistral7b_v03_L24.npy` | same, job 4811580 (first try 4811577 failed: no pad token → embed_prefixes.py now pads with EOS) | done 2026-10-15 |
| Ridge/logistic probes on OLMo-3 and Mistral layer-24 states (same grids as the Qwen controls) | `runs/goal2/gpqa25/sweep/feat_{olmo3_7b_L24,mistral7b_v03_L24}/results.csv` | `configs/sweep_feat_{olmo3_7b_L24,mistral7b_v03_L24}.yaml`; OLMo shards 4811638–42 + merge 4811643, Mistral 4811644–48 + merge 4811649 (start after the embed jobs) | done 2026-10-15 — see round-3 CI row |
| Long-trained embedders (round 3): UltraBERT-Embed-150M (32k, CLS, generic instruction, envs/icl), BGE-M3 (CLS, no instruction, cut to last 8192 tokens, envs/icl — goal1's torch 2.4 can't load its .bin), Granite-multilingual-R2-311M (32k, CLS, no instruction); all L2-normalised | `runs/goal2/gpqa25/embeddings/{ultrabert_embed_150m,bge_m3,granite_ml_r2_311m}.npy` | `10_embed.sh` with new `--pooling cls` (right padding, first token), jobs 4811682/83 + Granite rerun 4811808 unbatched (`--tokens-per-batch 1`; first run 4811684 gave NaN for 22 rows = padded rows in GPU batches; the same rows are finite one at a time on CPU, `runs/logs/round3/granite_nan_check.py`; old chunks kept in `granite_ml_r2_311m.chunks_batched_nan/`); BGE-M3 cut 1,088/24,773 inputs to 8192; batched GPU vectors vs one-at-a-time fp32 CPU (30 rows each: 10 longest, 10 shortest, 10 random; `runs/logs/round3/batch_vs_single.py`): cosine ≥0.9997 UltraBERT, ≥0.9995 BGE-M3; download job 4811665; CPU checks `runs/logs/round3/check_cls.py` (left truncation keeps CLS + end; UltraBERT all weights loaded). Not checked against sentence-transformers output (not installed) | done 2026-10-15 |
| Ridge/logistic probes on the 3 long-trained embedders | `runs/goal2/gpqa25/sweep/feat_{ultrabert_embed_150m,bge_m3,granite_ml_r2_311m}/results.csv` | `configs/sweep_feat_<name>.yaml`; shards+merge 4811689–94 (UltraBERT), 4811695–700 (BGE-M3), 4811809–14 (Granite; 4811701–06 cancelled) | UltraBERT + BGE-M3 done 2026-10-15 (mean 10–90%: entropy 0.260 / 0.245, rem_len_mean 0.376 / 0.435, committed 0.728 / 0.727; ours 0.289/0.524/0.741; Qwen3-Emb 0.183/0.358); Granite queued. Script `runs/logs/round3/compare_feats.py` |
| **Round-3 CIs**: other-family LLMs + long-trained embedders vs ours and vs TF-IDF (question bootstrap, 2000 draws, paired, mean 10–90%) | `runs/goal2/gpqa25/sweep/_round3/{ci_scores,ci_paired}.csv` | `src/viz/round3_ci.py` (new; same method as R5; ours−TF-IDF entropy +0.052 reproduces R5) | done 2026-10-15. **OLMo-3-7B ≈ ours** (all 5 targets n.s., abs. diff ≤0.008). **Mistral-7B slightly below ours** (entropy −0.018, p_mode −0.013, rem_len_mean −0.030 sig.; rem_len_std, committed n.s.) but well above TF-IDF. **UltraBERT** n.s. vs ours on entropy (−0.029 [−0.068,0.009]), p_mode, committed; sig. below on rem. length; ≈ TF-IDF on all (n.s.). **BGE-M3** sig. below ours on all 5; beats TF-IDF only on rem_len_mean (+0.044). **Granite-ML-R2** ≈ F2LLM, sig. below TF-IDF everywhere. Mean scores (entropy/p_mode/rem_len_mean/rem_len_std/committed): OLMo 0.282/0.245/0.521/0.306/0.741, Mistral 0.270/0.238/0.494/0.288/0.740, UltraBERT 0.260/0.232/0.376/0.208/0.728, BGE-M3 0.245/0.217/0.435/0.213/0.727, Granite 0.154/0.145/0.221/0.123/0.691; ours 0.289/0.251/0.524/0.298/0.741 |
| R1 frontier ICL (round 3): GPT-6 Astra (`gpt-6-astra`, reasoning high, forced JSON), k {10,50}, two versions: `ckpt` (as R1) and `nockpt` (examples hide their checkpoint, query shows it); 20 test rows (4 per checkpoint 10–90%, 20 different questions, from the R1 subsample) — cut from 100 rows / k 100 to fit a $100 credit | raw `runs/goal2/gpqa25/icl/raw/gpt6_astra_high[_nockpt]_k{10,50}.jsonl` (+ `_fold<f>.csv`); scored `icl/{preds,results}_gpt6_astra_high*`; comparison `icl/compare_gpt6_astra_20rows.csv`; log `runs/logs/round3/icl_gpt6_astra.log` | `src/baselines/icl_openai.py` (new; run / score / compare), run from login node 2026-10-16; key in `~/.frontier_api_key` (600) | done 2026-10-16. Cost ≈ $30 (tracker $25.9 + ~$4 cache-write surcharge not tracked). 80/80 calls completed, 0 parse failures, median ~1.0–1.4k reasoning tokens. First 3 calls had no cache hits (single message → implicit cache point at the end) → fixed with an explicit `prompt_cache_breakpoint` after the examples. **Pooled over the 20 rows, RAW (uncalibrated) predictions** (entropy/p_mode/rem_len_mean/rem_len_std/committed): k10 0.14/0.14/0.65/0.29/0.75, k50 0.28/0.31/0.75/0.44/0.75, nockpt k10 0.22/0.34/0.71/0.27/0.80, nockpt k50 0.27/0.34/0.75/0.39/0.75; ridge L21 on the same rows 0.31/0.27/0.75/0.31/0.65; TF-IDF 0.29/0.28/0.74/0.29/0.70. **No difference vs ridge is significant** (CIs ≈ ±0.3–0.5 R²); hiding the example checkpoints made no visible difference. Per-checkpoint scores (4 rows) and the per-fold calibration in `results_*` are meaningless at this n. Pooled R² mixes checkpoints (prefix_len gets 0.69 on rem_len_mean) **50-row extension (2026-10-19) STOPPED**: 30 new rows chosen to match the population (`--n50`, `subset50`: 10/checkpoint, 10/fold, subject mix + target quantiles matched; check `runs/logs/round3/icl_check50.py`), but credit ran out after 13 new k=10 calls (429 `credit_balance_exhausted`). Total tracked spend over all GPT-6 calls only ≈ $33 (incl. est. cache-write surcharge) vs the $100 credit — unexplained gap; check the OpenAI usage page. Results above still = the 20-row set; the 13 extra rows are in `gpt6_astra_high_k10.jsonl` |
| R1 frontier ICL, cheap model: GPT-6 Luna (`gpt-6-luna`, reasoning high), same 4 configs (ckpt/nockpt × k 10/50) on 10 of the 20 Astra rows (2 per fold, 2 per checkpoint, 10 questions; picked by `subset10` to match the population: subjects 50/40/10% vs 47/43/10%, target means within 0.46 sd; check `runs/logs/round3/icl_check10.py`) | raw `icl/raw/gpt6_luna_high[_nockpt]_k{10,50}.jsonl`; scored `icl/{preds,results}_gpt6_luna_high*`; comparison (Luna + Astra + probes on the same 10 rows) `icl/compare_gpt6_10rows.csv`; log `runs/logs/round3/icl_gpt6_luna.log` | `icl_openai.py run --model gpt-6-luna --n10` (new `--model`, `--n10`), 2026-10-21 | done 2026-10-21. 40/40 calls, 0 parse failures, cost $0.20 tracked, median ~2k reasoning tokens, caching worked. Pooled raw (entropy/p_mode/rem_len_mean/rem_len_std/committed): Luna k10 −0.11/0.09/0.69/0.17/0.70, k50 0.04/0.04/0.62/0.16/0.60, nockpt k10 0.03/0.26/0.75/0.01/0.60, nockpt k50 0.02/0.12/0.75/0.25/0.60; Astra on the same 10 rows −0.03…0.09 / −0.12…0.27 / 0.76–0.86 / 0.42–0.49 / 0.70–0.80; ridge 0.22/0.17/0.85/0.29/0.70. Only Luna k50 rem_len_mean is sig. below ridge; everything else n.s. (CIs ±0.3–1.5 at n=10). Luna ≤ Astra on rem. length |
| Multi-layer features (lab mate's recipe, 2026-10-17): per-layer PCA-64 of layers 13–15 (~45–55% depth) and 20–22, concatenated; controls L14/L21 PCA-64/192, full L14, full L13–15 and L20–22 concatenated; ridge + logistic, same grids as 4b / ref_fixed | `runs/goal2/gpqa25/sweep/multilayer/results.csv`; CIs `sweep/_multilayer/` | `configs/sweep_multilayer.yaml`; new `pca_per_layer` option in `sweep.py` (`Act`/`Feature.prep`; unit-checked = PCA per block); smoke 4840094 (20 questions, 22 min); shards 4840154–61 + merge 4840162; CIs `src/viz/multilayer_ci.py` (reuses round3_ci); shards 13–53 min | done 2026-10-17. **The lab mate's recipe does not transfer**: L13–15 PCA-64 each 0.245/0.215/0.447/0.244/0.733 (entropy/p_mode/rem_len_mean/rem_len_std/committed) vs our L21 ridge 0.289/0.251/0.524/0.298/0.741, sig. worse on all 5 (entropy −0.044, rem_len_mean −0.077); ≈ TF-IDF. PCA always hurts (every PCA variant sig. below its full-dimensional counterpart's level). Mid-depth layers are worse than L21 (L14 full −0.018 entropy, −0.037 rem_len_mean, sig.). **Only L20–22 full concatenated beats L21**, by a tiny but significant margin: 0.293/0.255/0.527/0.301/0.743 (diff +0.005/+0.004/+0.003 n.s./+0.004/+0.003 n.s.) |
| Job logs | `runs/logs/{gpqa25,sweep,setup,round2}/`; default Slurm `slurm-<id>.out` files moved from the repo root to `runs/logs/slurm/` (2026-10-12); all `runs/gpqa25/*.sh` now set `#SBATCH --output=runs/logs/slurm/slurm-%j.out` (relative to the submit dir — submit from the repo root) | — | — |

Sweep outputs: `results.csv` (one row per feature × probe × checkpoint × target; `--merge` after sharded runs) and `preds*.csv` (out-of-fold predictions).

## Current work
_User gives direction; track tasks here. Mark `[x]` when done._
- [x] Explore folder and cluster setup
- [x] Pull code from GitHub into this folder
- [x] Set up our own Python environments (`envs/goal1`, `envs/easysteer`; built 2026-10-03, imports OK on login node)
- [x] Point all paths at this folder (`env.sh` fallback)
- [x] GPU check of both envs — job 4672741 PASSED (2026-10-04): H100, both envs see GPU, weights visible, internet OK. Output: `runs/logs/setup/gpu_check_4672741.out`

### Baselines + probe sweeps (full plan: `PLAN.md`)
- [x] Check 0% checkpoint → **broken** (gives first sentence, drops 10%)
- [x] Write goals + plan (`PLAN.md`)
- [x] 0. Fix 0% checkpoint: code fixed, labels in `labels/checkpoint_labels_f0.csv` (198 rows, join on example_id; cot_id=-1)
- [x] 0b. 0% activations: job 4672844 done (8 min) → `runs/goal2/gpqa25/activations_ckpt_f0/` (198, 28, 3584); vectors ~0.96 correlated across questions (shared prompt template) — standardise before probing
- [x] 1. Sweep pipeline: `src/probing/sweep.py` + `configs/`. Reproduces `probe_checkpoints.csv` (90/90 cells, max diff 5e-5; checked independently)
- [x] 2. Pick sentence embedders → `docs/embedders.md` (shortlist verified on HF 2026-10-04)
- [x] 3. Embeddings + baseline 1 (ridge/logistic): Qwen3-Emb-8B, F2LLM-v2-8B, other-LLM controls Qwen2.5-Math-7B + Qwen2.5-7B-Instruct. Script `src/probing/embed_prefixes.py` (`--layer` = LLM hidden-state mode)
- [x] 4a–4c'. Probe architectures: MLP (+ wd rerun), PCA-k, random forest, gradient boosting — all done, none beats ridge
- [ ] 4d. Follow-ups: other layers; 0% checkpoint probes (needs sweep config for f0 data); ~~bootstrap CIs~~ (done as R5); best probes on embeddings/LLM controls; unrelated-LLM control (e.g. Llama-3.1-8B)
- [x] 1b. FIXED folds in sweep.py (default; 198/198 questions same fold everywhere incl. 0%; 40/40/40/39/39). `configs/sweep_baseline.yaml` keeps old folds (`fixed_folds: false`).
- [x] 1c. Reference ridge/logistic on fixed folds (done; scores within 0.04 of old folds, mean 0.007): jobs 4672913–17 → `runs/goal2/gpqa25/sweep/ref_fixed/` (the new baseline for all comparisons)
- [x] 5. LLM baselines 2 (zero-shot) and 3 (in-context) — done as R1 (k=0 is zero-shot)
- [x] 6a. Interim update doc + figures: `docs/update_2026-10-05.md`, `docs/figures/` (2026-10-05)
- [ ] 6. Final write-up
- On hold: baseline 4 (forced answer at each prefix → entropy)

### Round 2 (user list 2026-10-09; plan + checked facts: `PLAN.md` → ROUND 2)
- [x] R1. ICL baseline (full runs off+on done 2026-10-11; results in Results index): different LLM, k labeled examples (FULL prefixes, not shortened) → predict targets; three LLMs. k=1000 can't fit any context; max ~85 (256k) / ~350 (1M). k ∈ {0, 10, 30, max} for all models (max ≈ ours 40 / Gemma 85 / Qwen3.5 350). Models (user): our model (max k≈40, 131k ctx), Qwen3.5-9B, Gemma-4-31B. Do before R8.
- [x] R2. Check model bases (configs + weight similarity of R1-Distill-Qwen-7B, Qwen2.5-Math-7B, Qwen2.5-7B-Instruct)
- [x] R3. Diagnose why RF < ridge
- [x] R4. Embedder training lengths vs our input lengths
- [x] R5. Error bars (question bootstrap via `bootstrap_ci.py`; pass `--preds`/`--out-dir` under `sweep/`)
- [x] R6. Train vs test R² (overfitting) — from existing `results.csv`
- [x] R7. MSE alongside R² — from existing `preds*.csv`
- [x] R8 (done 2026-10-12; ridge on LoRA states, all 10–90%). LoRA + MLP-head probe on our model only (expand to other models later only if needed), same settings as existing probes (L21, last token, 5 targets, fixed folds) for direct comparison. `peft` install approved. After R1.

## Next steps
**Where we stopped (2026-10-15):** Round 1 and Round 2 (R1–R8) all DONE; round-2 write-up `docs/update_2026-10-12.md` (also a Claude doc, link in `docs/HANDOFF.md`). No jobs running. Since then only explanations to the user, plus: Slurm logs now go to `runs/logs/slurm/`, and a correction to the raw-ICL-R² claim.
- **Round 3 (2026-10-15):** other-family LLM controls (OLMo-3-7B, Mistral-7B-v0.3) and long-trained embedders (UltraBERT-Embed, BGE-M3, Granite-ML-R2) done — see "Round-3 CIs" in the Results index.
- **Open (pick with user):** (1) ~~unrelated-LLM control~~ (done: OLMo ≈ ours, Mistral slightly below); (2) 0% checkpoint probes (data ready); (3) other layers (partly done: multi-layer row — L20–22 concat ≈ L21 +0.005; mid-depth worse); (4) logistic C grid below 1e-4 for `committed_by_here` on our activations (chosen C sits at the grid edge); (5) LoRA seeds / other LoRA settings; (6) final write-up.
- **Not committed yet:** see `docs/HANDOFF.md` → "Git state". Last push 2026-10-12. Commit/push only when the user says (fetch + check origin/main first — see "Working alongside yuvraj").
- Several Claude sessions may work in this repo at once: use a distinct output folder per experiment, and don't commit other sessions' files without the user's say.

---

# Project notes (original repo CLAUDE.md, by yuvraj)

> **New session / new machine? Read [`.claude/README.md`](.claude/README.md) first.**
> This file is the authority on **scope and working practices**. `.claude/` is the
> authority on **current state and mechanics** -- what has actually been run
> ([`STATE.md`](.claude/STATE.md)), how to bootstrap a cluster
> ([`SETUP.md`](.claude/SETUP.md)), exact stage commands
> ([`PIPELINE.md`](.claude/PIPELINE.md)), silent-failure traps
> ([`GOTCHAS.md`](.claude/GOTCHAS.md)) and the agreed next steps
> ([`NEXT.md`](.claude/NEXT.md)). Where the two disagree about what has been run,
> trust `.claude/STATE.md`.
>
> **`runs/` is not in this repo.** All results described below and in `.claude/`
> live on the original UNC box; on a fresh clone they must be regenerated.

## Research question

Do internal representations at a chain-of-thought (CoT) prefix predict useful
properties of the **distribution over future continuations**, beyond what is
available from the prefix text itself?

Originally Goal 1 (behavioral labels only, no activations) was meant to be
validated before Goal 2 (activations/probing) started. **In practice a
collaborator (`yuvraj`) began Goal 2 on 2026-09-06/07 while Goal 1 was still
partial** — see "Goal 2 status" below. Treat the scope list as what was
originally agreed, not as a description of what has actually run; check
`runs/` before assuming a dataset/model/K combination hasn't been done.

## Scope (originally agreed — see actual state below before assuming gaps)

- **Models:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` and
  `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`. **Only Qwen-7B has been run, for
  everything, Goal 1 and Goal 2.** Llama-8B: zero runs anywhere.
- **Datasets:** AquaRAT, OpenBookQA, MMLU (college math) were the original set;
  GSM8K remains deferred. **Also actually run** (added by yuvraj without a
  CLAUDE.md update): GPQA-diamond, AIME2025, AIME 2022–2025, MATH500 — see
  `runs/goal1/full/`. GPQA-diamond is also the only dataset used for Goal 2.
- **Size:** 50–100 questions total was the target for the original 3 datasets;
  the added datasets vary in size (see `runs/goal1/full/*/raw/examples.csv`).
- **Base CoTs per question:** 3.
- **Checkpoints per CoT:** prefixes at 10 / 30 / 50 / 70 / 90% by sentence count.
- **Continuations per prefix (K):** originally 10. **AIME runs used K=100**
  (log records `K* = 300, ran at K = 100`) — K sufficiency findings from the
  original 3 datasets do not automatically apply; re-check per dataset.

### GPQA's K is two-tier, and the K=25 choice has a caveat (checked 2026-09-14)

`pipeline_gpqa.sh` ran phase A at K=100 on 20% of questions, measured K\* there,
then ran the rest at K\*. So the GPQA ground truth is **not** uniform:

| slice | questions | CoTs | K |
|---|---|---|---|
| `example_id % 5 == 0` | 40 | 120 | 100 |
| the rest | 155 | 455 | 25 |

`k_sufficiency/summary.csv` reports `k_bias=75`, `k_bias_mm=25`,
`k_precision=NaN` (the 0.05-bit SD tolerance was never met at K≤100),
`k_decision=100`, and the log states **`K* (max of the three criteria) = 100.0`**.
The pipeline then reads back `k_bias_mm` and runs at 25 — i.e. the K=25 choice is
justified only under the **Miller-Madow** bias correction, which `labels.py` was
not applying (`run_eval._entropy` is plug-in). Measured consequence: 66.5% of
K=25 rows were biased beyond the stated 0.02-bit tolerance, mean correction
0.022–0.053 bits depending on checkpoint.

**Fixed 2026-09-14:** `labels.py` now also emits `answer_entropy_bits_mm`
(`labels._entropy_mm`). Both estimators are stored so the gap stays auditable.
Prefer the `_mm` column as the entropy target; treat commitment labels on the
K=25 slice as ~82% stable (`k_sufficiency/commitment_agreement.csv`).

Any further expansion — more datasets/models, new Goal 2 stages beyond what's
described below — should still be flagged to the user before doing it.

## Go-forward design (decided 2026-09-14, NOT yet run)

Everything currently in `runs/` uses the original design: **3 CoTs per question,
5 checkpoints at 10/30/50/70/90%**. All Goal 2 results — `runs/goal2/probes_ckpt/`
included — describe that design.

New runs from here use instead:

- **1 CoT per question** (not 3)
- **6 checkpoints**, the first at **0%** — the rendered prompt before any
  reasoning has been emitted — then the existing five
- k rollouts from each checkpoint, k still set by the k-sufficiency analysis

Two consequences to keep in mind when comparing old and new numbers:

- **The 0% checkpoint removes the prefix-length confound.** Every 0% prefix has
  length zero, so the `prefix_len` baseline is constant and cannot predict
  anything. This matters because the strongest result so far — remaining
  reasoning length, R² 0.605 at the 10% checkpoint — is matched or beaten by
  `prefix_len` alone at the first three checkpoints, i.e. it is largely
  token-counting. At 0% that explanation is unavailable, so whatever a probe
  achieves there is clean. It also measures how much the model knows about where
  it is heading *before writing anything*.
- **Per-checkpoint training rows drop 575 → 195**, since each question now
  contributes one CoT rather than three. The learning curve in
  `runs/goal2/logs_ckpt/lcurve.log` shows entropy has already plateaued by ~97
  questions so it should be largely unaffected, but remaining-length was still
  improving with more data and should be expected to weaken.

Still open, parked rather than rejected: using the rollouts themselves as probe
training data. The activation at a checkpoint is **one vector shared by all k
rollouts** (causal attention: the prefix hidden states do not depend on what
follows), so repeating it per rollout adds rows but no information. The version
that would add real data is capturing activations at tokens *inside* each
continuation, where they genuinely diverge — that needs a new GPU pass and a
decision on what label a mid-continuation token carries.

## Current repo state (what already exists)

### Layout of `src/` (reorganized 2026-09-14)

Grouped by pipeline stage, not by goal. Every module is still runnable as
`python src/<package>/<module>.py` from the repo root; modules that import
across packages put `src/` on `sys.path` themselves, so `python -m
<package>.<module>` works too.

| package | what lives there |
|---|---|
| `src/generation/` | Everything that talks to the model (vLLM), plus maintenance of the raw records it writes: `run_eval`, `checkpoint`, `continuations`, `gpu_pool`, `merge_shards`, `reparse`. |
| `src/analysis/` | Pure CPU reductions over those records — no GPU, no model: `labels` (the Goal 1 behavioral labels), `robustness`, `k_sufficiency`, `year_effect`. |
| `src/probing/` | Goal 2: `verify_capture` → `extract_activations` → `probe_smoke` / `probe_linear` / `probe_goal1_labels`. |
| `src/viz/` | Figures, tables and PDFs: `aggregate_viz`, `plot_goal1`, `make_figures`, `report`, `make_timing_report`. |
| `src/setup/` | `download_dataset` — one-time fetch into `data/`. |

Cross-package imports are package-qualified (`from generation.run_eval import
...`); imports between siblings in the same package stay bare. Launcher scripts
under `runs/goal1/full/` and `runs/goal2/` were updated to the new paths.


- `src/generation/run_eval.py` — vLLM sampler. Per question, draws N **full** CoTs, parses
  the `{"Final Answer": ...}` JSON, normalizes MC/free-form answers, writes
  per-sample `sample_records.jsonl` + `example_metrics.csv`. Reuse its prompt
  rendering, `parse_output`, `normalize_mc` / `normalize_freeform`, entropy, and
  resume logic — do not reimplement these.
- `src/viz/report.py` — aggregate heatmaps / bars / PDF across a sweep.
- `src/setup/download_dataset.py` + `data/` — AquaRAT, OpenBookQA,
  MMLU college/pro math already downloaded locally.
- `archive/` — prior Qwen3 logit-lens / probing / activation-extraction code.
  **Reference only.** Note it hardcodes `/playpen-ssd/yuvraj/...` paths; this
  repo was copied from that original folder, so those paths are stale.

**The key gap:** existing code samples *whole* CoTs. Goal 1 needs
**prefix-conditioned continuation** — the main thing to build.

## Pipeline to build

New modules under `src/`, reusing `run_eval.py` helpers where possible.

1. **Base CoTs** — per question, generate 3 independent full CoTs (reuse
   `run_eval`).
2. **Checkpointing** — split each CoT's reasoning into sentences; take prefixes
   at 10/30/50/70/90% by sentence count. Store exact prefix text **and** its
   token offset. Use a **math-aware sentence splitter** — naive `.`-splitting
   breaks on decimals, equations, and abbreviations.
3. **Continuations** — from each prefix, generate K=10 continuations by feeding
   `prompt + prefix` and continuing generation (no chat-template re-application on
   the partial assistant turn).
4. **Parse & store** — raw output plus per-continuation answer / correctness /
   length, keyed by
   `(dataset, model, question_id, cot_id, checkpoint, continuation_id)`.

## Labels (computed per question × CoT × checkpoint)

1. **Answer distribution** — entropy over final answers; P(answer), P(gold),
   P(mode).
2. **Remaining reasoning length** — mean/median tokens to completion, and stdev
   across continuations (variability matters, not just central tendency).
3. **Value of added compute** — accuracy of a single continuation vs.
   majority-vote over 5 and 10 continuations from that prefix.
4. **Commitment juncture** (derived) — earliest checkpoint where one answer
   becomes sufficiently concentrated and *stays* concentrated at later
   checkpoints.

## Robustness analyses (the actual deliverable of Goal 1)

These are the questions that decide whether the labels are modelable:

- **K sufficiency** — subsample/bootstrap over K to find where entropy and length
  estimates stabilize. Report whether 10 is enough or K must grow.
- **Parser failure rate & patterns** — track parse-success per dataset/model;
  catalog failure modes, not just the rate.
- **Surface-form leakage** — how often the final answer is already literally
  present in the prefix text (a predictor could cheat off this).

## Storage & runs

- All outputs go **under this repo** (it was copied from the original
  `/playpen-ssd/yuvraj/...` tree — do not write back there). New Goal 1 outputs
  under `runs/goal1/...`.
- Raw records as JSONL; label tables as Parquet/CSV.
- Jobs must be **resumable/restartable** like `run_eval.py` (append-only JSONL +
  resume-by-count). Long GPU jobs should not lose work on interruption.
- Keep model / GPU / sampling config explicit and saved alongside each run
  (as `run_eval.py` does with `run_config.json`).

## Goal 2 status (started 2026-09-06/07, ahead of the original plan)

Baselines (mean-mode, surface-form-only, checkpoint-position/model-only
predictors) are still not built — that part of Goal 2 remains deferred.
Activation extraction and a first linear probe, however, already exist, on
GPQA-diamond / Qwen-7B only:

- `src/probing/verify_capture.py` — before trusting any captured activation, cross-checks
  EasySteer's (a forked vLLM) hidden-state capture against stock HF transformers
  on identical input, matching by cosine similarity (min 0.9969 over layers
  0–26). This pinned down two non-obvious conventions that are wrong by default:
  decoder layer L (1-indexed) → EasySteer index L−1, and token position t →
  EasySteer position t+1 (vLLM double-BOS). Also documents that vLLM's final
  layer is pre-RMSNorm, unlike HF's.
- `src/probing/extract_activations.py` — re-feeds Goal 1's already-generated CoT text
  through vLLM+EasySteer (`max_tokens=1`, prefix caching and chunked prefill
  both off — either silently truncates the captured sequence) to pull hidden
  states at layers 14/21/28, every 4th think-token. Output:
  `runs/goal2/activations/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B/` (5 shards,
  `.npy` activations + `.parquet` index).
- `src/probing/probe_linear.py` — mean-pools activations per (CoT, depth-bin) so each
  CoT contributes one vector per bin (not one per token — the first smoke
  attempt over-counted this way). Logistic regression, C swept over
  `[1e-4..1]`, `GroupKFold` grouped by `example_id` (a question's 3 CoTs share
  text, so ungrouped CV would leak). Includes a positive-control target
  (`depth`: early vs. late half of CoT) that must succeed for the setup itself
  to be trusted. Results: `runs/goal2/probes/gpqa_diamond/probe_results.csv`.
  Headline: `depth` control 0.81–0.82 acc (vs 0.50 base) — pipeline is sound;
  `correct` 0.69–0.72 (vs 0.53 base), flat across all 10 depth bins — real but
  doesn't sharpen with depth, unexplained; `answer` (which of 4 choices)
  0.25–0.34 (vs 0.264 base) — no reliable early signal.

- `src/probing/probe_goal1_labels.py` — the first attempt at the actual research
  question: predict the Goal 1 labels from prefix activations, within-checkpoint,
  GroupKFold by `example_id`, against a TF-IDF-on-prefix-text baseline. **It ran,
  and its result was undocumented until 2026-09-14.** Result in
  `runs/goal2/probes/gpqa_diamond/goal1_label_probes.csv`: **TF-IDF on the prefix
  text beats the activation probe on nearly every target and checkpoint** (e.g.
  entropy at 30%: text 0.263 vs best activation 0.105; `rem_len_mean` at 10%:
  text 0.475 vs 0.459). Only `committed_by_here` at the 10% checkpoint favours
  activations (0.736 vs 0.715). Taken at face value the answer to the research
  question is *no*.

  That comparison was, however, handicapped against activations in three ways,
  all fixed by the modules below: activations were read at the nearest captured
  token *at or before* the checkpoint (stride 4) while TF-IDF got the exact
  prefix; only 3 of 28 layers were stored; and regularisation was never swept
  (`Ridge(alpha=1000)` for activations vs `Ridge(alpha=1.0)` for text). It also
  scored 8 CoTs that have checkpoints but no labels as "not committed".

### Checkpoint-exact, all-layer probing (added 2026-09-14)

- `src/probing/extract_checkpoints.py` — captures **all 28 decoder layers** at
  the **exact checkpoint token** (last token of each prefix), one vector per
  `(example_id, cot_id, fraction)` — 1:1 with `checkpoint_labels.csv`. 2,915
  checkpoints over 583 CoTs, 0.54 GB fp16, versus 16 GB for the stride-4 store.
  Two facts that make this cheap and correct: EasySteer already returns all 28
  layers on every call (the old code discarded 25), and because the model is
  causal, one forward pass over each CoT's longest prefix yields every earlier
  checkpoint's state exactly. Output:
  `runs/goal2/activations_ckpt/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B/`.

  **Do not extract per continuation.** All K continuations of a prefix share that
  prefix, so with causal attention their checkpoint-token activations are
  bit-identical. The K continuations supply the *label*; the prefix supplies the
  *feature*.

- `src/probing/probe_checkpoints.py` — per (checkpoint × layer × target) probes
  with **nested** regularisation selection for every model including the
  baselines, so the comparison is about features rather than tuning effort.
  Ridge is solved along the whole alpha grid from one SVD of the training block
  (`RidgePath`), which is what makes 28 layers × 5 checkpoints affordable.
  Targets: `answer_entropy_bits_mm`, `rem_len_mean`, `rem_len_std`, `p_mode`,
  `committed_by_here`, `will_commit`. Also fits **[text ++ activations]** — the
  research question is about information *added* to the prefix text, which a
  head-to-head loss to TF-IDF does not settle. `--k100-only` restricts to the 40
  questions with K=100 labels.

Anything past this (more datasets/models, non-linear probes, the still-missing
baselines) should be flagged to the user before starting.

## Working practices

- **Do not assume.** Verify claims against the code and data before acting.
  Don't guess dataset schemas, column names, or model behavior — check.
- **Ask when a decision is genuinely the user's** (scope, model/dataset choices,
  anything hard to reverse). Pick sensible defaults for reversible details and
  say what you chose.
- **Report outcomes honestly.** If a job fails, parsing breaks, or a step was
  skipped, say so with the evidence. Don't paper over parser failures — they are
  a Goal 1 metric.
- **Reuse before rewriting.** Prefer existing `run_eval.py` helpers over new
  copies of prompt/parse/normalize logic.
- **Prefer resumable, config-logged jobs.** Assume any GPU run can be
  interrupted.
- **Under git as of 2026-09-18.** Commit before risky edits rather than making
  `.bak` copies (`archive/src_bak/` is the pre-git convention; don't grow it).
  Run outputs, datasets, venvs and the vendored EasySteer checkout are NOT
  tracked -- see `.gitignore` and `.claude/SETUP.md`.
