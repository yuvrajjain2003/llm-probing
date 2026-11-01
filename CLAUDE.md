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
_Last updated: 2026-11-01_

**New session? Read `NEXT_STEPS.md` first** - it is the live to-do list from the 2026-10-27 meeting. **Round 6 actions 1 and 2 and both concerns are now DONE** (2026-10-29, all for zero compute and zero dollars; write-up `docs/update_2026-10-29.md`). What is left in round 6 is actions 3 and 4, getting off GPQA, which need the user's decisions rather than compute. Then this file's "Next steps" for the longer-standing plan, then `learnings.md` (pitfalls). `docs/HANDOFF.md` is **stale** (predates rounds 4-5) - use it only for how the methods and folds work. For related work and how our numbers compare to published ones, read **`RESEARCH.md`**.

- **Folder:** `/proj/ssrivalab/projects/llm-probing` (group-shared, owned by user `yuvraj`). Git repo: `git@github.com:yuvrajjain2003/llm-probing.git`, branch `main`, last pushed commit `fe985b1` (2026-10-29: GlobalOpinionQA option + normalised entropy; before it full MMLU-Pro option, DATASET/SUBSET selection, `docs/GUIDE_new_datasets.md`). Uncommitted local work is listed in `docs/HANDOFF.md`.
- **Code:** `src/` (Python pipeline), `runs/gpqa25/*.sh` (current Slurm pipeline, stages 00–12; 10 embed, 11 ICL, 12 LoRA), `configs/sweep_*.yaml` (probe sweeps for `src/probing/sweep.py`), `src/baselines/` (ICL), `runs/goal1/full/`, `runs/goal2/` (older launchers), `env/` (setup), `env.sh` (paths).
- **Project docs** (from the repo — read when needed): `RESEARCH.md` (**related work; how our numbers compare to DyCon and five other papers; what is actually novel**), `.claude/README.md` (start here), `STATE.md` (results as of 2026-09-18, old UNC box), `NEXT.md` (agreed plan), `PIPELINE.md`, `SETUP.md`, `GOTCHAS.md`. The repo's original instructions are kept below under "Project notes".
- **Write-ups:** `docs/update_2026-10-05.md` (round 1), `docs/update_2026-10-12.md` (round 2), `docs/update_2026-10-23.md` (round 4, all three ideas), **`docs/update_2026-10-27.md` (round 5, the professor's ten asks)**. Round-4 experiment specs: `docs/plan_{redundancy,mlp_gap,literature}.md`. For the 2026-10-27 meeting: `docs/briefing_2026-10-27.md` (detailed background, plain language, file behind every number) and `docs/meeting_2026-10-27.md` (outline + the questions to ask him). **`NEXT_STEPS.md` (repo root) is the live to-do list from that meeting.** `docs/GUIDE_new_datasets.md` is a standalone guide for an agent on ANOTHER server to collect math500 / mmlu_pro data (written by a different session, committed `c8c301c`/`043218f`). `Yuvraj.md` is the user's own catch-up notes (not a team doc).
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
| Ridge/logistic probes on the 3 long-trained embedders | `runs/goal2/gpqa25/sweep/feat_{ultrabert_embed_150m,bge_m3,granite_ml_r2_311m}/results.csv` | `configs/sweep_feat_<name>.yaml`; shards+merge 4811689–94 (UltraBERT), 4811695–700 (BGE-M3), 4811809–14 (Granite; 4811701–06 cancelled) | UltraBERT + BGE-M3 done 2026-10-15 (mean 10–90%: entropy 0.260 / 0.245, rem_len_mean 0.376 / 0.435, committed 0.728 / 0.727; ours 0.289/0.524/0.741; Qwen3-Emb 0.183/0.358); Granite also done 2026-10-15 (0.154/0.145/0.221/0.123/0.691 — see round-3 CI row). Script `runs/logs/round3/compare_feats.py` |
| **Round-3 CIs**: other-family LLMs + long-trained embedders vs ours and vs TF-IDF (question bootstrap, 2000 draws, paired, mean 10–90%) | `runs/goal2/gpqa25/sweep/_round3/{ci_scores,ci_paired}.csv` | `src/viz/round3_ci.py` (new; same method as R5; ours−TF-IDF entropy +0.052 reproduces R5) | done 2026-10-15. **OLMo-3-7B ≈ ours** (all 5 targets n.s., abs. diff ≤0.008). **Mistral-7B slightly below ours** (entropy −0.018, p_mode −0.013, rem_len_mean −0.030 sig.; rem_len_std, committed n.s.) but well above TF-IDF. **UltraBERT** n.s. vs ours on entropy (−0.029 [−0.068,0.009]), p_mode, committed; sig. below on rem. length; ≈ TF-IDF on all (n.s.). **BGE-M3** sig. below ours on all 5; beats TF-IDF only on rem_len_mean (+0.044). **Granite-ML-R2** ≈ F2LLM, sig. below TF-IDF everywhere. Mean scores (entropy/p_mode/rem_len_mean/rem_len_std/committed): OLMo 0.282/0.245/0.521/0.306/0.741, Mistral 0.270/0.238/0.494/0.288/0.740, UltraBERT 0.260/0.232/0.376/0.208/0.728, BGE-M3 0.245/0.217/0.435/0.213/0.727, Granite 0.154/0.145/0.221/0.123/0.691; ours 0.289/0.251/0.524/0.298/0.741 |
| R1 frontier ICL (round 3): GPT-6 Astra (`gpt-6-astra`, reasoning high, forced JSON), k {10,50}, two versions: `ckpt` (as R1) and `nockpt` (examples hide their checkpoint, query shows it); 20 test rows (4 per checkpoint 10–90%, 20 different questions, from the R1 subsample) — cut from 100 rows / k 100 to fit a $100 credit | raw `runs/goal2/gpqa25/icl/raw/gpt6_astra_high[_nockpt]_k{10,50}.jsonl` (+ `_fold<f>.csv`); scored `icl/{preds,results}_gpt6_astra_high*`; comparison `icl/compare_gpt6_astra_20rows.csv`; log `runs/logs/round3/icl_gpt6_astra.log` | `src/baselines/icl_openai.py` (new; run / score / compare), run from login node 2026-10-16; key in `~/.frontier_api_key` (600) | done 2026-10-16. Cost ≈ $30 (tracker $25.9 + ~$4 cache-write surcharge not tracked). 80/80 calls completed, 0 parse failures, median ~1.0–1.4k reasoning tokens. First 3 calls had no cache hits (single message → implicit cache point at the end) → fixed with an explicit `prompt_cache_breakpoint` after the examples. **Pooled over the 20 rows, RAW (uncalibrated) predictions** (entropy/p_mode/rem_len_mean/rem_len_std/committed): k10 0.14/0.14/0.65/0.29/0.75, k50 0.28/0.31/0.75/0.44/0.75, nockpt k10 0.22/0.34/0.71/0.27/0.80, nockpt k50 0.27/0.34/0.75/0.39/0.75; ridge L21 on the same rows 0.31/0.27/0.75/0.31/0.65; TF-IDF 0.29/0.28/0.74/0.29/0.70. **No difference vs ridge is significant** (CIs ≈ ±0.3–0.5 R²); hiding the example checkpoints made no visible difference. Per-checkpoint scores (4 rows) and the per-fold calibration in `results_*` are meaningless at this n. Pooled R² mixes checkpoints (prefix_len gets 0.69 on rem_len_mean) **50-row extension (2026-10-19) STOPPED**: 30 new rows chosen to match the population (`--n50`, `subset50`: 10/checkpoint, 10/fold, subject mix + target quantiles matched; check `runs/logs/round3/icl_check50.py`), but credit ran out after 13 new k=10 calls (429 `credit_balance_exhausted`). Total tracked spend over all GPT-6 calls only ≈ $33 (incl. est. cache-write surcharge) vs the $100 credit — unexplained gap; check the OpenAI usage page. Results above still = the 20-row set; the 13 extra rows are in `gpt6_astra_high_k10.jsonl` |
| R1 frontier ICL, cheap model: GPT-6 Luna (`gpt-6-luna`, reasoning high), same 4 configs (ckpt/nockpt × k 10/50) on 10 of the 20 Astra rows (2 per fold, 2 per checkpoint, 10 questions; picked by `subset10` to match the population: subjects 50/40/10% vs 47/43/10%, target means within 0.46 sd; check `runs/logs/round3/icl_check10.py`) | raw `icl/raw/gpt6_luna_high[_nockpt]_k{10,50}.jsonl`; scored `icl/{preds,results}_gpt6_luna_high*`; comparison (Luna + Astra + probes on the same 10 rows) `icl/compare_gpt6_10rows.csv`; log `runs/logs/round3/icl_gpt6_luna.log` | `icl_openai.py run --model gpt-6-luna --n10` (new `--model`, `--n10`), 2026-10-21 | done 2026-10-21. 40/40 calls, 0 parse failures, cost $0.20 tracked, median ~2k reasoning tokens, caching worked. Pooled raw (entropy/p_mode/rem_len_mean/rem_len_std/committed): Luna k10 −0.11/0.09/0.69/0.17/0.70, k50 0.04/0.04/0.62/0.16/0.60, nockpt k10 0.03/0.26/0.75/0.01/0.60, nockpt k50 0.02/0.12/0.75/0.25/0.60; Astra on the same 10 rows −0.03…0.09 / −0.12…0.27 / 0.76–0.86 / 0.42–0.49 / 0.70–0.80; ridge 0.22/0.17/0.85/0.29/0.70. Only Luna k50 rem_len_mean is sig. below ridge; everything else n.s. (CIs ±0.3–1.5 at n=10). Luna ≤ Astra on rem. length |
| Multi-layer features (lab mate's recipe, 2026-10-17): per-layer PCA-64 of layers 13–15 (~45–55% depth) and 20–22, concatenated; controls L14/L21 PCA-64/192, full L14, full L13–15 and L20–22 concatenated; ridge + logistic, same grids as 4b / ref_fixed | `runs/goal2/gpqa25/sweep/multilayer/results.csv`; CIs `sweep/_multilayer/` | `configs/sweep_multilayer.yaml`; new `pca_per_layer` option in `sweep.py` (`Act`/`Feature.prep`; unit-checked = PCA per block); smoke 4840094 (20 questions, 22 min); shards 4840154–61 + merge 4840162; CIs `src/viz/multilayer_ci.py` (reuses round3_ci); shards 13–53 min | done 2026-10-17. **The lab mate's recipe does not transfer**: L13–15 PCA-64 each 0.245/0.215/0.447/0.244/0.733 (entropy/p_mode/rem_len_mean/rem_len_std/committed) vs our L21 ridge 0.289/0.251/0.524/0.298/0.741, sig. worse on all 5 (entropy −0.044, rem_len_mean −0.077); ≈ TF-IDF. PCA always hurts (every PCA variant sig. below its full-dimensional counterpart's level). Mid-depth layers are worse than L21 (L14 full −0.018 entropy, −0.037 rem_len_mean, sig.). **Only L20–22 full concatenated beats L21**, by a tiny but significant margin: 0.293/0.255/0.527/0.301/0.743 (diff +0.005/+0.004/+0.003 n.s./+0.004/+0.003 n.s.) |
| **Round 4, idea 1 — dataset redundancy** (cosine similarity, label ICC, between/within decomposition, noise floor, demeaned-target probe, CoT learning curve, question-vs-CoT exchange rate) | `runs/goal2/gpqa25/sweep/_redundancy/` (`cos_summary.csv`, `cos_auc.csv`, `text_tfidf_cos_auc.csv`, `icc.csv`, `decomposition{,_ci}.csv`, `noise_floor.csv`, `demeaned_probe*.csv`, `d1_f*.csv`, `d2_f*.csv`), figs `docs/figures/redundancy/` | `src/analysis/redundancy.py`, `redundancy_noise.py`, `redundancy_demeaned.py`, `redundancy_curves.py`; spec `docs/plan_redundancy.md`; jobs 4902834/46-54, srun | done 2026-10-23. Part C reproduces `ref_fixed` in 75/75 cells (≤1e-6). **Activations are moderately question-clustered** (standardised separation AUC L21 0.80→0.71 over 10-90%; raw 0.70→0.64; TF-IDF text AUC **1.000**, so text is far more clustered than activations; chain identity adds nothing beyond question identity, AUC 0.50-0.54). **Labels heavily clustered**: ICC 0.81→0.41 (entropy), 0.93→0.41 (rem_len_mean). **The probe is mostly a question-difficulty reader**: pooled R² is mostly between-question (entropy between 0.40-0.56, within R² −0.27→+0.06); noise is only 6-45% of within-question variance so it is NOT noise-limited, and the probe reaches just 0.3-12% of the noise-corrected within-question ceiling (0.54-0.94). Training on question-demeaned targets does NOT recover it (worse on entropy/p_mode at 30-90%; chosen α runs to the 1e6 grid edge); TF-IDF ≈ or > activations within-question. **D1: the 25 CoTs are NOT redundant** — entropy at 50% rises 0.152 (m=1) → 0.298 (m=25), monotonic, no plateau. **D2: but badly allocated** — at fixed rows, mean over ckpts entropy/rem_len: 300 rows as 12q×25 0.063/0.266, 30q×10 0.168/0.355, 60q×5 0.182/0.386, **150q×2 0.202/0.407**; 750 rows as 30q×25 0.196/0.406, 75q×10 0.227/0.451, **150q×5 0.243/0.463**. So **150q×2 (300 rows) ≈ 30q×25 (744 rows)** — same quality for 2.5× fewer rollouts. Recommendation for new collection: many more questions, 2-5 CoTs each |
| **Round 4, idea 2 — closing the MLP/ridge gap** (B1 linear-head correctness control, B2 closed-form kernel ceiling, residual/skip/ensemble MLPs, α-selection stability, demeaned kernels) | `runs/goal2/gpqa25/sweep/_mlpgap/` (`results.b1_all.csv`, `b1_diag_all.csv`, `results.b2all.csv`, `ci_{scores,paired}_b2_all.csv`, `results.mlp5.csv`, `dof.csv`, `alpha_curve.csv`, `demeaned_krr{,_L14,_L28}.csv`) | `src/probing/mlpgap_*.py`; spec `docs/plan_mlp_gap.md`; jobs 4902830/31/35, 4902858, 4902871/72, 4904477/79/84, 4906609 | done 2026-10-23. **B1: the training recipe was the culprit all along.** Over 80 fits L-BFGS reproduces ridge's closed form at cosine **1.0000** exactly, while the step-4a recipe behind every prior MLP/LoRA-head result sits at cosine **0.865 mean / 0.630 min**, rel. L2 error 0.600 mean / 2.83 max, costing 0.008-0.025 R². **B2 (closed-form, no optimiser confound): the ceiling is target-specific.** Degree-2 kernel beats linear ridge on `rem_len_mean` by **+0.0081 [+0.0058, +0.0103]** pooled (sig., growing with depth); `rem_len_std` +0.0124 at 90% only; **nothing on entropy, p_mode or committed** — those are genuinely linear-ceilinged. **MLP: `ens5_resid_h64_sq` +0.0039 [+0.0018, +0.0063]** pooled on `rem_len_mean` (and **+0.0026 [+0.0002, +0.0050] vs an ORACLE-tuned ridge**; kernel +0.0067 [+0.0044,+0.0089] vs oracle), sig. at 4/5 checkpoints; 4 of 5 cells have exactly zero ridge mistuning and the MLP wins at 3 of them, so it is not a tuning artefact. Needed all three of: cross-fitted residuals, square activation, seed ensembling (single seed sig. at 0/5 checkpoints). Ridge uses only 440/3584 directions (12.3%). Skip net instrumented: g is in the optimiser, 100% units live, and moves to ~0 because 0 is the regularised optimum — forcing L2 weak gives train 0.9995 / test 0.300. Its match to ridge elsewhere is **selection**: the inner CV picks the pure-linear fallback (26%) or trains 0 epochs (44%), i.e. the MLP contributes nothing in 70% of folds; the skip net's g does move, and moves to zero because zero is the regularised optimum. **α-selection stability**: per-fold CV costs only +0.0004 mean vs best fixed α (max +0.0071, `rem_len_mean` at 50%); 14/20 cells have all folds agreeing. **Negative results**: 1-SE stabiliser −0.016 (far worse than the instability it fixes — do not use), explicit-L2-only training +0.001 vs +0.005, sketched quadratic features on top-128 PCA +0.0002 vs the kernel's +0.0046 (so the quadratic signal is NOT in the leading PCA directions), layers 14 and 28 worse than 21 on demeaned targets |
| **Round 4, idea 3 — literature review** (7 papers verified against our setup) | `runs/goal2/gpqa25/sweep/_litreview/{papers.csv,findings.md,triage.md,QUESTIONS.md}` → **written up in `RESEARCH.md`** | web research + authors' released code; spec `docs/plan_literature.md` | done 2026-10-23/26. **Read `RESEARCH.md` for all of it.** Headlines: nothing in the paper list was fabricated; DyCon (ICML 2026, MATH 600 problems) is the only comparable number and its Table 8 gives raw **0.501 on our exact model vs our 0.524** (log is worth +0.14, measured; min-max is affine and cannot change R²); **DyCon's split leaks** (step-event level, no grouping, 83,342 events from 600 chains) so 0.501 is an upper bound; 0/6 run an other-model control, 0/6 report a trivial-scalar baseline, 0/7 report any CI; **pooled across depth as they pool, prefix_len 0.718 ≈ our 0.716 — never pool**; single-trajectory ceiling 0.731→0.538 raw vs ~0.97-0.99 for our 25-mean, so ours is ~53% of achievable. **Novelty narrowed**: paper 2's target is a mean over 8 rollouts (confirmed from code), so "every paper predicts a single trajectory" is FALSE; the surviving claim is that nobody predicts answer-distribution properties (entropy, p_mode, spread), with `rem_len_std` the sharpest case. Paper 2 probes the prompt end = **our unrun 0% checkpoint**, clean split, raw target, Spearman metric |
| **Master results table: all 22 approaches scored identically** (ours, 2 text baselines, 4 other-LLM controls, 5 embedders, 7 probe architectures, 3 layer variants), mean over 10-90%, paired CIs vs ours and vs TF-IDF | `runs/goal2/gpqa25/sweep/_master/{master_table.md,ci_scores.csv,ci_paired.csv,per_checkpoint.csv}` | `src/viz/master_table.py` (new; reuses round2_metrics/round3_ci method), srun `small` 4 CPUs | done 2026-10-26: reproduces published numbers exactly (ours 0.289/0.251/0.524/0.298/0.741; OLMo 0.282; TF-IDF 0.237; L20-22 0.293). Rows common to all 22 = 4,911-4,915 per cell. Only **L20-22 concatenated** beats ours (+0.005 entropy, +0.004 p_mode, both sig.). Ties (n.s. on all 5): Qwen2.5-Instruct, Qwen2.5-Math, OLMo-3. MLP h=256 n.s. on 4/5. Everything else sig. below ours |
| **Interactive results dashboard** (for the 2026-10-27 meeting): 7 tabs — overview, all 22 approaches, by depth, in-context LLMs, why linear wins, data redundancy, the preliminary question-difficulty finding | https://claude.ai/artifact/NCW8FuLt4GLE9gxTcCk46V ; datasets built by a one-off script from `_master`, `_mlpgap`, `_redundancy`, `icl`, `lora`, `ref_fixed` (CSVs in the session scratchpad, re-derivable) | Claude, 2026-10-26 | done 2026-10-26: 10 attached datasets, every figure marked back to one. Reproduces published numbers (ours 0.289/0.524; df 440/3584; learning curve 0.152→0.298; capacity 0.9995 train / 0.300 test). PRIVATE until shared from the page's Share menu. **Updated 2026-10-26 (evening)**: the ICL tab now includes all 8 full-set GPT-6 Luna configs (dataset `icl` re-uploaded as a new blob and the `datasets/icl` doc repointed; page edited in place via two `str_replace` calls on `files/index.html`, now at version 3). The type's automated render check produced no verdict (no `check` document after 2 reads), so the page is NOT confirmed to render - open it once before the meeting |
| **R1 frontier ICL, GPT-6 Luna, 100-row check** (8 configs: ckpt/blind k{0,10,50} + nockpt k{10,50}; new `blind` version hides the checkpoint on examples AND query; new `--full`/`--n100` flags) | raw `runs/goal2/gpqa25/icl/raw/gpt6_luna_high*_k*.jsonl`; scored `icl/{preds,results}_gpt6_luna_high*`; comparison `icl/compare_gpt6_luna_100rows.csv`; log `runs/logs/round5/icl_luna_100rows.log` | `icl_openai.py run/score/compare --model gpt-6-luna`, login node 2026-10-26 | done 2026-10-26, **$1.82**, 800/800 calls completed, 0 parse failures, cache 82-93% for k>0. Pooled over the 100 rows, calibrated: **Luna ties ridge on entropy, p_mode and committed, sig. worse on both length targets** (best = k50 cal: entropy 0.31 vs ridge 0.30, p_mode 0.28/0.28, committed 0.70/0.63, rem_len_mean 0.53 vs 0.69 sig., rem_len_std 0.16 vs 0.40 sig.). Same pattern as Gemma k0 in the full run. **Hiding the checkpoint changes nothing** (blind vs ckpt within noise at every k). Raw/uncalibrated still far below 0. CIs +/-0.15 at n=100. NB on these 100 rows prefix_len beats ridge on committed (+0.11 sig.) - sampling, vs 0.727/0.741 on the full set. Bug found+fixed: the cache-miss guard killed k=0 runs |
| **Logistic C-grid fix** (loose end: the published grid pinned at its 1e-4 bottom edge in 18/25 folds): C extended to 1e-8 (10 points), probe renamed `logistic_deep`, all 3 reference features, `committed_by_here` | `runs/goal2/gpqa25/sweep/ref_logistic_deep/{results,preds}.csv` | `configs/sweep_ref_logistic_deep.yaml`, job 4941954 (36 min, `small`) | done 2026-10-26: **the published 0.741 stands.** Ours 0.7409 -> 0.7403 (-0.0006), TF-IDF and prefix_len identical to 4 dp. The edge-pinning was a FLAT region, not a truncated optimum: the extra folds that picked 1e-8 (3/5 at 90%) scored very slightly worse, as a flat-but-noisy inner-CV surface predicts. Probe is not degenerate (2 classes predicted, AUC 0.78-0.80 at 10-70%). **New finding, worth reporting: at the 90% cut `committed_by_here` accuracy is just the base rate** - majority class 0.7609 vs ours 0.7583, TF-IDF 0.7595, prefix_len 0.7609 (AUC 0.656 / 0.701 / 0.523). Mean majority baseline over 10-90% is 0.6275, so the 0.741 headline is a real +0.11 lift, but it is carried by 10-70% and the 90% cell contributes nothing. **Under AUC our edge over TF-IDF on this target nearly vanishes** (0.7612 vs 0.7581, +0.003; TF-IDF is AHEAD at 90%, 0.701 vs 0.656) - report AUC alongside accuracy for commitment |
| **R1 frontier ICL, GPT-6 Luna, FULL 1,178-row run — all 8 configs** (`ckpt`/`blind` at k 0/10/50, `nockpt` at k 10/50) | raw `runs/goal2/gpqa25/icl/raw/gpt6_luna_high*_k*.jsonl`; scored `icl/{preds,results}_gpt6_luna_high*`; table `sweep/_icl/`; logs `runs/logs/round5/icl_luna_full{,2,3}.log` | `icl_openai.py run --model gpt-6-luna --full`, login node 2026-10-26 | done 2026-10-26, **$17.86**, 9,424 calls, **0 parse failures**, cache 91.8k/92k on k=50. Mean 10-90%, calibrated, vs ridge 0.309/0.261/0.563/0.328/0.727 on the same rows: **k=0 is the BEST config** (0.294/0.233/0.455/0.122/0.718; entropy -0.015 n.s.), k=50 0.255, k=10 0.216. **Examples HURT** (k=0 > k=50 > k=10) - the same shape Gemma showed (k=0 best, k=93 collapses to 0.158), so it reproduces across two unrelated families. Every config ties ridge on entropy/p_mode/committed and is **sig. worse on both length targets** (rem_len_std 0.03-0.13 vs 0.328). **Hiding the checkpoint depth (`blind`) costs nothing on entropy/p_mode/committed but is sig. worse on BOTH length targets at every k** (rem_len_mean -0.119/-0.239/-0.166 at k 0/10/50; rem_len_std -0.059/-0.095/-0.071) - the model does use the depth label where it is informative, which is a good internal check on the ICL setup. (The 100-row check's "hiding changes nothing" was right only for entropy/p_mode/committed.) Two run bugs found+fixed: an OpenAI `bio_policy` content refusal on a GPQA biology question escaped `pool.map` and killed 5 queued configs (now recorded as a dropped row; did NOT recur on retry, so the filter is non-deterministic), and the cache-miss guard's counter was global with threshold 2, so 2 transients in 3,833 calls stopped the run (now reset per fold) |
| **ICL comparison table** (in-context LLMs vs probe, scored by the master-table method on the ICL row set) | `runs/goal2/gpqa25/sweep/_icl/{table.md,ci_scores.csv,ci_paired.csv,per_checkpoint.csv}` | `src/viz/icl_table.py` (new; reuses `master_table.main`, now parameterised by SPEC/REFS/OUT/base_label - refactor verified byte-identical on all 3 published `_master` outputs) | done 2026-10-26: validated against the published R1 numbers (ridge 0.309/0.261/0.563/0.328/0.727 and Gemma k=0 0.262/0.237/0.283/0.069/0.742 on these rows, Gemma-vs-ridge entropy -0.047 [-0.126,+0.032] - all exact matches). Separate from `_master` on purpose: the ICL runs cover only the 1,178-row subsample, so folding them in would shrink all 22 approaches to those rows |
| **Slide figures** (7 standalone PNGs, the dashboard's findings for a deck) | `docs/figures/slides/fig{1..7}_*.png`; input CSVs copied to `runs/goal2/gpqa25/sweep/_slides/data/` | `src/viz/slide_figures.py` (new) | done 2026-10-26. fig1 ours vs TF-IDF (5 targets, paired CIs), fig2 all 21 alternatives vs ours on entropy, fig3 by depth (prefix_len beats us on length at 10-30%), fig4 in-context LLMs (zero-shot best; ties on distribution targets, loses on length), fig5 why linear wins (gains <=0.01; 440/3584 directions; capacity), fig6 data not redundant but badly allocated, fig7 question-vs-chain decomposition (**marked PRELIMINARY**). Colors are slots 1-3 of the dataviz skill's default palette, used unchanged; significance is carried by filled-vs-hollow markers, not color alone (the palette validator needs `node`, which is not installed here, so it was not re-run). Every figure rendered and visually checked |
| **Smaller-prompt ICL** (prof's ask 1): new `--ex-sents N` in `icl_predict.py` shortens each EXAMPLE's reasoning to its last N sentences, keeping the question + choices in full; query untouched. Gemma-4-31B, thinking off, N=2, k {10,30,93,400} | raw `runs/goal2/gpqa25/icl/raw/gemma4_31b_off_s2_k<k>_fold<f>.csv` (tag carries `_s2`, so it cannot resume into the full-prefix files); log `runs/logs/round2/icl_gemma4_31b_job4942955.log` | `HF_HOME=$PWD/cache/huggingface PY_ICL=$PWD/envs/icl/bin/python ICL_MODEL=gemma4_31b ICL_KS="10 30 93 400" ICL_EXTRA="--ex-sents 2" sbatch --gres=gpu:h100:2 --mem=192G -t 08:00:00 runs/gpqa25/11_icl.sh`, job 4942955 | done 2026-10-26 (1 h 11 min, 2 GPUs, 20/20 files, **0 parse failures**). **Shortening the examples helps a little, but does not change the conclusion.** Shortened scores HIGHER in 5 of 6 same-k cells (3 k x 2 targets). Entropy: k10 0.193 vs 0.176, k30 0.194 vs 0.144, k93 0.171 vs 0.158. rem_len_mean: k10 -0.060 vs -0.162 (+0.102, **sig.**), k30 +0.014 vs -0.006, k93 +0.039 vs +0.041. Only the k10 rem_len cell is significant on its own and 5/6 same-sign is not significant by a sign test, so **suggestive, not established** (an earlier note here wrongly said shortening DESTROYS rem_len - it improves it; what collapses rem_len is having any examples at all, 0.283 at k=0 to ~0 at k>=10). k=400, reachable only with shortening, is the WORST shortened config (0.156). Every shortened config stays sig. below our probe (entropy -0.115 to -0.153) and below Gemma k=0 (0.262). **So the answer to the ask is: smaller prompts buy 4x more examples and a small, mostly-not-significant improvement at fixed k; zero-shot remains the best in-context setting by a wide margin (0.262 vs the best shortened 0.194).** Reuses `generation.checkpoint.split_sentences` (exact round trip). Checked on CPU first: the 0% row's text is an exact prefix of all 24,566 other rows of that question, so question/reasoning split cleanly; shortening gives 8.5% of full example length and leaves 0% rows untouched. Longest fold-0 prompt: full k=93 250,604 tokens (at the 262k ctx limit) vs N=2 k=400 143,575 - **shortening buys k=93 -> k=400**. k 10/30/93 are same-k controls against the existing full-prefix runs (isolates example length); k=400 is only reachable with shortening. First try 4942953 failed instantly: env.sh resolves ENVS_DIR to /work for user `yuvraj` - see learnings.md |
| **Briefing notes for the meeting** (detailed background behind every slide, plain language, with the file behind each number) | `docs/briefing_2026-10-27.md` | Claude, 2026-10-26 | done 2026-10-26 |
| **Meeting deck (2026-10-27)**: 18 plain slides - bullets + figures only, no styling, per the user's instruction | https://claude.ai/artifact/JLTVCFvt2Go7XeU7qE9Dso ; **slide sources saved to `docs/deck_2026-10-27/project/` on 2026-10-29** (18 slides + `deck.json`; the original session scratchpad is gone, so this local copy or `Artifact action:"read"` on the URL are the only sources). Figures from `docs/figures/slides/`; outline + questions for the professor in `docs/meeting_2026-10-27.md` | Claude, 2026-10-26 | done 2026-10-26. Order follows HIS ask numbering: recap -> status slide mapping all 10 asks to slide numbers -> ask 1 (smaller prompt) -> 2 (frontier) -> 3 (hiding the checkpoint) -> 5/6/9 (22 approaches, 2 slides) -> 7 (LoRA) -> our own two investigations (data redundancy; why linear wins, 2 slides each) -> the preliminary question-difficulty finding (marked preliminary on the slide) -> asks 4 and 8, decisions needed. No em dashes (user's request). All figure slides use one height recipe so nothing is auto-shrunk. PRIVATE until shared from the page's Share menu. **Updated 2026-10-27 (v16-18)**: (a) ask-3 wording fixed - "drops 0.12 to 0.24" read as a rising range, now "R2 falls by 0.12, 0.24, 0.17 at k = 0, 10, 50" (same fix in `docs/meeting_2026-10-27.md` and the briefing's ask-3 table); (b) the approaches slide now LISTS the 22 by family (user's ask: legible at a glance), with the caveat that only OLMo-3 and Mistral are other FAMILIES - Qwen2.5-Instruct/Math are Qwen relatives; (c) **every slide now quotes BOTH answer entropy and remaining length** (user's standing ask, previously applied only to the figures): data-1 learning curve 0.15->0.30 / 0.38->0.55, data-2 allocation 0.20 vs 0.20 / 0.41 vs 0.41, frontier Luna k=0 0.294 vs 0.309 / 0.455 vs 0.563, smaller "5 of 6 cells" spelled out as entropy+length at k 10/30/93; (d) the LoRA slide now carries the full per-cut TABLE for both targets. All LoRA numbers re-verified this session: the 10 diffs independently from `lora/ci_f0.*.csv` and the LoRA-side absolutes by recomputing r2_score on `lora/f0*_L21_r16_ridge/preds.csv` - all match the published values. **2026-10-29: the live deck is version `1793299783-91af`, NEWER than what this repo's copy was published from** - the LoRA slide has been restyled in the Slides app (same numbers, different markup), so someone edited it after 2026-10-27. `docs/deck_2026-10-27/` holds that live state. **Before republishing, `Artifact action:"read"` the URL first and merge onto the live version** or the publish is refused |
| **Round-5 write-up** (the professor's ten asks, one loose end closed; audience: team) | `docs/update_2026-10-27.md` | Claude, 2026-10-27 | done 2026-10-27. Covers asks 1/2/3/5/6/7/9, the logistic C-grid fix and its two new findings, the 22-approach inventory, and the reporting conventions this round reinforced (score within cut point; always report entropy AND remaining length; AUC alongside accuracy for commitment). Asks 4 and 8 listed as needing his design call |
| **Question-difficulty bands (groundwork for the 2026-10-27 meeting actions 1-2)**: per-question difficulty from the 0% cut's `p_gold` (= fraction of the 25 base CoTs that are correct), two banding schemes, plus the fixed fold | `runs/goal2/gpqa25/sweep/_difficulty/question_difficulty.csv` (198 rows: `example_id`, `p_gold`, `n_correct` 0-25, `band_int`, `band_quintile`, `fold`) | Claude, 2026-10-29, from `labels/checkpoint_labels_f0.csv`; folds from `sweep.fixed_fold_of(run_dir,5,0)` | done 2026-10-29, **no new compute**. Spread is wide and usable (mean p_gold 0.494, 17 questions at 0/25, 28 at 25/25). `band_quintile` is ~40 q each; `band_int` is 17/49/56/48/28. **Difficulty vs entropy is NOT monotonic**: mean entropy by band_int v_easy 0.050, easy 0.508, medium 1.061, hard 1.028, **v_hard 0.642** (confidently wrong). In-sample R2 from `p_gold` alone (quadratic) - entropy 0.586/0.461/0.358/0.262/0.186 at 10/30/50/70/90%, p_mode 0.531->0.152, rem_len_mean 0.194->0.068, rem_len_std 0.084->0.047 (our probe: 0.289/0.251/0.524/0.298 mean). **CAVEAT, do not misreport: `p_gold` is label-derived (computed from rollouts), so this measures SHARED VARIANCE between difficulty and the targets, not a usable baseline** - and it is in-sample, so an upper bound. The legitimate version is the unrun question-text-only baseline. Smallest (band_quintile, fold) cell is **3 questions**, so per-fold calibration inside one band will be fragile |
| **Round 6, concern 2 (FREE, no compute, no API spend): is the frontier model's tie also a question-difficulty read? YES.** Part 1 row structure; part 2 difficulty partialled out of target AND prediction (quadratic in `p_gold`, per cut, refitted inside each of 2000 question-bootstrap draws); part 3 commitment accuracy + AUC with paired CIs; part 4 between-question vs within-question-ACROSS-DEPTH | `runs/goal2/gpqa25/sweep/_iclsplit/{p1_row_structure,p2_per_checkpoint,p2_ci_scores,p2_ci_paired,p3_committed,p3_committed_auc_ci,p3_agreement,p4_between_across_depth}.csv` | `src/analysis/icl_decompose.py` (new), login node, ~3 min | done 2026-10-29. **Cross-check: reproduces `_icl/ci_scores.csv` to 1.1e-15 over 28 cells.** (a) **The requested round-4 between/within decomposition is DEGENERATE on the ICL rows**: 1 CoT per question per cut, so within-question SS is exactly 0.000 at every cut and every cut's R2 is 100% between-question by construction (full `ref_fixed` set: within share 0.183/0.291/0.394/0.473/0.572). So the published Luna-vs-ridge comparison was never a test of within-question skill. (b) **Difficulty removed, the entropy tie vanishes for everyone**: ours 0.309 -> **0.092 [-0.010,0.180]**, Luna k=0 0.294 -> **0.037 [-0.072,0.140]**, TF-IDF 0.267 -> 0.066, Gemma k=0 0.262 -> 0.045; none sig. above 0. Only prefix_len stays sig. (0.247 -> 0.121 [0.008,0.221]). (c) **Remaining length barely moves and our edge survives**: ours 0.563 -> 0.528 [0.469,0.587], Luna 0.455 -> 0.397; ours beats Luna by +0.131 [0.070,0.195] and TF-IDF by +0.144 [0.092,0.194] AFTER partialling. p_mode 0.261->0.060 (ours) / 0.233->0.019 (Luna); rem_len_std 0.328->0.325 / 0.122->0.107. (d) **NEW, changes a published conclusion: on `committed_by_here` accuracy ties but AUC does not.** Ours AUC 0.773 vs 0.651-0.677 for ALL NINE ICL configs, diff -0.096 to -0.122, **all significant**, while accuracy is 0.698-0.742 vs our 0.727 (majority 0.634). prefix_len 0.761 and TF-IDF 0.757 are n.s. vs ours. Round 5's "commitment ties" was an accuracy artefact. (e) Part 4: across-depth variation is just depth - a depth-only reference (leave-one-fold-out mean per cut) scores 0.262 within-question-across-depth on entropy, ABOVE our probe's 0.216, so that dimension is uninformative. **CAVEAT: `p_gold` is label-derived so partialling over-removes; the partialled numbers are a LOWER bound** |
| **Round 6, action 2 done for $0: Luna and the probe scored INSIDE each difficulty band** (`band_quintile`, ~39 q each). No API calls needed - the query prompt does not depend on the band and `example_order` draws examples from all training-fold questions regardless, so subsetting the stored predictions IS the band rerun. Also removes the calibration worry: `icl_cal` was fitted per fold on the FULL other-fold rows, so nothing is refitted on a 3-question cell | `runs/goal2/gpqa25/sweep/_iclsplit/p5_bands_{scores,paired}.csv` | `src/analysis/icl_bands.py` (new) | done 2026-10-29, **CORRECTED the same day** (see below). **Confirms the falsifiable prediction written in `NEXT_STEPS.md` before it was run, on both rows.** **Headline numbers are now OFFSET CORRECTED** (`score_recentred` + `recentred_ci_*`, added to `icl_bands.py`): the stored predictions come from a fit on all 198 questions and are never recentred per band, so an uncorrected band score is charged for an offset the full-set fit never carried. Recentring costs ONE degree of freedom and leaves the slope alone, so under-confidence is still penalised. Entropy R2, corrected (all/v_hard/hard/medium/easy/v_easy): ours +0.311/+0.148/+0.174/+0.108/+0.110/-1.09, Luna k=0 +0.294/+0.062/+0.143/-0.071/+0.142/-0.49, prefix_len +0.248/+0.180/+0.172/-0.076/+0.210/-0.35 - **still collapses from ~0.30 to 0.11-0.17 inside every band**. rem_len_mean corrected: ours +0.563/+0.591/+0.564/+0.519/+0.512/+0.491 - **flat**; Luna +0.455/+0.537/+0.429/+0.350/+0.283/+0.501. (Uncorrected, for the record: entropy ours +0.309/+0.131/+0.057/-0.143/+0.096/**-2.71**, Luna .../-1.70; both columns are kept in `p5_bands_scores.csv`.) Corrected CI half-widths 0.14-0.26 for entropy in a band, so no single band is decisive alone. **The v_easy band stays negative after correction and that part is REAL, not an artefact**: there the truth is nearly constant (sd 0.265 vs 0.642 overall) while the predictions go on varying as much as ever, so they add error. The offset-free within-cut correlation is the most forgiving view - entropy corr ours 0.559 -> 0.392/0.422/0.382/0.398/0.234 and Luna 0.542 -> 0.321/0.416/0.193/0.430/0.356 (both drop ~1/3), while rem_len_mean corr ours 0.753 -> 0.779/0.757/0.722/0.730/0.719 is FLAT and ahead of Luna in all five bands |
| **Round 6, concern 1 (FREE): why do in-context examples HURT? They supply the SCALE and break the RANKING** | `runs/goal2/gpqa25/sweep/_iclwhy/{anchoring,vs_zero_shot,h2_example_noise_by_fold}.csv` | `src/analysis/icl_anchoring.py` (new); RAW (uncalibrated) predictions, 16 configs x 2 model families | done 2026-10-29. Across the 28 comparisons that have a same-model k=0 control (Luna `ckpt`/`blind`, Gemma `ckpt`, 4 targets): **within-cut correlation falls vs k=0 in 28 of 28**, mean -0.131, worst -0.335 (by target: entropy -0.075, p_mode -0.056, rem_len_mean -0.244, rem_len_std -0.147); spread ratio sd(pred)/sd(truth) moves CLOSER to 1 in 19/28 and raw R2 IMPROVES in 15/28 (mean +0.40), most clearly where zero-shot had no scale at all (Gemma rem_len_mean 0.057 -> 0.99, Luna blind 0.197 -> 0.78). **Mechanism: the published comparison is calibrated, so the scale gain is given away free and only the ranking damage counts -> k=0 wins, in both families.** H1 verdict: classic shrink-to-the-example-mean anchoring is NOT happening (spread widens; distance to the example mean stays 0.6-1.0 target sd). **Verbatim copying is real but modest and entropy-only**: excess over a chance baseline (same model's k=0 predictions matched to the example values that fold would have seen) is +0.08 to +0.29 on entropy at every k in both families, ~0 on both length targets. **The chance baseline is essential** - Gemma's raw p_mode match rate hits 0.94 at k=93 against a chance rate of 0.955 (p_mode is shown to 2 dp on a short range), so the uncorrected number is meaningless. Copying contributes but cannot explain it: rem_len_mean copying is ~0 yet Gemma's correlation falls 0.547 -> 0.234. **H2 (noisy example labels) is UNTESTABLE with these runs**: `example_order` depends only on (fold, k), so all rows in a fold saw an identical example set - 5 points per config, and the fold correlations are inconsistent in sign. **Consequence for ask 4: GEPA example selection optimises toward k=0 and is not worth running; instruction optimisation is untouched** |
| **Round-6 slide figures (3 PNGs): the difficulty-stratification result** | `docs/figures/slides/fig{9_difficulty_bands,10_difficulty_partialled,11_bands_paired}.png` | `src/viz/round6_figures.py` (new; reads only `_iclsplit/p2_ci_scores.csv`, `p5_bands_*.csv`, no refitting) | done 2026-10-29. Palette slots 1-3 unchanged from `fig1`-`fig8` so they drop into the existing deck; series identity carried by marker shape as well as hue; **every point has a 95% question-bootstrap interval** (the correlation CIs were added to `icl_bands.py` for this, `corr_ci_lo/hi` and `corr_significant`). fig9 is 2x2: R2 and the offset-free within-cut correlation, for entropy and remaining length, vs band, with "all questions" left of a divider. **Its top row was switched to the OFFSET-CORRECTED R2 on 2026-10-29** (`score_recentred`), so the absurd uncorrected v_easy values are gone; see the action-2 Results index row for why and for both sets of numbers. fig10 is the partialling dumbbell (hollow = as published, filled = difficulty removed). fig11 is the paired Luna-minus-probe forest per band. Figures say **"Best probe"** for our L21 ridge (user's wording); strictly, L20-22 concatenated beats it by +0.005 entropy R2 in the master table. The palette validator needs `node`, which is not installed here, so it was not re-run; the slots are the already-validated ones. All three rendered and visually checked. **Numbering: `slide_figures.py` already makes fig1-fig8** (`fig8_smaller_prompt`), so the round-6 set is fig9-fig11; an earlier draft collided on fig8 and was renamed. **`fig7_difficulty_preliminary.png` was also REBUILT 2026-10-29** (in `slide_figures.py`, its own generator): it had no error bars and an em dash in the title. It now reads `_redundancy/decomposition_ci.csv` for 95% intervals, shades the below-zero region with the label "worse than guessing the question's own average", moves the legend out of the data, and explains negative R2 in the caption. Still marked PRELIMINARY |
| **Round 6 follow-ups (user questions, 2026-10-29): (a) does the band result survive refitting? (b) are the probe and Luna just predicting the mean?** | `runs/goal2/gpqa25/sweep/_iclsplit/{p6_band_refit,p7_band_spread}.csv`; figure `docs/figures/slides/fig12_predicting_the_mean.png` | `src/analysis/icl_bands_refit.py`, `src/analysis/icl_bands_spread.py`, `fig12` in `src/viz/round6_figures.py` (all new) | done 2026-10-29. **Provenance confirmed: fig9 does NOT retrain anything** - `ref_fixed/preds*.csv` are out-of-fold predictions from a ridge fitted on all 198 questions minus the held-out fold (24,566 rows, one prediction per row, verified), and the bands are a row FILTER on them. **(a) Refitting the output map inside a band does not rescue entropy.** Best probe, mean over cuts, as-stored / +intercept / +intercept+slope - entropy: v_hard 0.122/0.134/0.150, hard -0.041/0.089/0.119, medium -0.181/0.041/0.109, easy 0.054/0.060/0.134, v_easy -2.250/-0.651/**0.091** (all questions 0.289). rem_len_mean: 0.559/0.566/0.571, 0.467/0.484/0.488, 0.422/0.434/0.435, 0.488/0.496/0.499, 0.150/0.350/0.416 (all 0.524). So the v_easy blow-up IS mostly the uncorrected offset, but even a best-case affine refit leaves entropy at 0.09-0.15 in every band while length stays 0.42-0.57. NB this check runs on the FULL ref_fixed rows (~975/cut/band, all 25 chains), so it is better powered than fig9 and is an independent confirmation. Probe WEIGHTS were NOT refit inside a band (39 questions vs 3,584 dims - would be noise). **(b) Neither is predicting a constant, but both are badly shrunk.** Entropy, inside bands: sd(pred)/sd(true) ~0.47-0.55 (a constant predictor would be 0) and within-cut corr 0.19-0.43. The real failure is the MEAN: true band means (v_hard->v_easy) 0.807/1.086/1.086/0.627/**0.100** vs probe 0.869/0.905/0.861/0.667/**0.404** and Luna 0.947/0.941/0.910/0.656/**0.365** - both compress the 0.99-bit true range into ~0.50 and are 4x too high on very easy questions. Remaining length is the opposite: predicted means track the truth and within-band corr stays 0.5-0.8. Luna shown raw AND calibrated (calibration rescales spread, so raw matters) |
| **GEPA prompt optimisation, PREP ONLY (round 5 ask 4)** - by a DIFFERENT session on 2026-10-31, running as user `rishp`. Not my work; recorded here so it is not duplicated | code `src/baselines/gepa_luna.py`, launcher `runs/gepa/run_luna.sh`, env `envs/dspy` (dspy 3.4.1, gepa 0.1.4, py 3.12.15); outputs `runs/goal2/gpqa25/gepa/{rows.jsonl (1,178 rows), seed_instructions_{ckpt,blind}.txt}`; logs `runs/logs/gepa/` | that session | **prep done, NOT RUN. BLOCKED on API credit.** `prep` has run and an offline `DummyLM` dry run completed end to end (1,512 task + 205 reflection calls, $0). Nothing real has been called: there are no `icl/raw/*gepa*` outputs. **The blocker (their `learnings.md` entry, 2026-10-31): `rishp`'s `~/.frontier_api_key` has NO credit.** The 2026-10-26 Luna runs went through the **`yuvraj`** account's key, which is the one that worked, so a session running as `yuvraj` may be able to proceed - **ask the user before spending.** Design: per outer fold, GEPA optimises on the other folds' questions (70/30 train/val by question) and is tested on that fold's rows, i.e. the same rows the Luna runs and the probes were scored on, writing `icl_predict`-format CSVs so the existing scorer is reused. **Note the round-6 finding that bears on this: examples HURT, so only INSTRUCTION optimisation is worth anything here, and this design optimises the k=0 instruction, which is the right call** |
| **Round-6 write-up (part 1)** | `docs/update_2026-10-29.md` | Claude, 2026-10-29 | done 2026-10-29. Audience: the team. Covers concern 2 (settled), concern 1 (mechanism), action 2 (done for $0), the commitment-AUC finding, and the calibration asymmetry that none of it settles |
; default Slurm `slurm-<id>.out` files moved from the repo root to `runs/logs/slurm/` (2026-10-12); all `runs/gpqa25/*.sh` now set `#SBATCH --output=runs/logs/slurm/slurm-%j.out` (relative to the submit dir — submit from the repo root) | — | — |

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
- [ ] 4d. Follow-ups: other layers (**24 of 28 still unprobed**; L14/L13–15/L20–22 done in the multilayer
      sweep); 0% checkpoint probes (**needs a ~30-line change in `sweep.py:88`, which hardcodes
      `labels/checkpoint_labels.csv`** — the f0 labels are a separate file, `cot_id=-1`, no commitment
      columns — plus a config); ~~bootstrap CIs~~ (done as R5); best probes on embeddings/LLM controls
      (**not done**: `feat_*` sweeps are ridge + `logistic_wide` only); ~~unrelated-LLM control~~ (done in
      round 3: OLMo-3, Mistral)
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

### Round 4 (user list 2026-10-22; run overnight 10-22/23 by three parallel agents; specs `docs/plan_*.md`, write-up `docs/update_2026-10-23.md`)
- [x] Idea 1. **Dataset redundancy.** Are the 25 chains per question too similar to learn from? Answer: no, but the data is badly allocated. Cosine separation AUC (standardised, L21) 0.80→0.71 over 10–90%, vs TF-IDF text at 1.000. Learning curve rises monotonically (entropy at 50%: 0.152 at m=1 → 0.298 at m=25, no plateau). **At fixed rows, 150q×2 (300 rows) ≈ 30q×25 (744 rows)**, i.e. same quality for 2.5× fewer rollouts; a new question is worth ~3.5× a new chain per row. Also found (**NOT yet shared with the team — user wants to review it first**) that the probe is mostly a question-difficulty reader: see the idea-1 Results index row.
- [x] Idea 2. **Close the MLP/ridge gap.** Done, and the gap closes. Prior "MLPs never beat ridge" was a **training-recipe artefact**: over 80 fits L-BFGS reproduces ridge at weight cosine 1.0000 while the step-4a recipe sits at 0.865 and costs 0.008–0.025 R². With cross-fitted residuals + square activation + 5-seed ensembling, the MLP beats ridge by **+0.0039 [+0.0018, +0.0063]** on `rem_len_mean` (+0.0026 vs an oracle-tuned ridge). Closed-form degree-2 **kernel ridge is the better probe at +0.0081**. **entropy, p_mode and committed are linear-ceilinged** (kernel gain n.s. at every checkpoint) — so qualify any blanket "ridge is best" claim. Do not spend more effort here.
- [x] Idea 3. **Literature review.** → `RESEARCH.md`.

### Round 5 (professor's feedback list, relayed by the user 2026-10-26)
**8 of 10 done. Write-up: `docs/update_2026-10-27.md`.** The two open ones need his design call, not compute.
His ten items, with status. Items 1–4 were already covered by rounds 3–4; see the Results index.
- [x] Activations from a different family — OLMo-3-7B (ties ours on all 5) + Mistral-7B (round 3)
- [x] Random forest hyperparameter grid search — `4c_rf_*`, `rf_reg`, plus HGB; R3 explains why trees lose
- [x] See if LoRA is working — R8; the head was the bottleneck, ridge-on-LoRA-states gains +0.01–0.026
- [x] Thorough embeddings analysis — 5 embedders; the best (UltraBERT) only reaches TF-IDF
- [x] Remove checkpoint location — new `blind` version in `icl_openai.py` (hides it on examples AND query).
      Full run: **no difference on entropy/p_mode/committed, but sig. WORSE on both length targets at
      every k** (rem_len_mean -0.12 to -0.24). The 100-row check's "no difference" held only for the
      non-length targets
- [x] Clearer picture of the results — master table (`sweep/_master/`) + interactive dashboard
- [x] Frontier model — Luna FULL 1,178-row run of all 8 configs DONE 2026-10-26 ($17.86, 9,424 calls,
      0 parse failures). Ties ridge on entropy/p_mode/committed, sig. worse on both length targets;
      k=0 beats k=10 and k=50, i.e. examples hurt. See the Results index row
- [x] In context with a SMALLER prompt — done 2026-10-26 (job 4942955). Shortening examples to their last
      2 sentences cuts them to 8.5% and lifts max k from 93 to 400+. It scores HIGHER in 5 of 6 same-k
      cells (only one sig. on its own, so suggestive), but k=400 is still the worst setting and zero-shot
      still wins by a wide margin. See the Results index row
- [ ] Black-box monitor with other REPRESENTATIONS — needs the professor's design call. Options written
      up for the 2026-10-27 meeting in `docs/meeting_2026-10-27.md` (question only / last-N sentences /
      trivial scalars). The last-N-sentences variant is now cheap: `--ex-sents` exists
- [ ] Prompt optimization with GEPA — **groundwork now EXISTS** (a different session, 2026-10-31, as
      user `rishp`): `src/baselines/gepa_luna.py`, `envs/dspy`, `runs/gepa/run_luna.sh`, and `prep` has
      run. **Blocked: that account's API key has no credit** (the Luna runs used the `yuvraj` key). It
      optimises the k=0 INSTRUCTION, which round 6 says is the only version worth running, since
      examples hurt. See the Results index row. Ask the user before spending.

### Round 6 (user's notes from the 2026-10-27 meeting, relayed 2026-10-29) — **full detail in `NEXT_STEPS.md`**
**Concern 2 is SETTLED and concern 1 has a mechanism, both for free (2026-10-29). Write-up:
`docs/update_2026-10-29.md`.** Four actions and two concerns, but **not six pieces of work**:
**actions 1 + 2 ARE the investigation of concern 2** (stratify by difficulty, then rerun Luna inside one
band - that is how you find out whether the frontier model's zero-shot tie with our probe is really both
of them reading question difficulty). Concern 1 is genuinely separate and free to test. Actions 3 + 4 are
about getting off GPQA. So: one investigation (1+2, answering concern 2), one free diagnostic (concern 1),
one data-collection track (3+4).
- [x] **Action 1. Stratify GPQA questions by difficulty** (how many of the 25 attempts are correct).
      Done: `_difficulty/question_difficulty.csv`, no new compute. **`band_quintile` used** (equal ~39 q
      sizes, so a between-band difference is not confounded with sample size). `band_int` kept for the
      interpretable extremes.
- [x] **Action 2. Score GPT-6 Luna inside each difficulty band — DONE FOR $0, no API calls.** The query
      prompt does not depend on the band and `example_order` draws examples from all training-fold
      questions regardless, so subsetting the stored predictions IS the rerun. That also removed the
      3-question-cell calibration worry (`icl_cal` was fitted per fold on the FULL other-fold rows, so
      nothing is refitted). All five bands, not one. **Entropy collapses inside every band for both of
      us; remaining length is flat for our probe and still ahead of Luna.** See the `_iclsplit/p5_*`
      Results index row. The requested between/within decomposition of Luna's predictions turned out to
      be **degenerate** (1 CoT per question per cut), which is itself a finding - see the `_iclsplit/`
      row. Not worth paying for: the only version that would need new calls is drawing the EXAMPLES from
      the query's band, which is a different question (and concern 1 says examples hurt anyway).
- [ ] **Action 3. Find a subjective dataset** (no objective answer). Nothing done. Needs a written proposal
      before any collection: what the label becomes when `p_gold` is undefined, how answers get parsed
      (`run_eval.parse_output` assumes a `{"Final Answer": ...}` JSON), and whether `committed_by_here`
      still means anything. The remaining-length targets survive unchanged and are the cleanest start.
- [ ] **Action 4. Get Scott to run MMLU-Pro + a subjective dataset.** `docs/GUIDE_new_datasets.md` already
      exists and is standalone, BUT it targets `math500` + `mmlu_pro`, not a subjective set, so the
      subjective half depends on action 3. **Ask the user whether to send it for MMLU-Pro now or wait.**
      Already decided, do not reopen: noisy 0% labels are accepted (commit `043218f`).
- [x] **Concern 1. Zero-shot beats in-context for frontier models — ANSWERED 2026-10-29, free.**
      **Examples supply the output SCALE and degrade the RANKING.** Within-cut correlation falls vs k=0 in
      **28 of 28** comparisons (two families), mean -0.131; spread ratio moves toward 1 in 19/28 and raw
      R2 improves in 15/28. Because the published comparison is calibrated, the scale gain is free and
      only the ranking damage shows. Classic anchoring (shrink to the example mean) is NOT happening;
      verbatim copying is real but modest and entropy-only (+0.08 to +0.29 over a chance baseline, ~0 on
      the length targets). The noisy-label hypothesis is **untestable with these runs** (one example set
      per fold). See the `_iclwhy/` Results index row. **Verdict for ask 4: GEPA example selection is not
      worth running; instruction optimisation is untouched.**
- **Concern 2 (NOT a separate task; settled by actions 1 + 2 above, both now done). SETTLED 2026-10-29:
      the tie IS question difficulty.** Partial a quadratic in `p_gold` out of target and prediction and
      entropy R2 falls 0.309 -> 0.092 for us and 0.294 -> 0.037 for Luna, neither sig. above 0; remaining
      length goes 0.563 -> 0.528 and our edge over Luna stays sig. at +0.131. Band stratification says the
      same thing independently. **The frontier
      model ties our probe on entropy, zero-shot** (0.294 vs 0.309, n.s.), with no
      activations and no examples. Recorded here for the reasoning. **Most likely already explained by our own round-4 finding**: the probe is mostly a
      question-difficulty reader, and difficulty is legible from the question text. Falsifiable prediction
      (table in `NEXT_STEPS.md`): stratifying should **kill the entropy tie** (difficulty shares 0.586->0.186
      of entropy variance) **and leave our remaining-length advantage intact** (only 0.194->0.068 shared). If
      Luna still ties us on entropy *inside* one band, difficulty was NOT the explanation and this reopens -
      the more interesting outcome. The one residual stratification does NOT settle: **ICL numbers are
      calibrated and our probe's are not** (raw Luna is far below zero) - defensible, since at k=0 it was
      never told the scale, but it must be said in any write-up.

### Loose ends verified 2026-10-26 (were stale or unrecorded)
- **Logistic C grid is edge-pinned for `committed_by_here` on our activations**: `logspace(-4,1,6)` picks
  C=1e-4 (the bottom edge) in **18 of 25 folds** (all 5 at both 10% and 90%). So the 0.741 committed accuracy
  in every write-up and on the dashboard comes from edge-pinned fits. Fix: extend to 1e-8, RENAME the probe
  (sweep.py skips by probe name), rerun. → **DONE 2026-10-26, job 4941954: the published 0.741 stands**
  (0.7409 -> 0.7403). Flat region, not a truncated optimum. But it surfaced that **`committed_by_here`
  accuracy at the 90% cut is just the base rate** (majority 0.7609 vs ours 0.7583) and that **under AUC
  our edge over TF-IDF on this target nearly vanishes** (0.7612 vs 0.7581). See the Results index row.
- **24 of the 28 stored layers have never been probed** (only L21, L14, L13–15, L20–22 + PCA variants).
- **No non-linear probe has been run on the other-LLM or embedder features** (those sweeps are ridge +
  `logistic_wide` only). Lower value now that kernels found nothing on 3/5 targets for our own features.
- `docs/HANDOFF.md` is **stale**: its open item 1 (unrelated-LLM control) was done in round 3 by OLMo-3 and
  Mistral. Fix when next editing it.

## Next steps

> **The live to-do list is `NEXT_STEPS.md` (repo root), written 2026-10-29 from the user's notes after
> the 2026-10-27 meeting. Start there.** The section below is the longer-standing plan; items 2 and 3 of
> it (the 0% checkpoint probes and the question-text-only baseline) bear directly on round 6's concern 2
> and are now MORE valuable than they were, not less.

**Where we stopped (2026-11-01).** Rounds 1-5 done. **Round 6: concern 2 is settled, concern 1 has a
mechanism, and actions 1 and 2 are done, all with zero compute and zero API spend**
(`docs/update_2026-10-29.md`; outputs in `sweep/_iclsplit/` and `sweep/_iclwhy/`). Still open in round 6:
actions 3 and 4 (getting off GPQA), which need the user's decisions. **No jobs are running and nothing
is queued** (`squeue` empty, checked 2026-11-01).

**Two things other sessions did that this file now records, so you do not redo them:**
- **GEPA prep (2026-10-31, as user `rishp`)**: code, env and `prep` all exist; **blocked on API credit**
  for that account. See the Results index row. Round 5 ask 4.
- **GlobalOpinionQA + normalised entropy** (commit `fe985b1`), which is groundwork for round 6 action 3
  (subjective dataset). Check with whoever owns it before starting action 3 from scratch.

**The headline to carry forward.** The frontier model's zero-shot tie with our probe on answer entropy
**was question difficulty**: remove a quadratic in `p_gold` and entropy R2 falls 0.309 -> 0.092 (ours)
and 0.294 -> 0.037 (Luna), neither significantly above zero. **Remaining length is the part that is
ours**: 0.563 -> 0.528 after partialling, flat across all five difficulty bands, and significantly ahead
of every in-context model and of TF-IDF. Also: on `committed_by_here` the ICL models tie us on accuracy
but lose by 0.10 to 0.12 AUC, all significant, so round 5's "commitment ties" was an accuracy artefact.
 The 2026-10-27 meeting happened; the deck,
dashboard and briefing are all built and both artifacts are **private** until the user shares them from
each page's Share menu. 8 of the professor's 10 asks are answered (`docs/update_2026-10-27.md`); the two
open ones (GEPA, black-box monitor) need his design call. A **different session** meanwhile added the
full MMLU-Pro option, dataset/subset selection and `docs/GUIDE_new_datasets.md` (commits `c8c301c`,
`043218f`), which is most of the groundwork for round 6's action 4.

**Where the research question stands.** The question is: do representations at a CoT prefix predict
properties of the distribution over future continuations, beyond the prefix text? **On its own terms
the answer is yes** - entropy R2 0.289, rem_len_mean 0.524, and we beat TF-IDF on the prefix by
+0.052 on entropy (significant). Round 4 changed the *characterisation*, not the answer. The sentence
to build the paper around:

> Activations beat text at judging **which questions** will produce uncertain rollouts. They do not
> beat text at judging **which chain of a given question** will.

(Within a question, activations are roughly level with TF-IDF and with prefix length; between
questions they are clearly ahead. Nobody else in this literature decomposes their R2 this way.)
The stronger "the model is tracking where *this* chain is heading" reading is **not** supported.

Round 5 sharpened the baseline picture in a way that matters for the paper: **the strongest baseline
is another LLM's activations on the same text** (OLMo-3-7B ties us on all 5 and shares no lineage with
our model), not the text baselines. The user asked explicitly that TF-IDF **not** be the headline -
the point of running 22 approaches was to find a strong baseline, and we did.

**Agreed next steps, in order.** Item 1 is now much better evidenced (see the round-6 work above), and
items 2 and 3 are the two experiments that would close the question-difficulty story for good:
1. **Discuss the question-difficulty finding with the user.** A conversation, not a job. It decides
   what the paper claims and everything else is downstream. It is deliberately **withheld from
   team-facing material** until then (internal docs only; it IS on slide "reading" of the deck and on
   the dashboard's last tab, both marked PRELIMINARY, and both still private).
2. **Run the 0% checkpoint probes. Highest-value single experiment.** Data ready since 2026-10-04
   (`runs/goal1/gpqa25/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B/labels/checkpoint_labels_f0.csv`,
   `runs/goal2/gpqa25/activations_ckpt_f0/`); needs a ~30-line change in `src/probing/sweep.py:88`
   (it hardcodes `labels/checkpoint_labels.csv`; the f0 labels are a separate file, `cot_id=-1`, no
   commitment columns) plus a config. A few CPU-hours. Three reasons: (a) prefix length is constant at
   0%, so the length confound vanishes entirely; (b) one row per question means the probe can *only*
   read question difficulty, so its R2 is a direct **measurement** of the difficulty floor rather than
   an inference; (c) paper 2 in `RESEARCH.md` probes exactly this position with a clean per-question
   split and a raw target, so our result becomes directly comparable to a published NeurIPS 2025
   number. **Report R2 and Spearman** (Spearman is what they published).
3. **Cheap companion:** predict each question's mean label from the **question text alone** (no
   prefix, TF-IDF). If that recovers most of our R2, the question-level story is settled beyond
   argument.
4. **Then decide the paper's framing** from 1-3, and the conventions in `RESEARCH.md` (report raw and
   log scales for length; score within checkpoint and say so).

**Also waiting on the professor (round 5, asks 4 and 8).** Options and the exact questions to settle
them are in `docs/meeting_2026-10-27.md`. Ask 8 (black-box monitor with other representations) is
cheap to run once he picks: question-only / last-N-sentences / trivial scalars, and `--ex-sents`
already exists for the middle one. Ask 4 (GEPA) has no groundwork; note our k-curve says more examples
HURT, so example selection would optimise toward k=0.

**Reporting conventions to keep (reinforced in round 5).**
- Score **within each cut point, then average. Never pool across depth** (pooled, prefix_len 0.718 vs
  our 0.716).
- **Always report BOTH answer entropy and remaining length** - the user asked for this explicitly, and
  the two genuinely diverge (ICL ties us on entropy, loses badly on length; the degree-2 kernel gains
  on length, nothing on entropy).
- For `committed_by_here`, **report AUC alongside accuracy**, and give the majority-class baseline
  (0.6275 mean over 10-90%). Our AUC edge over TF-IDF on that target is only +0.003.
- Units trap: R2 for entropy/p_mode/rem_len_mean/rem_len_std, **accuracy** for committed. They sit
  side by side in every table.
- The user wants **no em dashes** in slides and docs, and plain bullet-only slides (no boxes, no
  clever layout).

**Deliberately deprioritised:** more MLP/probe-architecture engineering (closed - closed-form kernels
establish the ceiling and 3 of 5 targets have nothing to find); more LoRA (settled in R8); the two
rungs agent 2 left unrun (bagging over question bootstraps, gap-vs-training-size - commands in
`_mlpgap/`); collecting more questions or new datasets, even though the allocation result points that
way, until step 4 settles what we are trying to show (it is weeks of GPU).

**Open housekeeping.**
- **Unverified security item:** earlier in this work the frontier API key was briefly written to
  `temp.txt` in this group-shared repo at mode 644. The file is now 0 bytes, but **key rotation was
  recommended and never confirmed** - ask the user. The key lives in `~/.frontier_api_key` (mode 600).
- `temp.txt` (empty) and two large `tmux-*.log` files (19 MB) sit untracked in the repo root and can be
  deleted once the user says so.
- `docs/HANDOFF.md` is **stale**: its open item 1 (unrelated-LLM control) was done in round 3 by OLMo-3
  and Mistral, and it predates rounds 4-5 entirely. Fix when next editing it.

**Everything is committed and pushed (2026-11-01).** `origin/main` is `f70bda7`. Rounds 4, 5 and 6, the
GEPA prep and all project docs went up in seven commits (`bead7c1`..`f70bda7`), authored
`Yuvraj <yuvrajjain2003@gmail.com>`. The working tree is clean.

**`git fetch` and `git push` now WORK from the `yuvraj` account.** The box's existing SSH key
(`~/.ssh/id_rsa`, the Sycamore login key) was added to the `yuvrajjain2003` GitHub account on
2026-11-01, which fixed the long-standing "Permission denied (publickey)" noted in `learnings.md`.
So `git log HEAD..origin/main` can be trusted again after a fetch.

`temp.txt` and the `tmux-*.log` files are now gitignored and left on disk.

Commit/push only when the user says, and **do not commit other sessions' files without asking** (the
2026-11-01 push included another session's GEPA prep at the user's explicit request).

- Several Claude sessions may work in this repo at once: use a distinct output folder per experiment,
  and don't commit other sessions' files without the user's say.

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