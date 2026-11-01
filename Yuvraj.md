# Yuvraj's notes — what the project actually is and why the baselines exist

_Written 2026-10-22, as a catch-up after two weeks away. Personal notes, not a team doc.
Numbers here were read out of the result CSVs listed at the bottom, not from memory._

---

## 1. What was collected

198 GPQA-Diamond questions. For each, the model (`DeepSeek-R1-Distill-Qwen-7B`, "our model")
wrote **25 full chains of thought**.

Each chain was cut at five points — **10 / 30 / 50 / 70 / 90%** of the way through, by sentence
count — plus a **0%** cut (the rendered prompt, before any reasoning). Each cut is a **checkpoint**.

**Correction to my earlier mental model:** it is 3584 dims × **28 layers**, not 5 layers.
All 28 decoder layers are stored at every checkpoint. The "5" is the number of *checkpoints*,
not layers. In practice nearly everything uses **one layer — layer 21** — and the feature is the
hidden state at the **last token of the prefix**: one 3584-dim vector per
(question, chain, checkpoint). ~24,773 rows in total including the 0% rows; 9 of them have no
valid continuation and are dropped everywhere.

## 2. Where the ground truth came from

From each cut point the model was restarted and allowed to finish **25 independent times**.
Over those 25 continuations we measured:

| label | meaning |
|---|---|
| `answer_entropy_bits_mm` | entropy of the final-answer distribution, Miller–Madow corrected. 0 = all 25 agree |
| `p_mode` | fraction of the 25 that give the most common answer |
| `rem_len_mean`, `rem_len_std` | mean and spread of how many more reasoning tokens were written before finishing |
| `committed_by_here` | 1 if the answer is locked in at this checkpoint *and stays* locked in at every later one |

So the label describes **the distribution of possible futures** from that point. That is the thing
we are asking whether the hidden state knows about.

## 3. The probes

Ridge regression from the 3584-dim vector to each label. Fit **separately per checkpoint**.
Cross-validated over **5 fixed folds split by question** — a question's 25 chains share text, so
splitting by row would leak. Hyperparameters picked by an inner 3-fold split, then refit on the
full training split. Reference run: `runs/goal2/gpqa25/sweep/ref_fixed/`.

Headline (mean over the 10–90% checkpoints): **entropy R² 0.289, p_mode 0.251,
rem_len_mean 0.524, rem_len_std 0.298, committed accuracy 0.741.** Real signal, not huge.

## 4. Why the baselines exist — the important part

"R² = 0.29" means nothing on its own, because an R² above zero does not tell you the
*activations* did the work. Three different things could explain it, and each needs its own
baseline.

### (a) Maybe it is just the text

A longer prefix means less reasoning left — no neural network needed for that.

**Baselines:** prefix length alone, TF-IDF bag-of-words, sentence embedders
(Qwen3-Embedding-8B, F2LLM, UltraBERT, BGE-M3, Granite).

**Result:** activations genuinely beat all of these. Entropy +0.052 over TF-IDF
[+0.021, +0.083], significant. **But** prefix length alone matches the activations on remaining
length at the 10–50% checkpoints — so that particular result is substantially token-counting.
Activations only pull ahead there at 70% and 90%.

### (b) Maybe any language model's internals would do — this is the one that bit us

Take a **different** 7B model, feed it the same prefix text, pull **its** hidden state, run the
identical ridge probe. Nothing about the probe changes; only the feature vector is swapped.

- First: Qwen2.5-Math-7B and Qwen2.5-7B-Instruct → both tie our model.
- Worry: maybe they are secretly the same weights. **R2** checked this — same architecture and
  tokenizer, but our model is a fine-tune of Math (rel. diff 0.142) while Instruct/Base sit
  0.787 away. The weights genuinely differ, so the tie is real.
- Remaining worry: shared *pretraining lineage*. So round 3 added **OLMo-3-7B** and
  **Mistral-7B-v0.3** — different families entirely.

**Result — OLMo-3 is statistically identical to our model on all five targets:**

| target | ours (L21) | OLMo-3 (L24) | paired diff (95% CI) |
|---|---|---|---|
| entropy | 0.289 | 0.282 | −0.006 [−0.016, +0.003] |
| p_mode | 0.251 | 0.245 | −0.006 [−0.016, +0.003] |
| rem_len_mean | 0.524 | 0.521 | −0.003 [−0.011, +0.005] |
| rem_len_std | 0.298 | 0.306 | +0.008 [−0.001, +0.016] |
| committed (acc) | 0.741 | 0.741 | +0.000 [−0.004, +0.005] |

None significant, and these paired CIs are tight (±0.01, not the ±0.05 of unpaired comparisons),
so this is **not** "we lacked power to tell them apart". Mistral is a touch below ours
(entropy −0.018, significant) but still well above TF-IDF.

**So the signal is "a competent 7B transformer read this text", not "a reasoning-trained model
knows where its own reasoning is going."** This reframes the answer to the research question and
the write-up has to say it plainly.

### (c) Maybe you do not need a probe at all

Show a strong LLM some worked examples and ask it for the five numbers directly.

**How it works:** one prompt = an instruction defining the five targets → k labelled examples
(question + choices + prefix + checkpoint %, then the true five numbers as JSON) → the query
prefix → the instruction repeated → answer forced to a compact-JSON regex. Examples are drawn
only from *other folds'* questions, round-robin over the six checkpoints, nested so k=10 is the
first 10 of the k=max list. Test set = 1 random CoT per question + the 0% row = **1,178 rows**.
Code: `src/baselines/icl_predict.py` (vLLM), `src/baselines/icl_openai.py` (OpenAI API).

**Important caveat:** the raw guesses are on the wrong scale, so raw R² is negative almost
everywhere. Reported scores are **calibrated** — `y ≈ a + b·guess` fitted per target and
checkpoint on the other four folds. That is a generous comparison: the LLM gets a free linear
rescaling the probe never needed. So "Gemma ties ridge on entropy" means it ties *after* being
handed a calibration.

**Result:** best is **Gemma-4-31B at k=0, thinking off** — entropy 0.262, p_mode 0.237,
rem_len_mean 0.283, committed 0.742. Tied with ridge on entropy / p_mode / committed, much worse
on remaining length. More examples *hurt*. Thinking *hurt*. Qwen3.5-9B ≈ 0 R² at every k.
Our own model produces degenerate answers (it tries to solve the GPQA question instead).
GPT-6 Astra/Luna ran on only 20 and 10 rows, so their CIs are ±0.3–1.5 — uninformative either way.

### Summary of the three

**(b) is the real threat to the claim.** (a) we pass. (c) is the weakest threat.

## 5. A separate category, not baselines

MLP / PCA→ridge / random forest / gradient boosting / LoRA are **not** baselines. They ask a
different question: *is ridge leaving signal on the table?*

**Answer: no.** Nothing beats it. **R3** explained why trees lose — the signal is a weak linear
trend spread over ~1,600 of the 3,584 dimensions, the worst possible shape for a tree. (Proof:
on a noise-free synthetic linear target on this same X, RF gets 0.68 where ridge gets 0.92.)
Regularising RF until train ≈ test only lowers test. The one thing that helps slightly: ridge on
**LoRA-adapted** layer-21 states, +0.01 to +0.026 R² at 30–90%, nothing at 10%.

## 6. Where the research question actually lands

Hidden states at a CoT prefix **do** predict properties of the future continuation distribution,
and better than the prefix text alone. But they do **not** do it because the model is a reasoner
introspecting its own trajectory — an unrelated model's states work just as well. The honest
version of the finding is about what is linearly readable from *any* transformer's representation
of the text.

Two confounds to keep in the write-up: prefix length explains much of the remaining-length result
at early checkpoints, and the 0%-checkpoint probes (which would be free of that confound) have
not been run yet even though the data has been ready since Oct 4.

## 7. Open items

1. **0% checkpoint probes** — labels + activations ready since Oct 4, needs a sweep config.
   The only experiment that removes the prefix-length confound entirely.
2. **Round-3 write-up** — does not exist yet; must deal with the OLMo result honestly.
3. **GPT-6 billing** — only ~$33 tracked against a $100 credit; the Oct 19 extension died on
   `credit_balance_exhausted`. Check the OpenAI usage page before planning more frontier calls.
4. Logistic C grid below 1e-4 for `committed_by_here` (chosen C sits at the grid edge).
5. LoRA seeds / other ranks.
6. Final write-up.

## Sources checked for this file

- `runs/goal2/gpqa25/sweep/_round3/{ci_scores,ci_paired}.csv` — OLMo/Mistral/embedder numbers and CIs
- `runs/goal2/gpqa25/sweep/_round2/r2_model_bases/{summary.csv,config_tokenizer.txt}` — R2 weight similarity
- `runs/goal2/gpqa25/sweep/_multilayer/ci_paired.csv` — layer comparisons
- `src/baselines/icl_predict.py`, `configs/sweep_feat_olmo3_7b_L24.yaml` — ICL and probe mechanics
- `runs/logs/round3/icl_gpt6_{astra,luna}.log` — GPT-6 costs and the credit error
- `CLAUDE.md` Results index and `docs/HANDOFF.md` — everything else
