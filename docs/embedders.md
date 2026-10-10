# Sentence embedders for baseline 1 (PLAN step 2)

Written 2026-10-04. Nothing here was run on a GPU; no weights were downloaded. "Checked" = I ran a command or read the page listed.

## How the numbers were obtained
- MTEB numbers come from the live leaderboard backend (the API behind https://huggingface.co/spaces/mteb/leaderboard, found in the page's `env.js`: `https://mteb-leaderboard-backend.hf.space/v1/benchmarks/<name>/scores`). I pulled `MTEB(eng, v2)`, `MTEB(Multilingual, v2)` and `LongEmbed` on 2026-10-04. Scores are x100.
- Model facts (params, license, files) come from `https://huggingface.co/api/models/<id>`, the model cards, and `config.json` files.
- "zero-shot %" is the leaderboard's own column: how much of the benchmark the model was NOT trained on. Low = more possible train/test overlap. Treat high MTEB numbers of low zero-shot models (QZhou 48, F2LLM-v2 43) with care.
- Caveat for all MTEB numbers: MTEB classification/STS/clustering texts are short (sentences, abstracts). None of them test 2-15k-token math/physics reasoning text. Only LongEmbed is long, and it is retrieval only. So MTEB ranks the models, but does not prove which is best for us. That is why we try several.

## Ranked table
Context column: "card/leaderboard" = what the page claims; "trained" = what the paper/code says they trained at (only where I found it). Mem = bf16 weights (leaderboard `memoryUsageMb` or params x 2).
Task-category scores: Cls = classification, Clu = clustering, STS, Rr = reranking, (E) = MTEB(eng, v2), (M) = MTEB(Multilingual, v2), LE = LongEmbed mean nDCG@10.

| # | Model (HF id) | Params | Dim | Context: card / trained | Key scores | License | Mem | Source |
|---|---|---|---|---|---|---|---|---|
| 1 | `microsoft/harrier-oss-v1-27b` | 27.0B | 5376 | card 32,768 (leaderboard lists 131,072); trained length: not found | (M) mean 74.3 (rank 1), Cls 79.9, Clu 58.9, STS 80.0, Rr 67.3. Not on (E) summary, not on LongEmbed | MIT | 51.5 GB | [card](https://huggingface.co/microsoft/harrier-oss-v1-27b) |
| 2 | `Qwen/Qwen3-Embedding-8B` | 7.57B | 4096 (32-4096 MRL) | card 32k; config max_pos 40,960; trained length: not stated in the paper text I could read | (E) mean 75.2, Cls 90.4, Clu 58.6, STS 88.6, Rr 51.6. (M) mean 70.6, Cls 74.0, Clu 57.6, STS 81.1. Not on LongEmbed | Apache-2.0 | 14.4 GB | [card](https://huggingface.co/Qwen/Qwen3-Embedding-8B), [paper](https://arxiv.org/abs/2506.05176) |
| 3 | `Kingsoft-LLM/QZhou-Embedding` (Qwen2.5-7B-Instruct, bidirectional) | 7.07B | 3584 | card 8,192; **trained at 1,536 passage tokens**, 8k "by RoPE extrapolation" (paper) | (E) mean 76.0 (rank 3), Cls 89.0, Clu 61.6, STS 91.6, Rr 51.8; zero-shot 48% | Apache-2.0 | 14.4 GB | [card](https://huggingface.co/Kingsoft-LLM/QZhou-Embedding), [paper](https://arxiv.org/abs/2508.21632) |
| 4 | `codefuse-ai/F2LLM-v2-8B` (Qwen3-8B base) | 7.57B | 4096 (MRL) | card/leaderboard 40,960; trained length: not found (repo v1 code default `max_seq_length=2048`, may not apply to v2) | (E) mean 72.9, Cls 91.5, Clu 62.8, STS 84.6. (M) mean 68.1, Rr 70.3, Clu 60.6. **LongEmbed 76.6 (best open 7-8B)**; zero-shot 43% | Apache-2.0 | ~15 GB (leaderboard says 28.9 GB, fp32) | [card](https://huggingface.co/codefuse-ai/F2LLM-v2-8B), [paper](https://arxiv.org/abs/2603.19223) |
| 5 | `Qwen/Qwen3-Embedding-4B` | 4.02B | 2560 | as #2 | (E) mean 74.6, Cls 89.8, Clu 57.5, STS 88.7. (M) 69.4 | Apache-2.0 | 7.7 GB | [card](https://huggingface.co/Qwen/Qwen3-Embedding-4B) |
| 6 | `Alibaba-NLP/gte-Qwen2-7B-instruct` | 7.07B | 3584 | leaderboard 32,768 | (E) mean 70.7, Cls 88.5, Clu 59.0, STS 82.7 (older, 2024) | Apache-2.0 | 14.5 GB (leaderboard 29 GB) | leaderboard |
| 7 | `nvidia/NV-Embed-v2` | 7.85B | 4096 | leaderboard 32,768 | (E) mean 69.8, Cls 87.2, Clu 47.7, STS 83.8 (weak clustering) | CC-BY-NC-4.0 | 15 GB | leaderboard |
| 8 | `microsoft/harrier-oss-v1-0.6b` | 0.60B | 1024 | card 32,768 | (M) mean 69.0, Cls 73.9, Clu 54.0, STS 77.1 | MIT | 1.1 GB | [card](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) |
| 9 | `Qwen/Qwen3-Embedding-0.6B` | 0.60B | 1024 | as #2 | (E) mean 70.5, Cls 84.6, Clu 54.1, STS 86.6. (M) 64.3 | Apache-2.0 | 1.1 GB | leaderboard |
| 10 | `BidirLM/BidirLM-1B-Embedding` | 1.0B | 1152 | evaluated on LongEmbed at 32k (76.4) | (M) mean 63.2; LE 76.4 | Apache-2.0 | 1.9 GB | [card](https://huggingface.co/BidirLM/BidirLM-1B-Embedding) |
| 11 | `nomic-ai/nomic-embed-text-v1.5` (classic floor, long) | 0.137B | 768 | **trained at 2,048**, runs to 8,192 with rope scaling (config `max_trained_positions: 2048`, `n_positions: 8192`) | (E) mean 62.2, Cls 75.7, Clu 47.6, STS 78.7 | Apache-2.0 | 0.5 GB | [card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) |
| 12 | `sentence-transformers/all-MiniLM-L6-v2` (classic floor, short) | 0.023B | 384 | 256 tokens | (E) mean 59.0, Cls 69.3, Clu 44.9, STS 78.9 | Apache-2.0 | 0.09 GB | leaderboard |

Others looked at and not chosen: `tencent/KaLM-Embedding-Gemma3-12B-2511` (M rank 2, 72.3; license field empty on leaderboard, so unclear), `jinaai/jina-embeddings-v5-text-small` and Linq/SFR/NV-Embed (CC-BY-NC), Giga-Embeddings (8k limit), `infgrad/Jasper-Token-Compression-600M` (rank 2 on E, 48% zero-shot, exotic token-compression model, not checked further), `BAAI/bge-m3` (LongEmbed 59.5, weaker on science clustering: ArXiv 52.8 vs ~64 for Qwen3-8B), closed APIs (Seed, Gemini, Voyage: excluded, not open weights).

Science-text clustering (closest to our domain; (E) leaderboard, x100): ArXivHierarchicalClusteringP2P / BiorxivP2P / MedrxivP2P
- Qwen3-Emb-8B 64.9 / 51.3 / 46.1; QZhou 66.2 / 54.5 / 49.9; Harrier-27B 63.4 / 57.1 / 48.8; F2LLM-v2-8B 63.5 / 69.4 / 57.0 (the 69.4 looks unusually high; low zero-shot %, possible training overlap); Qwen3-0.6B 63.7 / 47.3 / 42.2; nomic-1.5 60.1 / 41.9 / 38.0; MiniLM-L6 59.1 / 41.0 / 37.6.
- Math/code: StackOverflowQA (M): Harrier-27B 97.0, F2LLM-v2-8B 97.0, Qwen3-8B 94.8, nomic-1.5 63.6. No leaderboard task is math/LaTeX-specific; I found no evidence either way for LaTeX handling beyond these. All the 7-8B+ models are LLM-based (Qwen/Gemma tokenizers), which read LaTeX/code natively; the BERT-style tokenizers (nomic, MiniLM) do not.

## Recommended shortlist (4 + 1 floor)
1. **Qwen3-Embedding-8B** - top open model on MTEB(eng, v2) among those that are 95% zero-shot (75.2; Cls 90.4), 32k config, Apache, plain `transformers` works. The default "strong 7-8B".
2. **F2LLM-v2-8B** - best open 7-8B on LongEmbed (76.6, the only long-doc benchmark), best clustering; fully open data; a second 7-8B from a different training recipe so we don't depend on one model. Caveat: low zero-shot %.
3. **Harrier-oss-v1-27b** - #1 on MTEB(Multilingual, v2) (74.3), MIT, fits one H100 (51.5 GB weights). The "ceiling" run; larger than the probed LLM, so use it as an upper bound, not as the size-matched comparison.
4. **Qwen3-Embedding-0.6B** (add Qwen3-Embedding-4B if cheap) - same recipe as #1 at 0.6B/4B, gives a size curve inside one family: if embedding quality barely moves with size, size is not the confound. (Harrier-0.6b is the alternative small model; better on (M), no (E) mean on the leaderboard.)
5. **Floor: nomic-embed-text-v1.5** (8k, classic BERT-style, 137M; use prefix `classification: `). If a model this weak matches the big ones, the text carries little extra and the embedder is not the limit. MiniLM-L6 (256 tokens) is optional as an extreme floor, with tail truncation.
Optional 6th: **QZhou-Embedding** (E rank 3, 76.0; built on Qwen2.5-7B-Instruct, hidden 3584 = same as our probed model). Only 1,536 training length / 8k limit and low zero-shot %, and needs `trust_remote_code`; use only if there is time.

## Runnability here
Checked with `source env.sh; $PY ...`: transformers 4.57.6, torch 2.4.0+cu121, vllm 0.5.5, accelerate 1.13.0. **Not installed**: sentence-transformers, flash_attn, mteb. `Qwen3Model`, `Gemma3TextModel`, `Qwen2Model` import fine.
- No new environment is needed if we use plain `transformers` `AutoModel` (+ own pooling, ~15 lines, as in each model card): Qwen3-Embedding (card needs transformers >=4.51), F2LLM-v2 (card shows plain transformers code), Harrier (Gemma3 text; card shows plain transformers code), nomic (`trust_remote_code=True`, downloads custom code; needs internet on first load).
- vllm 0.5.5 is too old to serve these as embedding models; do not use it. `$PY_ES` has vLLM 0.17.1 but I did not check its embedding mode; plain transformers is simpler and enough.
- Use `attn_implementation="sdpa"` (no flash_attn installed). QZhou was evaluated with sdpa per its card. Gemma3 (Harrier) alternates sliding-window (1024) and global layers (config `layer_types`, `_sliding_window_pattern: 6`); HF handles this.
- BidirLM needs the `transformers-v4` branch for transformers 4.x per its card (main needs >=5.0). Skip unless wanted.
- sentence-transformers is optional (pure Python); I did not test that it installs cleanly. Not needed.

## Throughput estimate (calculation, not measured)
Assumption: mean prefix ~2.4k tokens (median at 50%), so ~24.6k x 2.4k = ~60M tokens in DeepSeek tokens; other tokenizers differ (check with the real tokenizer before running). Forward cost ~ 2 x params FLOPs/token. H100 bf16 dense peak is ~989 TFLOPS (NVIDIA datasheet); assume 30-40% utilisation (my assumption).
- 7-8B: ~15 GFLOP/token x 60M = 9e17 FLOP -> roughly 40-80 min on one H100.
- 27B: ~54 GFLOP/token -> 3.2e18 FLOP -> roughly 3-5 h.
- 0.6B: a few minutes. 4B: ~30 min.
- Memory: weights above + activations; batch by token budget (e.g. <=32k tokens per batch, sort by length). A single 15k-token sequence fits easily next to 15 GB (8B) or 54 GB (27B) of weights; for 27B use batch size 1-2 for the longest sequences. Measure on 100 texts first.

## Prompts / instructions
- Qwen3-Embedding, Harrier, F2LLM-v2, QZhou are instruction-aware; queries get `Instruct: <task>\nQuery: <text>`, documents get none (cards). Our texts are neither queries nor documents. Recommendation: use ONE fixed generic instruction for all texts of a model, chosen before seeing probe results (do not tune on labels): `Instruct: Retrieve semantically similar text\nQuery: ` (this exact string is Harrier's pre-configured `sts_query` prompt in its `config_sentence_transformers.json`; F2LLM-v2's card says symmetric tasks work with or without prompts). If cheap (small models), also run no-prompt as an ablation.
- nomic: prefix `classification: ` (card: "embedding texts into vectors that will be used as features for a classification model").
- I did not verify exactly which prompt MTEB used for classification per model.

## Truncation, pooling, normalisation
- **Truncation:** with 32k-context models (Qwen3, Harrier, F2LLM-v2) the max prefix (14.8k DeepSeek tokens) should fit without any truncation; check the token count with each model's own tokenizer and log how many prefixes get cut. Only the 8k-limited models (nomic; QZhou) and MiniLM (256) truncate. For those, **keep the END** (the most recent reasoning): the targets (future answer entropy, remaining length, commitment) depend on where the reasoning is now, while the start (question restatement, first plan) is the same across the 25 chains of a question and already carried by question identity. Also last-token pooling reads from the end. This is an argument, not tested: as a cheap check on one small model, compare end vs start vs start+end on the held-out probe score and report it. A model trained at 1.5-2k tokens (QZhou, nomic) beyond that length is unvalidated by its authors, so a 2k-token tail is an alternative setting to test for those two.
- **Pooling:** use each model's native pooling: last token (EOS) for Qwen3-Embedding, F2LLM-v2, Harrier (cards; Harrier's `1_Pooling/config.json` shows `pooling_mode_lasttoken`); mean for QZhou (its card script uses `POOLING_MODE=mean`); CLS/mean per config for nomic. Do not swap pooling: the models were trained for it. With left padding take position -1; with right padding take the last non-pad token (cards' code).
- **L2-normalise:** yes, as the primary features (all these models are trained/evaluated with cosine similarity; the cards normalise). Save the raw (un-normalised) fp32 too, and standardise features (z-score) inside the probe pipeline, same as for activations. Reason to keep raw: the norm might carry length information that normalising removes; cheap to check.

## Extra baseline: mean-pooled hidden states of another LLM (recommendation)
Yes, add one, small effort. Use **Qwen2.5-7B-Instruct** (non-reasoning, same architecture/hidden size 3584 as our model; Apache-2.0; hf id `Qwen/Qwen2.5-7B-Instruct`, ~15 GB download, so needs your OK) with mean-pooled hidden states from the same relative layers as the probed layers. Why: it separates "the activations of THIS reasoning model carry future information" from "any 7B LLM reading the text gets it". Our DeepSeek model is a distillation onto Qwen2.5-Math-7B (its README table); Qwen2.5-7B-Instruct is not the same base, so it is a related-but-different model; the true base `Qwen/Qwen2.5-Math-7B` would be the closest control (not checked for size/license). Run it with the same truncation-free long context (Qwen2.5 `max_position_embeddings` 32,768 for the Instruct model; not checked in this session).

## 2026-10-09 R4: trained length of the embedders vs our inputs

Files: `runs/goal2/gpqa25/sweep/_round2/r4_embed_len/` (`token_lengths.csv`, `length_stats.csv`, `length_bins.csv`, scripts).

Trained vs supported length (what I could verify):
- Qwen3-Embedding-8B: paper Table 1 "Sequence Length 32K" = max supported. Neither the paper (arXiv 2506.05176, text read) nor the GitHub README states the length used in contrastive training. README examples use `max_length = 8192`. Base Qwen3: "increase its maximum context length from 4,096 to 32,768 tokens" (Qwen3 report 2505.09388, long-context stage). Trained embedding length: NOT FOUND.
- F2LLM-v2-8B: paper (2603.19223) Table 1 lists lr/epochs/batch size only, no sequence length found. Repo `F2LLM/configs/config.json` (example, 4b): `"max_seq_length": 1024`, experiment id `...context.1024...`; `arguments.py` default `max_seq_length: int = 2048`; `tokenize_data_qwen.py`: `max_seq_length = 2047`, `truncation=True`. So the released training code trains at 1-2k tokens; the config of the v2-8B run itself was not seen. HF config max_position_embeddings 40960.
- Controls: Qwen2.5-Math-7B config max_position_embeddings 4096; Qwen2.5-7B-Instruct 32768 (Qwen2.5 report 2412.15115: "extend the context length from 4,096 to 32,768 tokens during the final" pretraining phase); R1-Distill-Qwen-7B config 131072 but rope_theta 10000 (training length not found; R1 paper text had no context-length line).

Our inputs (instruction + question + choices + prefix, each model's tokenizer; Qwen3 and F2LLM give identical counts): median 190 / 689 / 1.6k / 2.6k / 3.6k / 4.5k tokens at 0/10/30/50/70/90%; max 2.8k ... 14.9k. Share above 1024 tokens: 0.5/27/68/80/88/92%; above 2048: 0.5/0.6/39/60/69/75%; above 8192: 0 up to 16.6% at 90%; nothing above 32768. Nothing was truncated in the earlier runs.
