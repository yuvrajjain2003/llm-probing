---
title: "AUGUR: Answer Uncertainty and Generation Length from Unfinished Reasoning"
subtitle: "Predicting the Distribution of Future Continuations from Chain-of-Thought Representations"
author: "Yuvraj Jain"
date: "Master's Thesis Abstract"
geometry: margin=1.1in
fontsize: 11pt
linestretch: 1.08
---

Large reasoning models emit long chains of thought, but it is unclear whether a
model's internal state partway through a chain carries information about where that
chain is heading. This thesis asks whether hidden representations at a
chain-of-thought prefix predict properties of the *distribution* over the model's own
future continuations, beyond what is available from the prefix text itself.

Using DeepSeek-R1-Distill-Qwen-7B, I capture layer-21 activations at prefix cut points
and predict five properties of the distribution over $K$ sampled continuations ($K$
between 25 and 100): answer entropy, modal-answer probability, remaining reasoning
length and its spread, and whether the chain has already committed to an answer.
Across three datasets spanning 5,253 questions (GPQA-Diamond, MMLU-Pro and
GlobalOpinionQA), linear probes on activations significantly outperform tuned text
baselines on the same prefixes, and on the two larger datasets they also outperform a
frontier model asked to make the same predictions in context. Of 22 feature and probe
variants compared under identical folds, the only representation that matches
activations is an unrelated model's activations on the same text.
