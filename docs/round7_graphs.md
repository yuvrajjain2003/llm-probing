---
geometry: "landscape, margin=0.5in"
fontsize: 11pt
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \pagestyle{empty}
---

**MSE per difficulty band, GPQA.** R-squared inside a band divides by that band's own variance, which is the only reason very-easy read -1.090. On MSE it has the lowest error of any band.

![](/proj/ssrivalab/projects/llm-probing/docs/figures/round7/r7_1a_bands_mse_gpqa25.png){width=88%}

\newpage

**MSE per difficulty band, MMLU-Pro.** Same at 12.6x the questions. With 5 rollouts per question, 65 percent sit at accuracy exactly 0 or 1, so read the three middle bands.

![](/proj/ssrivalab/projects/llm-probing/docs/figures/round7/r7_1b_bands_mse_mmlupro.png){width=88%}

\newpage

**MSE per difficulty band, GlobalOpinionQA.** The probe beats the band average in all five bands, and no band is degenerate. No gold answer here, so bands come from a non-circular text-derived measure.

![](/proj/ssrivalab/projects/llm-probing/docs/figures/round7/r7_1c_bands_mse_globalop.png){width=88%}

\newpage

**Error against question accuracy, GPQA.** Continuous, no buckets. Error falls as accuracy rises. 198 questions, so bins are thin and intervals wide.

![](/proj/ssrivalab/projects/llm-probing/docs/figures/round7/r7_2a_err_vs_accuracy_gpqa25.png){width=88%}

\newpage

**Error against question accuracy, MMLU-Pro.** Entropy error falls 0.387 to 0.160 as accuracy goes 0 to 1. The probe is under the bin's own average in 5 of 6 bins, losing only where every attempt was correct.

![](/proj/ssrivalab/projects/llm-probing/docs/figures/round7/r7_2b_err_vs_accuracy_mmlupro.png){width=88%}

\newpage

**Remaining length against checkpoint.** Top row raw, bottom log scale; never compare the two. Prefix length alone matches us on the raw scale early and never on the log scale.

![](/proj/ssrivalab/projects/llm-probing/docs/figures/round7/r7_4a_length_vs_depth_mean.png){width=88%}

\newpage

**Spread of remaining length against checkpoint.** On the log scale this rises with depth on all three datasets, where the raw version decays or is flat.

![](/proj/ssrivalab/projects/llm-probing/docs/figures/round7/r7_4b_length_vs_depth_std.png){width=88%}
