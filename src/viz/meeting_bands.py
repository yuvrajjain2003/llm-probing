#!/usr/bin/env python3
"""Difficulty-BAND figures for the 2026-11-03 meeting, all three datasets.

Round 6 made this picture for GPQA only (docs/figures/slides/fig9_difficulty_bands.png and
fig12_predicting_the_mean.png). The frontier model has since been run on both new datasets, and
src/analysis/newds_bands_icl.py scored every approach inside difficulty bands there on IDENTICAL
rows, so the same picture can now be drawn for all three.

Reads only published CSVs and refits nothing.  NO fine-print caption under any figure: the user
asked for figures with nothing but the plot, and everything explanatory lives in the deck's speaker
notes.  House style copied from src/viz/meeting_figures.py.

    ./envs/goal1/bin/python src/viz/meeting_bands.py
Outputs: docs/figures/meeting/m9_bands_{gpqa,mmlupro,globalop}.png, m10_predicting_the_mean.png

BAND MEASURE DIFFERS BY DATASET, and the figures say so on themselves:
  GPQA, MMLU-Pro  : quintiles of p_gold, the share of the question's rollouts that are correct.
  GlobalOpinionQA : no gold answer exists, so quintiles of the OUT-OF-FOLD question-text difficulty
                    measure instead.  Equal-sized by construction and non-circular.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path("runs/goal2"); OUT = Path("docs/figures/meeting"); OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dd"
ENT, LEN = "answer_entropy_bits_mm", "rem_len_mean"
ORDER = ["all questions", "v_hard", "hard", "medium", "easy", "v_easy"]
NICE = {"all questions": "all\nquestions", "v_hard": "very\nhard", "hard": "hard",
        "medium": "medium", "easy": "easy", "v_easy": "very\neasy"}
OURS, LUNA, PLEN = "Our model L21 (ridge)", "Luna k=0", "Prefix length"
SERIES = [(OURS, "Our probe", BLUE, "o"), (LUNA, "GPT-6 Luna, zero-shot", ORANGE, "s"),
          (PLEN, "Prefix length alone", AQUA, "^")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14.5, "axes.labelcolor": MUTED,
    "axes.edgecolor": MUTED, "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 13.5,
})

DS = [("gpqa25", "GPQA", "_iclsplit/p5_bands_scores.csv", "_iclsplit/p7_band_spread.csv",
       "difficulty band: share of the 25 attempts that were correct"),
      ("mmlupro_v1", "MMLU-Pro", "_iclbands/p5_bands_scores.rollout.csv",
       "_iclbands/p7_band_spread.rollout.csv",
       "difficulty band: share of the attempts that were correct"),
      ("globalop_v1", "GlobalOpinionQA", "_iclbands/p5_bands_scores.qtext.csv",
       "_iclbands/p7_band_spread.qtext.csv",
       "difficulty band: out-of-fold question-text measure (no gold answer on this dataset)")]


def panel(ax, d, target, value, lo, hi, title, ylab):
    xs = range(len(ORDER))
    for lab, nice, col, mk in SERIES:
        x = d[(d.target == target) & (d.label == lab)].set_index("band").reindex(ORDER)
        v, l, h = x[value].to_numpy(float), x[lo].to_numpy(float), x[hi].to_numpy(float)
        ax.plot(xs, v, "-", color=col, lw=2.0, zorder=2)
        ax.errorbar(xs, v, yerr=[v - l, h - v], fmt=mk, ms=9, color=col, ecolor=col,
                    elinewidth=2.0, capsize=4, zorder=3, label=nice)
    ax.axvline(0.5, color=MUTED, lw=1.3, ls=(0, (5, 4)), zorder=1)
    ax.axhline(0, color=MUTED, lw=1.1, zorder=1)
    ax.set_xticks(list(xs)); ax.set_xticklabels([NICE[b] for b in ORDER])
    ax.set_xlim(-0.4, len(ORDER) - 0.6); ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left", pad=10); ax.set_ylabel(ylab)


def bands_fig(tag, name, scores, note, number, fname):
    d = pd.read_csv(R / tag / "sweep" / scores)
    fig, axs = plt.subplots(2, 2, figsize=(16.6, 10.2))
    panel(axs[0][0], d, ENT, "score_recentred", "recentred_ci_lo", "recentred_ci_hi",
          "Answer entropy: R$^2$ inside the band", "R$^2$")
    panel(axs[0][1], d, LEN, "score_recentred", "recentred_ci_lo", "recentred_ci_hi",
          "Remaining length: R$^2$ inside the band", "R$^2$")
    panel(axs[1][0], d, ENT, "within_cut_corr", "corr_ci_lo", "corr_ci_hi",
          "Answer entropy, offset-free view", "correlation")
    panel(axs[1][1], d, LEN, "within_cut_corr", "corr_ci_lo", "corr_ci_hi",
          "Remaining length, offset-free view", "correlation")
    axs[0][0].legend(loc="lower left")
    for ax in axs[1]:
        ax.set_xlabel(note)
    fig.suptitle(f"{name}: holding question difficulty roughly constant",
                 fontsize=19.5, y=1.015, x=0.012, ha="left", va="bottom")
    fig.tight_layout(); fig.savefig(OUT / fname); plt.close(fig)
    print(f"m{number}: {fname}")


def predicting_the_mean():
    """Are the predictors just emitting one flat number inside a band?  No, but they are too timid."""
    fig, axs = plt.subplots(2, 3, figsize=(17.4, 9.4))
    for j, (tag, name, _, spread, note) in enumerate(DS):
        s = pd.read_csv(R / tag / "sweep" / spread)
        s = s[s.target == ENT]
        lmap = {"Best probe": OURS, "Luna k=0 (calibrated)": LUNA}
        s = s.assign(label=s.label.map(lambda v: lmap.get(v, v)))
        for i, (col, ylab, title) in enumerate(
                [("mean_true", "bits", "average value"), ("sd_true", "bits", "spread (sd)")]):
            ax = axs[i][j]
            pc = "mean_pred" if i == 0 else "sd_pred"
            t = s[s.label == OURS].set_index("band").reindex(ORDER)
            xs = range(len(ORDER))
            ax.plot(xs, t[col], "-o", color=INK, lw=2.0, ms=8, zorder=3,
                    label="what actually happened")
            for lab, nice, c, mk in SERIES[:2]:
                x = s[s.label == lab].set_index("band").reindex(ORDER)
                if x[pc].notna().sum() == 0:
                    continue
                ax.plot(xs, x[pc], "--", color=c, lw=2.0, zorder=2)
                ax.plot(xs, x[pc], mk, color=c, ms=8, zorder=3, label=nice)
            ax.axvline(0.5, color=MUTED, lw=1.3, ls=(0, (5, 4)), zorder=1)
            ax.set_xticks(list(xs)); ax.set_xticklabels([NICE[b] for b in ORDER])
            ax.set_xlim(-0.4, len(ORDER) - 0.6); ax.grid(axis="x", visible=False)
            ax.set_title(f"{name}: {title}" if i == 0 else title, loc="left", pad=10)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 1:
                ax.set_ylim(bottom=0)
                ax.annotate("a predictor that had learned nothing\ninside a band would sit at zero",
                            (len(ORDER) - 0.55, 0.03), ha="right", va="bottom",
                            fontsize=11.5, color=MUTED)
                ax.set_xlabel("question difficulty band")
    axs[0][0].legend(loc="lower left", fontsize=12.5)
    fig.suptitle("They are not predicting one flat number, but they are far too timid. "
                 "Answer entropy.", fontsize=19.5, y=1.015, x=0.012, ha="left", va="bottom")
    fig.tight_layout(); fig.savefig(OUT / "m10_predicting_the_mean.png"); plt.close(fig)
    print("m10: m10_predicting_the_mean.png")


if __name__ == "__main__":
    for k, (tag, name, sc, sp, note) in enumerate(DS):
        bands_fig(tag, name, sc, note, 9, f"m9{chr(97 + k)}_bands_{tag.split('_')[0]}.png")
    predicting_the_mean()
    print("\nwrote to", OUT.resolve())
