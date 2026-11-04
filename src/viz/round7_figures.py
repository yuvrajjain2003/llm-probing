#!/usr/bin/env python3
"""Round 7 figures, items 1 to 4, all three datasets.

Reads ONLY published CSVs and refits nothing. Sources:
  item 1  <tag>/sweep/_r7bands/p5_bands_scores*.csv   (src/analysis/r7_bands_mse.py)
  items 2, 3  <tag>/sweep/_r7err/{binned,x_measures}.csv   (src/analysis/r7_err_vs_x.py)
  item 4  <tag>/sweep/_r7depth/per_cut_ci.csv         (src/analysis/r7_depth_ci.py)

House style copied from src/viz/meeting_figures.py: palette slots 1 to 3 of the dataviz skill's
default categorical palette unchanged, white background, Arial first in the font list (Arial is not
installed here, Liberation Sans is metric compatible and is what renders), significance and the
reference lines carried by marker fill and line style rather than by colour alone, and NO fine-print
caption baked into any figure (the user removed all of those on 2026-11-03; method text belongs in
prose or speaker notes). No em dashes.

ROW SETS ARE NAMED ON EVERY PANEL. Three mutually incomparable sets are in play: the full sweep
rows, the ICL rows the frontier model was actually called on (1 chain per question per cut), and
the band subsets of those. Mixing them is this project's most repeated error.

SCALES ARE NEVER MIXED ON ONE AXIS. Raw and log1p length R2 live in separate rows of item 4, and
length error is shown as RMSE in tokens so it is readable beside an entropy MSE in bits squared.

    ./envs/goal1/bin/python src/viz/round7_figures.py            # all items
    ./envs/goal1/bin/python src/viz/round7_figures.py --items 1 4
Outputs: docs/figures/round7/r7_*.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("runs/goal2")
OUT = Path("docs/figures/round7")
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#e3e2dd", "#8a8884"

ENT, ENTN = "answer_entropy_bits_mm", "answer_entropy_bits_mm_norm"
LEN, LSD = "rem_len_mean", "rem_len_std"
DSNAME = {"gpqa25": "GPQA", "mmlupro_v1": "MMLU-Pro", "globalop_v1": "GlobalOpinionQA"}

BANDS = ["all questions", "v_hard", "hard", "medium", "easy", "v_easy"]
BNICE = {"all questions": "all\nquestions", "v_hard": "very\nhard", "hard": "hard",
         "medium": "medium", "easy": "easy", "v_easy": "very\neasy"}

OURS_B, LUNA_B, PLEN_B, TFIDF_B = ("Our model L21 (ridge)", "Luna k=0", "Prefix length",
                                   "TF-IDF on prefix")
OURS, LUNA_C, LUNA_R = "Our probe", "Luna k=0 (calibrated)", "Luna k=0 (raw)"
TFIDF, PLEN = "TF-IDF on prefix", "Prefix length"
BINMEAN, SETMEAN = "that bin's own average", "the full set's average"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "font.size": 14, "axes.titlesize": 15.5, "axes.labelsize": 14, "axes.labelcolor": MUTED,
    "axes.edgecolor": MUTED, "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 12.5,
})


# ------------------------------------------------------------------ item 1 ----
BAND_SRC = [("gpqa25", "_r7bands/p5_bands_scores.csv",
             "difficulty band: share of the 25 attempts that were correct"),
            ("mmlupro_v1", "_r7bands/p5_bands_scores.rollout.csv",
             "difficulty band: share of the attempts that were correct"),
            ("globalop_v1", "_r7bands/p5_bands_scores.qtext.csv",
             "difficulty band: out-of-fold question-text measure (no gold answer here)")]
BAND_SERIES = [(OURS_B, "Our probe", BLUE, "o"), (LUNA_B, "GPT-6 Luna, zero-shot", ORANGE, "s"),
               (PLEN_B, "Prefix length alone", AQUA, "^")]


def band_panel(ax, d, target, kind, title, ylab, root=False):
    """kind='mse' plots the error (as RMSE in the target's own units when root=True, which is the
    only readable scale for a token count), kind='corr' the offset-free within-cut correlation."""
    xs = np.arange(len(BANDS))
    scale = (lambda v: np.sqrt(v)) if root else (lambda v: v)
    for lab, nice, col, mk in BAND_SERIES:
        x = d[(d.target == target) & (d.label == lab)].set_index("band").reindex(BANDS)
        if kind == "mse":
            v, lo, hi = (scale(x["mse"].to_numpy(float)), scale(x.mse_ci_lo.to_numpy(float)),
                         scale(x.mse_ci_hi.to_numpy(float)))
        else:
            v, lo, hi = (x.within_cut_corr.to_numpy(float), x.corr_ci_lo.to_numpy(float),
                         x.corr_ci_hi.to_numpy(float))
        ax.plot(xs, v, "-", color=col, lw=2.0, zorder=3)
        ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt=mk, ms=9, color=col, ecolor=col,
                    elinewidth=2.0, capsize=4, zorder=4, label=nice)
    if kind == "mse":
        ref = d[d.target == target].drop_duplicates("band").set_index("band").reindex(BANDS)
        ax.plot(xs, scale(ref.mse_band_mean.to_numpy(float)), ls=(0, (5, 3)), color=INK, lw=1.8,
                zorder=2, label="guessing that band's own average")
        ax.plot(xs, scale(ref.mse_global_mean.to_numpy(float)), ls=(0, (1, 2.4)), color=GREY,
                lw=1.8, zorder=2, label="guessing the whole set's average")
        top = max(max(np.nanmax(ln.get_ydata()) for ln in ax.get_lines() if len(ln.get_ydata())),
                  np.nanmax(scale(d[d.target == target].mse_ci_hi.to_numpy(float))))
        ax.set_ylim(0, top * 1.10)
    else:
        ax.axhline(0, color=MUTED, lw=1.1, zorder=1)
    ax.axvline(0.5, color=MUTED, lw=1.3, ls=(0, (5, 4)), zorder=1)
    ax.set_xticks(xs); ax.set_xticklabels([BNICE[b] for b in BANDS])
    ax.set_xlim(-0.4, len(BANDS) - 0.6); ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left", pad=9); ax.set_ylabel(ylab)


def item1():
    for k, (tag, src, note) in enumerate(BAND_SRC):
        d = pd.read_csv(R / tag / "sweep" / src)
        fig, axs = plt.subplots(2, 2, figsize=(16.6, 10.2))
        band_panel(axs[0][0], d, ENT, "mse", "Answer entropy: squared error inside the band",
                   "mean squared error (bits$^2$)")
        band_panel(axs[0][1], d, LEN, "mse", "Remaining length: error inside the band",
                   "root mean squared error (tokens)", root=True)
        band_panel(axs[1][0], d, ENT, "corr", "Answer entropy, offset-free view", "correlation")
        band_panel(axs[1][1], d, LEN, "corr", "Remaining length, offset-free view", "correlation")
        for ax in axs[1]:
            ax.set_xlabel(note)
        fig.suptitle(f"{DSNAME[tag]}: error has the same units in every band, R$^2$ does not. "
                     f"ICL row set.", fontsize=19, y=1.015, x=0.012, ha="left", va="bottom")
        fig.tight_layout()
        h, l = axs[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.035))
        f = OUT / f"r7_1{chr(97 + k)}_bands_mse_{tag.split('_')[0]}.png"
        fig.savefig(f); plt.close(fig)
        print("item 1:", f.name)


# ------------------------------------------------------------- items 2 and 3 ----
ERR_SERIES = [(OURS, "Our probe", BLUE, "o", "-"),
              (LUNA_C, "GPT-6 Luna, zero-shot, calibrated", ORANGE, "s", "-"),
              (LUNA_R, "GPT-6 Luna, zero-shot, raw", ORANGE, "D", (0, (4, 2))),
              (TFIDF, "TF-IDF on prefix", AQUA, "^", "-"),
              (PLEN, "Prefix length alone", GREY, "v", "-")]
REF_SERIES = [(BINMEAN, "guessing that bin's own average", INK, (0, (5, 3))),
              (SETMEAN, "guessing the whole set's average", GREY, (0, (1, 2.4)))]


MIN_BIN_FRAC = 0.02   # a bin holding under 2 percent of the dataset's questions is kept in the CSV
                      # but not plotted. On MMLU-Pro the p_gold values 0.25 / 0.5 / 0.75, and on
                      # GlobalOpinionQA the p_mode0 values 0.333 / 0.5 / 0.75, exist only for the
                      # questions with fewer than 5 valid rollouts: 9 to 25 questions each, pure
                      # noise. The threshold is a fraction so GPQA's 198 questions survive it.


def err_panel(ax, d, xm, row_set, target, transform, metric, title, ylab, root=False,
              scale=1.0, show_n=True):
    x = d[(d.x_measure == xm) & (d.row_set == row_set) & (d.target == target)
          & (d["transform"] == transform) & (d.metric == metric)]
    if len(x):
        tot = x[x.label == OURS].n_questions.sum()
        x = x[x.n_questions >= max(10, MIN_BIN_FRAC * tot)]
    if not len(x):
        ax.set_axis_off(); return
    f = (lambda v: np.sqrt(v) / scale) if root else (lambda v: v / scale)
    for lab, nice, col, mk, ls in ERR_SERIES:
        s = x[x.label == lab].sort_values("x")
        if not len(s):
            continue
        v, lo, hi = f(s.value.to_numpy(float)), f(s.ci_lo.to_numpy(float)), f(s.ci_hi.to_numpy(float))
        ax.plot(s.x, v, ls=ls, color=col, lw=2.0, zorder=3)
        ax.errorbar(s.x, v, yerr=[v - lo, hi - v], fmt=mk, ms=8, color=col, ecolor=col,
                    elinewidth=1.8, capsize=3, zorder=4, label=nice)
    for lab, nice, col, ls in REF_SERIES:
        s = x[x.label == lab].sort_values("x")
        if len(s):
            ax.plot(s.x, f(s.value.to_numpy(float)), ls=ls, color=col, lw=1.8, zorder=2, label=nice)
    if show_n:
        s = x[x.label == OURS].sort_values("x")
        top = ax.get_ylim()[1]
        for _, r in s.iterrows():
            ax.annotate(f"{int(r.n_questions)}", (r.x, top), ha="center", va="bottom",
                        fontsize=10, color=MUTED, annotation_clip=False)
    ax.set_ylim(bottom=0)
    ax.set_title(title, loc="left", pad=16); ax.set_ylabel(ylab)


def item2():
    for k, tag in enumerate(["gpqa25", "mmlupro_v1"]):
        d = pd.read_csv(R / tag / "sweep/_r7err/binned.csv")
        xname = d[d.x_measure == "p_gold"].x_name.iloc[0]
        fig, axs = plt.subplots(2, 2, figsize=(16.4, 10.4))
        err_panel(axs[0][0], d, "p_gold", "icl", ENT, "raw", "mse",
                  "Answer entropy, ICL row set (the monitor's rows)",
                  "mean squared error (bits$^2$)")
        err_panel(axs[0][1], d, "p_gold", "full", ENT, "raw", "mse",
                  "Answer entropy, full sweep row set (no monitor here)",
                  "mean squared error (bits$^2$)")
        err_panel(axs[1][0], d, "p_gold", "icl", LEN, "raw", "mse",
                  "Remaining length, ICL row set", "root mean squared error (tokens)", root=True)
        err_panel(axs[1][1], d, "p_gold", "full", LEN, "raw", "mse",
                  "Remaining length, full sweep row set", "root mean squared error (tokens)",
                  root=True)
        for ax in axs[1]:
            ax.set_xlabel(f"question accuracy: {xname}")
        fig.suptitle(f"{DSNAME[tag]}: prediction error against question accuracy, "
                     f"continuously. Number above each point is its question count.",
                     fontsize=18, y=1.015, x=0.012, ha="left", va="bottom")
        fig.tight_layout()
        h, l = axs[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.055))
        f = OUT / f"r7_2{chr(97 + k)}_err_vs_accuracy_{tag.split('_')[0]}.png"
        fig.savefig(f); plt.close(fig)
        print("item 2:", f.name)


def item3():
    tag = "globalop_v1"
    d = pd.read_csv(R / tag / "sweep/_r7err/binned.csv")
    cols = [("p_mode0", "model self-consistency at the prompt"),
            ("human_disagree_multi", "disagreement among the human survey answers"),
            ("qtext_ent_norm_mean", "uncertainty predicted from the question words alone")]
    fig, axs = plt.subplots(2, 3, figsize=(18.2, 10.4))
    for j, (xm, nice) in enumerate(cols):
        err_panel(axs[0][j], d, xm, "full", ENTN, "raw", "mse",
                  "Normalised answer entropy" if j == 0 else " ",
                  "mean squared error (bits$^2$)" if j == 0 else "")
        err_panel(axs[1][j], d, xm, "full", LEN, "log1p", "mse",
                  "Remaining length, log scale" if j == 0 else " ",
                  "mean squared error (log tokens)$^2$" if j == 0 else "", root=False)
        axs[1][j].set_xlabel(nice)
    fig.suptitle("GlobalOpinionQA has no gold answer, so the x axis is consistency, not accuracy. "
                 "Full sweep row set, 2,555 questions.",
                 fontsize=17.5, y=1.015, x=0.012, ha="left", va="bottom")
    fig.tight_layout()
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.055))
    fig.savefig(OUT / "r7_3a_globalop_consistency.png"); plt.close(fig)
    print("item 3: r7_3a_globalop_consistency.png")

    fig, axs = plt.subplots(1, 2, figsize=(16.0, 6.0))
    err_panel(axs[0], d, "p_mode0", "icl", ENT, "raw", "mse",
              "Answer entropy, raw scale", "mean squared error (bits$^2$)")
    err_panel(axs[1], d, "p_mode0", "icl", LEN, "raw", "mse",
              "Remaining length, raw scale", "root mean squared error (tokens)", root=True)
    for ax in axs:
        ax.set_xlabel("model self-consistency at the prompt")
    fig.suptitle("GlobalOpinionQA, the black-box monitor on identical rows. ICL row set, "
                 "500 questions. Raw entropy, the only entropy target Luna was asked for.",
                 fontsize=17, y=1.02, x=0.012, ha="left", va="bottom")
    fig.tight_layout()
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.10))
    fig.savefig(OUT / "r7_3b_globalop_monitor.png"); plt.close(fig)
    print("item 3: r7_3b_globalop_monitor.png")


# ------------------------------------------------------------------ item 4 ----
DEPTH_SERIES = [("act_L21", "Our probe", BLUE, "o"), ("tfidf", "TF-IDF on prefix", AQUA, "^"),
                ("prefix_len", "Prefix length alone", GREY, "v")]
TAGS4 = ["gpqa25", "mmlupro_v1", "globalop_v1"]


def item4():
    C = pd.concat([pd.read_csv(R / t / "sweep/_r7depth/per_cut_ci.csv") for t in TAGS4],
                  ignore_index=True)
    for k, (target, nice) in enumerate([(LEN, "Remaining length"),
                                        (LSD, "Spread of remaining length")]):
        fig, axs = plt.subplots(2, 3, figsize=(17.4, 9.6), sharey="row")
        for i, tr in enumerate(["raw", "log1p"]):
            for j, tag in enumerate(TAGS4):
                ax = axs[i][j]
                x = C[(C.tag == tag) & (C.target == target) & (C["transform"] == tr)]
                for ft, lab, col, mk in DEPTH_SERIES:
                    s = x[x.feature == ft].sort_values("fraction")
                    v = s.score.to_numpy(float)
                    ax.plot(s.fraction * 100, v, "-", color=col, lw=2.0, zorder=3)
                    ax.errorbar(s.fraction * 100, v,
                                yerr=[v - s.ci_lo.to_numpy(float), s.ci_hi.to_numpy(float) - v],
                                fmt=mk, ms=8, color=col, ecolor=col, elinewidth=1.8, capsize=3,
                                zorder=4, label=lab)
                ax.set_xticks([10, 30, 50, 70, 90])
                ax.grid(axis="x", visible=False)
                ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
                scale = "raw token counts" if tr == "raw" else "log1p of the token count"
                ax.set_title(f"{DSNAME[tag]}, {scale}" if i == 0 else f"{DSNAME[tag]}, {scale}",
                             loc="left", pad=9)
                if j == 0:
                    ax.set_ylabel("R$^2$")
                if i == 1:
                    ax.set_xlabel("checkpoint, percent of the chain written")
        axs[0][0].legend(loc="lower left", fontsize=12)
        fig.suptitle(f"{nice} against checkpoint depth. Top row raw, bottom row log scale: "
                     f"never compare one to the other. Full sweep row set.",
                     fontsize=17.5, y=1.015, x=0.012, ha="left", va="bottom")
        fig.tight_layout()
        f = OUT / f"r7_4{chr(97 + k)}_length_vs_depth_{'mean' if target == LEN else 'std'}.png"
        fig.savefig(f); plt.close(fig)
        print("item 4:", f.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", type=int, default=[1, 2, 3, 4])
    a = ap.parse_args()
    fns = {1: item1, 2: item2, 3: item3, 4: item4}
    for i in a.items:
        fns[i]()
    print("\nwrote to", OUT.resolve())
