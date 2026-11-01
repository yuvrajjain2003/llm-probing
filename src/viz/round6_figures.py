#!/usr/bin/env python3
"""Round-6 slide figures: the difficulty-stratification result.

Reads only the stored CSVs from src/analysis/icl_{decompose,bands}.py. No refitting.

    ./envs/goal1/bin/python src/viz/round6_figures.py
Outputs: docs/figures/slides/fig{9,10,11}_*.png

Colors are slots 1-3 of the dataviz skill's default categorical palette, used unchanged and
identical to fig1-7, so these drop straight into the existing deck. Series identity is carried
by marker shape as well as hue, and EVERY point carries a 95% question-bootstrap interval.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S = Path("runs/goal2/gpqa25/sweep/_iclsplit")
OUT = Path("docs/figures/slides"); OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#e3e2dd", "#8a8884"
ENT, LEN = "answer_entropy_bits_mm", "rem_len_mean"
BANDS = ["v_hard", "hard", "medium", "easy", "v_easy"]
XS = ["all questions"] + BANDS
XLAB = ["all\nquestions", "very\nhard", "hard", "medium", "easy", "very\neasy"]
SERIES = [("Our model L21 (ridge)", "Best probe", BLUE, "o"),
          ("Luna k=0", "GPT-6 Luna, zero-shot", ORANGE, "s"),
          ("Prefix length", "Prefix length alone", AQUA, "^")]

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11, "axes.labelcolor": MUTED,
    "axes.edgecolor": MUTED, "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "legend.frameon": False, "legend.fontsize": 10,
})


def band_panel(ax, B, target, vcol, lo_col, hi_col, ylim, ylabel, title):
    """One panel: a measure against difficulty band, 3 series, every point with a 95% interval.
    'all questions' sits left of a divider because it is not part of the difficulty ordering."""
    off = {0: -0.13, 1: 0.0, 2: 0.13}
    for si, (lab, nice, col, mk) in enumerate(SERIES):
        d = B[(B.label == lab) & (B.target == target)].set_index("band").reindex(XS)
        x = np.arange(len(XS)) + off[si]
        v, lo, hi = d[vcol].to_numpy(), d[lo_col].to_numpy(), d[hi_col].to_numpy()
        # bands only, joined by a line because difficulty is ordered; "all" stays a lone marker
        ax.plot(x[1:], v[1:], color=col, lw=1.6, alpha=.75, zorder=2)
        inside = (v >= ylim[0]) & (v <= ylim[1])
        ylo = np.where(np.isfinite(lo), np.maximum(lo, ylim[0]), ylim[0])
        yhi = np.where(np.isfinite(hi), np.minimum(hi, ylim[1]), ylim[1])
        ax.errorbar(x[inside], v[inside], yerr=[(v - ylo)[inside], (yhi - v)[inside]],
                    fmt="none", ecolor=col, elinewidth=1.8, capsize=3, capthick=1.4, zorder=3)
        ax.plot(x[inside], v[inside], mk, ms=7.5, color=col, markeredgecolor="white",
                markeredgewidth=1.4, zorder=4, label=nice)
        for xi, vi in zip(x[~inside], v[~inside]):      # off scale: caret + the value, never hidden
            ax.plot([xi], [ylim[0] + .02 * (ylim[1] - ylim[0])], marker="v", ms=8, color=col,
                    markeredgecolor="white", markeredgewidth=1.2, clip_on=False, zorder=5)
            ax.annotate(f"{vi:.1f}", (xi, ylim[0] + .02 * (ylim[1] - ylim[0])), fontsize=8.5,
                        color=col, ha="center", va="bottom", zorder=6,
                        xytext=(0, 8 + 11 * si), textcoords="offset points")
    ax.axvline(0.5, color=GREY, lw=1.1, ls=(0, (3, 3)), zorder=1)
    ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
    ax.set_xticks(np.arange(len(XS))); ax.set_xticklabels(XLAB, fontsize=9.5)
    ax.set_xlim(-0.55, len(XS) - 0.45); ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel); ax.set_title(title, loc="left", pad=8)
    ax.grid(axis="x", visible=False)


def fig9(B):
    fig, ax = plt.subplots(2, 2, figsize=(12.4, 8.2))
    band_panel(ax[0, 0], B, ENT, "score_recentred", "recentred_ci_lo", "recentred_ci_hi",
               (-1.2, 0.45), "R2", "Answer entropy: R2 collapses inside every band")
    band_panel(ax[0, 1], B, LEN, "score_recentred", "recentred_ci_lo", "recentred_ci_hi",
               (0.0, 0.75), "R2", "Remaining length: R2 holds up")
    band_panel(ax[1, 0], B, ENT, "within_cut_corr", "corr_ci_lo", "corr_ci_hi", (-0.1, 0.72),
               "correlation", "Answer entropy, offset-free view: same story")
    band_panel(ax[1, 1], B, LEN, "within_cut_corr", "corr_ci_lo", "corr_ci_hi", (-0.1, 0.95),
               "correlation", "Remaining length, offset-free view: flat")
    for a in ax[1]:
        a.set_xlabel("question difficulty band")
    ax[0, 0].legend(loc="lower left", ncol=1)
    fig.suptitle("Holding question difficulty constant kills the entropy tie and leaves remaining "
                 "length intact", fontsize=14.5, y=0.995, x=0.012, ha="left")
    fig.text(0.012, -0.035,
             "Scored within each cut point (10 to 90%) then averaged. Bars are 95% intervals from "
             "2,000 question bootstrap draws. Bands are quintiles of how many of the 25 base attempts "
             "at a question were\ncorrect, about 39 questions each, so roughly 39 rows per cut: "
             "intervals inside a band are wide by construction and it is the agreement across all five "
             "bands that carries the result.\nTOP ROW IS OFFSET CORRECTED. The predictions come from a "
             "fit on all 198 questions and are never recentred per band, which penalised a band for an "
             "offset the full-set fit never\ncarried; the prediction's mean is now matched to the "
             "truth's inside each band, which is one degree of freedom and leaves the slope alone. The "
             "very-easy band stays negative for a real\nreason, not an artefact: there the truth is "
             "almost constant (sd 0.27 against 0.64 overall) while the predictions still vary as much "
             "as ever, so they add error. The bottom row drops the\nscale too, and is the most "
             "forgiving view available.",
             fontsize=8.6, color=MUTED, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT / "fig9_difficulty_bands.png"); plt.close(fig)


def fig10(SC):
    """Independent route to the same conclusion: partial difficulty out of target and prediction."""
    rows = [("Best probe", "Our model L21 (ridge)"), ("GPT-6 Luna, zero-shot", "Luna k=0"),
            ("Gemma-4-31B, zero-shot", "Gemma-4-31B k=0"), ("Prefix length alone", "Prefix length")]
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.3))
    for a, t, ttl in [(ax[0], ENT, "Answer entropy"), (ax[1], LEN, "Remaining length")]:
        for i, (nice, lab) in enumerate(rows):
            g = SC[(SC.label == lab) & (SC.target == t)].set_index("metric")
            for met, col, mec, mfc, nm in [("r2", GREY, GREY, "white", "as published"),
                                           ("r2_partialled", BLUE, BLUE, BLUE, "question difficulty removed")]:
                r = g.loc[met]
                a.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=2.2, solid_capstyle="butt",
                       zorder=2, alpha=.9)
                for e in (r.ci_lo, r.ci_hi):            # explicit caps, so an interval reads as one
                    a.plot([e, e], [i - .12, i + .12], color=col, lw=1.4, zorder=2)
                a.plot([r.value], [i], "o", ms=8.5, color=mfc, markeredgecolor=mec,
                       markeredgewidth=1.9, zorder=4, label=nm if i == 0 else None)
            a.annotate("", xy=(g.loc["r2_partialled"].value, i - .3),
                       xytext=(g.loc["r2"].value, i - .3), zorder=3,
                       arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, alpha=.7,
                                       shrinkA=2, shrinkB=2))
        a.axvline(0, color=MUTED, lw=1.0, zorder=1)
        a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows], fontsize=10.5, color=INK)
        a.invert_yaxis(); a.set_xlabel("R2"); a.set_title(ttl, loc="left", pad=8)
        a.grid(axis="y", visible=False)
        a.set_ylim(len(rows) - 0.1, -0.6)      # blank band under the last row, for the legend
    ax[0].set_xlim(-0.14, 0.42); ax[1].set_xlim(0.1, 0.66)
    ax[0].legend(loc="lower left", ncol=2, handletextpad=0.4, columnspacing=1.6)
    fig.suptitle("Remove question difficulty and the entropy scores collapse. The length scores barely "
                 "move.", fontsize=14.5, y=1.045, x=0.012, ha="left")
    fig.text(0.012, -0.10,
             "A quadratic in the question's p_gold (the share of its 25 base attempts that were "
             "correct) is regressed out of BOTH the target and each prediction, within each cut point, "
             "refitted inside every\nbootstrap draw. Bars are 95% intervals, 2,000 question bootstrap "
             "draws, 196 questions. On entropy, nothing is significantly above zero afterwards except "
             "prefix length. Caveat: p_gold is\ncomputed from rollouts, so it is label-derived and this "
             "over-removes. The remaining values are a lower bound, not a measurement of usable skill.",
             fontsize=8.6, color=MUTED, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(OUT / "fig10_difficulty_partialled.png"); plt.close(fig)


def fig11(PR):
    """Paired: Luna zero-shot minus the best probe, inside each band. Paired CIs are the convention."""
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 3.9))
    for a, t, ttl in [(ax[0], ENT, "Answer entropy"), (ax[1], LEN, "Remaining length")]:
        d = PR[(PR.label == "Luna k=0") & (PR.target == t)].set_index("band").reindex(XS)
        for i, (b, r) in enumerate(d.iterrows()):
            if not np.isfinite(r.corr_ci_lo):
                continue
            sig = bool(r.corr_significant)
            col = BLUE
            a.plot([r.corr_ci_lo, r.corr_ci_hi], [i, i], color=col, lw=2.2,
                   alpha=1.0 if sig else 0.5, solid_capstyle="round", zorder=2)
            for e in (r.corr_ci_lo, r.corr_ci_hi):
                a.plot([e, e], [i - .12, i + .12], color=col, lw=1.4, alpha=1.0 if sig else 0.5, zorder=2)
            a.plot([r.corr_diff], [i], "o", ms=8, color=col if sig else "white",
                   markeredgecolor=col, markeredgewidth=1.9, zorder=4)
        a.axvline(0, color=MUTED, lw=1.0, zorder=1)
        a.axhline(0.5, color=GREY, lw=1.0, ls=(0, (3, 3)), zorder=1)
        a.set_yticks(range(len(XS)))
        a.set_yticklabels([x.replace("v_", "very ") for x in XS], fontsize=10.5, color=INK)
        a.invert_yaxis(); a.set_xlabel("Luna's correlation minus the best probe's")
        a.set_title(ttl, loc="left", pad=8); a.grid(axis="y", visible=False)
        a.set_ylim(len(XS) - 0.1, -0.6)        # blank band under the last row, for the legend
    ax[0].plot([], [], "o", ms=8, color=BLUE, label="significant")
    ax[0].plot([], [], "o", ms=8, color="white", markeredgecolor=BLUE, markeredgewidth=1.9,
               label="not significant")
    ax[0].legend(loc="lower left", ncol=2, handletextpad=0.4, columnspacing=1.6)
    fig.suptitle("Paired: the frontier model against the best probe, inside each difficulty band",
                 fontsize=14, y=1.05, x=0.012, ha="left")
    fig.text(0.012, -0.12,
             "Offset-free within-cut correlation, paired on identical rows and identical bootstrap "
             "draws (2,000). Left of zero means the best probe is ahead. On answer entropy the two "
             "are statistically indistinguishable in every single band, and over all\nquestions. On "
             "remaining length the best probe is ahead in four of the five bands and significantly so "
             "in three of them, as well as over all questions.",
             fontsize=8.6, color=MUTED, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(OUT / "fig11_bands_paired.png"); plt.close(fig)


def fig12(SP):
    """Are they just predicting the mean? Show what the predictions themselves look like."""
    SER = [("truth", INK, "o", "-"), ("Best probe", BLUE, "s", "--"), ("Luna k=0 (calibrated)", ORANGE, "^", "--")]
    fig, ax = plt.subplots(2, 2, figsize=(12.4, 8.0))
    for row, (t, nice, dec) in enumerate([(ENT, "Answer entropy", 2), (LEN, "Remaining length", 0)]):
        for col, (stat, sname) in enumerate([("mean", "average value"), ("sd", "spread (standard deviation)")]):
            a = ax[row, col]
            for lab, c, mk, ls in SER:
                src = "Best probe" if lab == "truth" else lab      # the truth is the same whoever predicts it
                d = SP[(SP.label == src) & (SP.target == t)].set_index("band").reindex(XS)
                k = f"{stat}_{'true' if lab == 'truth' else 'pred'}"
                v, lo, hi = d[k].to_numpy(), d[k + "_lo"].to_numpy(), d[k + "_hi"].to_numpy()
                x = np.arange(len(XS)) + {0: -0.12, 1: 0.0, 2: 0.12}[SER.index((lab, c, mk, ls))]
                a.plot(x[1:], v[1:], color=c, lw=1.8, ls=ls, alpha=.85, zorder=2)
                a.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor=c, elinewidth=1.8,
                           capsize=3, capthick=1.4, zorder=3)
                a.plot(x, v, mk, ms=7.5, color=c, markeredgecolor="white", markeredgewidth=1.4,
                       zorder=4, label=("what actually happened" if lab == "truth" else lab))
            a.axvline(0.5, color=GREY, lw=1.1, ls=(0, (3, 3)), zorder=1)
            if stat == "sd":
                a.set_ylim(bottom=0)
                a.annotate("a predictor that always said the same number\nwould sit on this line",
                           (len(XS) - 0.55, 0), fontsize=8.5, color=MUTED, ha="right", va="bottom",
                           xytext=(0, 5), textcoords="offset points")
            a.set_xticks(np.arange(len(XS))); a.set_xticklabels(XLAB, fontsize=9.5)
            a.set_xlim(-0.55, len(XS) - 0.45); a.grid(axis="x", visible=False)
            a.set_title(f"{nice}: {sname}", loc="left", pad=8)
            a.set_ylabel("bits" if t == ENT else "tokens")
    for a in ax[1]:
        a.set_xlabel("question difficulty band")
    ax[0, 0].legend(loc="lower left", ncol=1)
    fig.suptitle("They are not predicting one flat number, but they are far too timid",
                 fontsize=14.5, y=0.995, x=0.012, ha="left")
    fig.text(0.012, -0.03,
             "Bars are 95% intervals from 2,000 question bootstrap draws. LEFT: what actually happens "
             "versus what each method says will happen. On entropy both follow the real pattern only "
             "about half as far as\nthey should, and on the very easy questions they predict roughly "
             "0.4 bits where the truth is 0.1. RIGHT: how much the predictions move around. A method "
             "that had learned nothing inside a band would sit\nat zero; both sit at about half the "
             "real spread. Remaining length, the bottom row, tracks the truth far better on both counts.",
             fontsize=8.6, color=MUTED, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT / "fig12_predicting_the_mean.png"); plt.close(fig)


if __name__ == "__main__":
    B = pd.read_csv(S / "p5_bands_scores.csv")
    SC = pd.read_csv(S / "p2_ci_scores.csv")
    PR = pd.read_csv(S / "p5_bands_paired.csv")
    fig9(B); fig10(SC); fig11(PR)
    fig12(pd.read_csv(S / "p7_band_spread.csv"))
    for f in ["fig9_difficulty_bands", "fig10_difficulty_partialled", "fig11_bands_paired",
              "fig12_predicting_the_mean"]:
        print("wrote", OUT / f"{f}.png")
