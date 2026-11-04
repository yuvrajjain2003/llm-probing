#!/usr/bin/env python3
"""Slide figures for the new-dataset results (work packages P1, P4, P5, P6).

Reads only stored CSVs. Nothing is refitted: every number here was produced by
`newds_ref_table.py`, `qtext_baseline.py`/`qtext_compare.py`, `newds_difficulty_scoring.py`
and `loglen_refit.py`, and is read out of their outputs.

    ./envs/goal1/bin/python src/viz/newds_figures.py
Outputs: docs/figures/newds/fig{1..8}_*.png

House style follows src/viz/slide_figures.py and src/viz/round6_figures.py: same palette slots
(1-3 of the dataviz skill's default categorical palette), same rcParams, significance carried by
filled-vs-hollow markers rather than colour alone, every point that has an interval shows it, and
any point that has no interval available says so on the figure. No em dashes.

Two source choices worth knowing, both deliberate:
  * `committed_by_here` is taken from the SHALLOW logistic grid (`ref_fixed`), which is the
    published convention. `_qtext/cmp_scores.csv` carries the DEEP-grid AUC instead, which differs
    on GPQA (0.761 against 0.784) and GlobalOpinionQA (0.671 against 0.683). See the report.
  * GPQA's reference CIs come from `_newds_reports/_gpqa_method_check/`, which is the same script
    as the new datasets' `_ref_ci/` with its root repointed, so the three datasets are comparable.
"""
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("runs/goal2")
OUT = Path("docs/figures/newds"); OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, BAD = "#1baf7a", "#e34948"
INK, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#e3e2dd", "#8a8884"

ENT, ENTN = "answer_entropy_bits_mm", "answer_entropy_bits_mm_norm"
PMODE, LEN, LSD, COMM = "p_mode", "rem_len_mean", "rem_len_std", "committed_by_here"
NICE = {ENT: "Answer entropy", ENTN: "Answer entropy, normalised", PMODE: "P(mode answer)",
        LEN: "Remaining length", LSD: "Length spread", COMM: "Commitment"}

# dataset tag -> (display name, question count)
DS = [("gpqa25", "GPQA\n198 questions", 198),
      ("mmlupro_v1", "MMLU-Pro\n2,499 questions", 2499),
      ("globalop_v1", "GlobalOpinionQA\n2,555 questions", 2555)]
DSSHORT = {"gpqa25": "GPQA", "mmlupro_v1": "MMLU-Pro", "globalop_v1": "GlobalOpinionQA"}

OURS, TFIDF, PLEN = "Our model L21 (ridge)", "TF-IDF on prefix", "Prefix length"
SERIES = [(OURS, "Activations, layer 21", BLUE, "o"),
          (TFIDF, "TF-IDF on the same prefix text", ORANGE, "s"),
          (PLEN, "Prefix length alone", AQUA, "^")]

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11, "axes.labelcolor": MUTED,
    "axes.edgecolor": MUTED, "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "legend.frameon": False, "legend.fontsize": 10,
})


# ---------------------------------------------------------------- loaders ----
def ref_ci(tag):
    """Reference scores with 95% question-bootstrap CIs, same script for all three datasets."""
    if tag == "gpqa25":
        return pd.read_csv(R / "_newds_reports/_gpqa_method_check/ci_scores.csv")
    return pd.read_csv(R / tag / "sweep/_ref_ci/ci_scores.csv")


def ref_paired(tag):
    if tag == "gpqa25":
        return pd.read_csv(R / "_newds_reports/_gpqa_method_check/ci_paired.csv")
    return pd.read_csv(R / tag / "sweep/_ref_ci/ci_paired.csv")


def ref_auc(tag):
    """Commitment: AUC, accuracy and the majority baseline, shallow grid (`ref_fixed`)."""
    p = (R / "_newds_reports/_gpqa_method_check/committed_auc.csv" if tag == "gpqa25"
         else R / tag / "sweep/_ref_ci/committed_auc.csv")
    return pd.read_csv(p)


def ref_norm(tag):
    p = R / tag / "sweep/_ref_ci/entropy_norm.csv"
    return pd.read_csv(p) if p.exists() else None


def f0_cmp(tag):
    """0%-cut head-to-head with CIs: activations, question text, prefix-length control."""
    return pd.read_csv(R / tag / "sweep/_qtext/cmp_f0.csv")


def p6(tag, kind, name):
    """P6 difficulty files. `kind` is 'rollout' (label-derived) or 'qtext' (text-derived)."""
    if tag == "gpqa25":
        return pd.read_csv(R / f"_newds_reports/_p6_gpqa_ref/{kind}/{name}.{kind}.csv")
    return pd.read_csv(R / tag / f"sweep/_iclsplit/{name}.{kind}.csv")


def get(d, **kw):
    for k, v in kw.items():
        d = d[d[k] == v]
    assert len(d) == 1, f"{kw} matched {len(d)} rows"
    return d.iloc[0]


def errpoint(ax, x, v, lo, hi, col, mk, label=None, filled=True, ms=7.5, horiz=False):
    """One point with its 95% interval. Hollow marker = interval crosses the reference."""
    if horiz:
        ax.plot([lo, hi], [x, x], color=col, lw=2.0, solid_capstyle="butt", zorder=2)
        for e in (lo, hi):
            ax.plot([e, e], [x - .12, x + .12], color=col, lw=1.3, zorder=2)
        ax.plot([v], [x], mk, ms=ms, color=col if filled else "white", markeredgecolor=col,
                markeredgewidth=1.8, zorder=4, label=label)
    else:
        ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt="none", ecolor=col, elinewidth=1.8,
                    capsize=3, capthick=1.3, zorder=3)
        ax.plot([x], [v], mk, ms=ms, color=col if filled else "white", markeredgecolor=col,
                markeredgewidth=1.6, zorder=4, label=label)


def caption(fig, text, y=-0.045):
    fig.text(0.012, y, text, fontsize=8.6, color=MUTED, ha="left", va="top")


# ======================================================== fig 1 ==============
def fig1():
    """Three datasets, five targets, three approaches. MMLU-Pro beats GPQA everywhere,
    GlobalOpinionQA is below it everywhere."""
    panels = [(ENT, "R2"), (PMODE, "R2"), (LEN, "R2"), (LSD, "R2"),
              ("comm_acc", "accuracy"), ("comm_auc", "AUC")]
    fig, axs = plt.subplots(2, 3, figsize=(13.4, 8.0))
    for ax, (t, unit) in zip(axs.ravel(), panels):
        for si, (lab, nice, col, mk) in enumerate(SERIES):
            xs, vs = [], []
            for di, (tag, _, _) in enumerate(DS):
                x = di + (si - 1) * 0.17
                if t in ("comm_acc", "comm_auc"):
                    r = get(ref_auc(tag), label=lab)
                    if t == "comm_acc":
                        v, lo, hi = r.mean_accuracy, np.nan, np.nan
                    else:
                        v, lo, hi = r.mean_auc, r.ci_lo, r.ci_hi
                else:
                    r = get(ref_ci(tag), label=lab, target=t)
                    v, lo, hi = r.mean_score, r.ci_lo, r.ci_hi
                if np.isfinite(lo):
                    errpoint(ax, x, v, lo, hi, col, mk, label=nice if di == 0 else None)
                else:                      # accuracy: the bootstrap file carries no interval
                    ax.plot([x], [v], mk, ms=7.5, color="white", markeredgecolor=col,
                            markeredgewidth=1.6, zorder=4, label=nice if di == 0 else None)
                xs.append(x); vs.append(v)
            ax.plot(xs, vs, color=col, lw=1.1, alpha=.45, zorder=2)
        if t in ("comm_acc", "comm_auc"):
            for di, (tag, _, _) in enumerate(DS):
                m = get(ref_auc(tag), label=OURS).mean_majority
                ax.plot([di - .36, di + .36], [m, m], color=GREY, lw=1.6, ls=(0, (4, 3)),
                        zorder=1, label="majority class baseline" if di == 0 else None)
            if t == "comm_auc":
                ax.axhline(0.5, color=GREY, lw=1.0, ls=":", zorder=1)
        ax.set_xticks(range(len(DS)))
        ax.set_xticklabels([d[1] for d in DS], fontsize=9.2)
        ax.set_xlim(-0.5, len(DS) - 0.5); ax.grid(axis="x", visible=False)
        ax.set_ylabel(unit)
        ttl = {"comm_acc": "Commitment: accuracy",
               "comm_auc": "Commitment: AUC"}.get(t) or NICE[t]
        ax.set_title(ttl, loc="left", pad=8)
    axs[0, 0].legend(loc="lower center", ncol=1, fontsize=9.2)
    h, l = axs[1, 1].get_legend_handles_labels()      # only the baseline, the rest is above
    keep = [(hh, ll) for hh, ll in zip(h, l) if "majority" in ll]
    axs[1, 1].legend([k[0] for k in keep], [k[1] for k in keep], loc="upper left",
                     fontsize=9.2)
    fig.suptitle("MMLU-Pro beats GPQA on all five targets. GlobalOpinionQA is below GPQA on all "
                 "five.", fontsize=14.5, y=1.0, x=0.012, ha="left")
    caption(fig,
            "Ridge on layer 21, last token of the prefix, fixed per-question folds. Scored within "
            "each cut point (10 to 90%) then averaged, never pooled across depth. Bars are 95% "
            "intervals from 2,000 question bootstrap draws.\nUNITS DIFFER BETWEEN PANELS: the first "
            "four are R2, the last two are accuracy and AUC on the same binary target, so do not "
            "read heights across the panel boundary. Commitment accuracy is plotted HOLLOW because "
            "the stored\nbootstrap file carries an interval for the AUC only; the dashed line is "
            "the majority-class baseline, which is most of what accuracy on that target measures. "
            "GlobalOpinionQA's remaining-length interval is wide for a\nreason: ten rows of 12,736 "
            "carry 55.6% of that target's sum of squares on the raw scale. See figure 7, which is "
            "the scale those numbers should be read on.",
            y=-0.055)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig1_three_datasets.png"); plt.close(fig)


# ======================================================== fig 2 ==============
def fig2():
    """The precision story: the gap is flat, the interval shrinks 3x."""
    fig = plt.figure(figsize=(13.0, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.28)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # LEFT: ours minus TF-IDF on entropy, one interval per dataset. The visual subject is length.
    for di, (tag, _, nq) in enumerate(DS):
        r = get(ref_paired(tag), label=OURS, minus=TFIDF, target=ENT)
        hw = (r.ci_hi - r.ci_lo) / 2
        errpoint(ax, di, r["diff"], r.ci_lo, r.ci_hi, BLUE, "o",
                 filled=bool(r.significant), horiz=True, ms=9)
        ax.annotate(f"+{r['diff']:.3f}", (r["diff"], di), xytext=(0, 13),
                    textcoords="offset points", ha="center", fontsize=10.5, color=INK)
        ax.annotate(f"paired half-width {hw:.3f}", (r.ci_hi, di), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=9.5, color=MUTED)
    ax.axvline(0, color=MUTED, lw=1.1, zorder=1)
    ax.set_yticks(range(len(DS)))
    ax.set_yticklabels([f"{DSSHORT[t]}\n{n:,} questions" for t, _, n in DS], fontsize=10.5,
                       color=INK)
    ax.invert_yaxis(); ax.set_ylim(len(DS) - 0.35, -0.75)
    ax.set_xlim(-0.012, 0.135); ax.grid(axis="y", visible=False)
    ax.set_xlabel("activations minus TF-IDF on the same prefix text, answer entropy R2")
    ax.set_title("The gap does not grow. The interval around it shrinks.", loc="left", pad=14)
    ax.plot([], [], "o", ms=8, color=BLUE, label="significant")
    ax.legend(loc="lower left")

    # RIGHT: half-width of our probe's own interval, per target, against question count.
    tg = [(ENT, BLUE, "o"), (PMODE, ORANGE, "s"), (LEN, AQUA, "^"), (LSD, GREY, "D")]
    for t, col, mk in tg:
        xs, ys = [], []
        for tag, _, nq in DS:
            r = get(ref_ci(tag), label=OURS, target=t)
            xs.append(nq); ys.append((r.ci_hi - r.ci_lo) / 2)
        ax2.plot(xs, ys, color=col, lw=1.5, alpha=.8, zorder=2)
        ax2.plot(xs, ys, mk, ms=7.5, color=col, markeredgecolor="white", markeredgewidth=1.3,
                 zorder=4, label=NICE[t])
        if t == ENT:                       # the headline tightening, spelled out
            for xi, yi, dy in zip(xs, ys, [10, 10, -16]):
                ax2.annotate(f"{yi:.3f}", (xi, yi), xytext=(4 if dy > 0 else 10, dy),
                             textcoords="offset points", ha="left", fontsize=9, color=BLUE)
        if t == LEN:                       # the one outlier, and why
            ax2.annotate("GlobalOpinionQA remaining length is the\none exception: ten extreme rows "
                         "dominate\nthe raw target, so it does not tighten.\nFigure 7 is the scale "
                         "to read it on.",
                         (xs[2], ys[2]), xytext=(-14, -2), textcoords="offset points", ha="right",
                         va="top", fontsize=8.4, color=AQUA)
    n0 = 198
    base = (get(ref_ci("gpqa25"), label=OURS, target=ENT).ci_hi
            - get(ref_ci("gpqa25"), label=OURS, target=ENT).ci_lo) / 2
    xx = np.array([198, 2555])
    ax2.plot(xx, base * np.sqrt(n0 / xx), color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1,
             label="1 / sqrt(questions), anchored on GPQA entropy")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xticks([198, 2527]); ax2.set_xticklabels(["198", "2,499 and 2,555"])
    ax2.set_xticks([], minor=True)
    ax2.set_xlim(165, 4200)
    ax2.set_yticks([0.01, 0.02, 0.04, 0.08])
    ax2.set_yticklabels(["0.01", "0.02", "0.04", "0.08"])
    ax2.set_yticks([], minor=True)
    ax2.set_ylim(0.0085, 0.105)
    ax2.set_xlabel("questions in the dataset (log scale)")
    ax2.set_ylabel("half-width of the 95% interval (log scale)")
    ax2.set_title("Why: the intervals follow the question count", loc="left", pad=14)
    ax2.legend(loc="lower left", fontsize=9.2)

    fig.suptitle("More questions did not widen the gap between activations and text. They made it "
                 "measurable.", fontsize=14.5, y=1.045, x=0.012, ha="left")
    caption(fig,
            "LEFT: our probe minus TF-IDF on the identical prefix text, answer entropy, mean "
            "over the five cut points, paired 95% question-bootstrap intervals (2,000 draws).\n"
            "The three point estimates sit within 0.019 R2 of each other while the interval around "
            "them shrinks by a factor of 2.8 (0.031 to 0.011); the single-score interval on\nthe "
            "right shrinks by 3.7 (0.056 to 0.015). RIGHT: the half-width of our probe's own "
            "single-score interval on each target, against the number of questions, both axes\n"
            "logarithmic. A question bootstrap resamples QUESTIONS, so power tracks question count "
            "and not row count: 12.6 times the questions buys about sqrt(12.6) = 3.5 times\nthe "
            "precision, which is what the dashed reference line draws. The practical consequence is "
            "that effects of 0.01 R2 are now measurable, where on GPQA only effects above\nabout "
            "0.05 were.",
            y=-0.115)
    fig.savefig(OUT / "fig2_precision.png"); plt.close(fig)


# ======================================================== fig 3 ==============
def fig3():
    """0% cut against the 10-90% mean, per target per dataset, as a share."""
    tg = [ENT, PMODE, LEN, LSD]
    fig, axs = plt.subplots(1, 3, figsize=(14.2, 4.5), sharex=True)
    fig.subplots_adjust(wspace=0.08)
    for ci, (ax, (tag, dname, nq)) in enumerate(zip(axs, DS)):
        F, S = f0_cmp(tag), ref_ci(tag)
        for i, t in enumerate(tg):
            f = get(F, target=t, label="Our model L21 @0%")
            s = get(S, label=OURS, target=t)
            hl = t == LSD
            # 10-90% mean, hollow; 0% cut, filled; joined, so the share is the line's length
            ax.plot([f.r2, s.mean_score], [i, i], color=GREY, lw=1.4, zorder=2, alpha=.9)
            errpoint(ax, i, s.mean_score, s.ci_lo, s.ci_hi, MUTED, "s", horiz=True, filled=False,
                     label="10 to 90% cuts, averaged" if i == 0 else None)
            errpoint(ax, i, f.r2, f.ci_lo, f.ci_hi, BAD if hl else BLUE, "o", horiz=True,
                     label="0% cut, before any reasoning" if i == 0 else None)
            share = f.r2 / s.mean_score
            ax.annotate(f"{share:.0%}", (0.985, i), xycoords=("axes fraction", "data"),
                        ha="right", va="center", fontsize=10.5,
                        color=BAD if hl else INK, fontweight="bold" if hl else "normal")
        ax.axvline(0, color=MUTED, lw=1.0, zorder=1)
        ax.axvline(0.845, color=GRID, lw=1.2, zorder=1)      # share column sits to the right
        ax.set_yticks(range(len(tg)))
        ax.set_yticklabels([NICE[t] for t in tg] if ci == 0 else [""] * len(tg), fontsize=10.6,
                           color=INK)
        if ci == 0:
            for lab, t in zip(ax.get_yticklabels(), tg):
                if t == LSD:
                    lab.set_color(BAD)
        ax.invert_yaxis(); ax.set_ylim(len(tg) - 0.4, -0.95)
        ax.set_xlim(-0.03, 1.00); ax.grid(axis="y", visible=False)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
        ax.set_xlabel("R2")
        ax.set_title(f"{DSSHORT[tag]}, {nq:,} questions", loc="left", pad=8)
        ax.annotate("share of the\n10 to 90% mean", (0.985, -0.80),
                    xycoords=("axes fraction", "data"), ha="right", va="center", fontsize=8.6,
                    color=MUTED)
    axs[0].legend(loc="upper center", fontsize=9.4, bbox_to_anchor=(0.46, -0.17), ncol=2)
    fig.suptitle("Two thirds to nine tenths of the probe's score exists before a single reasoning "
                 "token", fontsize=14.5, y=1.035, x=0.012, ha="left")
    caption(fig,
            "At the 0% cut the model has emitted nothing: the prefix is the empty string and there "
            "is ONE row per question, so the probe can only read the question. Prefix length scores "
            "-0.0005 to -0.019 there, as a degenerate\ncontrol must. The percentage is the 0% score "
            "divided by the mean of the five 10 to 90% scores; it is a ratio of a single-cut score "
            "to a five-cut average, so read it as an approximation and note that it has NO interval "
            "of its own,\nthough both of the points it is built from do (95%, 2,000 question "
            "bootstrap draws). LENGTH SPREAD, in red, is the exception on every dataset and the one "
            "that most needs the prefix: 69%, 70% and 24%. That is also the target\nRESEARCH.md "
            "identifies as the project's sharpest novelty claim. MMLU-Pro's 0% entropy score alone "
            "(0.301) is higher than GPQA's entire 10 to 90% average (0.289).",
            y=-0.10)
    fig.savefig(OUT / "fig3_zero_pct_share.png"); plt.close(fig)


# ======================================================== fig 4 ==============
def fig4():
    """Activations against the question text, at the 0% cut: same one row per question."""
    tg = [ENT, PMODE, LEN, LSD]
    OURS0, TXT0, CTRL0 = "Our model L21 @0%", "Question text + choices @0%", \
        "Prefix length @0% (control)"
    fig, axs = plt.subplots(2, 3, figsize=(13.6, 7.8),
                            gridspec_kw=dict(height_ratios=[1.25, 1.0]))
    for col, (tag, dname, nq) in enumerate(DS):
        F = f0_cmp(tag)
        a, b = axs[0, col], axs[1, col]
        for i, t in enumerate(tg):
            o, x = get(F, target=t, label=OURS0), get(F, target=t, label=TXT0)
            c = get(F, target=t, label=CTRL0)
            errpoint(a, i, o.r2, o.ci_lo, o.ci_hi, BLUE, "o", horiz=True,
                     label="activations, layer 21" if i == 0 else None)
            errpoint(a, i, x.r2, x.ci_lo, x.ci_hi, ORANGE, "s", horiz=True,
                     label="question text + choices, TF-IDF" if i == 0 else None)
            a.plot([c.r2], [i], "x", ms=6, color=GREY, zorder=4,
                   label="prefix length (degenerate control)" if i == 0 else None)
            sig = bool(x.significant)
            cc = GOOD if x.diff_vs_probe < 0 else BAD
            errpoint(b, i, x.diff_vs_probe, x.diff_lo, x.diff_hi, cc, "o", horiz=True,
                     filled=sig, ms=8)
        for ax in (a, b):
            ax.set_yticks(range(len(tg)))
            ax.set_yticklabels([NICE[t] for t in tg] if col == 0 else [""] * len(tg),
                               fontsize=10.2, color=INK)
            ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        a.set_ylim(len(tg) + 0.55, -0.6); b.set_ylim(len(tg) + 0.15, -0.5)
        a.axvline(0, color=MUTED, lw=1.0, zorder=1); b.axvline(0, color=MUTED, lw=1.1, zorder=1)
        a.set_title(f"{DSSHORT[tag]}, {nq:,} questions", loc="left", pad=8)
        a.set_xlim(-0.04, 0.72); b.set_xlim(-0.38, 0.06)
        a.set_xlabel("R2 at the 0% cut", labelpad=2)
        b.set_xlabel("text minus activations (R2)")
    axs[0, 0].legend(loc="lower right", fontsize=9.2)
    axs[1, 0].plot([], [], "o", ms=8, color=GOOD, label="text is behind, significant")
    axs[1, 0].plot([], [], "o", ms=8, color="white", markeredgecolor=GOOD, markeredgewidth=1.8,
                   label="interval crosses zero")
    axs[1, 0].legend(loc="lower left", fontsize=9.0)
    fig.suptitle("Before emitting a token, layer 21 knows things about its own rollouts that the "
                 "question's words do not say", fontsize=14.5, y=1.0, x=0.012, ha="left")
    caption(fig,
            "The cleanest head-to-head available: the 0% cut gives both sides exactly the same "
            "information, one row per question, no prefix at all. TOP ROW, absolute R2 with 95% "
            "question-bootstrap intervals (2,000 draws). BOTTOM ROW,\nthe same comparison paired on "
            "identical rows and identical draws; left of zero means the activations are ahead. On "
            "MMLU-Pro the activations keep 1.5 to 2.5 times the text's R2 and the gap is significant "
            "on all four targets shown\n(and on normalised entropy, not plotted). GlobalOpinionQA "
            "inverts it: the text recovers 79 to 96% of the probe, and on BOTH length targets (and "
            "on normalised entropy, not plotted) the gap is not significant at 2,555 questions, so "
            "there\nthe probe is largely reading the question's words. Its two small entropy and "
            "P(mode) gaps do clear significance, which only that sample size makes visible. GPQA "
            "points the same way as MMLU-Pro but only remaining length\nclears significance at 198 "
            "questions: do NOT read its wide crossing intervals as nulls. The text baseline is a "
            "nested-selection TF-IDF fit on question + choices, which is the\nheadline version "
            "because the number of options varies from 2 to 18.",
            y=-0.055)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT / "fig4_zero_pct_vs_text.png"); plt.close(fig)


# ======================================================== fig 5 ==============
def fig5():
    """The depth effect in the partialling: never drawn before."""
    fig, axs = plt.subplots(2, 2, figsize=(12.0, 7.8), sharex=True)
    rows = [(ENT, "Answer entropy", (-0.26, 0.64)), (LEN, "Remaining length", (-0.05, 0.86))]
    for ri, (t, nice, ylim) in enumerate(rows):
        for ci, tag in enumerate(["mmlupro_v1", "gpqa25"]):
            ax = axs[ri, ci]
            d = p6(tag, "rollout", "p2_per_checkpoint")
            d = d[(d.label == OURS) & (d.target == t)].sort_values("fraction")
            x = d.fraction.to_numpy() * 100
            for col, mk, key, lab in [
                    (BLUE, "o", "r2", "as published"),
                    (ORANGE, "s", "r2_of_difficulty_alone",
                     "question difficulty alone (in sample)"),
                    (AQUA, "^", "r2_partialled", "what survives difficulty removal")]:
                ax.plot(x, d[key], color=col, lw=1.8, zorder=2)
                ax.plot(x, d[key], mk, ms=8, color=col, markeredgecolor="white",
                        markeredgewidth=1.4, zorder=4, label=lab)
            ax.fill_between(x, d.r2_partialled, d.r2, color=BLUE, alpha=0.07, zorder=1)
            ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
            ax.set_xticks(x); ax.set_xticklabels([f"{int(v)}%" for v in x])
            ax.set_ylim(*ylim); ax.grid(axis="x", visible=False)
            ax.set_title(f"{DSSHORT[tag]}: {nice.lower()}", loc="left", pad=8)
            if ci == 0:
                ax.set_ylabel("R2")
            if ri == 1:
                ax.set_xlabel("cut point, fraction of the chain the model has written")
    for ax in axs[0]:
        ax.annotate("difficulty explains less and less;\nwhat the probe keeps grows with depth",
                    (0.5, 0.97), xycoords="axes fraction", ha="center", va="top", fontsize=9,
                    color=MUTED)
    axs[0, 0].legend(loc="lower right", fontsize=9.4, bbox_to_anchor=(1.0, 0.0))
    fig.suptitle("The difficulty explanation is a SHALLOW-CUT story. Deeper in the chain, most of "
                 "the entropy signal is not difficulty.", fontsize=14.0, y=0.995, x=0.012,
                 ha="left")
    caption(fig,
            "A quadratic in the question's difficulty is regressed out of BOTH the target and the "
            "prediction, within each cut point. Difficulty here is label-derived (p_gold, the share "
            "of a question's rollouts that were correct), which\nover-removes, so the green line is "
            "a LOWER bound on what survives. NO INTERVALS ARE AVAILABLE PER CUT: the stored "
            "bootstrap covers the mean over cuts only, so read the shapes and not small differences "
            "between adjacent\npoints. For scale, the mean-over-cuts half-width on entropy is 0.026 "
            "(MMLU-Pro) and 0.066 (GPQA). The orange line is an in-sample fit, so it is an upper "
            "bound on difficulty's own share, and partialled R2 is NOT a subset of\npublished R2: "
            "partialling changes the denominator too, so the two must not be read as a "
            "decomposition. Remaining length, the bottom row, has almost nothing to remove at any "
            "depth: difficulty alone never exceeds 0.053 on\nMMLU-Pro and 0.194 on GPQA.",
            y=-0.035)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT / "fig5_depth_partialled.png"); plt.close(fig)


# ======================================================== fig 6 ==============
def fig6():
    """The bracket: the two difficulty measures bound the answer from opposite sides."""
    fig, axs = plt.subplots(1, 2, figsize=(13.0, 3.5))
    rows = [("mmlupro_v1", "MMLU-Pro"), ("gpqa25", "GPQA")]
    for ax, t, nice, xlim in [(axs[0], ENT, "Answer entropy", (-0.09, 0.47)),
                              (axs[1], LEN, "Remaining length", (0.38, 0.78))]:
        for i, (tag, dn) in enumerate(rows):
            pub = get(p6(tag, "rollout", "p2_ci_scores"), label=OURS, target=t, metric="r2")
            lowb = get(p6(tag, "rollout", "p2_ci_scores"), label=OURS, target=t,
                       metric="r2_partialled")
            upb = get(p6(tag, "qtext", "p2_ci_scores"), label=OURS, target=t,
                      metric="r2_partialled")
            # the bracket itself: the span the honest answer lies in
            w = upb.value - lowb.value
            ax.add_patch(plt.Rectangle((lowb.value, i - 0.26), w, 0.52, facecolor=BLUE,
                                       alpha=0.12, edgecolor="none", zorder=1))
            msg = (f"the answer is in here: {lowb.value:.2f} to {upb.value:.2f}" if w > 0.02
                   else f"barely moves: {lowb.value:.2f} to {upb.value:.2f}")
            ax.annotate(msg, ((lowb.value + upb.value) / 2, i + 0.26), xytext=(0, -3),
                        textcoords="offset points", ha="center", va="top", fontsize=9.0,
                        color=BLUE)
            for r, col, mk, lab, filled in [
                    (pub, GREY, "o", "as published", True),
                    (upb, BLUE, "^", "text-derived difficulty removed (UPPER bound)", True),
                    (lowb, ORANGE, "v", "label-derived difficulty removed (LOWER bound)", True)]:
                sig = bool(r.ci_lo > 0)
                errpoint(ax, i, r.value, r.ci_lo, r.ci_hi, col, mk, horiz=True, ms=9,
                         filled=sig, label=lab if i == 0 else None)
        ax.axvline(0, color=MUTED, lw=1.1, zorder=1)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[1] for r in rows], fontsize=11.5, color=INK)
        ax.invert_yaxis(); ax.set_ylim(len(rows) - 0.42, -0.55)
        ax.set_xlim(*xlim); ax.grid(axis="y", visible=False)
        ax.set_xlabel("R2, mean over the five cut points")
        ax.set_title(nice, loc="left", pad=8)
    axs[0].plot([], [], "o", ms=8, color="white", markeredgecolor=MUTED, markeredgewidth=1.8,
                label="hollow: interval includes zero")
    axs[0].legend(loc="upper center", fontsize=9.6, ncol=2, bbox_to_anchor=(1.07, -0.30))
    fig.suptitle("Report the pair, not either number. Entropy on MMLU-Pro survives difficulty "
                 "removal; on GPQA it may not.", fontsize=14.0, y=1.05, x=0.012, ha="left")
    caption(fig,
            "Two difficulty measures, biased in OPPOSITE directions, so together they bracket the "
            "answer and neither alone is honest. The label-derived measure is a quadratic in the "
            "question's p_gold, computed from the very rollouts\nthat define the target, so it "
            "over-removes and gives a lower bound. The text-derived measure is an out-of-fold TF-IDF "
            "prediction of the question's own mean label, which for a held-out question uses none of "
            "its rollouts, so it is\nattenuated and gives an upper bound. Both are refitted inside "
            "each of 2,000 question bootstrap draws; bars are 95% intervals. ENTROPY: on MMLU-Pro "
            "even the harshest removal leaves 0.200 [0.174, 0.226], half the published\n0.410 and "
            "comfortably above zero; on GPQA the same operation leaves 0.025 [-0.045, +0.086], which "
            "is not a measured zero but an unmeasurable small number. REMAINING LENGTH: both "
            "datasets barely move, so difficulty explains\nalmost none of it. Round 6's \"the tie "
            "was question difficulty\" stands for GPQA and does NOT replicate on the 12.6 times "
            "larger dataset.",
            y=-0.44)
    fig.savefig(OUT / "fig6_difficulty_bracket.png"); plt.close(fig)


# ======================================================== fig 7 ==============
LOGLEN = R / "globalop_v1/sweep/_loglen"
LOGLOG = Path("runs/logs/newds/loglen_globalop_4998355.log")
FEATNAME = {"act_L21": OURS, "tfidf": TFIDF, "prefix_len": PLEN}


def loglen_results():
    """Prefer the finished CSV. Fall back to the running job's own log, and say which was used."""
    p = LOGLEN / "results.csv"
    if p.exists():
        d = pd.read_csv(p)
        return d[["feature", "target", "transform", "fraction", "test"]], "csv"
    rows = []
    pat = re.compile(r"^\s+(0\.\d)\s+(\S+)\s+(\S+)\s+(\S+)\s+R2\s+([-+][\d.]+)")
    for ln in LOGLOG.read_text().splitlines():
        m = pat.match(ln)
        if m:
            rows.append(dict(fraction=float(m[1]), feature=m[2], target=m[3], transform=m[4],
                             test=float(m[5])))
    return pd.DataFrame(rows), "log"


def fig7():
    D, src = loglen_results()
    ci = pd.read_csv(LOGLEN / "ci.csv") if (LOGLEN / "ci.csv").exists() else None
    tail = pd.read_csv(LOGLEN / "tail_mass.csv")
    cuts = sorted(D.fraction.unique())
    partial = sorted(set([0.1, 0.3, 0.5, 0.7, 0.9]) - set(cuts))

    TR = [("raw", "RAW token counts: do not quote these"),
          ("log1p", "log(1 + tokens): the scale to use"),
          ("winsor99", "clipped at the 99th percentile")]
    FEATS = [("act_L21", BLUE, "o", "Activations, layer 21"),
             ("tfidf", ORANGE, "s", "TF-IDF on the same prefix"),
             ("prefix_len", AQUA, "^", "Prefix length alone")]
    fig = plt.figure(figsize=(13.2, 10.4))
    gs = fig.add_gridspec(3, 4, width_ratios=[3.1, 1.0, 3.1, 1.0], wspace=0.10, hspace=0.42)
    for ri, (tr, ttl) in enumerate(TR):
        for ci_, (t, nice) in enumerate([(LEN, "Remaining length"), (LSD, "Length spread")]):
            ax = fig.add_subplot(gs[ri, ci_ * 2])
            axm = fig.add_subplot(gs[ri, ci_ * 2 + 1], sharey=ax)
            for feat, col, mk, flab in FEATS:
                d = D[(D.feature == feat) & (D.target == t) & (D["transform"] == tr)]
                d = d.sort_values("fraction")
                if not len(d):
                    continue
                x = d.fraction.to_numpy() * 100
                ax.plot(x, d.test, color=col, lw=1.8, zorder=2)
                ax.plot(x, d.test, mk, ms=8, color=col, markeredgecolor="white",
                        markeredgewidth=1.4, zorder=4, label=flab)
                if ci is not None:                 # mean over cuts, with its 95% interval
                    r = ci[(ci.feature == feat) & (ci.target == t)
                           & (ci["transform"] == tr)].iloc[0]
                    xm = {"act_L21": 0, "tfidf": 1, "prefix_len": 2}[feat]
                    errpoint(axm, xm, r.score, r.ci_lo, r.ci_hi, col, mk, ms=8)
            tm = tail[(tail.target == t) & (tail["transform"] == tr)].top10_share_of_ss
            ax.annotate(f"ten most extreme rows of 12,736 carry\n"
                        f"{tm.min():.0%} to {tm.max():.0%} of this target's sum of squares",
                        (0.5, 0.985), xycoords="axes fraction", ha="center", va="top",
                        fontsize=8.6, color=BAD if tr == "raw" else MUTED)
            ax.set_xticks([10, 30, 50, 70, 90])
            ax.set_xticklabels(["10%", "30%", "50%", "70%", "90%"])
            ax.set_ylim(-0.07, 0.90); ax.grid(axis="x", visible=False)
            ax.set_title(nice, loc="left", pad=7, fontsize=11.5)
            axm.set_xlim(-0.8, 2.8); axm.set_xticks([1]); axm.set_xticklabels(["mean"])
            axm.grid(axis="x", visible=False)
            plt.setp(axm.get_yticklabels(), visible=False)
            axm.spines["left"].set_visible(False); axm.tick_params(left=False)
            if ci_ == 0:
                ax.set_ylabel("R2")
            if ri == 2:
                ax.set_xlabel("cut point")
            if ri == 0 and ci_ == 0:
                leg_ax = ax
                axm.annotate("mean over the\nfive cuts, with\nits 95% interval", (1, 0.87),
                             ha="center", va="top", fontsize=8.2, color=MUTED)
            if ci_ == 1:
                axm.set_title(ttl, loc="right", pad=7, fontsize=11.5,
                              color=BAD if tr == "raw" else MUTED)
    h, lb = leg_ax.get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=3, fontsize=10.5, bbox_to_anchor=(0.5, 0.075))
    note = ""
    if partial:
        note = ("  PARTIAL: cuts " + ", ".join(f"{int(c*100)}%" for c in partial)
                + " are missing from results.csv.")
    if ci is None:
        note += ("  NO INTERVALS ARE SHOWN ANYWHERE: the fits completed but the bootstrap step of "
                 "the job crashed, so ci.csv does not exist.")
    fig.suptitle("GlobalOpinionQA length: on the raw scale prefix length wins, on any sane scale it "
                 "collapses", fontsize=14.0, y=0.995, x=0.012, ha="left")
    caption(fig,
            "Same probes, same alpha grids, same fixed folds; only the target's scale changes, so "
            "`raw` is a control and it reproduces the reference run to four decimals. SCORES ON "
            "DIFFERENT TARGET SCALES ARE NOT COMPARABLE IN\nABSOLUTE TERMS: what is comparable, and "
            "what the figure is about, is the ORDERING of the three approaches within one row. On "
            "the raw scale prefix length beats our probe on remaining length at the 30% and 50% cuts "
            "(0.387\nand 0.464 against 0.341 and 0.377), which is what produced the reference run's "
            "non-result of -0.013 [-0.150, +0.212]. That ordering is an artefact of a handful of "
            "5,000 to 12,000 token rows: the longest prefixes sit with the\nlongest remainders. Both "
            "transforms bring the top-ten share into the 2 to 4% range that MMLU-Pro's raw target "
            "already occupies, and both give the same answer, which is the robustness check that "
            "matters. The PER-CUT points carry NO interval:\nthe stored bootstrap covers "
            "the mean over the five cuts only, which is the narrow column on the right of "
            "each panel, where every point does have its 95% question-bootstrap interval "
            "(2,000 draws). On that mean, on log1p, ours is\n0.628 [0.617, 0.639] against "
            "prefix length's 0.107 [0.037, 0.157], a gap of +0.521 [+0.472, +0.591]; on the "
            "raw scale the same gap is -0.013 [-0.150, +0.212] and not significant." + note,
            y=-0.03)
    fig.savefig(OUT / "fig7_globalop_loglen.png"); plt.close(fig)
    return src, partial, ci


# ======================================================== fig 8 ==============
def fig8():
    """Does the activations-beat-text claim survive difficulty removal? On MMLU-Pro it grows."""
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 5.6), sharex="col")
    specs = [("as published", "r2", GREY, "o"),
             ("text-derived difficulty removed (upper bound)", "r2_partialled", BLUE, "^"),
             ("label-derived difficulty removed (lower bound)", "r2_partialled", ORANGE, "v")]
    for ri, (tag, dn) in enumerate([("mmlupro_v1", "MMLU-Pro"), ("gpqa25", "GPQA")]):
        for ci, (t, nice, xlim) in enumerate([(ENT, "answer entropy", (-0.025, 0.10)),
                                              (LEN, "remaining length", (-0.01, 0.19))]):
            ax = axs[ri, ci]
            for i, (lab, metric, col, mk) in enumerate(specs):
                kind = "qtext" if "text-derived" in lab else "rollout"
                r = get(p6(tag, kind, "p2_ci_paired"), label=OURS, minus=TFIDF, target=t,
                        metric=metric)
                errpoint(ax, i, r["diff"], r.ci_lo, r.ci_hi, col, mk, horiz=True, ms=8.5,
                         filled=bool(r.significant))
            ax.axvline(0, color=MUTED, lw=1.1, zorder=1)
            ax.set_yticks(range(len(specs)))
            ax.set_yticklabels([x[0] for x in specs] if ci == 0 else [""] * len(specs),
                               fontsize=9.8, color=INK)
            ax.invert_yaxis(); ax.set_ylim(len(specs) - 0.45, -0.6)
            ax.set_xlim(*xlim); ax.grid(axis="y", visible=False)
            ax.set_title(f"{dn}: {nice}", loc="left", pad=7, fontsize=11.5)
            if ri == 1:
                ax.set_xlabel("activations minus TF-IDF on the same prefix text (R2)")
    a0 = axs[0, 0]
    a0.plot([], [], "o", ms=8, color=MUTED, label="significant")
    a0.plot([], [], "o", ms=8, color="white", markeredgecolor=MUTED, markeredgewidth=1.8,
            label="interval crosses zero")
    a0.legend(loc="upper center", fontsize=9.6, ncol=2, bbox_to_anchor=(1.12, -1.48))
    fig.suptitle("On MMLU-Pro the activations-beat-text gap is not a shared difficulty read. It "
                 "grows when difficulty is removed.", fontsize=13.8, y=1.0, x=0.012, ha="left")
    caption(fig,
            "Paired on identical rows and identical bootstrap draws (2,000 over questions); right of "
            "zero means the activations are ahead. On MMLU-Pro the entropy gap goes from +0.055 "
            "[+0.044, +0.068] to +0.063 [+0.048, +0.078]\nwith the harshest, label-derived removal: "
            "it does not shrink, it widens, because the activations were better at the part of the "
            "variance that is NOT difficulty-shaped. On GPQA the same gap falls to +0.032 [-0.010, "
            "+0.075] and\nstops being significant, which is the GPQA-only version of round 6's "
            "conclusion. Remaining length holds up on both. NOTE on direction: section 2 of "
            "`p6_difficulty_scoring.md` quotes these two entropy numbers as -0.032 and\n-0.063, "
            "which is the sign of the TF-IDF-minus-ours rows of the same file; the magnitudes are "
            "right and the direction is the one drawn here, with the activations ahead.",
            y=-0.17)
    fig.savefig(OUT / "fig8_vs_text_after_difficulty.png"); plt.close(fig)


if __name__ == "__main__":
    fig1(); print("wrote fig1_three_datasets.png")
    fig2(); print("wrote fig2_precision.png")
    fig3(); print("wrote fig3_zero_pct_share.png")
    fig4(); print("wrote fig4_zero_pct_vs_text.png")
    fig5(); print("wrote fig5_depth_partialled.png")
    fig6(); print("wrote fig6_difficulty_bracket.png")
    src, partial, ci = fig7()
    print(f"wrote fig7_globalop_loglen.png  (source: {src}, missing cuts: {partial})")
    fig8(); print("wrote fig8_vs_text_after_difficulty.png")
