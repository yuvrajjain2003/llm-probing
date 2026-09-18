#!/usr/bin/env python3
"""Publication figures for the Goal 1 results.

Each figure answers one question; none of them restate a table. Palette is the
reference categorical instance (slots 1-3, fixed order, never cycled), text wears
ink colours rather than series colours, grid is recessive.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42          # never Type 3 (renders blank in many viewers)
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("runs/goal1/full/figures"); OUT.mkdir(parents=True, exist_ok=True)
RUN = "runs/goal1/full/{}/DeepSeek-R1-Distill-Qwen-7B"

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots, fixed order
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a85", "#dedede"
DS = [("math500", "MATH-500", S1), ("gpqa_diamond", "GPQA-Diamond", S2),
      ("aime2025", "AIME-2025", S3)]


def _ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK2)
    ax.grid(alpha=.30, color=GRID, lw=.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def labels(name):
    return pd.read_csv(f"{RUN.format(name)}/labels/checkpoint_labels.csv")


# ── 1. the headline: converges on AN answer, not the RIGHT one ────────────────
def fig_divergence():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (name, label, col) in zip(axes, DS[:2]):
        ck = labels(name)
        g = ck.groupby("fraction").agg(pm=("p_mode", "mean"), pg=("p_gold", "mean"))
        x = g.index * 100
        ax.fill_between(x, g.pg, g.pm, color=col, alpha=.13, lw=0,
                        label="converging on a wrong answer")
        ax.plot(x, g.pm, "o-", color=col, lw=2, ms=8, label="P(mode) — self-agreement")
        ax.plot(x, g.pg, "s--", color=INK2, lw=2, ms=8, label="P(gold) — correctness")
        ax.annotate(f"gap {g.pm.iloc[-1]-g.pg.iloc[-1]:.3f}",
                    xy=(90, (g.pm.iloc[-1]+g.pg.iloc[-1])/2), fontsize=9,
                    color=INK2, ha="right", va="center")
        _ax(ax, label, "checkpoint (% through the reasoning)", "share of continuations")
        ax.set_ylim(0.35, 1.0); ax.legend(fontsize=8.5, loc="lower left", frameon=False)
    fig.suptitle("Reasoning increases agreement, not accuracy", fontsize=13,
                 color=INK, x=.02, ha="left", y=.99)
    fig.tight_layout(); fig.savefig(OUT/"fig1_mode_vs_gold.png", dpi=160); plt.close(fig)


# ── 2. why the flat average hides two opposite movements ──────────────────────
def fig_decomposition():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (name, label, col) in zip(axes, DS[:2]):
        ck = labels(name)
        fin = ck[ck.fraction == .9][["example_id", "cot_id", "p_mode", "p_gold"]].copy()
        fin["right"] = (fin.p_gold > 0) & ((fin.p_mode - fin.p_gold).abs() < 1e-9)
        m = ck.merge(fin[["example_id", "cot_id", "right"]], on=["example_id", "cot_id"])
        for flag, c, ls, nm in ((True, S3, "-", "ends on the correct answer"),
                                (False, "#c0392b", "-", "ends on a wrong answer")):
            g = m[m.right == flag].groupby("fraction")["p_gold"].mean()
            n = m[m.right == flag].groupby("fraction").size().iloc[0]
            ax.plot(g.index*100, g.values, "o"+ls, color=c, lw=2, ms=8,
                    label=f"{nm}  (n={n})")
        # The aggregate the two cancel to -- without it the title claims a
        # cancellation the figure never shows. It is not a category, so it wears
        # ink and a dashed stroke, matching how P(gold) is drawn in fig 1.
        allg = ck.groupby("fraction")["p_gold"].mean()
        ax.plot(allg.index*100, allg.values, "s--", color=INK2, lw=2, ms=8,
                label="all CoTs pooled — the flat average")
        _ax(ax, label, "checkpoint (% through the reasoning)", "P(gold)")
        ax.set_ylim(0, 1)
    # One shared legend under both panels: the series are identical, and any
    # in-axes placement collided with a line in at least one panel.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(.5, -.02))
    fig.suptitle("The flat average is two opposite movements cancelling",
                 fontsize=13, color=INK, x=.02, ha="left", y=.99)
    fig.tight_layout(rect=[0, .06, 1, 1])
    fig.savefig(OUT/"fig2_decomposition.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ── 3. extra compute pays only while the distribution is wide ─────────────────
def fig_compute_value():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, label, col in DS:
        ck = labels(name)
        g = ck.groupby("fraction").agg(a1=("acc_single", "mean"), a10=("acc_maj10", "mean"))
        ax.plot(g.index*100, (g.a10-g.a1)*100, "o-", color=col, lw=2, ms=8, label=label)
    ax.axhline(0, color=INK2, lw=1.2, ls=":")
    ax.annotate("voting no longer helps", xy=(11, -.75), fontsize=8.5, color=MUTED)
    _ax(ax, "Value of majority voting decays to zero",
        "checkpoint (% through the reasoning)",
        "accuracy gain, majority@10 vs single (pp)")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout(); fig.savefig(OUT/"fig3_compute_value.png", dpi=160); plt.close(fig)


# ── 4. K* is set by the answer space, not the model ───────────────────────────
def fig_ksuff():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, label, col in DS:
        p = Path(f"{RUN.format(name)}/k_sufficiency/curves.csv")
        if not p.exists():
            continue
        c = pd.read_csv(p)
        o = c[c.fraction == "ALL"].sort_values("k")
        ref = o.mm_mean.iloc[-1]
        # Drop the final point: it IS the reference, so its error is 0 by
        # construction and would draw a spurious cliff at the right edge.
        o = o.iloc[:-1]
        err = (o.mm_mean - ref).abs()
        ax.plot(o.k, err, "o-", color=col, lw=2, ms=7, label=label)
        below = o.k[err <= 0.02]
        if len(below):
            kstar = int(below.iloc[0])
            ax.plot([kstar], [err[err <= 0.02].iloc[0]], "o", ms=15, mfc="none",
                    mec=col, mew=2.2)
            ax.annotate(f"K*={kstar}", xy=(kstar, 0.021), fontsize=9, color=col,
                        ha="center", va="bottom", weight="bold")
    ax.axhline(0.02, color=INK2, lw=1.2, ls=":")
    ax.annotate("0.02 bit tolerance", xy=(2.2, .023), fontsize=8.5, color=MUTED)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([2, 5, 10, 25, 50, 100]); ax.set_xticklabels([2, 5, 10, 25, 50, 100])
    _ax(ax, "How many continuations each dataset needs",
        "K (continuations per prefix)",
        "entropy error vs the K\u2192\u221e estimate (bits)")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout(); fig.savefig(OUT/"fig4_k_sufficiency.png", dpi=160); plt.close(fig)


# ── 5. the confound Goal 2 has to control for ─────────────────────────────────
def fig_leakage():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    series = [("math500", "MATH-500: answer text in prefix", S1, "gold_answer_text_present"),
              ("gpqa_diamond", "GPQA: choice text in prefix", S2, "gold_choice_text_present"),
              ("gpqa_diamond", "GPQA: answer letter declared", S3, "gold_letter_declared")]
    for name, label, col, column in series:
        lk = pd.read_csv(f"{RUN.format(name)}/robustness/surface_leakage.csv")
        ax.plot(lk.fraction*100, lk[column]*100, "o-", color=col, lw=2, ms=8, label=label)
    _ax(ax, "The answer is often already in the prefix text",
        "checkpoint (% through the reasoning)", "% of prefixes containing the answer")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT/"fig5_leakage.png", dpi=160); plt.close(fig)


# ── 6. remaining length is a distribution, not a countdown ────────────────────
def fig_remaining_len():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (name, label, col) in zip(axes, DS[:2]):
        ck = labels(name)
        fr = [.1, .3, .5, .7, .9]
        data = [ck[ck.fraction == f]["rem_len_mean"].dropna().values for f in fr]
        bp = ax.boxplot(data, positions=[f*100 for f in fr], widths=11,
                        patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set(facecolor=col, alpha=.22, edgecolor=col, linewidth=1.5)
        for part in ("whiskers", "caps"):
            for it in bp[part]:
                it.set(color=col, linewidth=1.2)
        for med in bp["medians"]:
            med.set(color=INK, linewidth=2)
        _ax(ax, label, "checkpoint (% through the reasoning)",
            "remaining reasoning tokens")
        ax.set_xlim(0, 100); ax.set_xticks([int(f*100) for f in fr])
        ax.set_xticklabels([str(int(f*100)) for f in fr])
    fig.suptitle("Remaining length stays widely spread even near the end",
                 fontsize=13, color=INK, x=.02, ha="left", y=.99)
    fig.tight_layout(); fig.savefig(OUT/"fig6_remaining.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    fig_divergence(); fig_decomposition(); fig_compute_value()
    fig_ksuff(); fig_leakage(); fig_remaining_len()
    print("figures written to", OUT)
