#!/usr/bin/env python3
"""Figures for idea 2 (close the MLP/ridge gap) -> docs/figures/mlpgap/.

    ./envs/goal1/bin/python src/probing/mlpgap_figures.py

fig1_b1_recipe   B1: weight cosine to ridge's closed form, per optimiser, and the score cost.
fig2_b2_ceiling  B2: paired difference vs the LINEAR kernel, per target, with 95% CIs.
fig3_ladder      The ladder: paired difference vs ridge, per variant and target, with 95% CIs.
Each reads only CSVs already written into runs/goal2/gpqa25/sweep/_mlpgap/; missing ones are
skipped with a note rather than faked.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MG = Path("runs/goal2/gpqa25/sweep/_mlpgap")
OUT = Path("docs/figures/mlpgap"); OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY, INK, INK2, GRID, SURF = "#a8a7a0", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
TGT = {"answer_entropy_bits_mm": "Answer\nentropy", "p_mode": "P(mode)",
       "rem_len_mean": "Remaining\nlength", "rem_len_std": "Rem. length\nspread",
       "committed_by_here": "Committed\n(acc)"}
ORDER = list(TGT)


def fig1():
    ps = [q for q in [MG / "b1_diag.csv", MG / "b1_diag_all.csv"] if q.exists()]
    if not ps:
        return print("  skip fig1 (no b1_diag*.csv)")
    d = pd.concat([pd.read_csv(q) for q in ps], ignore_index=True)   # ALL checkpoints
    hows = [h for h in ["lbfgs", "adam", "recipe4a"] if h in set(d.how)]
    col = {"lbfgs": BLUE, "adam": AQUA, "recipe4a": ORANGE}
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
    for i, h in enumerate(hows):
        x = d[d.how == h]
        j = np.random.default_rng(0).normal(0, 0.06, len(x))
        ax[0].scatter(1 - x.cos_w + 1e-7, i + j, s=26, c=col[h], alpha=.85,
                      edgecolors=SURF, linewidths=.6)
    ax[0].set_xscale("log")
    ax[0].set_yticks(range(len(hows)))
    ax[0].set_yticklabels([{"lbfgs": "L-BFGS\n(control)", "adam": "full-batch\nAdam",
                            "recipe4a": "step-4a recipe\n(all prior MLP work)"}[h] for h in hows])
    ax[0].axvline(1e-3, color=GRAY, ls=":", lw=1)
    ax[0].set_xlabel("1 − cosine(trained weights, ridge closed form)   [log]")
    ax[0].set_title("B1: does a linear head find ridge's optimum?\n"
                    f"one point per fold × target × checkpoint; dotted line = cos 0.999",
                    loc="left", fontsize=9)
    rs = [q for q in [MG / "results.b1.csv", MG / "results.b1_all.csv"] if q.exists()]
    if rs:
        b = pd.concat([pd.read_csv(q) for q in rs], ignore_index=True)
        b["t"] = b.variant.str.replace("b1_", "", regex=False)
        # Take the difference PER CELL, then average over checkpoints. Averaging scores first is
        # wrong whenever an optimiser was not run at every checkpoint (adam is 50% only), and the
        # 50% cell alone is not representative: recipe4a beats ridge there on rem_len_mean
        # (+0.004) and loses at every other checkpoint (5-checkpoint mean -0.008).
        w = b.pivot_table(index=["target", "fraction"], columns="t", values="test")
        dif = pd.DataFrame({h: w[h] - w["ridge"] for h in hows if h in w.columns})
        dif = dif.groupby(level="target").mean()
        nck = {h: int(w[h].notna().groupby(level="target").sum().max()) for h in dif.columns}
        tg = [t for t in ORDER if t in dif.index]
        xs = np.arange(len(tg))
        for i, h in enumerate(hows):
            if h in dif.columns:
                ax[1].bar(xs + (i - 1) * .26, dif.loc[tg, h].to_numpy(), .25,
                          color=col[h], label=f"{h} ({nck[h]} ckpt)")
        ax[1].axhline(0, color=INK2, lw=1)
        ax[1].set_xticks(xs); ax[1].set_xticklabels([TGT[t] for t in tg], fontsize=7.5)
        ax[1].set_ylabel("test R² − ridge")
        ax[1].legend(frameon=False, fontsize=7.5)
        ax[1].set_title("…and what that costs in score\n(per-cell difference, averaged over "
                        "checkpoints)", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig1_b1_recipe.png", dpi=170); plt.close(fig)
    print(f"  wrote {OUT/'fig1_b1_recipe.png'}")


def _forest(ax, d, labels, title, xlabel):
    y = np.arange(len(labels))[::-1]
    for yy, lab in zip(y, labels):
        r = d[d.variant == lab]
        if not len(r):
            continue
        r = r.iloc[0]
        c = BLUE if r.significant and r["diff"] > 0 else (ORANGE if r.significant else GRAY)
        ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=c, lw=2, solid_capstyle="round")
        ax.scatter([r["diff"]], [yy], s=34, c=c, zorder=3, edgecolors=SURF, linewidths=1.2)
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xlabel(xlabel)


def fig2():
    ps = sorted(MG.glob("ci_paired_b2*.csv"))
    if not ps:
        return print("  skip fig2 (no ci_paired_b2*.csv)")
    d = pd.read_csv(ps[-1])
    labs = [v for v in d.variant.unique() if v not in ("ref_ridge",)]
    labs = ([l for l in labs if l == "krr_honest_joint"]
            + sorted([l for l in labs if l.startswith("krr_poly")])
            + sorted([l for l in labs if l.startswith("krr_rbf")]))
    tg = [t for t in ORDER if t in set(d.target)]
    fig, ax = plt.subplots(1, len(tg), figsize=(3.1 * len(tg), 3.4), sharey=True)
    ax = np.atleast_1d(ax)
    for a, t in zip(ax, tg):
        _forest(a, d[d.target == t], labs, TGT[t].replace("\n", " "),
                "difference vs LINEAR kernel")
    fig.suptitle("B2 ceiling: closed-form kernel ridge minus the linear kernel (= ridge exactly)."
                 "  Paired question bootstrap, 2,000 draws; blue = significant gain, orange = "
                 "significant loss.", fontsize=9, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(OUT / "fig2_b2_ceiling.png", dpi=170); plt.close(fig)
    print(f"  wrote {OUT/'fig2_b2_ceiling.png'}")


def fig3():
    ps = [MG / "ci_paired_mlp5.csv", MG / "ci_paired_ladder.csv", MG / "ci_paired_q05.csv"]
    ps = [p for p in ps if p.exists()]
    if not ps:
        return print("  skip fig3 (no ci_paired_mlp5/ladder/q05.csv)")
    d = pd.concat([pd.read_csv(p) for p in ps], ignore_index=True).drop_duplicates(
        ["variant", "target"], keep="first")
    labs = [v for v in d.variant.unique() if v != "ref_ridge"]
    tg = [t for t in ORDER if t in set(d.target)]
    fig, ax = plt.subplots(1, len(tg), figsize=(3.1 * len(tg), 0.33 * len(labs) + 2.1), sharey=True)
    ax = np.atleast_1d(ax)
    for a, t in zip(ax, tg):
        _forest(a, d[d.target == t], labs, TGT[t].replace("\n", " "), "difference vs ridge")
    fig.suptitle("The ladder: each MLP variant minus ridge, same rows and folds, paired question "
                 "bootstrap (2,000 draws). Mean over the five checkpoints where available.",
                 fontsize=9, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUT / "fig3_ladder.png", dpi=170); plt.close(fig)
    print(f"  wrote {OUT/'fig3_ladder.png'}")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
