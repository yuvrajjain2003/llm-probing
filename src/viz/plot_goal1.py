#!/usr/bin/env python3
"""Static Goal-1 plots. Entropy-vs-checkpoint (the headline) + a small-multiple
panel of the other key labels. Colorblind-safe (Okabe-Ito) categorical hues."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse

_ap = argparse.ArgumentParser(description=__doc__)
# Positionals kept so the original `plot_goal1.py [viz] [out]` invocation still works.
_ap.add_argument("viz", nargs="?", default="runs/goal1/goal1_viz.json")
_ap.add_argument("out", nargs="?", default="runs/goal1/figures")
_ap.add_argument("--datasets", default=None,
                 help="comma-separated subset to plot (default: all present)")
_ap.add_argument("--combined", metavar="STEM", default=None,
                 help="write ONE 2x2 figure to STEM.png instead of the usual two figures")
_ap.add_argument("--title", default=None, help="suptitle for --combined")
_args = _ap.parse_args()

VIZ, OUT = Path(_args.viz), Path(_args.out)
OUT.mkdir(parents=True, exist_ok=True)
data = json.loads(VIZ.read_text())
if _args.datasets:
    want = [d.strip() for d in _args.datasets.split(",") if d.strip()]
    missing = [d for d in want if d not in data]
    if missing:
        sys.exit(f"not in {VIZ}: {missing}\navailable: {sorted(data)}")
    data = {d: data[d] for d in want}

# Okabe-Ito, fixed order — CVD-safe, assigned by dataset identity (never cycled).
COLOR = {"aqua_rat": "#0072B2", "openbookqa": "#D55E00", "mmlu_college_math": "#009E73",
         "math500": "#CC79A7", "aime2025": "#E69F00", "gpqa_diamond": "#56B4E9"}
LABEL = {"aqua_rat": "AquaRAT", "openbookqa": "OpenBookQA", "mmlu_college_math": "MMLU college math",
         "math500": "MATH-500", "aime2025": "AIME 2025", "gpqa_diamond": "GPQA-Diamond"}
# Free-form datasets have no answer choices, so the letter/choice-text leakage
# signals are structurally zero for them; panel (c) plots answer-text instead.
FREEFORM = {"math500", "aime2025"}
# Fixed order for anything known; plot only what the viz file actually contains, so
# a dataset still mid-run simply doesn't appear rather than raising a KeyError.
_ORDER = ["aqua_rat", "openbookqa", "mmlu_college_math", "math500", "aime2025", "gpqa_diamond"]
DSETS = [d for d in _ORDER if d in data] + [d for d in data if d not in _ORDER]
X = [10, 30, 50, 70, 90]
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def xs(d): return [r["fraction"] * 100 for r in data[d]["by_fraction"]]


def col(d, key):
    """A by_fraction column as floats, JSON null -> NaN so matplotlib skips the point.

    Nulls are real here: acc_maj10 is undefined for a checkpoint with fewer than 10
    valid answers, which happens on the harder datasets.
    """
    return [float("nan") if x[key] is None else float(x[key])
            for x in data[d]["by_fraction"]]


def draw_entropy(ax, *, legend=True):
    for d in DSETS:
        y, se = col(d, "entropy_mean"), col(d, "entropy_se")
        ax.fill_between(xs(d), [a - b for a, b in zip(y, se)], [a + b for a, b in zip(y, se)],
                        color=COLOR[d], alpha=0.15, linewidth=0)
        ax.plot(xs(d), y, "-o", color=COLOR[d], lw=2, ms=7, label=LABEL[d])
    # End-of-line labels were readable with 3 datasets but collide with 6; the legend
    # carries the identity instead.
    ax.set_xlabel("CoT prefix checkpoint (% of reasoning by sentence)")
    ax.set_ylabel("Answer-distribution entropy (bits)")
    ax.set_xticks(X); ax.set_ylim(bottom=0); ax.set_xlim(5, 95)
    if legend:
        ax.legend(frameon=False, loc="upper right", fontsize=9)


def draw_rem_len(ax, *, legend=True):
    for d in DSETS:
        ax.plot(xs(d), col(d, "rem_len"), "-o", color=COLOR[d], lw=2, ms=6, label=LABEL[d])
    ax.set_title("Remaining reasoning length"); ax.set_ylabel("tokens to completion (mean)")
    ax.set_xlabel("checkpoint (%)"); ax.set_xticks(X); ax.set_ylim(bottom=0)
    if legend:
        ax.legend(frameon=False, fontsize=8)


def draw_accuracy(ax):
    """single vs majority@10, with the majority@10 coverage confound made visible.

    acc_maj10 is only defined where >=10 continuations parsed, so on the harder
    datasets it averages an easier subset of checkpoints than acc_single does. The
    dotted line is acc_single re-computed on exactly that subset -- the like-for-like
    comparison. Dashed above dotted is a real gain from extra compute; dashed above
    solid alone may just be the subset being easier.
    """
    for d in DSETS:
        ax.plot(xs(d), col(d, "acc_single"), "-o", color=COLOR[d], lw=2, ms=6)
        ax.plot(xs(d), col(d, "acc_maj10"), "--^", color=COLOR[d], lw=1.6, ms=6, alpha=0.8)
        ax.plot(xs(d), col(d, "acc_single_maj10sub"), ":", color=COLOR[d], lw=1.4, alpha=0.75)
    ax.set_title("Accuracy: single (—•) vs majority@10 (--▲)\n"
                 "dotted = single on the same >=10-parsed subset", fontsize=10)
    ax.set_ylabel("accuracy"); ax.set_xlabel("checkpoint (%)"); ax.set_xticks(X)
    # Full 0-1 range: the original 0.6 floor predates AIME/GPQA and would clip them
    # off the axis entirely (AIME sits near 0.30).
    ax.set_ylim(0, 1.02)


def draw_coverage(ax):
    """Share of checkpoints where all 10 continuations parsed -- the confound itself."""
    for d in DSETS:
        ax.plot(xs(d), [v * 100 for v in col(d, "maj10_coverage")],
                "-o", color=COLOR[d], lw=2, ms=6, label=LABEL[d])
    ax.set_title("majority@10 coverage\n(checkpoints with 10/10 parsed)", fontsize=10)
    ax.set_ylabel("% of checkpoints"); ax.set_xlabel("checkpoint (%)")
    ax.set_xticks(X); ax.set_ylim(0, 102)


def draw_leakage(ax, *, legend=True):
    for d in DSETS:
        lk = sorted(data[d]["leakage"], key=lambda z: z["fraction"])
        lx = [z["fraction"] * 100 for z in lk]
        key = "mode_answer_text_present" if d in FREEFORM else "mode_choice_text_present"
        ax.plot(lx, [(z[key] or 0) * 100 for z in lk], "-o", color=COLOR[d], lw=2, ms=6,
                label=LABEL[d])
        if d not in FREEFORM:
            ax.plot(lx, [(z["mode_letter_declared"] or 0) * 100 for z in lk],
                    "--s", color=COLOR[d], lw=1.6, ms=5, alpha=0.7)
    ax.set_title("Surface leakage: answer text (—•) vs letter declared (--■)", fontsize=10)
    ax.set_ylabel("% of prefixes"); ax.set_xlabel("checkpoint (%)")
    ax.set_xticks(X); ax.set_ylim(0, 100)
    if legend:
        ax.legend(frameon=False, fontsize=8)


if _args.combined:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    draw_entropy(axes[0][0]); axes[0][0].set_title("Answer-distribution entropy", fontsize=11)
    draw_rem_len(axes[0][1], legend=False)
    draw_accuracy(axes[1][0])
    draw_leakage(axes[1][1], legend=False)
    fig.suptitle(_args.title or "Goal 1 — key labels by checkpoint", fontsize=14, y=0.99)
    fig.tight_layout()
    path = OUT / f"{_args.combined}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)
    raise SystemExit

# ── Figure 1: entropy vs checkpoint (headline) ──────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.2))
draw_entropy(ax)
ax.set_title("Answer entropy collapses as the CoT prefix grows\n"
             "mean over CoTs ± SE; entropy of the 10 continuations' final answers", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "entropy_by_checkpoint.png", dpi=150)
print("wrote", OUT / "entropy_by_checkpoint.png")

# ── Figure 2: other key labels (small multiples) ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
draw_rem_len(axes[0])
draw_accuracy(axes[1])
draw_leakage(axes[2], legend=False)
fig.suptitle("Goal 1 — other key labels by checkpoint (Qwen-7B)", fontsize=13, y=1.02)
fig.tight_layout(); fig.savefig(OUT / "other_labels_by_checkpoint.png", dpi=150, bbox_inches="tight")
print("wrote", OUT / "other_labels_by_checkpoint.png")
