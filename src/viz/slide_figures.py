#!/usr/bin/env python3
"""Standalone PNGs for slides - the dashboard's findings as static figures.

Reads the same curated CSVs the dashboard uses (copied to sweep/_slides/data/, each one
derived from sweep/_master, _icl, _mlpgap, _redundancy and ref_fixed). No refitting.

    ./envs/goal1/bin/python src/viz/slide_figures.py
Outputs: docs/figures/slides/fig{1..7}_*.png

Colors are slots 1-3 of the dataviz skill's default categorical palette, used unchanged;
better/worse use its status greens/reds with filled-vs-hollow markers so significance is
never carried by color alone.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = Path("runs/goal2/gpqa25/sweep/_slides/data")
M = Path("runs/goal2/gpqa25/sweep/_master")
I = Path("runs/goal2/gpqa25/sweep/_icl")
OUT = Path("docs/figures/slides"); OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, BAD, INK, MUTED, GRID = "#1baf7a", "#e34948", "#0b0b0b", "#52514e", "#e3e2dd"
LONG = {"answer_entropy_bits_mm": "entropy", "p_mode": "p_mode", "rem_len_mean": "rem_len_mean",
        "rem_len_std": "rem_len_std", "committed_by_here": "committed"}
NICE = {"entropy": "Answer entropy", "p_mode": "P(mode)", "rem_len_mean": "Remaining length",
        "rem_len_std": "Length spread", "committed": "Committed (acc.)"}

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11.5, "axes.labelcolor": MUTED,
    "axes.edgecolor": MUTED, "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "legend.frameon": False, "legend.fontsize": 10.5,
})


def forest(ax, rows, xlabel, note_sig=True, labels=True):
    """rows: list of (name, diff, lo, hi, significant), drawn top to bottom."""
    y = range(len(rows))
    for i, (_, d, lo, hi, sig) in zip(y, rows):
        c = GOOD if d > 0 else BAD
        ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round",
                alpha=1.0 if sig else 0.45)
        ax.plot([d], [i], marker="o", ms=7, color=c if sig else "white",
                markeredgecolor=c, markeredgewidth=1.8, zorder=3)
    ax.axvline(0, color=MUTED, lw=1.1, zorder=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows] if labels else [""] * len(rows),
                       fontsize=10.5, color=INK)
    ax.invert_yaxis(); ax.set_xlabel(xlabel); ax.grid(axis="y", visible=False)
    if note_sig:
        seen = {(d > 0, sig) for _, d, _, _, sig in rows}
        if (True, True) in seen:
            ax.plot([], [], "o", ms=7, color=GOOD, label="better, significant")
        if any(not sig for _, _, _, _, sig in rows):
            ax.plot([], [], "o", ms=7, color="white", markeredgecolor=MUTED,
                    markeredgewidth=1.8, label="interval crosses zero")
        if (False, True) in seen:
            ax.plot([], [], "o", ms=7, color=BAD, label="worse, significant")


# ---- fig 1: the core claim - activations vs the prefix text ------------------
pr = pd.read_csv(M / "ci_paired.csv")
x = pr[(pr.label == "Our model L21 (ridge)") & (pr.minus == "TF-IDF on prefix")]
rows = [(NICE[LONG[t]], r["diff"], r.ci_lo, r.ci_hi, bool(r.significant))
        for t in LONG for r in [x[x.target == t].iloc[0]]]
fig, ax = plt.subplots(figsize=(8.6, 4.3))
forest(ax, rows, "Our probe minus TF-IDF on the same prefix text  (R², accuracy for the last)")
ax.set_title("Activations beat the prefix text on every target", loc="left", color=INK, pad=12)
ax.legend(loc="lower right", ncol=1)
fig.text(0, -0.04, "Mean over the 10–90% cut points. 95% intervals from 2,000 question bootstrap draws, "
         "paired.", fontsize=9, color=MUTED)
fig.savefig(OUT / "fig1_vs_text.png"); plt.close(fig)

# ---- fig 2: everything we tried, against our probe --------------------------
ap = pd.read_csv(D / "approaches.csv")
ap = ap.assign(label=ap.label.str.replace(" -> ", " \u2192 ", regex=False))
order = (ap[(ap.target == "entropy") & (ap.label != "Our model L21 (ridge)")]
         .sort_values("diff").label.tolist())
fig, axs = plt.subplots(1, 2, figsize=(15.4, 7.6))
for ax, t, first in zip(axs, ["entropy", "rem_len_mean"], [True, False]):
    d = ap[ap.target == t].set_index("label")
    rows = [(l, d.loc[l, "diff"], d.loc[l, "diff_lo"], d.loc[l, "diff_hi"],
             bool(d.loc[l, "significant"])) for l in order]
    forest(ax, rows, "Difference from our probe (R²)", note_sig=first, labels=first)
    ax.set_title(NICE[t], loc="left", color=INK)
axs[0].legend(loc="lower left", ncol=1, fontsize=9.5)
fig.suptitle("21 alternatives, scored on identical rows and folds",
             x=0.005, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.text(0, -0.01, "Same row order in both panels, sorted by the entropy difference. Mean over the "
         "10–90% cut points; 4,911–4,915 rows per cell.", fontsize=9, color=MUTED)
fig.savefig(OUT / "fig2_all_approaches.png"); plt.close(fig)

# ---- fig 3: by depth, where the text baselines catch up ----------------------
bc = pd.read_csv(D / "by_checkpoint.csv")
keep = [("Our model L21 (ridge)", "our probe", BLUE, "o", "-"),
        ("TF-IDF on prefix", "TF-IDF on the text", ORANGE, "s", "--"),
        ("Prefix length", "prefix length alone", AQUA, "^", ":")]
fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.4))
for ax, t in zip(axs, ["entropy", "rem_len_mean"]):
    for lab, nice, c, mk, ls in keep:
        g = bc[(bc.label == lab) & (bc.target == t)].sort_values("fraction")
        ax.plot(g.fraction * 100, g.score, color=c, lw=2, ls=ls, marker=mk, ms=7,
                label=nice, clip_on=False)
    ax.set_title(NICE[t], loc="left", color=INK)
    ax.set_xlabel("how far into the chain (%)"); ax.set_xticks([10, 30, 50, 70, 90])
axs[0].set_ylabel("R²")
axs[1].legend(loc="upper right")
fig.suptitle("Prefix length alone matches us on remaining length early on — so score within a cut point, "
             "never pooled", x=0.005, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "fig3_by_depth.png"); plt.close(fig)

# ---- fig 4: black-box LLMs given the task in context ------------------------
icl = pd.read_csv(D / "icl.csv")
fig, axs = plt.subplots(1, 3, figsize=(16.2, 4.6))
ks = [0, 10, 50]
for ax, t in zip(axs[:2], ["entropy", "rem_len_mean"]):
    for name, pre, c, mk in [("GPT-6 Luna (frontier)", "gpt6_luna_k", BLUE, "o"),
                             ("Gemma-4-31B (local)", "gemma4_31b_off_k", ORANGE, "s")]:
        kk = ks if "luna" in pre else [0, 10, 30]
        v = [icl[(icl.config == f"{pre}{k}") & (icl.target == t)].score for k in kk]
        v = [float(x.iloc[0]) for x in v if len(x)]
        ax.plot(kk[:len(v)], v, color=c, lw=2, marker=mk, ms=7, label=name, clip_on=False)
    ridge = float(icl[(icl.kind == "probe") & (icl.target == t)
                      & (icl.config.str.contains("activations"))].score.iloc[0])
    ax.axhline(ridge, color=MUTED, lw=1.3, ls="--")
    ax.annotate("our probe", (0, ridge), xytext=(4, 5), textcoords="offset points",
                fontsize=10, color=MUTED, ha="left")
    ax.set_xlabel("labelled examples in the prompt (k)")
    ax.set_title(NICE[t], loc="left", color=INK)
axs[0].set_ylabel("R²"); axs[0].legend(loc="lower left", fontsize=9.5)

best = {"GPT-6 Luna, k=0": "gpt6_luna_k0", "Gemma, k=0": "gemma4_31b_off_k0"}
rows = []
for nice, cfg in best.items():
    for t in ["entropy", "p_mode", "committed", "rem_len_mean", "rem_len_std"]:
        r = icl[(icl.config == cfg) & (icl.target == t)].iloc[0]
        rows.append((f"{NICE[t]} — {nice.split(',')[0]}", r.diff_vs_ridge, r.diff_lo, r.diff_hi,
                     not (r.diff_lo <= 0 <= r.diff_hi)))
forest(axs[2], rows, "Best in-context setting minus our probe")
axs[2].legend(loc="upper left", fontsize=9.5)
axs[2].set_title("Best setting vs our probe", loc="left", color=INK)
fig.suptitle("Told the task in words: zero-shot is the best setting, and remaining length "
             "collapses as soon as examples are added",
             x=0.005, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "fig4_in_context.png"); plt.close(fig)

# ---- fig 8: shorter examples let us fit more of them, and change nothing ----
sc = pd.read_csv(I / "ci_scores.csv")
get = lambda lab, t: float(sc[(sc.label == lab) & (sc.target == t)].mean_score.iloc[0])
fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.9))
ks = [10, 30, 93]
for ax, t in zip(axs, ["answer_entropy_bits_mm", "rem_len_mean"]):
    ax.plot(ks, [get(f"Gemma, k={k}", t) for k in ks], color=BLUE, lw=2, marker="o", ms=8,
            label="full examples", clip_on=False)
    ax.plot(ks + [400], [get(f"Gemma, k={k}, examples shortened", t) for k in ks]
            + [get("Gemma, k=400, examples shortened", t)], color=ORANGE, lw=2, ls="--",
            marker="s", ms=8, label="examples shortened to 2 sentences", clip_on=False)
    k0 = get("Gemma-4-31B, k=0 (best local ICL)", t)
    ax.axhline(k0, color=AQUA, lw=2, ls=":")
    ax.annotate("no examples at all (k=0)", (400, k0), xytext=(0, 7), textcoords="offset points",
                ha="right", fontsize=10.5, color=AQUA)
    ax.axvline(93, color=MUTED, lw=1, ls="-", alpha=0.5)
    ax.annotate("full examples stop fitting here", xy=(93, 0.02), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", fontsize=9.5, color=MUTED,
                rotation=90, va="bottom")
    ax.set_xscale("log"); ax.set_xticks(ks + [400]); ax.set_xticklabels(["10", "30", "93", "400"])
    ax.set_xlabel("labelled examples in the prompt (k)")
    ax.set_title(NICE[LONG[t]], loc="left", color=INK)
axs[0].set_ylabel("R²"); axs[0].legend(loc="lower left", fontsize=9.5)
fig.suptitle("Shorter examples fit 4x more of them, and score slightly higher at the same k",
             x=0.005, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.text(0, -0.04, "Gemma-4-31B, mean over the 10-90% cut points. Shortened is higher in 5 of 6 same-k "
         "comparisons, but only remaining length at k=10 is significant on its own, and no setting "
         "with examples reaches the no-example line.", fontsize=9, color=MUTED)
fig.savefig(OUT / "fig8_smaller_prompt.png"); plt.close(fig)

# ---- fig 5: why a linear probe is still the best ---------------------------
gains = pd.read_csv(D / "gains.csv")
g = gains[(gains.scope == "mean 10-90%")].copy()
g["name"] = (g.method.str.replace("degree-2 kernel ridge", "degree-2 kernel")
             + " — " + g.target.map(lambda t: NICE[t].lower()))
g = g[g.target.isin(["entropy", "rem_len_mean"])].sort_values("diff")
rows = [(r["name"], r["diff"], r["ci_lo"], r["ci_hi"], bool(r["significant"]))
        for _, r in g.iterrows()]
fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.4))
forest(axs[0], rows, "Difference from the matched ridge baseline (R²)")
axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=9.5)
axs[0].set_title("Every non-linear gain is ≤0.01", loc="left", color=INK)
cap = pd.read_csv(D / "capacity.csv")
ax = axs[1]
yy = range(len(cap))
ax.barh([i - 0.19 for i in yy], cap.train, height=0.34, color=ORANGE, label="training score")
ax.barh([i + 0.19 for i in yy], cap.test, height=0.34, color=BLUE, label="test score")
ax.set_yticks(list(yy))
ax.set_yticklabels([i.replace("rem_len_mean", "remaining length") for i in cap.item],
                   fontsize=10.5, color=INK)
ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlabel("R²")
ax.set_title("There is no headroom to find", loc="left", color=INK)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
dof = pd.read_csv(D / "dof.csv")
fig.suptitle(f"Ridge already discards most of itself — it uses {dof.directions_used.mean():.0f} of "
             f"{int(dof.input_dims.iloc[0]):,} directions on average", x=0.005, ha="left",
             fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "fig5_why_linear.png"); plt.close(fig)

# ---- fig 6: the data is not redundant, but it is badly allocated ------------
lc = pd.read_csv(D / "learning_curve.csv")
al = pd.read_csv(D / "allocation.csv")
fig, axs = plt.subplots(2, 2, figsize=(11.4, 8.4))
for row, t in zip(axs, ["entropy", "rem_len_mean"]):
    ax = row[0]
    g = lc[(lc.target == t) & (lc.fraction == 0.5)].sort_values("m_per_q")
    ax.plot(g.m_per_q, g.score, color=BLUE, lw=2, marker="o", ms=7, clip_on=False)
    ax.set_ylabel("R²")
    ax.set_title(f"{NICE[t]}: still rising at 25 chains", loc="left", color=INK)
    ax = row[1]
    a = al[al.target == t].sort_values(["budget", "rows"])
    for bud, c, mk in zip(sorted(a.budget.unique()), [BLUE, ORANGE], ["o", "s"]):
        g = a[a.budget == bud].sort_values("m_per_q")
        ax.plot(g.m_per_q, g.score, color=c, lw=2, marker=mk, ms=7, label=bud, clip_on=False)
    for _, r in a.iterrows():
        ax.annotate(f"{int(r.q_train)}q", (r.m_per_q, r.score), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9, color=MUTED)
    ax.set_xscale("log"); ax.set_xticks([2, 5, 10, 25]); ax.set_xticklabels(["2", "5", "10", "25"])
    ax.margins(y=0.18)
    ax.set_title(f"{NICE[t]}: more questions win at a fixed budget", loc="left", color=INK)
axs[0][1].legend(fontsize=9.5)
axs[1][0].set_xlabel("chains per question used for training")
axs[1][1].set_xlabel("chains per question (same total rows along each line)")
fig.suptitle("The 25 chains per question are not redundant, but the rollouts are badly spent",
             x=0.005, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT / "fig6_data.png"); plt.close(fig)

# ---- fig 7: what the probe is actually reading (preliminary) ----------------
# Reads the full CI file rather than the curated copy, which carries no intervals.
RT = {"entropy": "answer_entropy_bits_mm", "rem_len_mean": "rem_len_mean"}
dc = pd.read_csv(Path("runs/goal2/gpqa25/sweep/_redundancy/decomposition_ci.csv"))
SER = [("between_r2", BLUE, "o", "-", "Telling QUESTIONS apart"),
       ("within_r2", ORANGE, "s", "--", "Telling CHAINS of one question apart")]
fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.8))
for ax, t in zip(axs, ["entropy", "rem_len_mean"]):
    g = dc[(dc.feature == "act_L21") & (dc.target == RT[t])]
    for met, col, mk, ls, lab in SER:
        d = g[g.metric == met].sort_values("fraction")
        x = d.fraction.to_numpy() * 100
        ax.plot(x, d.value, color=col, lw=2, ls=ls, zorder=3)
        ax.errorbar(x, d.value, yerr=[d.value - d.lo, d.hi - d.value], fmt="none", ecolor=col,
                    elinewidth=1.8, capsize=4, capthick=1.5, zorder=3)
        ax.plot(x, d.value, mk, ms=8, color=col, markeredgecolor="white", markeredgewidth=1.4,
                zorder=4, label=lab)
    lo = min(g.lo.min(), 0) * 1.12
    ax.axhspan(lo, 0, color=GRID, alpha=.55, zorder=0)     # worse than guessing the group average
    ax.axhline(0, color=MUTED, lw=1.2, zorder=2)
    ax.set_ylim(lo, max(g.hi.max() * 1.1, 0.05))
    ax.set_title(NICE[t], loc="left", color=INK, pad=8)
    ax.set_xlabel("how far into the chain (%)"); ax.set_xticks([10, 30, 50, 70, 90])
    ax.set_xlim(2, 98); ax.set_ylabel("R2")
    ax.annotate("worse than guessing\nthe question's own average", (96, lo * 0.94), fontsize=8.5,
                color=MUTED, ha="right", va="bottom")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.004, 0.945), ncol=2, frameon=False,
           fontsize=10.5, handletextpad=0.4, columnspacing=2.2)
fig.suptitle("PRELIMINARY. The score is mostly about which question, not which chain",
             x=0.005, ha="left", fontsize=13.5, color=INK)
fig.text(0.005, -0.09,
         "Our layer-21 ridge probe, all 25 chains per question. Bars are 95% intervals from 2,000 "
         "question bootstrap draws. Blue: how well the probe ranks QUESTIONS, scoring only each "
         "question's average.\nOrange: how well it ranks the 25 CHAINS inside a question, after "
         "removing that question's average from both the truth and the prediction. Negative R2 means "
         "the prediction does WORSE than simply\nguessing the question's own average, which is what "
         "R2 = 0 corresponds to here: the probe's chain-to-chain wiggle is real in size but barely "
         "correlated with the truth, so it actively hurts.",
         fontsize=8.6, color=MUTED, ha="left", va="top")
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(OUT / "fig7_difficulty_preliminary.png"); plt.close(fig)

print("wrote:")
for f in sorted(OUT.glob("fig*.png")):
    print(f"  {f}  ({f.stat().st_size // 1024} KB)")
