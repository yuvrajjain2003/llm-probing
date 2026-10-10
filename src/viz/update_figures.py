#!/usr/bin/env python3
"""Figures for docs/update_2026-10-05.md, read straight from the sweep results CSVs.

    $PY src/viz/update_figures.py            # -> docs/figures/*.png

Every number is a test score from fixed per-question 5-fold CV, averaged over the 5 checkpoints
(10/30/50/70/90%) unless the figure says otherwise. Layer 21 activations throughout.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S = Path("runs/goal2/gpqa25/sweep")
OUT = Path("docs/figures"); OUT.mkdir(parents=True, exist_ok=True)

# reference palette (dataviz skill, light mode), slots in documented order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY, INK, INK2, GRID, SURF = "#a8a7a0", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})

TARGETS = [("answer_entropy_bits_mm", "Answer entropy (R²)"), ("p_mode", "P(mode answer) (R²)"),
           ("rem_len_mean", "Remaining length (R²)"), ("rem_len_std", "Remaining-length spread (R²)"),
           ("committed_by_here", "Committed yet? (accuracy)")]


def read(*names):
    d = pd.concat([pd.read_csv(S / n / "results.csv") for n in names], ignore_index=True)
    return d.replace({"feature": {"emb_qwen3_8b": "qwen3_emb_8b"}})   # same embedding, two run names


def mean_by(df, key):                      # mean test over checkpoints, per (key, target)
    return df.groupby([key, "target"]).test.mean().unstack()


def dotplot(ax, labels, vals, colors, title, ref=None):
    y = np.arange(len(labels))[::-1]
    ax.hlines(y, min(vals) - 0.02, vals, color=GRID, lw=1)
    ax.scatter(vals, y, s=42, c=colors, zorder=3, edgecolors=SURF, linewidths=1.5)
    for yi, v in zip(y, vals):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1, ls=":")
    ax.set_yticks(y, labels); ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    lo, hi = min(vals), max(vals); ax.set_xlim(lo - 0.02, hi + (hi - lo) * 0.35 + 0.02)


def fig_baselines():
    d = read("ref_fixed", "emb_qwen3_8b", "feat_f2llm_v2_8b", "feat_qwen25_math_7b_L21",
             "feat_qwen25_7b_instruct_L21")
    d = d[~((d.feature == "qwen3_emb_8b") & (d.probe == "logistic"))]    # use the wider-grid logistic
    d = pd.concat([d, read("feat_qwen3_emb_8b")], ignore_index=True)
    m = mean_by(d, "feature")
    rows = [("act_L21", "Our model (R1-Distill-Qwen-7B)", BLUE),
            ("qwen25_7b_instruct_L21", "Qwen2.5-7B-Instruct hidden state", ORANGE),
            ("qwen25_math_7b_L21", "Qwen2.5-Math-7B hidden state*", ORANGE),
            ("tfidf", "TF-IDF on prefix", GRAY), ("prefix_len", "Prefix length only", GRAY),
            ("qwen3_emb_8b", "Qwen3-Embedding-8B", GRAY), ("f2llm_v2_8b", "F2LLM-v2-8B embedding", GRAY)]
    fig, axs = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
    for ax, (t, title) in zip(axs, TARGETS):
        dotplot(ax, [r[1] for r in rows], [m.loc[r[0], t] for r in rows], [r[2] for r in rows], title)
    fig.suptitle("Baselines vs our model's activations — layer 21, ridge/logistic, mean over checkpoints",
                 x=0.01, ha="left", fontsize=10)
    fig.text(0.01, -0.04, "Blue = probed model; orange = other 7B LLMs reading the same text (*inputs cut to "
             "last 4,096 tokens); gray = text-only baselines.", fontsize=8, color=INK2)
    fig.tight_layout(); fig.savefig(OUT / "fig1_baselines.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_checkpoints():
    d = read("ref_fixed", "emb_qwen3_8b", "feat_qwen25_7b_instruct_L21", "feat_qwen25_math_7b_L21")
    series = [("act_L21", "Our model", BLUE, "o"), ("qwen25_7b_instruct_L21", "Qwen2.5-7B-Instruct", ORANGE, "s"),
              ("tfidf", "TF-IDF", AQUA, "^"), ("prefix_len", "Prefix length", YELLOW, "D"),
              ("qwen3_emb_8b", "Qwen3-Embedding-8B", MAGENTA, "v")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (t, title) in zip(axs, [TARGETS[0], TARGETS[2]]):
        for f, lab, c, mk in series:
            x = d[(d.feature == f) & (d.target == t) & (d.metric == "R2")].sort_values("fraction")
            ax.plot(x.fraction * 100, x.test, color=c, lw=2, marker=mk, ms=6, label=lab,
                    markeredgecolor=SURF, markeredgewidth=1)
        ax.set_title(title, loc="left", fontsize=9); ax.set_xlabel("checkpoint (% of reasoning)")
        ax.set_xticks([10, 30, 50, 70, 90]); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.suptitle("Across checkpoints: our activations vs baselines (ridge)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.savefig(OUT / "fig2_checkpoints.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_probes():
    ref = read("ref_fixed"); ref = ref[ref.feature == "act_L21"]
    mlp_new, mlp_old = read("4a_mlp_wd_selected"), read("4a_mlp_selected")
    rf, hgb, pca = read("4c_rf_selected"), read("4c_hgb_selected"), read("4b_pca")
    best = lambda df, key: mean_by(df, key).max()               # best variant per target (mean over ckpts)
    rows = [("Ridge / logistic (all 3584 dims)", mean_by(ref, "feature").iloc[0], BLUE),
            ("MLP, strong weight decay (best hidden size)", best(mlp_new, "probe"), GRAY),
            ("MLP, weak weight decay (best hidden size)", best(mlp_old, "probe"), GRAY),
            ("Gradient boosting (best input)", best(hgb, "feature"), GRAY),
            ("Random forest (best input)", best(rf, "feature"), GRAY),
            ("PCA-256 → ridge", mean_by(pca, "feature").loc["act_L21_pca256"], GRAY),
            ("PCA-15 → ridge", mean_by(pca, "feature").loc["act_L21_pca15"], GRAY)]
    fig, axs = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
    for ax, (t, title) in zip(axs, TARGETS):
        dotplot(ax, [r[0] for r in rows], [r[1][t] for r in rows], [r[2] for r in rows], title)
    fig.suptitle("Probe architectures on our activations (layer 21) — fair scores, mean over checkpoints",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig3_probes.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_pca():
    pca, ref = read("4b_pca"), read("ref_fixed")
    m = mean_by(pca, "feature"); full = mean_by(ref[ref.feature == "act_L21"], "feature").iloc[0]
    ks = [5, 8, 10, 12, 15, 64, 256]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ends = []
    for (t, title), c, mk in zip(TARGETS[:4], [BLUE, ORANGE, AQUA, YELLOW], ["o", "s", "^", "D"]):
        y = [m.loc[f"act_L21_pca{k}", t] for k in ks] + [full[t]]
        ax.plot(ks + [3584], y, color=c, lw=2, marker=mk, ms=6, markeredgecolor=SURF, markeredgewidth=1)
        ends.append([y[-1], title.replace(" (R²)", "")])
    ends.sort()                                   # nudge end labels apart (min gap 0.03)
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.03)
    for yv, lab in ends:
        ax.text(4200, yv, lab, va="center", fontsize=8, color=INK2)
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xticks([5, 15, 64, 256, 3584], ["5", "15", "64", "256", "all\n3584"])
    ax.set_xlabel("PCA dimensions kept (fit on training split) → ridge"); ax.set_ylabel("test R²")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlim(4, 3584 * 8)
    ax.set_title("Compressing activations loses signal (mean over checkpoints)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig4_pca.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_mlp_ablation():
    a = read("4a_mlp_ablation", "4a_mlp_wd_ablation")
    a[["h", "do", "wd"]] = a.probe.str.extract(r"mlp_h(\d+)_do([\d.]+)_wd([\d.e+-]+)_s\d")
    a = a[a.do.isin(["0.0", "0.3"])]; a["h"] = a.h.astype(int); a["wd"] = a.wd.astype(float)
    ref = read("ref_fixed"); ref = mean_by(ref[ref.feature == "act_L21"], "feature").iloc[0]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, (t, title) in zip(axs, [TARGETS[0], TARGETS[2]]):
        p = a[a.target == t].pivot_table(index="h", columns="wd", values="test")
        im = ax.imshow(p.values, cmap="Blues", aspect="auto")
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                v = p.values[i, j]; dark = im.norm(v) > 0.55          # white text on dark cells
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color=SURF if dark else INK)
        ax.set_xticks(range(p.shape[1]), [f"{w:g}" for w in p.columns]); ax.set_xlabel("weight decay (AdamW)")
        ax.set_yticks(range(p.shape[0]), p.index); ax.set_ylabel("hidden units")
        ax.set_title(f"{title} — ridge = {ref[t]:.3f}", loc="left", fontsize=9)
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("MLP ablation: test score by hidden size × weight decay (mean over checkpoints, 3 seeds, dropout 0/0.3)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig5_mlp_ablation.png", dpi=160, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_baselines, fig_checkpoints, fig_probes, fig_pca, fig_mlp_ablation):
        f(); print("wrote", f.__name__)
