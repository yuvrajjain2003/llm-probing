#!/usr/bin/env python3
"""Figures for docs/update_2026-10-12.md (round 2), read from the saved results CSVs.

    $PY src/viz/update_1012_figures.py       # -> docs/figures/round2/fig{6,7,8}_*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path("runs/goal2/gpqa25")
OUT = Path("docs/figures/round2"); OUT.mkdir(parents=True, exist_ok=True)

# same palette as src/viz/update_figures.py
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2, GRID, SURF = "#a8a7a0", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
TARGETS = [("answer_entropy_bits_mm", "Answer entropy (R²)"), ("p_mode", "P(mode answer) (R²)"),
           ("rem_len_mean", "Remaining length (R²)"), ("rem_len_std", "Remaining-length spread (R²)"),
           ("committed_by_here", "Committed yet? (accuracy)")]


def fig_icl():
    """R1: calibrated ICL score vs number of examples, thinking off (solid) / on (dashed)."""
    d = pd.read_csv(R / "icl/ci_vs_probes.csv")
    models = [("gemma4_31b", "Gemma-4-31B", BLUE), ("qwen35_9b", "Qwen3.5-9B", ORANGE), ("r1qwen7b", "Our model", AQUA)]
    tg = [TARGETS[0], TARGETS[2], TARGETS[4]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (t, name) in zip(axes, tg):
        sub = d[d.target == t]
        lo = 0.55 if t == "committed_by_here" else -0.2
        for key, lab, c in models:
            for mode, ls in (("off", "-"), ("on", "--")):
                m = sub[sub.feature.str.startswith(f"{key}_{mode}_k") & (sub.probe == "icl_cal")].copy()
                m["k"] = m.feature.str.split("_k").str[-1].astype(int)
                m = m.sort_values("k")
                ax.plot(range(len(m)), m.score.clip(lower=lo), ls, color=c, marker="o", ms=4,
                        label=f"{lab}, thinking {mode}")
        for feat, lab, c in (("act_L21", "ridge, our activations", INK), ("tfidf", "TF-IDF", GRAY)):
            v = sub[sub.feature == feat].score.iloc[0]
            ax.axhline(v, color=c, lw=1, ls=":", label=lab)
        ax.set_xticks(range(4), ["0", "10", "30", "max"]); ax.set_xlim(-0.2, 3.2)
        ax.set_ylim(lo - 0.02, None); ax.set_xlabel("in-context examples (k)")
        ax.set_title(name, loc="left", fontsize=9); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("R1: in-context LLMs vs ridge (10–90% mean, calibrated; values below the axis are clipped)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig6_icl.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_rf_reg():
    """Regularised RF: train vs test as leaves grow (no depth limit), ridge for reference."""
    d = pd.read_csv(R / "sweep/rf_reg/results.csv")
    ref = pd.read_csv(R / "sweep/ref_fixed/results.csv")
    ref = ref[(ref.feature == "act_L21") & ref.probe.isin(["ridge_act", "logistic"])]
    tg = [TARGETS[0], TARGETS[2], TARGETS[4]]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (t, name) in zip(axes, tg):
        m = d[(d.target == t) & d.probe.str.startswith("rf_dNone")].groupby("probe")[["train", "test"]].mean()
        m.index = m.index.str.split("_l").str[-1].astype(int); m = m.sort_index()
        x = range(len(m))
        ax.plot(x, m.train, "o-", color=GRAY, label="RF train"); ax.plot(x, m.test, "o-", color=BLUE, label="RF test")
        r = ref[ref.target == t][["train", "test"]].mean()
        ax.axhline(r.test, color=INK, lw=1, ls=":"); ax.text(len(m) - 0.95, r.test, "ridge test", va="bottom", fontsize=7)
        ax.set_xticks(list(x), [str(v) for v in m.index]); ax.set_xlim(-0.2, len(m) - 0.2)
        ax.set_xlabel("min. rows per leaf"); ax.set_title(name, loc="left", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Regularised random forest: train falls to test, test falls too (mean over checkpoints)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig7_rf_reg.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_lora():
    """R8: ridge on LoRA-adapted states minus ridge on stored states, 95% paired bootstrap CI."""
    d = pd.concat([pd.read_csv(R / f"lora/ci_f0.{f}.csv").assign(c=10 * f) for f in (1, 3, 5, 7, 9)])
    d = d[d.model.str.endswith("_L21_r16_ridge")]
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.0), sharey=True)
    for ax, (t, name) in zip(axes, TARGETS):
        m = d[d.target == t].sort_values("c")
        for _, r in m.iterrows():
            c = BLUE if r.lo > 0 or r.hi < 0 else GRAY
            ax.vlines(r.c, r.lo, r.hi, color=c, lw=2); ax.plot(r.c, r["diff"], "o", color=c, ms=5)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks([10, 30, 50, 70, 90]); ax.set_xlabel("checkpoint (%)")
        ax.set_title(name.replace(" (R²)", "").replace(" (accuracy)", ""), loc="left", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("gain over ridge on stored states")
    fig.suptitle("R8: ridge on LoRA-adapted layer-21 states vs ridge on stored states "
                 "(95% CI; blue = CI excludes 0; R², accuracy for committed)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig8_lora_ridge.png", dpi=160, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_icl(); fig_rf_reg(); fig_lora()
    print("wrote", sorted(p.name for p in OUT.glob("fig[678]_*.png")))
