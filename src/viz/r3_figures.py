"""R3 figures: why RF < ridge. Reads runs/goal2/gpqa25/sweep/_round2/r3_rf/ + 4c_rf_ablation."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY, INK, INK2, GRID, SURF = "#a8a7a0", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
R = Path("runs/goal2/gpqa25/sweep/_round2/r3_rf"); O = Path("docs/figures/round2"); O.mkdir(parents=True, exist_ok=True)
d = pd.concat([pd.read_csv(f) for f in sorted(R.glob("r3_f0.[0-9].csv"))])
w = pd.concat([pd.read_csv(f) for f in sorted(R.glob("w_f0.[0-9].csv"))])
T = {"answer_entropy_bits_mm": "entropy", "rem_len_mean": "remaining length (mean)"}

# Fig 1 (50% checkpoint): test R2 of every variant
groups = [("ridge", "ridge (reference)", BLUE), ("rf_raw", "RF, raw 3584 dims", ORANGE),
          ("rf_mf0.3", "RF, raw, 30% dims/split", ORANGE), ("rf_mf1.0", "RF, raw, all dims/split", ORANGE),
          ("rf_ridgepred", "RF on 1 feature: ridge prediction", AQUA),
          ("pls5_ols", "PLS-5 + OLS", GRAY), ("pls5_rf", "RF on PLS-5", AQUA), ("pls16_rf", "RF on PLS-16", AQUA),
          ("top64_ridge", "ridge, top-64 |w| dims", GRAY), ("top64_rf", "RF, top-64 |w| dims", MAGENTA),
          ("top256_ridge", "ridge, top-256 |w| dims", GRAY), ("top256_rf", "RF, top-256 |w| dims", MAGENTA)]
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, (t, lab) in zip(axs, T.items()):
    x = d[(d.fraction == 0.5) & (d.target == t)].set_index("variant")
    y = np.arange(len(groups))[::-1]
    for yi, (v, l, c) in zip(y, groups):
        ax.hlines(yi, 0, x.test[v], color=GRID, lw=1); ax.scatter(x.test[v], yi, s=42, c=c, zorder=3, edgecolors=SURF, linewidths=1.5)
        ax.scatter(x.train[v], yi, s=22, facecolors="none", edgecolors=c, zorder=2)
        ax.text(x.test[v] + 0.012, yi, f"{x.test[v]:.3f}", va="center", fontsize=8, color=INK2)
    ax.axvline(x.test["ridge"], color=INK2, lw=1, ls=":"); ax.set_title(lab, loc="left", fontsize=9)
    ax.set_xlabel("out-of-fold R$^2$ (filled) / train R$^2$ (open)"); ax.grid(axis="x", color=GRID, lw=.6); ax.set_axisbelow(True)
axs[0].set_yticks(np.arange(len(groups))[::-1], [g[1] for g in groups])
fig.suptitle("Where RF loses to ridge (50% checkpoint, layer 21)", x=0.01, ha="left", fontsize=10)
fig.tight_layout(); fig.savefig(O / "r3_fig1_variants.png", dpi=160); plt.close(fig)

# Fig 2: can RF learn a known linear function?  (mean over checkpoints)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
ax = axs[0]
syn = d[d.variant.isin(["ridge_syn_matched", "syn_matched", "ridge_syn_noiseless", "syn_noiseless", "rf_on_ridgeoof"])]
m = syn.groupby(["variant", "target"]).test.mean().unstack()
lab = [("syn_matched", "noisy linear target\n(ceiling = real ridge R$^2$)"), ("syn_noiseless", "noise-free linear target\n(ceiling = 1)")]
xs = np.arange(2); bw = .36
for k, (v, l) in enumerate(lab):
    pass
for j, (t, tl) in enumerate(T.items()):
    for k, (v, c, nm) in enumerate([("ridge_", BLUE, "ridge"), ("", ORANGE, "RF")]):
        vals = [m.loc[v + s, t] for s, _ in lab]
        ax.bar(xs + (j * 2 + k - 1.5) * .2, vals, .2, color=c, alpha=1 if j == 0 else .55,
               label=f"{nm}, {tl}" if True else None)
ax.set_xticks(xs, [l for _, l in lab]); ax.set_ylabel("recovered R$^2$"); ax.legend(fontsize=7, frameon=False)
ax.set_title("Fit to a known linear function of X", loc="left", fontsize=9); ax.grid(axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)
# weight spread
ax = axs[1]
ws = w.groupby(["target", "fraction"])[["participation_ratio", "dims_50pct_energy", "dims_90pct_energy"]].mean()
for t, c in zip(T, (BLUE, ORANGE)):
    s = ws.loc[t]
    ax.plot(s.index * 100, s.dims_90pct_energy, color=c, marker="o", lw=2, label=f"{T[t]}: dims for 90% of w$^2$")
    ax.plot(s.index * 100, s.participation_ratio, color=c, marker="s", lw=1.5, ls="--", label=f"{T[t]}: participation ratio")
ax.axhline(3584, color=GRAY, ls=":"); ax.text(52, 3350, "3584 dims", fontsize=8, color=INK2)
ax.set_xlabel("checkpoint (%)"); ax.set_ylabel("# dimensions"); ax.set_ylim(0, 3800); ax.legend(fontsize=7, frameon=False, loc="center right")
ax.set_title("Ridge weights are spread out", loc="left", fontsize=9); ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(O / "r3_fig2_linear_and_spread.png", dpi=160); plt.close(fig)

# Fig 3: top-k |w| dims, ridge vs RF (mean over checkpoints)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
for ax, (t, tl) in zip(axs, T.items()):
    x = d[d.target == t].groupby("variant").test.mean()
    ks = [16, 64, 256]
    ax.plot(ks + [1024], [x[f"top{k}_ridge"] for k in ks] + [x["top1024_ridge"]], color=BLUE, marker="o", lw=2, label="ridge on top-k dims")
    ax.plot(ks, [x[f"top{k}_rf"] for k in ks], color=MAGENTA, marker="s", lw=2, label="RF on top-k dims")
    ax.axhline(x["ridge"], color=BLUE, ls=":", label="ridge, all 3584"); ax.axhline(x["rf_raw"], color=ORANGE, ls=":", label="RF, all 3584")
    ax.set_xscale("log"); ax.set_xlabel("k = dims kept (by |w| in the training fold)"); ax.set_ylabel("R$^2$ (mean over checkpoints)")
    ax.set_title(tl, loc="left", fontsize=9); ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
axs[0].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(O / "r3_fig3_topk.png", dpi=160); plt.close(fig)
print("ok")
