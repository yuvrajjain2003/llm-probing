#!/usr/bin/env python3
"""Round 2 items R5 (bootstrap CIs), R6 (train vs test), R7 (MSE next to R2).
Reads existing sweep results.csv / preds*.csv only; no refitting.

    sbatch ... --wrap '. ./env.sh; $PY src/viz/round2_metrics.py'
Outputs: runs/goal2/gpqa25/sweep/_round2/{r5_ci,r6_overfit,r7_mse}/*.csv, docs/figures/round2/*.png
"""
import glob, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import update_figures as U                       # palette + rcParams (importing does not draw)
import matplotlib.pyplot as plt

S = Path("runs/goal2/gpqa25/sweep"); R2D = S / "_round2"
FIG = Path("docs/figures/round2"); FIG.mkdir(parents=True, exist_ok=True)
REG = ["answer_entropy_bits_mm", "p_mode", "rem_len_mean", "rem_len_std"]; BIN = "committed_by_here"
TARGETS = REG + [BIN]; KEYS = ["example_id", "cot_id", "fraction"]
B, SEED = 2000, 0

# label -> list of (sweep, feature regex, probe regex); per target the best candidate (mean test
# over the 5 checkpoints) is used, so "best h" / "best input" rows are chosen the way fig. 3 does.
SPEC = {
 "Our model L21 (ridge/logistic)": [("ref_fixed", "act_L21", ".*")],
 "Qwen2.5-7B-Instruct L21": [("feat_qwen25_7b_instruct_L21", ".*", ".*")],
 "Qwen2.5-Math-7B L21": [("feat_qwen25_math_7b_L21", ".*", ".*")],
 "TF-IDF": [("ref_fixed", "tfidf", ".*")],
 "Prefix length": [("ref_fixed", "prefix_len", ".*")],
 "Qwen3-Embedding-8B": [("emb_qwen3_8b", ".*", "ridge_act"), ("feat_qwen3_emb_8b", ".*", ".*")],
 "F2LLM-v2-8B": [("feat_f2llm_v2_8b", ".*", ".*")],
 "MLP strong wd (best h)": [("4a_mlp_wd_selected", ".*", ".*")],
 "MLP weak wd (best h)": [("4a_mlp_selected", ".*", ".*")],
 "Gradient boosting (raw)": [("4c_hgb_selected", "act_L21", ".*")],
 "Random forest (best input)": [("4c_rf_selected", ".*", ".*")],
 "PCA-256 -> ridge": [("4b_pca", "act_L21_pca256", ".*")],
 "PCA-15 -> ridge": [("4b_pca", "act_L21_pca15", ".*")],
}
OURS = "Our model L21 (ridge/logistic)"
BASELINES = ["Qwen2.5-7B-Instruct L21", "Qwen2.5-Math-7B L21", "TF-IDF", "Prefix length",
             "Qwen3-Embedding-8B", "F2LLM-v2-8B"]
VS = BASELINES + ["MLP strong wd (best h)", "Gradient boosting (raw)"]


def load_results():
    d = pd.concat([pd.read_csv(f).assign(sweep=f.parent.name) for f in map(Path, glob.glob(str(S / "*/results.csv")))])
    d = d.replace({"feature": {"emb_qwen3_8b": "qwen3_emb_8b"}}) if False else d
    return d[d.target.isin(TARGETS)]


def choose(res):
    """-> {(label, target): (sweep, feature, probe)}"""
    out = {}
    for lab, cands in SPEC.items():
        for t in TARGETS:
            m = pd.DataFrame()
            for sw, fr, pr in cands:
                x = res[(res.sweep == sw) & (res.target == t) & res.feature.str.fullmatch(fr) & res.probe.str.fullmatch(pr)]
                m = pd.concat([m, x])
            g = m.groupby(["sweep", "feature", "probe"]).test.agg(["mean", "size"])
            assert len(g) and (g["size"] == 5).all(), (lab, t, g)
            out[(lab, t)] = g["mean"].idxmax()
    return out


def load_preds(chosen):
    need = set(chosen.values()); P = {}
    for sw in {c[0] for c in need}:
        parts = []
        for f in glob.glob(str(S / sw / "preds*.csv")):
            d = pd.read_csv(f); d = d[d.target.isin(TARGETS)]
            parts.append(d[[ (r, c) in {(x[1], x[2]) for x in need if x[0] == sw} for r, c in zip(d.feature, d.probe)]])
        P[sw] = pd.concat(parts).drop_duplicates(["feature", "probe", "target"] + KEYS)
    return P


def main():
    res = load_results(); chosen = choose(res); P = load_preds(chosen)
    pd.DataFrame([dict(label=l, target=t, sweep=c[0], feature=c[1], probe=c[2]) for (l, t), c in chosen.items()]
                 ).to_csv(R2D / "chosen_probes.csv", index=False)
    fold_of = fixed_fold_of()

    # --- aligned vectors per (label, target, fraction): common rows across all labels ---
    data = {}                                          # (target, frac) -> {"key": df, label: y_pred}
    for t in TARGETS:
        for f in sorted(res.fraction.unique()):
            cols = {}
            for lab in SPEC:
                sw, ft, pr = chosen[(lab, t)]; d = P[sw]
                d = d[(d.feature == ft) & (d.probe == pr) & (d.target == t) & np.isclose(d.fraction, f)]
                cols[lab] = d.set_index(["example_id", "cot_id"])[["y_true", "y_pred"]]
            idx = cols[OURS].index
            for c in cols.values(): idx = idx.intersection(c.index)
            n0 = len(cols[OURS])
            assert len(idx) >= 0.99 * n0, (t, f, len(idx), n0)
            y = cols[OURS].loc[idx, "y_true"].to_numpy()
            for c in cols.values(): assert np.allclose(c.loc[idx, "y_true"], y)
            data[(t, f)] = dict(idx=idx, y=y, q=idx.get_level_values(0).to_numpy(),
                                p={l: c.loc[idx, "y_pred"].to_numpy() for l, c in cols.items()}, n_full=n0)
    fracs = sorted({f for _, f in data})

    # ---------------- R7: MSE ----------------
    rows = []
    for (t, f), D in data.items():
        y = D["y"]; q = D["q"]; fo = np.array([fold_of[int(i)] for i in q])
        for lab, p in D["p"].items():
            sw, ft, pr = chosen[(lab, t)]
            stored = res[(res.sweep == sw) & (res.feature == ft) & (res.probe == pr) & (res.target == t) & np.isclose(res.fraction, f)].test.iloc[0]
            r = dict(label=lab, target=t, fraction=f, n=len(y), stored_test=stored)
            if t in REG:
                mse = ((p - y) ** 2).mean(); var = y.var()
                r.update(mse=mse, rmse=np.sqrt(mse), var_y=var, sd_y=np.sqrt(var), r2_1_minus_mse_over_var=1 - mse / var,
                         r2_foldmean=np.mean([1 - ((p[fo == k] - y[fo == k]) ** 2).mean() / y[fo == k].var() for k in range(5)]))
            else:
                pc = np.clip(p, 1e-6, 1 - 1e-6); pi = y.mean()
                r.update(acc=((p >= .5) == y).mean(), brier=((p - y) ** 2).mean(),
                         logloss=-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean(),
                         brier_const=((pi - y) ** 2).mean(),
                         logloss_const=-(y * np.log(pi) + (1 - y) * np.log(1 - pi)).mean(),
                         p_in_01=bool(((p >= 0) & (p <= 1)).all()))
            rows.append(r)
    r7 = pd.DataFrame(rows); (R2D / "r7_mse").mkdir(exist_ok=True)
    r7.to_csv(R2D / "r7_mse" / "mse_by_checkpoint.csv", index=False)
    m = r7.groupby(["label", "target"]).mean(numeric_only=True).drop(columns=["fraction"]).reset_index()
    m.to_csv(R2D / "r7_mse" / "mse_mean_over_checkpoints.csv", index=False)
    fig_mse(r7)

    # ---------------- R6: train vs test ----------------
    rows = []
    for (lab, t), (sw, ft, pr) in chosen.items():
        x = res[(res.sweep == sw) & (res.feature == ft) & (res.probe == pr) & (res.target == t)].sort_values("fraction")
        for _, r in x.iterrows():
            rows.append(dict(label=lab, target=t, fraction=r.fraction, train=r.train, test=r.test, gap=r.train - r.test))
    r6 = pd.DataFrame(rows); (R2D / "r6_overfit").mkdir(exist_ok=True)
    r6.to_csv(R2D / "r6_overfit" / "train_test_by_checkpoint.csv", index=False)
    g6 = r6.groupby(["label", "target"])[["train", "test", "gap"]].mean().reset_index()
    g6.to_csv(R2D / "r6_overfit" / "train_test_mean_over_checkpoints.csv", index=False)
    # every individual probe in the architecture sweeps, too
    allp = res[res.sweep.isin(["4a_mlp_selected", "4a_mlp_wd_selected", "4c_rf_selected", "4c_hgb_selected", "4b_pca", "ref_fixed"])]
    allp.assign(gap=allp.train - allp.test).groupby(["sweep", "feature", "probe", "target"])[["train", "test", "gap"]].mean().reset_index() \
        .to_csv(R2D / "r6_overfit" / "train_test_all_probes.csv", index=False)
    fig_overfit(g6)

    # ---------------- R5: bootstrap over questions ----------------
    (R2D / "r5_ci").mkdir(exist_ok=True)
    qs = np.unique(np.concatenate([D["q"] for D in data.values()])); Q = len(qs)
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(Q, np.ones(Q) / Q, size=B).astype(float)        # B x Q question counts, shared by everything
    qi = {int(v): i for i, v in enumerate(qs)}

    def stats(t, f, lab):
        D = data[(t, f)]; y, p = D["y"], D["p"][lab]; qq = np.array([qi[int(v)] for v in D["q"]])
        a = lambda v: np.bincount(qq, weights=v, minlength=Q)
        return (a(np.ones(len(y))), a(y), a(y * y), a((p - y) ** 2)) if t in REG else (a(np.ones(len(y))), a(((p >= .5) == y).astype(float)))

    def draws(t, f, lab):                      # B bootstrap scores
        s = stats(t, f, lab); N = W @ s[0]
        if t in REG:
            return 1 - (W @ s[3]) / ((W @ s[2]) - (W @ s[1]) ** 2 / N)
        return (W @ s[1]) / N

    def point(t, f, lab):
        s = stats(t, f, lab)
        return (1 - s[3].sum() / (s[2].sum() - s[1].sum() ** 2 / s[0].sum())) if t in REG else s[1].sum() / s[0].sum()

    def foldscores(t, f, lab):
        D = data[(t, f)]; y, p = D["y"], D["p"][lab]; fo = np.array([fold_of[int(i)] for i in D["q"]]); o = []
        for k in range(5):
            a, b = y[fo == k], p[fo == k]
            o.append(1 - ((b - a) ** 2).sum() / ((a - a.mean()) ** 2).sum() if t in REG else ((b >= .5) == a).mean())
        return np.array(o)

    srows, prows = [], []
    ci = lambda v: np.percentile(v, [2.5, 97.5])
    for t in TARGETS:
        cache = {}
        for lab in SPEC:
            dr = {f: draws(t, f, lab) for f in fracs}; pt = {f: point(t, f, lab) for f in fracs}
            fs = {f: foldscores(t, f, lab) for f in fracs}
            cache[lab] = (dr, pt)
            for f in fracs + ["mean"]:
                if f == "mean":
                    d_ = np.mean([dr[x] for x in fracs], 0); p_ = np.mean([pt[x] for x in fracs]); fsd = np.mean([fs[x] for x in fracs], 0)
                else:
                    d_, p_, fsd = dr[f], pt[f], fs[f]
                lo, hi = ci(d_)
                srows.append(dict(label=lab, target=t, fraction=f, score=p_, ci_lo=lo, ci_hi=hi, boot_sd=d_.std(),
                                  fold_std=fsd.std(ddof=1), fold_min=fsd.min(), fold_max=fsd.max()))
        for lab in VS:
            for f in fracs + ["mean"]:
                gd = lambda l: np.mean([cache[l][0][x] for x in fracs], 0) if f == "mean" else cache[l][0][f]
                gp = lambda l: np.mean([cache[l][1][x] for x in fracs]) if f == "mean" else cache[l][1][f]
                d_ = gd(OURS) - gd(lab); lo, hi = ci(d_)
                prows.append(dict(vs=lab, target=t, fraction=f, diff_ours_minus_vs=gp(OURS) - gp(lab), ci_lo=lo, ci_hi=hi,
                                  significant=bool(lo > 0 or hi < 0), p_boot_two_sided=min(1, 2 * min((d_ <= 0).mean(), (d_ >= 0).mean()))))
    sc = pd.DataFrame(srows); pd_ = pd.DataFrame(prows)
    sc.to_csv(R2D / "r5_ci" / "ci_scores.csv", index=False); pd_.to_csv(R2D / "r5_ci" / "ci_paired_vs_ours.csv", index=False)
    fig_ci(sc, pd_)
    print("done")


def fixed_fold_of(k=5, seed=0):                  # copy of sweep.fixed_fold_of (cfg: folds 5, fold_seed 0)
    ids = np.sort(pd.read_csv("runs/goal1/gpqa25/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B/raw/examples.csv").example_id.unique())
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {int(q): int(r % k) for q, r in zip(ids[perm], range(len(ids)))}


TT = dict(zip(TARGETS, ["Answer entropy", "P(mode answer)", "Remaining length", "Length spread", "Committed yet?"]))
COL = {OURS: U.BLUE, "Qwen2.5-7B-Instruct L21": U.ORANGE, "Qwen2.5-Math-7B L21": U.ORANGE}


def fig_ci(sc, pr):
    labs = [OURS] + BASELINES
    fig, axs = plt.subplots(2, 5, figsize=(17, 6.2), sharey="row")
    for j, t in enumerate(TARGETS):
        ax = axs[0, j]; y = np.arange(len(labs))[::-1]
        for yi, l in zip(y, labs):
            r = sc[(sc.label == l) & (sc.target == t) & (sc.fraction == "mean")].iloc[0]
            c = COL.get(l, U.GRAY)
            ax.hlines(yi, r.ci_lo, r.ci_hi, color=c, lw=2); ax.scatter(r.score, yi, c=c, s=36, zorder=3, edgecolors=U.SURF)
        ax.set_yticks(y, labs); ax.set_title(TT[t] + (" (acc)" if t == BIN else " (R²)"), loc="left", fontsize=9)
        ax.grid(axis="x", color=U.GRID, lw=.6); ax.set_axisbelow(True)
        ax = axs[1, j]; vs = VS; y = np.arange(len(vs))[::-1]
        for yi, l in zip(y, vs):
            r = pr[(pr.vs == l) & (pr.target == t) & (pr.fraction == "mean")].iloc[0]
            c = U.BLUE if r.significant else U.GRAY
            ax.hlines(yi, r.ci_lo, r.ci_hi, color=c, lw=2); ax.scatter(r.diff_ours_minus_vs, yi, c=c, s=36, zorder=3, edgecolors=U.SURF)
        ax.axvline(0, color=U.INK2, lw=1, ls=":"); ax.set_yticks(y, vs)
        ax.set_title("our model minus ...", loc="left", fontsize=8); ax.grid(axis="x", color=U.GRID, lw=.6); ax.set_axisbelow(True)
    fig.suptitle("Top: score with 95% CI (bootstrap over 198 questions, 2000 draws), mean over checkpoints.  "
                 "Bottom: paired difference, ours (ridge L21) minus other; blue = CI excludes 0", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "r5_ci_forest.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_mse(r7):
    labs = [OURS] + BASELINES; m = r7.groupby(["label", "target"]).mean(numeric_only=True)
    fig, axs = plt.subplots(2, 4, figsize=(15, 5.6), sharey=True)
    for j, t in enumerate(REG):
        for i, (col, nm) in enumerate([("stored_test", "R²"), ("mse", "MSE")]):
            ax = axs[i, j]; y = np.arange(len(labs))[::-1]
            v = [m.loc[(l, t), col] for l in labs]
            ax.hlines(y, min(v), v, color=U.GRID, lw=1)
            ax.scatter(v, y, c=[COL.get(l, U.GRAY) for l in labs], s=40, zorder=3, edgecolors=U.SURF)
            for yi, x in zip(y, v): ax.text(x, yi + .28, f"{x:.3f}" if nm == "R²" else f"{x:.3g}", fontsize=7, color=U.INK2, ha="center")
            ax.set_yticks(y, labs); ax.set_title(f"{TT[t]} — {nm}" + (" (lower is better)" if nm == "MSE" else ""), loc="left", fontsize=9)
            ax.grid(axis="x", color=U.GRID, lw=.6); ax.set_axisbelow(True)
    fig.suptitle("R² and MSE side by side, mean over checkpoints (MSE in target units²)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "r7_r2_vs_mse.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_overfit(g6):
    labs = list(SPEC)
    fig, axs = plt.subplots(1, 5, figsize=(17, 4.2), sharey=True)
    for ax, t in zip(axs, TARGETS):
        y = np.arange(len(labs))[::-1]; x = g6[g6.target == t].set_index("label").loc[labs]
        ax.hlines(y, x.test, x.train, color=U.GRID, lw=3)
        ax.scatter(x.train, y, c=U.ORANGE, s=38, zorder=3, label="train (in-sample)", edgecolors=U.SURF)
        ax.scatter(x.test, y, c=U.BLUE, s=38, zorder=3, label="test (out-of-fold)", edgecolors=U.SURF)
        ax.set_yticks(y, labs); ax.set_title(TT[t] + (" (acc)" if t == BIN else " (R²)"), loc="left", fontsize=9)
        ax.grid(axis="x", color=U.GRID, lw=.6); ax.set_axisbelow(True)
    fig.suptitle("Train vs test score per probe, mean over checkpoints (long bar = overfitting)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.savefig(FIG / "r6_train_vs_test.png", dpi=160, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
