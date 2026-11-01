#!/usr/bin/env python3
"""One table for every approach tried on the layer-21 prediction task, scored identically.

Same method as R5 / round3_ci: stored out-of-fold predictions, rows common to all labels, 2000
shared question-resampling draws, pooled R2 (accuracy for committed), mean over the 5 checkpoints
(10-90%), paired differences against our probe. No refitting.

    srun -p small -c 4 --mem=32G -t 60 bash -c './envs/goal1/bin/python src/viz/master_table.py'
Outputs: runs/goal2/gpqa25/sweep/_master/{ci_scores.csv, ci_paired.csv, master_table.md}

Labels whose predictions do not cover the common row set are reported and dropped, never silently
averaged over a different set of rows.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from round2_metrics import REG, TARGETS, KEYS, B, SEED

S = Path("runs/goal2/gpqa25/sweep"); OUT = S / "_master"; OUT.mkdir(exist_ok=True)

# label -> (sweep, feature, probe or None = the only one for that target, family)
SPEC = {
    "Our model L21 (ridge)":        ("ref_fixed", "act_L21", None, "ours"),
    "TF-IDF on prefix":             ("ref_fixed", "tfidf", None, "text baseline"),
    "Prefix length":                ("ref_fixed", "prefix_len", None, "text baseline"),
    "Qwen2.5-7B-Instruct L21":      ("feat_qwen25_7b_instruct_L21", "qwen25_7b_instruct_L21", None, "other LLM"),
    "Qwen2.5-Math-7B L21":          ("feat_qwen25_math_7b_L21", "qwen25_math_7b_L21", None, "other LLM"),
    "OLMo-3-7B L24":                ("feat_olmo3_7b_L24", "olmo3_7b_L24", None, "other LLM"),
    "Mistral-7B-v0.3 L24":          ("feat_mistral7b_v03_L24", "mistral7b_v03_L24", None, "other LLM"),
    "Qwen3-Embedding-8B":           ("emb_qwen3_8b", "emb_qwen3_8b", None, "embedder"),
    "F2LLM-v2-8B":                  ("feat_f2llm_v2_8b", "f2llm_v2_8b", None, "embedder"),
    "UltraBERT-Embed-150M":         ("feat_ultrabert_embed_150m", "ultrabert_embed_150m", None, "embedder"),
    "BGE-M3":                       ("feat_bge_m3", "bge_m3", None, "embedder"),
    "Granite-ML-R2-311M":           ("feat_granite_ml_r2_311m", "granite_ml_r2_311m", None, "embedder"),
    "MLP h=256 (tuned wd)":         ("4a_mlp_wd_selected", "act_L21", "mlp_h256_selwd", "probe architecture"),
    "MLP h=5 (tuned wd)":           ("4a_mlp_wd_selected", "act_L21", "mlp_h5_selwd", "probe architecture"),
    "Random forest (raw)":          ("4c_rf_selected", "act_L21", "rf_sel_raw", "probe architecture"),
    "Random forest (regularised)":  ("rf_reg", "act_L21", "rf_dNone_l100", "probe architecture"),
    "Gradient boosting (raw)":      ("4c_hgb_selected", "act_L21", "hgb_sel", "probe architecture"),
    "PCA-15 -> ridge":              ("4b_pca", "act_L21_pca15", None, "probe architecture"),
    "PCA-256 -> ridge":             ("4b_pca", "act_L21_pca256", None, "probe architecture"),
    "L20-22 concatenated":          ("multilayer", "act_L20_22", None, "other layers"),
    "L14 (mid-depth)":              ("multilayer", "act_L14", None, "other layers"),
    "L13-15 PCA-64 each":           ("multilayer", "act_L13_15_pp64", None, "other layers"),
}
REFS = ["Our model L21 (ridge)", "TF-IDF on prefix"]


def preds_for(sweep, feat, probe, t):
    d = (pd.read_csv(Path("runs/goal2/gpqa25/icl") / f"preds_{sweep[4:]}.csv") if sweep.startswith("icl:")
         else pd.concat([pd.read_csv(x) for x in sorted((S / sweep).glob("preds*.csv"))]))
    d = d[(d.feature == feat) & (d.target == t)]
    if probe is not None:
        d = d[d.probe == probe]
    else:
        pr = d.probe.unique()
        assert len(pr) == 1, f"{sweep}/{feat}/{t}: pick a probe among {pr}"
    return d.drop_duplicates(KEYS).assign(fraction=lambda x: x.fraction.round(3))


def main(SPEC=SPEC, REFS=REFS, OUT=OUT, base_label=None, title="Master results table"):
    """base_label: the label whose rows define the common set (default: ours). Pass the smallest
    label's name when some approaches were only run on a subsample, as the ICL runs were."""
    rows = []
    for lab, (sw, ft, pr, fam) in SPEC.items():
        for t in TARGETS:
            d = preds_for(sw, ft, pr, t)
            rows.append(d.assign(label=lab)[["label", "target"] + KEYS + ["y_true", "y_pred"]])
    P = pd.concat(rows, ignore_index=True)
    P = P[P.fraction > 0]
    fracs = sorted(P.fraction.unique()); assert len(fracs) == 5, fracs

    # coverage: keep only labels that cover (almost) all of our rows, at every checkpoint
    ours = "Our model L21 (ridge)"
    base_label = base_label or ours
    base = P[(P.label == base_label) & (P.target == TARGETS[0])].set_index(["example_id", "cot_id", "fraction"]).index
    keep, dropped = [], {}
    for lab in SPEC:
        cov = min((P[(P.label == lab) & (P.target == t)].set_index(["example_id", "cot_id", "fraction"]).index
                   .isin(base).sum() / len(base)) for t in TARGETS)
        (keep.append(lab) if cov >= 0.99 else dropped.update({lab: cov}))
    if dropped:
        print("DROPPED (predictions do not cover our rows):")
        for lab, c in dropped.items():
            print(f"  {lab}: {c:.1%} of our rows")
    labels = [l for l in SPEC if l in keep]
    P = P[P.label.isin(labels)]

    qs = np.sort(P.example_id.unique()); Q = len(qs); qi = {int(v): i for i, v in enumerate(qs)}
    W = np.random.default_rng(SEED).multinomial(Q, np.ones(Q) / Q, size=B).astype(float)
    sc, dr = [], {}
    for t in TARGETS:
        for f in fracs:
            x = P[(P.target == t) & (P.fraction == f)]
            wide = x.pivot_table(index=["example_id", "cot_id"], columns="label", values="y_pred").dropna()
            yt = x.drop_duplicates(["example_id", "cot_id"]).set_index(["example_id", "cot_id"]).y_true.loc[wide.index].to_numpy()
            n_base = (x.label == base_label).sum(); assert len(wide) >= 0.99 * n_base, (t, f, len(wide), n_base)
            qq = np.array([qi[int(e)] for e in wide.index.get_level_values(0)])
            a = lambda v: np.bincount(qq, weights=v, minlength=Q)
            for lab in labels:
                p = wide[lab].to_numpy()
                if t in REG:
                    n, s1, s2, se = a(np.ones(len(yt))), a(yt), a(yt * yt), a((p - yt) ** 2)
                    dr[(lab, t, f)] = 1 - (W @ se) / ((W @ s2) - (W @ s1) ** 2 / (W @ n))
                    pt = 1 - se.sum() / (s2.sum() - s1.sum() ** 2 / n.sum())
                else:
                    n, c = a(np.ones(len(yt))), a(((p >= .5) == yt).astype(float))
                    dr[(lab, t, f)] = (W @ c) / (W @ n); pt = c.sum() / n.sum()
                sc.append(dict(label=lab, target=t, fraction=f, score=pt, n=len(yt)))
    sc = pd.DataFrame(sc)
    sc.to_csv(OUT / "per_checkpoint.csv", index=False)
    m = sc.groupby(["label", "target"]).score.mean().reset_index()

    out_sc, out_pr = [], []
    for lab in labels:
        for t in TARGETS:
            d = np.mean([dr[(lab, t, f)] for f in fracs], 0)
            r = m[(m.label == lab) & (m.target == t)].iloc[0]
            out_sc.append(dict(label=lab, family=SPEC[lab][3], target=t, mean_score=r.score,
                               ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5)))
            for ref in REFS:
                if ref == lab: continue
                dd = d - np.mean([dr[(ref, t, f)] for f in fracs], 0)
                rr = m[(m.label == ref) & (m.target == t)].iloc[0].score
                lo, hi = np.percentile(dd, [2.5, 97.5])
                out_pr.append(dict(label=lab, family=SPEC[lab][3], minus=ref, target=t, diff=r.score - rr,
                                   ci_lo=lo, ci_hi=hi, significant=bool(lo > 0 or hi < 0)))
    SC = pd.DataFrame(out_sc); SC.to_csv(OUT / "ci_scores.csv", index=False)
    PR = pd.DataFrame(out_pr); PR.to_csv(OUT / "ci_paired.csv", index=False)

    # markdown: score table, then the paired-vs-ours table
    short = {"answer_entropy_bits_mm": "entropy", "p_mode": "p_mode", "rem_len_mean": "rem_len_mean",
             "rem_len_std": "rem_len_std", "committed_by_here": "committed (acc)"}
    lines = [f"# {title}", "",
             f"Mean over the 10-90% checkpoints. {B} question-bootstrap draws, rows common to all "
             f"{len(labels)} approaches. R2 for the first four targets, accuracy for the last.", ""]
    piv = SC.pivot(index="label", columns="target", values="mean_score").reindex(labels)[TARGETS]
    fam = {l: SPEC[l][3] for l in labels}
    lines += ["| approach | family | " + " | ".join(short[t] for t in TARGETS) + " |",
              "|---|---|" + "---|" * len(TARGETS)]
    for l in labels:
        lines.append(f"| {l} | {fam[l]} | " + " | ".join(f"{piv.loc[l, t]:.3f}" for t in TARGETS) + " |")
    for ref in REFS:
        x = PR[PR.minus == ref]
        lines += ["", f"## Difference from **{ref}** (95% CI; bold = significant)", "",
                  "| approach | " + " | ".join(short[t] for t in TARGETS) + " |", "|---|" + "---|" * len(TARGETS)]
        for l in labels:
            if l == ref: continue
            cells = []
            for t in TARGETS:
                r = x[(x.label == l) & (x.target == t)].iloc[0]
                c = f"{r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
                cells.append(f"**{c}**" if r.significant else c)
            lines.append(f"| {l} | " + " | ".join(cells) + " |")
    (OUT / "table.md" if OUT.name != "_master" else OUT / "master_table.md").write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 250)
    print(piv.round(3).to_string())
    print(f"\nwrote {OUT}  ({len(labels)} approaches, n per cell = "
          f"{sc.n.min()}-{sc.n.max()} rows)")


if __name__ == "__main__":
    main()
