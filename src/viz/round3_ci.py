#!/usr/bin/env python3
"""Round 3: question-bootstrap CIs for the new features (other-family LLMs, long-trained embedders).
Same method as R5 (src/viz/round2_metrics.py): stored out-of-fold predictions, rows common to all
features, 2000 shared question-resampling draws, pooled R2 / accuracy, paired differences.
Mean over the 5 checkpoints (10-90%). No refitting.

    sbatch -p small -c 2 --mem=16G --wrap '. ./env.sh; $PY src/viz/round3_ci.py'
Outputs: runs/goal2/gpqa25/sweep/_round3/{ci_scores.csv, ci_paired.csv}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from round2_metrics import REG, TARGETS, KEYS, B, SEED, load_results

S = Path("runs/goal2/gpqa25/sweep"); OUT = S / "_round3"; OUT.mkdir(exist_ok=True)
SPEC = {  # label -> (sweep, feature); probe = ridge for R2 targets, logistic for committed
    "Our model L21": ("ref_fixed", "act_L21"),
    "Qwen2.5-7B-Instruct L21": ("feat_qwen25_7b_instruct_L21", "qwen25_7b_instruct_L21"),
    "Qwen2.5-Math-7B L21": ("feat_qwen25_math_7b_L21", "qwen25_math_7b_L21"),
    "OLMo-3-7B L24": ("feat_olmo3_7b_L24", "olmo3_7b_L24"),
    "Mistral-7B-v0.3 L24": ("feat_mistral7b_v03_L24", "mistral7b_v03_L24"),
    "TF-IDF": ("ref_fixed", "tfidf"),
    "Prefix length": ("ref_fixed", "prefix_len"),
    "Qwen3-Embedding-8B": ("emb_qwen3_8b", "emb_qwen3_8b"),
    "F2LLM-v2-8B": ("feat_f2llm_v2_8b", "f2llm_v2_8b"),
    "UltraBERT-Embed-150M": ("feat_ultrabert_embed_150m", "ultrabert_embed_150m"),
    "BGE-M3": ("feat_bge_m3", "bge_m3"),
    "Granite-ML-R2-311M": ("feat_granite_ml_r2_311m", "granite_ml_r2_311m"),
}
REFS = ["Our model L21", "TF-IDF"]          # paired differences: every label minus each of these


def probe_for(res, sw, ft, t):
    """ridge for R2 targets; for committed the logistic (Qwen3-emb: the wider grid in feat_qwen3_emb_8b)."""
    if sw == "emb_qwen3_8b" and t not in REG:
        sw, ft = "feat_qwen3_emb_8b", "qwen3_emb_8b"
    x = res[(res.sweep == sw) & (res.feature == ft) & (res.target == t)]
    pr = x.probe.unique(); assert len(pr) == 1, (sw, ft, t, pr)
    return sw, ft, pr[0]


def main():
    res = load_results()
    rows = []
    for lab, (sw, ft) in SPEC.items():
        for t in TARGETS:
            s, f, p = probe_for(res, sw, ft, t)
            d = pd.concat([pd.read_csv(x) for x in sorted((S / s).glob("preds*.csv"))])
            d = d[(d.feature == f) & (d.probe == p) & (d.target == t)].drop_duplicates(KEYS)
            rows.append(d.assign(label=lab, fraction=d.fraction.round(3))[["label", "target"] + KEYS + ["y_true", "y_pred"]])
    P = pd.concat(rows, ignore_index=True)
    P = P[P.fraction > 0]
    fracs = sorted(P.fraction.unique()); assert len(fracs) == 5, fracs

    qs = np.sort(P.example_id.unique()); Q = len(qs); qi = {int(v): i for i, v in enumerate(qs)}
    W = np.random.default_rng(SEED).multinomial(Q, np.ones(Q) / Q, size=B).astype(float)
    sc, dr = [], {}
    for t in TARGETS:
        for f in fracs:
            x = P[(P.target == t) & (P.fraction == f)]
            wide = x.pivot_table(index=["example_id", "cot_id"], columns="label", values="y_pred").dropna()
            yt = x.drop_duplicates(["example_id", "cot_id"]).set_index(["example_id", "cot_id"]).y_true.loc[wide.index].to_numpy()
            n_ours = (x.label == "Our model L21").sum(); assert len(wide) >= 0.99 * n_ours, (t, f, len(wide), n_ours)
            qq = np.array([qi[int(e)] for e in wide.index.get_level_values(0)])
            a = lambda v: np.bincount(qq, weights=v, minlength=Q)
            for lab in SPEC:
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
    m = sc.groupby(["label", "target"]).score.mean().reset_index().assign(fraction="mean")
    out_sc, out_pr = [], []
    for lab in SPEC:
        for t in TARGETS:
            d = np.mean([dr[(lab, t, f)] for f in fracs], 0)
            r = m[(m.label == lab) & (m.target == t)].iloc[0]
            out_sc.append(dict(label=lab, target=t, mean_score=r.score, ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5)))
            for ref in REFS:
                if ref == lab: continue
                dd = d - np.mean([dr[(ref, t, f)] for f in fracs], 0)
                rr = m[(m.label == ref) & (m.target == t)].iloc[0].score
                lo, hi = np.percentile(dd, [2.5, 97.5])
                out_pr.append(dict(label=lab, minus=ref, target=t, diff=r.score - rr, ci_lo=lo, ci_hi=hi,
                                   significant=bool(lo > 0 or hi < 0)))
    pd.DataFrame(out_sc).to_csv(OUT / "ci_scores.csv", index=False)
    pr = pd.DataFrame(out_pr); pr.to_csv(OUT / "ci_paired.csv", index=False)
    pd.set_option("display.width", 200)
    for ref in REFS:
        x = pr[pr.minus == ref].copy()
        x["cell"] = [f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if s else ''}" for d, lo, hi, s in zip(x["diff"], x.ci_lo, x.ci_hi, x.significant)]
        print(f"\n== label minus {ref} (mean over 10-90%; * = 95% CI excludes 0)")
        print(x.pivot(index="label", columns="target", values="cell")[TARGETS].to_string())


if __name__ == "__main__":
    main()
