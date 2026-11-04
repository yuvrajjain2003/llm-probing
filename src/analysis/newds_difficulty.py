#!/usr/bin/env python3
"""Question-difficulty measures and bands for MMLU-Pro and GlobalOpinionQA (plan_newdatasets P2.1/P2.2).

Replicates the GPQA recipe behind runs/goal2/gpqa25/sweep/_difficulty/question_difficulty.csv:
per-question difficulty from the 0% cut, two banding schemes, plus the fixed outer fold, written as
one row per question.  Two things differ from GPQA and both are handled here rather than assumed away:

  * the 0% cut has 5 continuations, not 25, so every 0%-cut measure is coarse and noisy (`noise.csv`);
  * GlobalOpinionQA has no gold answer, so `p_gold` is null on every row and a proxy is required.
    All candidates are built and scored side by side (`candidates.csv`, `shared_variance.csv`); the
    one used for the bands is `--measure`.

CAVEAT carried from GPQA, do not drop it when reporting: a difficulty measure computed from rollouts
is label-derived, so regressing a target on it measures SHARED VARIANCE, not a usable baseline, and
these numbers are in-sample so they are an upper bound.  The GlobalOpinionQA proxies are worse on this
axis than p_gold was: they are the same statistic as the entropy target, read at cut 0.

    ./envs/goal1/bin/python src/analysis/newds_difficulty.py --tag mmlupro_v1
    ./envs/goal1/bin/python src/analysis/newds_difficulty.py --tag globalop_v1
    ./envs/goal1/bin/python src/analysis/newds_difficulty.py --tag gpqa25      # cross-check only
Outputs: runs/goal2/<tag>/sweep/_difficulty/
"""
from __future__ import annotations

import argparse, ast, glob, json, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.labels import _entropy_mm
from probing.sweep import fixed_fold_of

REG = ["answer_entropy_bits_mm", "p_mode", "rem_len_mean", "rem_len_std"]
NORM = "answer_entropy_bits_mm_norm"   # reported alongside when n_options varies (see plan_newdatasets)
BIN = "committed_by_here"
FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]
BANDS = ["v_hard", "hard", "medium", "easy", "v_easy"]
EASY_WHEN_HIGH = {"p_gold", "p_mode0"}   # every other measure is an uncertainty: high value = harder
DRAWS = 200
GLOBALOP_RAW = "cache/huggingface/hub/datasets--Anthropic--llm_global_opinions/snapshots/*/data/global_opinions.csv"

# tag -> dataset dir name, default difficulty measure, and whether p_gold exists
DS = {
    "mmlupro_v1":  dict(ds="mmlu_pro", measure="p_gold"),
    "globalop_v1": dict(ds="global_opinions", measure="ent0_norm"),
    "gpqa25":      dict(ds="gpqa_diamond", measure="p_gold"),
}


# ---------------------------------------------------------------- candidate measures
def human_measures(examples: pd.DataFrame) -> pd.DataFrame:
    """Non-circular difficulty candidates for GlobalOpinionQA from the SURVEY data, not the model.

    `selections` in Anthropic/llm_global_opinions is a defaultdict(country -> probability vector over
    that question's options).  Pool with equal weight per country, then:
      human_ent        entropy in bits of the pooled human response distribution (how contested);
      human_ent_norm   the same over log2(n_options), comparable across questions;
      human_disagree   mean total-variation distance of a country from the pooled distribution.
    Joined on question text: examples.csv prepends an instruction, and `dataset_index` is NOT a
    reliable index into the released file (it disagrees for the last ~17 rows, checked)."""
    paths = glob.glob(GLOBALOP_RAW)
    if not paths:
        print(f"   WARNING no GlobalOpinionQA source at {GLOBALOP_RAW}; human measures skipped")
        return pd.DataFrame(columns=["example_id"])
    R = pd.read_csv(paths[0])

    def one(s):
        i, j = s.find("{"), s.rfind("}")
        sel = ast.literal_eval(s[i:j + 1])
        P = np.array([np.asarray(v, float) for v in sel.values()])
        P = P / P.sum(1, keepdims=True)
        q = P.mean(0)
        p = q[q > 0]
        return pd.Series(dict(human_ent=float(-(p * np.log2(p)).sum()),
                              human_disagree=float(np.abs(P - q).sum(1).mean() / 2),
                              n_countries=len(P), n_options_raw=len(q)))

    H = R.join(R.selections.apply(one))
    H["q"] = H.question.astype(str).str.strip()
    H = H.groupby("q", as_index=False)[["human_ent", "human_disagree", "n_countries",
                                        "n_options_raw"]].mean()   # 1 question appears twice
    ex = examples.copy()
    # examples.csv prepends one instruction paragraph; take everything after the FIRST blank line
    # (splitting on every blank line would truncate multi-paragraph questions)
    ex["q"] = ex.question.astype(str).str.split("\n\n", n=1).str[-1].str.strip()
    m = ex[["example_id", "q"]].merge(H, on="q", how="left")
    assert m.example_id.is_unique and len(m) == len(ex), "human join is not 1:1"
    print(f"   human measures joined for {m.human_ent.notna().sum()}/{len(m)} questions")
    m["human_ent_norm"] = m.human_ent / np.log2(m.n_options_raw.where(m.n_options_raw > 1))
    return m.drop(columns="q")


def candidates(f0: pd.DataFrame, examples: pd.DataFrame, tag: str) -> tuple[pd.DataFrame, list[str]]:
    """One row per question with every candidate difficulty measure."""
    f0 = f0.copy()
    for c in ["n_options", "answer_entropy_bits_mm_norm"]:      # absent in the older GPQA f0 file
        if c not in f0.columns:
            f0[c] = np.nan
    d = f0[["example_id", "n_continuations", "n_valid", "n_options", "p_gold", "p_mode",
            "answer_entropy_bits_mm", "answer_entropy_bits_mm_norm", "answer_dist"]].copy()
    d = d.rename(columns={"p_mode": "p_mode0", "answer_entropy_bits_mm": "ent0",
                          "answer_entropy_bits_mm_norm": "ent0_norm"})
    # two counts, both kept: n_correct = correct among the n_valid parsed answers (the honest count);
    # n_correct_k = correct out of the K attempted, which is what GPQA's published n_correct column is
    # (identical whenever n_valid == K, which is 100% of GPQA and 95% of MMLU-Pro).
    d["n_correct"] = (d.p_gold * d.n_valid).round().astype("Int64")
    d["n_correct_k"] = (d.p_gold * d.n_continuations).round().astype("Int64")
    names = ["ent0", "p_mode0"] + (["ent0_norm"] if d.ent0_norm.notna().any() else [])
    if d.p_gold.notna().any():
        names = ["p_gold"] + names
    if tag == "globalop_v1":
        d = d.merge(human_measures(examples), on="example_id", how="left")
        names += ["human_ent", "human_ent_norm", "human_disagree"]
    return d, names


# ---------------------------------------------------------------- bands
def make_bands(d: pd.DataFrame, measure: str, tiebreak: str | None) -> pd.DataFrame:
    """band_int: five fixed bands on the measure, GPQA's edges when the measure is p_gold.
    band_quintile: five equal-size bands by rank, ties inside one value broken by `tiebreak`.
    `band_quintile_arbitrary` flags questions whose quintile came from a tie no column could break."""
    x = d[measure]
    # ascending order of the measure runs v_hard -> v_easy for an accuracy, v_easy -> v_hard for an
    # uncertainty, so the two banding schemes must share one orientation or they disagree.
    labels = BANDS if measure in EASY_WHEN_HIGH else BANDS[::-1]
    if measure == "p_gold":
        # GPQA's band_int, exactly: thresholds on the COUNT of correct base samples out of K --
        # 0 | 1..K/5 | ..3K/5 | ..K-1 | K (K=25 there gives 0 | 1-5 | 6-15 | 16-24 | 25, and
        # reproduces the published 17/49/56/48/28).  Counts, not n_valid rates, so n_valid<K is
        # treated by how many were right, as GPQA did.
        frac_k = d.n_correct_k.astype(float) / d.n_continuations
        d["band_int"] = pd.cut(frac_k, [-1e-9, 1e-9, 0.2, 0.6, 1 - 1e-9, 1.0], labels=BANDS)
    else:                              # equal-width on the measure's range
        lo, hi = x.min(), x.max()
        d["band_int"] = pd.cut(x, np.linspace(lo - 1e-9, hi + 1e-9, 6), labels=labels)
    order = [x] + ([d[tiebreak]] if tiebreak else []) + [d.example_id]
    rank = pd.DataFrame({f"k{i}": v for i, v in enumerate(order)}).sort_values(
        [f"k{i}" for i in range(len(order))]).index
    q = pd.Series(index=d.index, dtype=object)
    n = x.notna().sum()
    edges = [round(n * i / 5) for i in range(6)]
    for b, (a, z) in zip(labels, zip(edges[:-1], edges[1:])):
        q.loc[rank[a:z]] = b
    d["band_quintile"] = pd.Categorical(q, categories=BANDS, ordered=True)
    # a question's quintile is arbitrary if its (measure, tiebreak) group straddles a quintile edge
    key = [measure] + ([tiebreak] if tiebreak else [])
    grp = d.groupby(key, dropna=False).band_quintile.nunique()
    d["band_quintile_arbitrary"] = d.set_index(key).index.map(grp > 1).to_numpy()
    d.loc[x.isna(), ["band_int", "band_quintile"]] = np.nan
    return d


# ---------------------------------------------------------------- diagnostics
def quad_r2(y: np.ndarray, x: np.ndarray, deg: int = 2) -> float:
    """In-sample R2 of a degree-`deg` polynomial in x.  Label-derived x => shared variance, not a baseline."""
    X = np.column_stack([x ** k for k in range(deg + 1)])
    p = X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def reg_targets(lab: pd.DataFrame) -> list[str]:
    return REG + ([NORM] if NORM in lab.columns and lab[NORM].notna().any() else [])


def shared_variance(lab: pd.DataFrame, d: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Per candidate x target x cut: in-sample R2 (linear and quadratic) of difficulty alone.
    `committed_by_here` gets AUC and the majority baseline instead of R2."""
    from sklearn.metrics import roc_auc_score
    idx = d.set_index("example_id")
    rows = []
    for nm in names:
        for t in reg_targets(lab) + [BIN]:
            for f in FRACS:
                s = lab[(lab.fraction == f) & lab[t].notna()]
                xv = idx[nm].reindex(s.example_id).to_numpy(float)
                ok = np.isfinite(xv)
                y, xv = s[t].to_numpy(float)[ok], xv[ok]
                if len(y) < 50:
                    continue
                r = dict(measure=nm, target=t, fraction=f, n=len(y),
                         n_questions=int(pd.unique(s.example_id.to_numpy()[ok]).size))
                if t == BIN:
                    r.update(metric="auc", linear=np.nan,
                             quadratic=(roc_auc_score(y, xv) if len(np.unique(y)) > 1 else np.nan),
                             majority_baseline=max(y.mean(), 1 - y.mean()))
                else:
                    r.update(metric="r2", linear=quad_r2(y, xv, 1), quadratic=quad_r2(y, xv, 2),
                             majority_baseline=np.nan)
                rows.append(r)
    return pd.DataFrame(rows)


def band_means(lab: pd.DataFrame, d: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean target by band, per cut and over cuts: the diagnostic that made the GPQA finding legible
    (there, difficulty vs entropy turned out to be non-monotonic)."""
    j = lab.merge(d[["example_id", col]], on="example_id")
    rows = []
    for t in reg_targets(lab) + [BIN]:
        for f in list(FRACS) + ["mean_over_cuts"]:
            s = j if f == "mean_over_cuts" else j[j.fraction == f]
            g = s.groupby(col, observed=True)[t]
            for b, v in g.mean().items():
                rows.append(dict(target=t, fraction=f, band=b, mean=v, sd=g.std().loc[b],
                                 n_rows=int(g.size().loc[b]),
                                 n_questions=int(s[s[col] == b].example_id.nunique())))
    return pd.DataFrame(rows)


def _boot(a):
    dist, n, seed = a
    keys = list(dist)
    p = np.array([dist[k] for k in keys], float)
    p /= p.sum()
    cnt = np.random.default_rng(seed).multinomial(n, p, size=DRAWS)
    ent = [_entropy_mm([k for k, c in zip(keys, row) for _ in range(c)]) for row in cnt]
    return float(np.var(ent, ddof=1)), float(np.var(cnt.max(1) / n, ddof=1))


def noise(d: pd.DataFrame, nproc: int = 16) -> pd.DataFrame:
    """How much of the spread in each 0%-cut measure is sampling noise from its n_valid draws.
    entropy / p_mode: parametric bootstrap from the row's own answer_dist (same method as
    redundancy_noise.py).  p_gold: binomial variance p(1-p)/n_valid.  reliability = 1 - noise/observed."""
    ok = d[d.n_valid.fillna(0) >= 2].copy()
    with Pool(nproc) as pool:
        r = pool.map(_boot, [(json.loads(x), int(n), i)
                             for i, (x, n) in enumerate(zip(ok.answer_dist, ok.n_valid))], chunksize=200)
    nv = {"ent0": np.array([a for a, _ in r]), "p_mode0": np.array([b for _, b in r])}
    if ok.n_options.notna().any():
        nv["ent0_norm"] = nv["ent0"] / np.log2(ok.n_options.where(ok.n_options > 1).to_numpy(float)) ** 2
    if ok.p_gold.notna().any():
        pg = ok.p_gold.to_numpy(float)
        nv["p_gold"] = pg * (1 - pg) / ok.n_valid.to_numpy(float)
    rows = []
    for nm, v in nv.items():
        x = ok[nm].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(v)
        obs, nse = float(np.var(x[m], ddof=1)), float(np.mean(v[m]))
        rows.append(dict(measure=nm, n_questions=int(m.sum()), mean_n_valid=float(ok.n_valid[m].mean()),
                         observed_var=obs, sampling_noise_var=nse, noise_share=nse / obs,
                         reliability=max(1 - nse / obs, 0.0), sd=float(np.std(x[m], ddof=1)),
                         noise_sd=float(np.sqrt(nse))))
    return pd.DataFrame(rows)


def cand_table(d: pd.DataFrame, names: list[str], SV: pd.DataFrame, measure: str,
               tiebreak: str | None) -> pd.DataFrame:
    """Usability of each candidate for banding, its n_options confound, and its shared variance."""
    rows = []
    for nm in names:
        x = d[nm].dropna()
        vc = x.value_counts(normalize=True)
        tmp = make_bands(d.copy(), nm, tiebreak if tiebreak != nm else None)
        qc = tmp.band_quintile.value_counts()
        xo = d[[nm, "n_options"]].dropna()
        r = dict(measure=nm, n_questions=len(x), n_distinct=int(x.nunique()),
                 largest_atom_frac=float(vc.iloc[0]), sd=float(x.std()),
                 corr_n_options=float(np.corrcoef(xo[nm], xo.n_options)[0, 1]) if xo[nm].std() else np.nan,
                 spearman_n_options=float(xo[nm].corr(xo.n_options, method="spearman")),
                 quintile_min_n=int(qc.min()), quintile_max_n=int(qc.max()),
                 quintile_arbitrary_frac=float(tmp.band_quintile_arbitrary.mean()),
                 chosen=(nm == measure))
        for t in sorted(SV.target.unique()):
            if t == BIN:
                continue
            s = SV[(SV.measure == nm) & (SV.target == t)]
            r[f"r2_{t}"] = float(s.quadratic.mean())
        s = SV[(SV.measure == nm) & (SV.target == BIN)]
        r[f"auc_{BIN}"] = float(s.quadratic.mean())
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(DS))
    ap.add_argument("--measure", default=None, help="difficulty column used for the bands")
    ap.add_argument("--tiebreak", default="ent0", help="column breaking ties inside one measure value")
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--out", default=None, help="output dir (default runs/goal2/<tag>/sweep/_difficulty)")
    a = ap.parse_args()
    cfg = DS[a.tag]
    measure = a.measure or cfg["measure"]
    run = Path(f"runs/goal1/{a.tag}/{cfg['ds']}/DeepSeek-R1-Distill-Qwen-7B")
    out = Path(a.out or f"runs/goal2/{a.tag}/sweep/_difficulty")
    out.mkdir(parents=True, exist_ok=True)

    f0 = pd.read_csv(run / "labels/checkpoint_labels_f0.csv")
    examples = pd.read_csv(run / "raw/examples.csv")
    lab = pd.read_csv(run / "labels/checkpoint_labels.csv")
    cm = pd.read_csv(run / "labels/commitment.csv")
    lab["fraction"] = lab.fraction.round(3)
    lab = lab[lab.fraction.isin(FRACS)].merge(
        cm[["example_id", "cot_id", "commitment_fraction"]], on=["example_id", "cot_id"], how="inner")
    lab[BIN] = (lab.commitment_fraction.notna() & (lab.fraction >= lab.commitment_fraction)).astype(int)
    print(f"{a.tag}: {len(f0)} questions at the 0% cut, {len(lab)} checkpoint rows over {FRACS}")

    d, names = candidates(f0, examples, a.tag)
    tb = a.tiebreak if (a.tiebreak in d.columns and a.tiebreak != measure) else None
    assert measure in d.columns and d[measure].notna().any(), f"measure {measure} unusable"
    fold = fixed_fold_of(run, 5, 0)
    d["fold"] = d.example_id.map(fold)
    assert d.fold.notna().all()

    SV = shared_variance(lab, d, names)
    SV.to_csv(out / "shared_variance.csv", index=False)
    CT = cand_table(d, names, SV, measure, tb)
    CT.to_csv(out / "candidates.csv", index=False)
    d = make_bands(d, measure, tb)
    d["difficulty"] = d[measure]
    d["difficulty_measure"] = measure
    cols = (["example_id", "difficulty", "difficulty_measure", "band_int", "band_quintile",
             "band_quintile_arbitrary", "fold", "n_continuations", "n_valid", "n_options",
             "p_gold", "n_correct", "n_correct_k"] + names)
    d[list(dict.fromkeys(cols))].to_csv(out / "question_difficulty.csv", index=False)
    for col in ["band_int", "band_quintile"]:
        band_means(lab, d, col).to_csv(out / f"band_target_means.{col}.csv", index=False)
        pd.crosstab(d[col], d.fold, dropna=False).to_csv(out / f"band_fold_counts.{col}.csv")
    NZ = noise(d, a.nproc)
    NZ.to_csv(out / "noise.csv", index=False)

    pd.set_option("display.width", 230)
    print("\n== candidate difficulty measures (r2_* / auc_* are IN-SAMPLE shared variance, mean over cuts) ==")
    print(CT.round(3).to_string(index=False))
    print("\n== sampling noise in the 0%-cut measures ==")
    print(NZ.round(4).to_string(index=False))
    print(f"\n== bands from `{measure}` (ties broken by {tb}) ==")
    for col in ["band_int", "band_quintile"]:
        g = d.groupby(col, observed=True)
        print(f"-- {col}: counts {g.size().to_dict()}")
        print(f"   range of {measure}: " + ", ".join(
            f"{b} [{v.min():.3f},{v.max():.3f}]" for b, v in g[measure]))
        ct = pd.crosstab(d[col], d.fold)
        print(f"   smallest (band, fold) cell: {int(ct.to_numpy().min())} questions")
        if col == "band_quintile":
            print("   share of each band assigned by an arbitrary tie-break: "
                  f"{g.band_quintile_arbitrary.mean().round(3).to_dict()}")
    print(f"\n== mean target by band_int, per cut ({measure}) ==")
    bm = band_means(lab, d, "band_int")
    for t in ["answer_entropy_bits_mm", "rem_len_mean"]:
        print(f"-- {t}")
        print(bm[bm.target == t].pivot_table(index="band", columns="fraction", values="mean",
                                             observed=True).reindex(BANDS).round(3).to_string())
    print(f"\n== shared variance of `{measure}` alone (quadratic, IN-SAMPLE => upper bound) ==")
    print(SV[SV.measure == measure].pivot_table(index="target", columns="fraction",
                                                values="quadratic").round(3).to_string())
    print("\nwrote", out)


if __name__ == "__main__":
    main()
