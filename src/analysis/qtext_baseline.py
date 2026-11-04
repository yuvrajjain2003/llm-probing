#!/usr/bin/env python3
"""Question-text-only baseline (CLAUDE.md next-steps item 3), and a NON-CIRCULAR difficulty measure.

Two purposes, one fit.

PURPOSE A (the baseline).  Predict each row's label from the QUESTION STATEMENT alone: TF-IDF on the
question (optionally plus its answer choices), no CoT prefix, no activations.  The existing `tfidf`
baseline in the sweeps vectorises the PREFIX, so it knows which chain it is on and how far along it
is; this one cannot know either.  Its score is therefore a direct measurement of how much of the
probe's R2 is question-level difficulty legible from the question statement.  Scored on the same
rows, the same fixed folds and the same metrics as `ref_fixed`, within each cut point and never
pooled across depth, plus a question-level view against the per-question mean label.

PURPOSE B (a difficulty measure that is not circular).  `runs/goal2/<tag>/sweep/_difficulty/` builds
difficulty from the model's own rollouts (`p_gold`, or on GlobalOpinionQA the 0%-cut entropy, which
IS the entropy target read at cut 0).  An OUT-OF-FOLD question-text prediction breaks that: for a
held-out question the value is a function of its words only, the mapping having been learned from
other questions' labels, so partialling it out removes "what the statement implies" rather than
"this question's own measured entropy".  The output mirrors `_difficulty/question_difficulty.csv`
column for column, and reuses that script's own banding and diagnostics, so `icl_bands.py` /
`icl_decompose.py` can consume it unchanged and the two measures are directly comparable.

PROTOCOL (and where it deviates from src/probing/sweep.py, deliberately)
    Rows, folds and NaN handling come from sweep.py itself (`load_meta`, `fixed_fold_of`), so the
    row set is identical to `ref_fixed`.  Fits are at QUESTION level with sample weights equal to the
    question's row count, which is EXACTLY equivalent to fitting on the duplicated rows (a question's
    rows share one feature vector, so the within-question sum of squares is constant in the
    coefficients); `--verify` checks that against a direct row-level sklearn fit.  Inner selection is
    over the TF-IDF configuration AND the regularisation strength jointly, on 3 inner folds of the
    outer training questions only.  The inner folds are a fixed seeded question-level split rather
    than sweep's per-cell GroupKFold over rows, so every (cut, target) cell shares its splits and one
    vectoriser fit serves them all.  Vectorisers are always fitted on training questions only.

    ./envs/goal1/bin/python src/analysis/qtext_baseline.py --tag gpqa25 --verify
    ./envs/goal1/bin/python src/analysis/qtext_baseline.py --tag mmlupro_v1
    ./envs/goal1/bin/python src/analysis/qtext_baseline.py --tag globalop_v1
Outputs: runs/goal2/<tag>/sweep/_qtext/
"""
from __future__ import annotations

import argparse, ast, importlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.sweep import fixed_fold_of, load_meta

REG = ["answer_entropy_bits_mm", "p_mode", "rem_len_mean", "rem_len_std"]
NORM = "answer_entropy_bits_mm_norm"
BIN = "committed_by_here"
FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]
ALPHAS = np.logspace(-3, 3, 7).tolist()        # = configs/sweep_ref*.yaml, probe `ridge_text`
CS = np.logspace(-8, 1, 10).tolist()           # = probe `logistic_deep`
N_INNER = 3

DS = {"gpqa25": "gpqa_diamond", "mmlupro_v1": "mmlu_pro", "globalop_v1": "global_opinions"}

# TF-IDF configurations, crossed with the regularisation grid in the inner selection.
# w12 is identical to probe_checkpoints.text_features, i.e. to the prefix-TF-IDF baseline.
VCFGS = {
    "w12": dict(max_features=20000, ngram_range=(1, 2), sublinear_tf=True),
    "w1":  dict(max_features=20000, ngram_range=(1, 1), sublinear_tf=True),
    "c35": dict(analyzer="char_wb", ngram_range=(3, 5), max_features=30000, sublinear_tf=True),
}
VARIANTS = ["q", "qc"]                          # question only / question + answer choices
DIFF_VARIANT = "qc"                             # variant the difficulty columns are taken from


# ------------------------------------------------------------------ text
def question_text(examples: pd.DataFrame, tag: str) -> dict[str, np.ndarray]:
    """Per question, the two text variants, in the row order of `examples`.

    GlobalOpinionQA's `question` column has ONE prepended instruction paragraph, identical on all
    2,555 rows (asserted), so it is dropped with maxsplit=1: splitting on every blank line would
    truncate the 279 questions whose body is itself multi-paragraph."""
    q = examples.question.fillna("").astype(str)
    if tag == "globalop_v1":
        pre = q.str.split("\n\n", n=1).str[0]
        assert pre.nunique() == 1, f"expected one instruction paragraph, got {pre.nunique()}"
        q = q.str.split("\n\n", n=1).str[-1]
    q = q.str.strip()
    ch = examples.choices.fillna("[]").astype(str).map(
        lambda s: "\n".join(str(x) for x in _lit(s, [])) if isinstance(_lit(s, []), list) else str(s))
    return {"q": q.to_numpy(object), "qc": (q + "\n" + ch).to_numpy(object)}


def _lit(s, default):
    try:
        return ast.literal_eval(s)
    except Exception:
        return default


def n_options_of(examples: pd.DataFrame) -> np.ndarray:
    return examples.choices.fillna("[]").astype(str).map(
        lambda s: float(len(_lit(s, []))) if isinstance(_lit(s, []), list) else np.nan).to_numpy(float)


# ------------------------------------------------------------------ rows
def load_rows(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(examples, rows).  `rows` = the exact ref_fixed row set over 10-90%, plus the 0% cut
    (one row per question, cot_id -1; no commitment target exists there)."""
    ds = DS[tag]
    run = Path(f"runs/goal1/{tag}/{ds}/DeepSeek-R1-Distill-Qwen-7B")
    act = Path(f"runs/goal2/{tag}/activations_ckpt/{ds}/DeepSeek-R1-Distill-Qwen-7B")
    M = load_meta(act, run)
    keep = ["example_id", "cot_id", "fraction", "n_continuations"] + REG + [BIN] + (
        [NORM] if NORM in M.columns else [])
    M = M[[c for c in keep if c in M.columns]].copy()
    f0 = pd.read_csv(run / "labels/checkpoint_labels_f0.csv")
    f0["p_gold_f0"], f0["n_valid_f0"] = f0.p_gold, f0.n_valid
    f0 = f0[[c for c in keep if c in f0.columns] + ["p_gold_f0", "n_valid_f0"]].copy()
    f0[BIN] = np.nan
    R = pd.concat([f0, M], ignore_index=True)
    R["fraction"] = R.fraction.round(3)
    return pd.read_csv(run / "raw/examples.csv"), R


# ------------------------------------------------------------------ fits
def _vec(txt, trmask, vkw):
    v = TfidfVectorizer(dtype=np.float32, **vkw)
    v.fit(txt[trmask])
    return v.transform(txt)


def _fit_predict(Xtr, Xall, agg, task, hp):
    """agg: (ybar, w) for reg, (n1, n0) for bin, already restricted to the rows of Xtr.
    Returns predictions for every row of Xall."""
    if task == "reg":
        ybar, w = agg
        return Ridge(alpha=hp).fit(Xtr, ybar, sample_weight=w).predict(Xall)
    n1, n0 = agg
    X2 = sp.vstack([Xtr, Xtr])
    y2 = np.r_[np.ones(Xtr.shape[0]), np.zeros(Xtr.shape[0])]
    w2 = np.r_[n1, n0]
    k = w2 > 0
    if len(np.unique(y2[k])) < 2:
        return np.full(Xall.shape[0], float(y2[k][0]))
    m = LogisticRegression(max_iter=3000, C=hp).fit(X2[k], y2[k], sample_weight=w2[k])
    return m.predict_proba(Xall)[:, 1]


def _aggregate(y, qp, nq, task):
    """Row target -> (fit aggregate, per-question mean, per-question row count)."""
    cnt = np.bincount(qp, minlength=nq).astype(float)
    s = np.bincount(qp, weights=y, minlength=nq)
    mean = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
    return ((mean, cnt) if task == "reg" else (s, cnt - s)), mean, cnt


def _sub(agg, mask):
    return tuple(v[mask] for v in agg)


def _row_score(y, p, task):
    if task == "reg":
        ss = ((y - y.mean()) ** 2).sum()
        return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")
    return float(((p >= 0.5).astype(int) == y).mean())


def run_cells(R, texts, fold, inner, qids, verify=False):
    """-> (results, long row-level predictions for the `sel` label, OOF question-level table)."""
    pos = {int(q): i for i, q in enumerate(qids)}
    nq = len(qids)
    fo = np.array([fold[int(q)] for q in qids])
    ino = np.array([inner[int(q)] for q in qids])
    reg_targets = REG + ([NORM] if NORM in R.columns and R[NORM].notna().any() else [])

    info = {}
    for f in [0.0] + FRACS:
        for t in reg_targets + ([BIN] if f > 0 else []):
            if t not in R.columns:
                continue
            m = ((R.fraction == f) & R[t].notna()).to_numpy()
            if int(m.sum()) < 50:
                continue
            idx = np.where(m)[0]
            qp = np.array([pos[int(q)] for q in R.example_id.to_numpy()[idx]])
            info[(f, t)] = dict(idx=idx, qp=qp, y=R[t].to_numpy(float)[idx],
                                qmask=np.bincount(qp, minlength=nq) > 0,
                                rfold=fo[qp], task="bin" if t == BIN else "reg")
    groups = {}
    for k, v in info.items():
        groups.setdefault(v["qmask"].tobytes(), []).append(k)
    print(f"  {len(info)} cells in {len(groups)} question-set group(s); {nq} questions, {len(R)} rows")

    inner_sc, oof, chosen = {}, {}, {}
    for variant in VARIANTS:
        txt = texts[variant]
        for vc, vkw in VCFGS.items():
            t0 = time.time()
            for g, cks in groups.items():
                qm = np.frombuffer(g, dtype=bool)
                X = {}
                for k in range(5):
                    tr = qm & (fo != k)
                    X[("o", k)] = _vec(txt, tr, vkw)
                    for j in range(N_INNER):
                        X[("i", k, j)] = _vec(txt, tr & (ino != j), vkw)
                for ck in cks:
                    d = info[ck]
                    task, qp, y, rf = d["task"], d["qp"], d["y"], d["rfold"]
                    grid = CS if task == "bin" else ALPHAS
                    p_out, hps = np.full(len(y), np.nan), []
                    for k in range(5):
                        trm = qm & (fo != k)
                        rtr, rte = rf != k, rf == k
                        sc = np.zeros(len(grid))
                        for j in range(N_INNER):
                            itr, ite = trm & (ino != j), trm & (ino == j)
                            ri, rj = itr[qp], ite[qp]
                            if ri.sum() < 10 or rj.sum() < 10:
                                continue
                            agg = _sub(_aggregate(y[ri], qp[ri], nq, task)[0], itr)
                            A = X[("i", k, j)]
                            for h, hp in enumerate(grid):
                                pq = _fit_predict(A[itr], A, agg, task, hp)
                                sc[h] += _row_score(y[rj], pq[qp[rj]], task)
                        inner_sc[(ck, k, variant, vc)] = sc
                        best = grid[int(np.argmax(sc))] if sc.any() else grid[len(grid) // 2]
                        hps.append(best)
                        agg = _sub(_aggregate(y[rtr], qp[rtr], nq, task)[0], trm)
                        A = X[("o", k)]
                        p_out[rte] = _fit_predict(A[trm], A, agg, task, best)[qp[rte]]
                    oof[(ck, variant, vc)] = p_out
                    chosen[(ck, variant, vc)] = hps
                del X
            print(f"  {variant}/{vc}: {len(info)} cells in {time.time() - t0:.0f}s")

    # jointly selected configuration: per (cell, fold) pick argmax over (vectoriser, hp) together.
    # Given the winning vectoriser, the winning hp is that vectoriser's own best, so the OOF
    # prediction is already computed and nothing is refitted here.
    for ck, d in info.items():
        for variant in VARIANTS:
            p_out, hps = np.full(len(d["y"]), np.nan), []
            for k in range(5):
                cand = [(inner_sc[(ck, k, variant, vc)].max(), vc) for vc in VCFGS]
                vc = max(cand)[1] if any(c[0] for c in cand) else "w12"
                sel = d["rfold"] == k
                p_out[sel] = oof[(ck, variant, vc)][sel]
                hps.append(f"{vc}:{chosen[(ck, variant, vc)][k]:g}")
            oof[(ck, variant, "sel")] = p_out
            chosen[(ck, variant, "sel")] = hps

    if verify:
        _verify_equivalence(texts, fo, nq, info)

    res, preds, oofq = [], [], []
    for (f, t), d in info.items():
        task, qp, y, idx = d["task"], d["qp"], d["y"], d["idx"]
        _, qb, cnt = _aggregate(y, qp, nq, task)
        use = cnt > 0
        # ceiling for ANY question-level predictor at row level: predict each row by its own
        # question's mean.  Uses the row's own question, so an upper bound, not a baseline.
        oracle = _row_score(y, qb[qp], "reg")
        for variant in VARIANTS:
            for vc in list(VCFGS) + ["sel"]:
                p = oof[((f, t), variant, vc)]
                ok = ~np.isnan(p)
                pq = np.full(nq, np.nan)
                pq[qp[ok]] = p[ok]
                row = dict(variant=variant, vectorizer=vc, fraction=f, target=t,
                           metric="acc" if task == "bin" else "R2",
                           n=int(ok.sum()), n_questions=int(use.sum()),
                           row_score=_row_score(y[ok], p[ok], task),
                           q_score=_row_score(qb[use], pq[use], "reg"),
                           q_mean_oracle_row_r2=oracle,
                           row_auc=np.nan, majority_baseline=np.nan,
                           hp=json.dumps(chosen[((f, t), variant, vc)]))
                if task == "bin":
                    row["row_auc"] = (float(roc_auc_score(y[ok], p[ok]))
                                      if len(np.unique(y[ok])) > 1 else np.nan)
                    row["majority_baseline"] = float(max(y[ok].mean(), 1 - y[ok].mean()))
                res.append(row)
                if vc == "sel":
                    pr = R.iloc[idx][["example_id", "cot_id", "fraction"]].copy()
                    pr.insert(0, "target", t)
                    pr.insert(0, "probe", "logistic_qtext" if task == "bin" else "ridge_qtext")
                    pr.insert(0, "feature", f"qtext_{variant}")
                    pr["y_true"], pr["y_pred"] = y, p
                    preds.append(pr)
                    oofq.append(pd.DataFrame(dict(example_id=np.asarray(qids)[use], fraction=f,
                                                  target=t, variant=variant,
                                                  y_true_qmean=qb[use], y_pred=pq[use])))
    return (pd.DataFrame(res), pd.concat(preds, ignore_index=True),
            pd.concat(oofq, ignore_index=True))


def _verify_equivalence(texts, fo, nq, info):
    """The weighted question-level fit must reproduce a direct row-level sklearn fit."""
    for key in [(0.5, "answer_entropy_bits_mm"), (0.9, "rem_len_mean"), (0.5, BIN)]:
        if key not in info:
            continue
        d = info[key]
        task, qp, y, qm = d["task"], d["qp"], d["y"], d["qmask"]
        trm = qm & (fo != 0)
        rtr, rte = d["rfold"] != 0, d["rfold"] == 0
        A = _vec(texts["qc"], trm, VCFGS["w12"])
        hp = 1.0 if task == "reg" else 0.01
        pg = _fit_predict(A[trm], A, _sub(_aggregate(y[rtr], qp[rtr], nq, task)[0], trm),
                          task, hp)[qp[rte]]
        Xtr, Xte = A[qp[rtr]], A[qp[rte]]
        if task == "reg":
            pr = Ridge(alpha=hp).fit(Xtr, y[rtr]).predict(Xte)
        else:
            pr = LogisticRegression(max_iter=3000, C=hp).fit(
                Xtr, y[rtr].astype(int)).predict_proba(Xte)[:, 1]
        sd = float(np.std(y)) or 1.0
        err = float(np.abs(pg - pr).max()) / sd       # relative: rem_len is ~1e3, entropy ~1
        print(f"  VERIFY {key[1]} at {key[0]}: max |grouped - row-level| / sd(y) = {err:.2e}  "
              f"{'OK' if err < 1e-3 else 'MISMATCH'}")


# ------------------------------------------------------------------ difficulty interface
def _pgold_oof(texts, R, fo, qids, ino):
    """OOF question-text prediction of the 0%-cut p_gold (p_gold is null on GlobalOpinionQA)."""
    f0 = R[(R.fraction == 0.0) & R.p_gold_f0.notna()] if "p_gold_f0" in R.columns else R.iloc[:0]
    if not len(f0):
        return pd.Series(dtype=float)
    pos = {int(q): i for i, q in enumerate(qids)}
    y = f0.p_gold_f0.to_numpy(float)
    qp = np.array([pos[int(q)] for q in f0.example_id])
    qm = np.zeros(len(qids), bool)
    qm[qp] = True
    out = np.full(len(y), np.nan)
    for k in range(5):
        trm = qm & (fo != k)
        sc = np.zeros(len(ALPHAS))
        for j in range(N_INNER):
            itr, ite = trm & (ino != j), trm & (ino == j)
            ri, rj = itr[qp], ite[qp]
            if ri.sum() < 10 or rj.sum() < 10:
                continue
            A = _vec(texts, itr, VCFGS["w12"])
            for h, a in enumerate(ALPHAS):
                sc[h] += _row_score(y[rj], Ridge(alpha=a).fit(A[qp[ri]], y[ri]).predict(A[qp[rj]]),
                                    "reg")
        best = ALPHAS[int(np.argmax(sc))] if sc.any() else ALPHAS[len(ALPHAS) // 2]
        A = _vec(texts, trm, VCFGS["w12"])
        rtr, rte = trm[qp], (fo == k)[qp]
        out[rte] = Ridge(alpha=best).fit(A[qp[rtr]], y[rtr]).predict(A[qp[rte]])
    return pd.Series(out, index=f0.example_id.to_numpy())


def difficulty_table(oofq, examples, R, fold, texts, fo, ino, qids, measure=None):
    nd = importlib.import_module("analysis.newds_difficulty")
    nd.EASY_WHEN_HIGH = set(nd.EASY_WHEN_HIGH) | {"qtext_pgold", "qtext_pmode0"}

    w = oofq[oofq.variant == DIFF_VARIANT].pivot_table(
        index="example_id", columns=["target", "fraction"], values="y_pred")
    have = set(w.columns)
    cand = pd.DataFrame(index=w.index)
    if ("answer_entropy_bits_mm", 0.0) in have:
        cand["qtext_ent0"] = w[("answer_entropy_bits_mm", 0.0)]
    if ("p_mode", 0.0) in have:
        cand["qtext_pmode0"] = w[("p_mode", 0.0)]
    if (NORM, 0.0) in have:
        cand["qtext_ent0_norm"] = w[(NORM, 0.0)]
    cand["qtext_ent_mean"] = w["answer_entropy_bits_mm"][FRACS].mean(1)
    if NORM in w.columns.get_level_values(0):
        cand["qtext_ent_norm_mean"] = w[NORM][FRACS].mean(1)
    cand["qtext_remlen_mean"] = w["rem_len_mean"][FRACS].mean(1)
    pg = _pgold_oof(texts[DIFF_VARIANT], R, fo, qids, ino)
    if len(pg):
        cand["qtext_pgold"] = pg.reindex(cand.index)

    names = [c for c in cand.columns if cand[c].notna().any()]
    measure = measure or ("qtext_pgold" if "qtext_pgold" in names else
                          "qtext_ent_norm_mean" if "qtext_ent_norm_mean" in names else
                          "qtext_ent_mean")
    d = cand.reset_index()
    nopt = pd.Series(n_options_of(examples), index=examples.example_id)
    f0 = R[R.fraction == 0.0].drop_duplicates("example_id").set_index("example_id")
    d["n_options"] = d.example_id.map(nopt)
    d["n_continuations"] = d.example_id.map(f0.n_continuations) if "n_continuations" in f0 else np.nan
    d["n_valid"] = d.example_id.map(f0.n_valid_f0) if "n_valid_f0" in f0 else np.nan
    d["p_gold"] = d.example_id.map(f0.p_gold_f0) if "p_gold_f0" in f0 else np.nan
    d["n_correct"] = (d.p_gold * d.n_valid).round()
    d["n_correct_k"] = (d.p_gold * d.n_continuations).round()
    d["fold"] = d.example_id.map(fold)
    assert d.fold.notna().all()

    lab = R[R.fraction.isin(FRACS)].copy()
    SV = nd.shared_variance(lab, d, names)
    CT = nd.cand_table(d, names, SV, measure, None)
    d = nd.make_bands(d, measure, None)
    d["difficulty"] = d[measure]
    d["difficulty_measure"] = measure
    return d, names, measure, SV, CT


def _inner_folds(qids, seed=1):
    """Fixed question-level 3-fold used for every inner selection (target independent)."""
    ids = np.sort(np.unique(qids))
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {int(q): int(r % N_INNER) for q, r in zip(ids[perm], range(len(ids)))}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(DS))
    ap.add_argument("--measure", default=None, help="difficulty column copied into `difficulty`")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out or f"runs/goal2/{a.tag}/sweep/_qtext")
    out.mkdir(parents=True, exist_ok=True)
    examples, R = load_rows(a.tag)
    run = Path(f"runs/goal1/{a.tag}/{DS[a.tag]}/DeepSeek-R1-Distill-Qwen-7B")
    fold = fixed_fold_of(run, 5, 0)
    qids = examples.example_id.to_numpy()
    assert set(R.example_id) <= set(qids)
    texts = question_text(examples, a.tag)
    inner = _inner_folds(qids)
    fo = np.array([fold[int(q)] for q in qids])
    ino = np.array([inner[int(q)] for q in qids])
    print(f"{a.tag}: {len(examples)} questions in examples.csv, {R.example_id.nunique()} with rows; "
          f"mean text length (chars) q {np.mean([len(x) for x in texts['q']]):.0f}, "
          f"qc {np.mean([len(x) for x in texts['qc']]):.0f}")

    RES, PRED, OOFQ = run_cells(R, texts, fold, inner, qids, verify=a.verify)
    RES.insert(0, "dataset_tag", a.tag)
    RES.to_csv(out / "results.csv", index=False)
    PRED.to_csv(out / "preds.csv", index=False)
    OOFQ.to_csv(out / "oof_question.csv", index=False)

    d, names, measure, SV, CT = difficulty_table(OOFQ, examples, R, fold, texts, fo, ino, qids,
                                                 a.measure)
    cols = (["example_id", "difficulty", "difficulty_measure", "band_int", "band_quintile",
             "band_quintile_arbitrary", "fold", "n_continuations", "n_valid", "n_options",
             "p_gold", "n_correct", "n_correct_k"] + names)
    d[list(dict.fromkeys(cols))].to_csv(out / "question_difficulty.csv", index=False)
    SV.to_csv(out / "shared_variance.csv", index=False)
    CT.to_csv(out / "candidates.csv", index=False)
    for col in ["band_int", "band_quintile"]:
        pd.crosstab(d[col], d.fold, dropna=False).to_csv(out / f"band_fold_counts.{col}.csv")

    dp = Path(f"runs/goal2/{a.tag}/sweep/_difficulty/question_difficulty.csv")
    if dp.exists():
        o = pd.read_csv(dp).set_index("example_id")
        mine = d.set_index("example_id")
        rows = []
        for nm in names:
            for om in [c for c in ("difficulty", "p_gold", "ent0", "p_mode0", "ent0_norm",
                                   "human_ent_norm", "human_disagree") if c in o.columns]:
                x = pd.concat([mine[nm].rename("a"), o[om].rename("b")], axis=1).dropna()
                if len(x) > 50 and x.a.std() > 0 and x.b.std() > 0:
                    rows.append(dict(qtext_measure=nm, rollout_measure=om, n=len(x),
                                     pearson=float(np.corrcoef(x.a, x.b)[0, 1]),
                                     spearman=float(x.a.corr(x.b, method="spearman"))))
        pd.DataFrame(rows).to_csv(out / "vs_rollout_difficulty.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n== jointly-selected vectoriser, mean over 10-90% ==")
    print(RES[(RES.vectorizer == "sel") & (RES.fraction > 0)]
          .groupby(["variant", "target"])[["row_score", "q_score", "q_mean_oracle_row_r2"]]
          .mean().round(3).to_string())
    print("\n== row score per variant x vectoriser, entropy and remaining length (mean 10-90%) ==")
    print(RES[(RES.fraction > 0) & RES.target.isin(["answer_entropy_bits_mm", "rem_len_mean"])]
          .pivot_table(index=["variant", "vectorizer"], columns="target", values="row_score")
          .round(3).to_string())
    print("\n== 0% cut (one row per question) ==")
    print(RES[(RES.fraction == 0) & (RES.vectorizer == "sel")]
          .pivot_table(index="target", columns="variant", values="row_score").round(3).to_string())
    print("\n== difficulty candidates (shared variance is IN-SAMPLE, of an OUT-OF-FOLD measure) ==")
    print(CT.round(3).to_string(index=False))
    print(f"\nchosen difficulty measure: {measure}")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
