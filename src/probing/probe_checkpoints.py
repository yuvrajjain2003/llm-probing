#!/usr/bin/env python3
"""Do prefix activations predict the Goal 1 continuation labels, beyond the text?

Successor to ``probe_goal1_labels.py``, which answered "no" for entropy and
length but under three handicaps that all favoured the text baseline:

  1. Activations were read at the nearest captured token AT OR BEFORE the
     checkpoint (capture stride was 4), while TF-IDF got the exact prefix.
  2. Only decoder layers 14/21/28 of 28 were stored.
  3. Regularisation was never swept -- Ridge(alpha=1000) for activations against
     Ridge(alpha=1.0) for TF-IDF, both fixed by hand.

This version reads the checkpoint-exact, all-layer store written by
``extract_checkpoints.py``, sweeps regularisation for EVERY model by nested CV,
and adds the comparison the research question actually asks for: whether
[text + activations] beats [text] alone.  "Beyond the prefix text" is a claim
about added information, and an activation probe losing to TF-IDF head-to-head
does not settle it either way.

DESIGN

  Within-checkpoint training.  Entropy falls 0.955 -> 0.403 and remaining length
  4735 -> 952 across the five checkpoints, and depth is itself decodable from the
  activations (0.82 in probe_linear.py).  A probe pooled over checkpoints could
  score well by reading depth and predicting the per-depth mean while knowing
  nothing about the individual CoT.  Training separately per checkpoint holds
  depth constant by construction.

  Grouped CV.  A question's 3 CoTs share its text, so ungrouped folds leak.
  GroupKFold on example_id, with the SAME fold assignment across every model at
  a checkpoint so scores are paired.

  Nested alpha/C selection.  The outer fold's test rows never participate in
  choosing regularisation.  Picking alpha on the score being reported is how you
  manufacture a result at n=575, d=3584.

  Miller-Madow entropy is the default target.  The stored plug-in labels are
  biased low by ~0.05 bits at K=25, which is 2.5x the tolerance the K=25 choice
  was justified against -- see labels._entropy_mm.

TARGETS
  answer_entropy_bits_mm  entropy over the K continuation answers (MM-corrected)
  rem_len_mean            mean tokens from the checkpoint to </think>
  rem_len_std             spread of that, across continuations
  p_mode                  share of continuations landing on the modal answer
  committed_by_here       has this CoT committed by this checkpoint (binary)
  will_commit             will it ever commit, at any checkpoint (binary, per-CoT)

Usage
-----
    python src/probing/probe_checkpoints.py
    python src/probing/probe_checkpoints.py --k100-only    # sensitivity check
"""
from __future__ import annotations

import argparse, builtins, functools, glob, json
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(builtins.print, flush=True)

REG_TARGETS = ["answer_entropy_bits_mm", "rem_len_mean", "rem_len_std", "p_mode"]
BIN_TARGETS = ["committed_by_here", "will_commit"]
ALPHAS = np.logspace(0, 6, 13)          # ridge, on standardised features
CS = np.logspace(-4, 1, 6)              # logistic


# ── data ──────────────────────────────────────────────────────────────────────

def load(act_dir: Path, run_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Activations (n, n_layers, hidden) + metadata joined to the Goal 1 labels.

    The join is INNER on (example_id, cot_id, fraction): a handful of CoTs have
    checkpoints but no labels (their continuations never produced a parseable
    answer).  probe_goal1_labels.py kept them and, because its commitment target
    was built with .notna(), silently scored them as "not committed" -- 8 CoTs of
    fabricated negatives.  Dropping them is the whole fix.
    """
    idxs, arrs = [], []
    for ip in sorted(glob.glob(str(act_dir / "index.shard*.parquet"))):
        sfx = ip.split("index.")[1].replace(".parquet", "")
        a = np.load(act_dir / f"activations.{sfx}.npy")
        i = pd.read_parquet(ip)
        assert len(i) == a.shape[0], f"{sfx}: index/activation mismatch"
        i = i.sort_values("row_idx")
        arrs.append(a[i.row_idx.to_numpy()])
        idxs.append(i.drop(columns="row_idx"))
    if not arrs:
        raise SystemExit(f"no activation shards under {act_dir}")
    X = np.concatenate(arrs, axis=0)
    M = pd.concat(idxs, ignore_index=True)
    print(f"  activations {X.shape}  ({X.nbytes/2**30:.2f} GB in memory as "
          f"{X.dtype})")

    lab = pd.read_csv(run_dir / "labels" / "checkpoint_labels.csv")
    cm = pd.read_csv(run_dir / "labels" / "commitment.csv")
    M["_row"] = np.arange(len(M))
    M = M.merge(lab, on=["example_id", "cot_id", "fraction"], how="inner")
    M = M.merge(cm[["example_id", "cot_id", "commitment_fraction"]],
                on=["example_id", "cot_id"], how="inner")
    X = X[M._row.to_numpy()]
    M = M.drop(columns="_row").reset_index(drop=True)

    M["committed_by_here"] = (M.commitment_fraction.notna() &
                              (M.fraction >= M.commitment_fraction)).astype(int)
    M["will_commit"] = M.commitment_fraction.notna().astype(int)

    # Prefix text, for the baseline that matters.
    cps = pd.DataFrame([json.loads(l) for l in
                        open(run_dir / "raw" / "checkpoints.jsonl")])
    cps = cps.drop_duplicates(["example_id", "cot_id", "fraction"])
    M = M.merge(cps[["example_id", "cot_id", "fraction", "prefix_text"]],
                on=["example_id", "cot_id", "fraction"], how="left")
    print(f"  joined: {len(M)} (CoT, checkpoint) rows, "
          f"{M.example_id.nunique()} questions, "
          f"{M.groupby(['example_id','cot_id']).ngroups} CoTs")
    return X, M


# ── ridge, with every alpha nearly free ───────────────────────────────────────

class RidgePath:
    """Ridge over a whole alpha grid from one SVD of the training matrix.

    With n=460 train rows and d=3584 features the economy SVD is n x n, so the
    grid costs one decomposition instead of one fit per alpha.  That is what
    makes honest nested selection affordable across 28 layers x 5 checkpoints.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, standardize: bool = True):
        # On coordinates that _rowspace already produced, standardize MUST be
        # False: those columns are principal directions whose scales carry the
        # singular-value spectrum.  Dividing them by their own std whitens the
        # block, which silently changes the ridge penalty into a different
        # estimator entirely and overfits catastrophically (R^2 ~ -600).
        if standardize:
            self.mu = X.mean(0)
            self.sd = X.std(0); self.sd[self.sd < 1e-8] = 1.0
        else:
            self.mu = np.zeros(X.shape[1]); self.sd = np.ones(X.shape[1])
        Z = (X - self.mu) / self.sd
        self.ybar = float(y.mean())
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        self.s, self.Vt, self.Uty = s, Vt, U.T @ (y - self.ybar)

    def coef(self, alpha: float) -> np.ndarray:
        return self.Vt.T @ (self.s / (self.s ** 2 + alpha) * self.Uty)

    def predict(self, X: np.ndarray, alpha: float) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.coef(alpha) + self.ybar


def _r2(y, yhat) -> float:
    ss = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - yhat) ** 2).sum() / ss) if ss > 0 else float("nan")


def ridge_nested(X, y, g, outer, rng_alphas=ALPHAS) -> float:
    """Out-of-fold R^2; alpha chosen on inner grouped folds of the train part."""
    from sklearn.model_selection import GroupKFold
    oof = np.full(len(y), np.nan)
    for tr, te in outer:
        gi = g[tr]
        n_in = min(4, len(np.unique(gi)))
        if n_in < 2:
            best = float(np.median(rng_alphas))
        else:
            inner = list(GroupKFold(n_splits=n_in).split(X[tr], y[tr], gi))
            sc = np.zeros(len(rng_alphas))
            for itr, ite in inner:
                rp = RidgePath(X[tr][itr], y[tr][itr])
                for j, a in enumerate(rng_alphas):
                    sc[j] += _r2(y[tr][ite], rp.predict(X[tr][ite], a))
            best = float(rng_alphas[int(np.argmax(sc))])
        oof[te] = RidgePath(X[tr], y[tr]).predict(X[te], best)
    return _r2(y, oof)


def _rowspace(Xtr: np.ndarray, *others: np.ndarray):
    """Standardise on Xtr, then rotate into Xtr's row space.

    An L2-penalised linear model's solution always lies in the span of the
    training rows, so with n=460 train rows and d=3584 features, fitting on the
    460-dim rotated coordinates gives IDENTICAL predictions and an identical
    penalty -- V is orthonormal, so ||V a|| = ||a||.  It is an exact
    reparameterisation, not an approximation, and it makes the logistic sweep
    ~8x cheaper, which is what puts 28 layers x 5 checkpoints in reach.
    """
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-8] = 1.0
    Z = (Xtr - mu) / sd
    _, s, Vt = np.linalg.svd(Z, full_matrices=False)
    keep = s > (s.max() * 1e-10 if s.size else 0)
    V = Vt[keep].T
    return (Z @ V,) + tuple(((o - mu) / sd) @ V for o in others)


class FoldViews:
    """Row-space projections for one feature block, cached across targets.

    The projection depends only on X and the fold split, never on y, so the SVDs
    are computed once here and reused by every target sharing this split.  That
    is the difference between minutes and hours over 28 layers x 5 checkpoints:
    naively, each target re-decomposed the same 460x3584 block.

    Both ridge and logistic then run in ~460 dims instead of 3584, exactly -- see
    _rowspace.
    """

    def __init__(self, X, g, outer, n_inner=3):
        from sklearn.model_selection import GroupKFold
        self.folds = []
        for tr, te in outer:
            Ztr, Zte = _rowspace(X[tr], X[te])
            gi = g[tr]
            inners = []
            k = min(n_inner, len(np.unique(gi)))
            if k >= 2:
                for itr, ite in GroupKFold(n_splits=k).split(X[tr], gi, gi):
                    Itr, Ite = _rowspace(X[tr][itr], X[tr][ite])
                    inners.append((itr, ite, Itr, Ite))
            self.folds.append((tr, te, Ztr, Zte, inners))


def ridge_cached(fv, y, alphas=ALPHAS) -> float:
    """Nested-alpha out-of-fold R^2 using pre-projected fold views."""
    oof = np.full(len(y), np.nan)
    for tr, te, Ztr, Zte, inners in fv.folds:
        if inners:
            sc = np.zeros(len(alphas))
            for itr, ite, Itr, Ite in inners:
                rp = RidgePath(Itr, y[tr][itr], standardize=False)
                for j, a in enumerate(alphas):
                    sc[j] += _r2(y[tr][ite], rp.predict(Ite, a))
            best = float(alphas[int(np.argmax(sc))])
        else:
            best = float(np.median(alphas))
        oof[te] = RidgePath(Ztr, y[tr], standardize=False).predict(Zte, best)
    return _r2(y, oof)


def logistic_cached(fv, y, cs=CS):
    """Nested-C out-of-fold (accuracy, AUC) using pre-projected fold views."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    oof = np.full(len(y), np.nan)
    for tr, te, Ztr, Zte, inners in fv.folds:
        best = 0.01
        if inners and len(np.unique(y[tr])) > 1:
            sc = np.zeros(len(cs))
            for itr, ite, Itr, Ite in inners:
                if len(np.unique(y[tr][itr])) < 2:
                    continue
                for j, c in enumerate(cs):
                    m = LogisticRegression(max_iter=3000, C=c).fit(Itr, y[tr][itr])
                    sc[j] += float((m.predict(Ite) == y[tr][ite]).mean())
            best = float(cs[int(np.argmax(sc))])
        oof[te] = LogisticRegression(max_iter=3000, C=best).fit(
            Ztr, y[tr]).predict_proba(Zte)[:, 1]
    acc = float(((oof >= 0.5).astype(int) == y).mean())
    auc = float(roc_auc_score(y, oof)) if len(np.unique(y)) > 1 else float("nan")
    return acc, auc


def logistic_nested(X, y, g, outer, cs=CS) -> tuple[float, float]:
    """Out-of-fold (accuracy, AUC); C chosen on inner grouped folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    def mk(c):
        return LogisticRegression(max_iter=3000, C=c)

    oof = np.full(len(y), np.nan)
    for tr, te in outer:
        gi = g[tr]
        Ztr, Zte = _rowspace(X[tr], X[te])
        n_in = min(3, len(np.unique(gi)))
        if n_in < 2 or len(np.unique(y[tr])) < 2:
            best = 0.01
        else:
            inner = list(GroupKFold(n_splits=n_in).split(Ztr, y[tr], gi))
            sc = np.zeros(len(cs))
            for itr, ite in inner:
                if len(np.unique(y[tr][itr])) < 2:
                    continue
                # Re-derive the row space inside the inner split too, so the
                # held-out inner rows never influence the rotation.
                Itr, Ite = _rowspace(X[tr][itr], X[tr][ite])
                for j, c in enumerate(cs):
                    m = mk(c).fit(Itr, y[tr][itr])
                    sc[j] += float((m.predict(Ite) == y[tr][ite]).mean())
            best = float(cs[int(np.argmax(sc))])
        oof[te] = mk(best).fit(Ztr, y[tr]).predict_proba(Zte)[:, 1]
    acc = float(((oof >= 0.5).astype(int) == y).mean())
    auc = float(roc_auc_score(y, oof)) if len(np.unique(y)) > 1 else float("nan")
    return acc, auc


def text_features(txt_tr, txt_te):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    return v.fit_transform(txt_tr), v.transform(txt_te)


def text_nested(txt, y, g, outer, binary: bool):
    """Same nested protocol for the TF-IDF baseline -- the competitor gets the
    same budget, otherwise the comparison is about tuning effort, not features."""
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    grid = np.logspace(-3, 3, 7) if not binary else CS
    oof = np.full(len(y), np.nan)
    for tr, te in outer:
        gi = g[tr]
        n_in = min(3, len(np.unique(gi)))
        best = grid[len(grid) // 2]
        if n_in >= 2:
            inner = list(GroupKFold(n_splits=n_in).split(txt[tr], y[tr], gi))
            sc = np.zeros(len(grid))
            for itr, ite in inner:
                if binary and len(np.unique(y[tr][itr])) < 2:
                    continue
                Xt, Xe = text_features(txt[tr][itr], txt[tr][ite])
                for j, a in enumerate(grid):
                    if binary:
                        m = LogisticRegression(max_iter=3000, C=a).fit(Xt, y[tr][itr])
                        sc[j] += float((m.predict(Xe) == y[tr][ite]).mean())
                    else:
                        m = Ridge(alpha=a).fit(Xt, y[tr][itr])
                        sc[j] += _r2(y[tr][ite], m.predict(Xe))
            best = float(grid[int(np.argmax(sc))])
        Xt, Xe = text_features(txt[tr], txt[te])
        if binary:
            oof[te] = LogisticRegression(max_iter=3000, C=best).fit(
                Xt, y[tr]).predict_proba(Xe)[:, 1]
        else:
            oof[te] = Ridge(alpha=best).fit(Xt, y[tr]).predict(Xe)
    if binary:
        acc = float(((oof >= 0.5).astype(int) == y).mean())
        auc = float(roc_auc_score(y, oof)) if len(np.unique(y)) > 1 else float("nan")
        return acc, auc
    return _r2(y, oof), float("nan")


def _combo_blocks(txt_tr, txt_te, A_tr, A_te):
    """TF-IDF ++ standardised activations, the dense block norm-matched to the
    sparse one so neither view wins on scale alone before regularisation sees it."""
    import scipy.sparse as sp
    Xt, Xe = text_features(txt_tr, txt_te)
    mu, sd = A_tr.mean(0), A_tr.std(0); sd[sd < 1e-8] = 1.0
    Zt, Ze = (A_tr - mu) / sd, (A_te - mu) / sd
    s = np.sqrt(sp.linalg.norm(Xt) ** 2 / max((Zt ** 2).sum(), 1e-9))
    return (sp.hstack([Xt, sp.csr_matrix(Zt * s)]).tocsr(),
            sp.hstack([Xe, sp.csr_matrix(Ze * s)]).tocsr())


def combo_nested(txt, A, y, g, outer, binary: bool):
    """[TF-IDF prefix text ++ activations] -- does the activation ADD anything?

    Nested-tuned like every other model here.  A combined model handed a fixed
    hyper-parameter while its components were swept would lose for reasons that
    have nothing to do with whether activations carry extra information.
    """
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    grid = CS if binary else np.logspace(-3, 3, 7)
    oof = np.full(len(y), np.nan)
    for tr, te in outer:
        gi = g[tr]
        n_in = min(3, len(np.unique(gi)))
        best = grid[len(grid) // 2]
        if n_in >= 2:
            inner = list(GroupKFold(n_splits=n_in).split(A[tr], y[tr], gi))
            sc = np.zeros(len(grid))
            for itr, ite in inner:
                if binary and len(np.unique(y[tr][itr])) < 2:
                    continue
                Ft, Fe = _combo_blocks(txt[tr][itr], txt[tr][ite],
                                       A[tr][itr], A[tr][ite])
                for j, a in enumerate(grid):
                    if binary:
                        m = LogisticRegression(max_iter=3000, C=a).fit(Ft, y[tr][itr])
                        sc[j] += float((m.predict(Fe) == y[tr][ite]).mean())
                    else:
                        m = Ridge(alpha=a).fit(Ft, y[tr][itr])
                        sc[j] += _r2(y[tr][ite], m.predict(Fe))
            best = float(grid[int(np.argmax(sc))])
        Ft, Fe = _combo_blocks(txt[tr], txt[te], A[tr], A[te])
        if binary:
            oof[te] = LogisticRegression(max_iter=3000, C=best).fit(
                Ft, y[tr]).predict_proba(Fe)[:, 1]
        else:
            oof[te] = Ridge(alpha=best).fit(Ft, y[tr]).predict(Fe)
    if binary:
        acc = float(((oof >= 0.5).astype(int) == y).mean())
        auc = float(roc_auc_score(y, oof)) if len(np.unique(y)) > 1 else float("nan")
        return acc, auc
    return _r2(y, oof), float("nan")


# ── driver ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--act-dir", default="runs/goal2/activations_ckpt/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--run-dir", default="runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--out-dir", default="runs/goal2/probes_ckpt/gpqa_diamond")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--layers", default="all",
                   help="'all' or comma-separated 1-indexed decoder layers")
    p.add_argument("--k100-only", action="store_true",
                   help="restrict to the 40 questions labelled at K=100")
    p.add_argument("--no-combo", action="store_true",
                   help="skip the [text ++ activations] model (the slow part)")
    a = p.parse_args()

    from sklearn.model_selection import GroupKFold

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    X, M = load(Path(a.act_dir), Path(a.run_dir))

    if a.k100_only:
        keep = (M.n_continuations == 100).to_numpy()
        X, M = X[keep], M[keep].reset_index(drop=True)
        print(f"  --k100-only: {len(M)} rows, {M.example_id.nunique()} questions")

    n_layers = X.shape[1]
    layers = (list(range(1, n_layers + 1)) if a.layers == "all"
              else [int(x) for x in a.layers.split(",")])

    res = []
    for frac in sorted(M.fraction.unique()):
        m = (M.fraction == frac).to_numpy()
        Mi, Xi = M[m].reset_index(drop=True), X[m]
        g = Mi.example_id.to_numpy()
        print(f"\n=== checkpoint {frac:.0%}   n={len(Mi)} CoTs, "
              f"{len(set(g))} questions ===")

        # Targets whose usable rows coincide share a fold split, hence share the
        # cached row-space projections.  In practice this collapses to two
        # groups: the regression targets (a few NaN rows) and the binary ones.
        groups: dict[bytes, list[str]] = {}
        for tgt in REG_TARGETS + BIN_TARGETS:
            y_all = Mi[tgt].to_numpy(dtype=float)
            ok = ~np.isnan(y_all)
            binary = tgt in BIN_TARGETS
            if ok.sum() < 50 or (binary and len(np.unique(y_all[ok])) < 2):
                print(f"   {tgt:<24} skipped (n={int(ok.sum())})")
                continue
            groups.setdefault(ok.tobytes(), []).append(tgt)

        for okbytes, tgts in groups.items():
            ok = np.frombuffer(okbytes, dtype=bool)
            Xo, go = Xi[ok], g[ok]
            txt = Mi.prefix_text.fillna("").to_numpy()[ok]
            L = Mi.prefix_tokens.to_numpy(dtype=float)[ok].reshape(-1, 1)
            # One fold assignment shared by every model and target in this group,
            # so all the scores below are paired.
            o = list(GroupKFold(n_splits=a.folds).split(Xo, go, go))

            # Decompose each layer once; every target in this group reuses it.
            print(f"   [projecting {len(layers)} layers x {len(o)} folds ...]")
            fv_len = FoldViews(L, go, o)
            fvs = {L1: FoldViews(Xo[:, L1 - 1, :].astype(np.float32), go, o)
                   for L1 in layers}

            for tgt in tgts:
                binary = tgt in BIN_TARGETS
                y = Mi[tgt].to_numpy(dtype=float)[ok]
                yi = y.astype(int) if binary else y
                base = float(max(y.mean(), 1 - y.mean())) if binary else 0.0

                if binary:
                    s_len, _ = logistic_cached(fv_len, yi)
                    s_txt, _ = text_nested(txt, yi, go, o, True)
                else:
                    s_len = ridge_cached(fv_len, y, np.logspace(-6, 3, 10))
                    s_txt, _ = text_nested(txt, y, go, o, False)

                row = dict(fraction=frac, target=tgt, n=int(ok.sum()),
                           metric="acc" if binary else "R2",
                           baseline=round(base, 4), prefix_len=round(s_len, 4),
                           text=round(s_txt, 4))
                best_l, best_s = None, -np.inf
                for L1 in layers:
                    s = (logistic_cached(fvs[L1], yi)[0] if binary
                         else ridge_cached(fvs[L1], y))
                    row[f"act_L{L1}"] = round(s, 4)
                    if s > best_s:
                        best_s, best_l = s, L1
                row["act_best_layer"] = best_l
                row["act_best"] = round(best_s, 4)

                if not a.no_combo:
                    A = Xo[:, best_l - 1, :].astype(np.float32)
                    c, _ = combo_nested(txt, A, yi, go, o, binary)
                    row["text_plus_act"] = round(c, 4)
                    row["combo_gain_over_text"] = round(c - s_txt, 4)
                res.append(row)
                extra = ("" if a.no_combo
                         else f"  text+act {row['text_plus_act']:>7}")
                print(f"   {tgt:<24} [{row['metric']}]  base {base:>6.3f}  "
                      f"len {s_len:>7.3f}  text {s_txt:>7.3f}  "
                      f"act(L{best_l}) {best_s:>7.3f}{extra}")

    df = pd.DataFrame(res)
    tag = "_k100" if a.k100_only else ""
    df.to_csv(out / f"probe_checkpoints{tag}.csv", index=False)
    print(f"\n  wrote {out / f'probe_checkpoints{tag}.csv'}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
