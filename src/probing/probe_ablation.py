#!/usr/bin/env python3
"""Three probe families on identical features: linear, RBF kernel, MLP.

The question this answers is not "how well can we predict" but "does a more
expressive model find structure the linear probe misses".  So the three families
see EXACTLY the same inputs (raw 3,584-dim activations, no dimensionality
reduction) and EXACTLY the same fold assignment.  Only the model class varies --
otherwise a gap could just be the input representation.

  linear   ridge / logistic, alpha chosen on inner folds        (the current probe)
  kernel   kernel ridge with an RBF kernel / RBF-SVC            (fixed non-linear basis)
  mlp      one hidden layer, DELIBERATELY UNREGULARISED         (learned features)

The MLP is meant to overfit.  With ~3,900 training rows and 3,584 inputs it has far
more parameters than data, and that is the point: if it cannot even drive TRAIN error
down there is no non-linear structure to find, whereas high train / low test means
there is structure but not enough data to generalise from it.  Both scores are
reported for every model for exactly this reason -- a weak test score alone cannot
distinguish "no signal" from "could not fit".

Baseline is `prefix_len` (prefix token count), the competitor that matched the
activation probe on remaining-length in earlier work.  No TF-IDF: it is a
bag-of-words with 20k features against the probe's 3,584, so it confounds
information with capacity.
"""
from __future__ import annotations

import argparse, builtins, functools, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.probe_checkpoints import load, RidgePath          # noqa: E402

print = functools.partial(builtins.print, flush=True)

REG_TARGETS = ["answer_entropy_bits_mm", "rem_len_mean", "rem_len_std", "p_mode"]
BIN_TARGETS = ["committed_by_here"]
ALPHAS = np.logspace(0, 6, 13)          # ridge
KR_ALPHAS = np.logspace(-2, 3, 6)       # kernel ridge
KR_GAMMA_SCALE = [0.25, 1.0, 4.0]       # multiples of the 1/(d*var) heuristic
CS = np.logspace(-4, 1, 6)              # logistic


def _r2(y, p):
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _acc(y, p):
    return float((y == p).mean())


# ── RBF kernels: squared distances once, every gamma for free ─────────────────
def _sqdist(A, B):
    """||a-b||^2 without forming the difference: the dominant cost, done once."""
    return (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)


def fit_kernel(Xtr, ytr, Xte, gtr, binary, seed=0):
    """Kernel ridge / RBF-SVC, hyper-parameters chosen on inner folds only."""
    from sklearn.model_selection import GroupKFold
    from sklearn.svm import SVC

    d2_tr = _sqdist(Xtr, Xtr)
    med = np.median(d2_tr[d2_tr > 0]) if (d2_tr > 0).any() else 1.0
    gammas = [g / med for g in KR_GAMMA_SCALE]          # median heuristic

    n_in = min(3, len(np.unique(gtr)))
    inner = list(GroupKFold(n_splits=n_in).split(Xtr, ytr, gtr)) if n_in >= 2 else []

    best, best_s = None, -np.inf
    for gam in gammas:
        K_tr = np.exp(-gam * d2_tr)
        for al in (CS if binary else KR_ALPHAS):
            s = []
            for itr, ite in inner:
                if binary and len(np.unique(ytr[itr])) < 2:
                    continue
                Kii, Koi = K_tr[np.ix_(itr, itr)], K_tr[np.ix_(ite, itr)]
                if binary:
                    m = SVC(C=al, kernel="precomputed").fit(Kii, ytr[itr])
                    s.append(_acc(ytr[ite], m.predict(Koi)))
                else:
                    a = np.linalg.solve(Kii + al * np.eye(len(itr)), ytr[itr])
                    s.append(_r2(ytr[ite], Koi @ a))
            m_ = np.mean(s) if s else -np.inf
            if m_ > best_s:
                best_s, best = m_, (gam, al)
    gam, al = best
    K_tr = np.exp(-gam * d2_tr)
    K_te = np.exp(-gam * _sqdist(Xte, Xtr))
    if binary:
        m = SVC(C=al, kernel="precomputed").fit(K_tr, ytr)
        return m.predict(K_te), m.predict(K_tr), f"gamma*med={gam*med:.2f},C={al:g}"
    a = np.linalg.solve(K_tr + al * np.eye(len(Xtr)), ytr)
    return K_te @ a, K_tr @ a, f"gamma*med={gam*med:.2f},alpha={al:g}"


def _mlp(hidden, iters, alpha, binary, seed=0):
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    kw = dict(hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
              alpha=alpha, learning_rate_init=1e-3, max_iter=iters,
              early_stopping=False, n_iter_no_change=iters, tol=0.0,
              random_state=seed)
    return MLPClassifier(**kw) if binary else MLPRegressor(**kw)


def _fit_quiet(m, X, y):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # convergence warnings are expected at 50 iters
        return m.fit(X, y)


def fit_mlp(Xtr, ytr, Xte, gtr, binary, hidden, iters, alphas, n_inner, seed=0):
    """One hidden layer; L2 strength chosen on inner folds (or fixed if one alpha).

    Unregularised, this reaches train R^2 ~0.99 while test sits at 0.16-0.26 -- it
    memorises all ~3,900 training rows.  That is the capacity/data mismatch: 3,584
    inputs x 256 hidden is ~917k weights behind only 198 independent questions.  The
    L2 grid asks whether penalising that recovers anything; the inner folds are
    grouped by question so the selection cannot see the held-out questions.
    """
    from sklearn.model_selection import GroupKFold
    best = alphas[0]
    if len(alphas) > 1:
        n_in = min(n_inner, len(np.unique(gtr)))
        inner = list(GroupKFold(n_splits=n_in).split(Xtr, ytr, gtr)) if n_in >= 2 else []
        best_s = -np.inf
        for al in alphas:
            s = []
            for itr, ite in inner:
                if binary and len(np.unique(ytr[itr])) < 2:
                    continue
                m = _fit_quiet(_mlp(hidden, iters, al, binary, seed), Xtr[itr], ytr[itr])
                pr = m.predict(Xtr[ite])
                s.append(_acc(ytr[ite], pr) if binary else _r2(ytr[ite], pr))
            if s and np.mean(s) > best_s:
                best_s, best = np.mean(s), al
    m = _fit_quiet(_mlp(hidden, iters, best, binary, seed), Xtr, ytr)
    return m.predict(Xte), m.predict(Xtr), f"h={hidden},iters={iters},l2={best:g}"


def fit_linear(Xtr, ytr, Xte, gtr, binary):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    if binary:
        n_in = min(3, len(np.unique(gtr)))
        inner = list(GroupKFold(n_splits=n_in).split(Xtr, ytr, gtr)) if n_in >= 2 else []
        best, best_s = CS[len(CS) // 2], -np.inf
        for c in CS:
            s = []
            for itr, ite in inner:
                if len(np.unique(ytr[itr])) < 2:
                    continue
                m = LogisticRegression(C=c, max_iter=2000).fit(Xtr[itr], ytr[itr])
                s.append(_acc(ytr[ite], m.predict(Xtr[ite])))
            if s and np.mean(s) > best_s:
                best_s, best = np.mean(s), c
        m = LogisticRegression(C=best, max_iter=2000).fit(Xtr, ytr)
        return m.predict(Xte), m.predict(Xtr), f"C={best:g}"
    rp = RidgePath(Xtr, ytr)
    n_in = min(3, len(np.unique(gtr)))
    from sklearn.model_selection import GroupKFold
    inner = list(GroupKFold(n_splits=n_in).split(Xtr, ytr, gtr)) if n_in >= 2 else []
    # One decomposition per inner fold, reused across the whole alpha grid -- that is
    # the entire point of RidgePath. Rebuilding it inside the alpha loop costs 13x.
    rps = [(RidgePath(Xtr[itr], ytr[itr]), ite) for itr, ite in inner]
    best, best_s = ALPHAS[len(ALPHAS) // 2], -np.inf
    for al in ALPHAS:
        s = [_r2(ytr[ite], rpi.predict(Xtr[ite], al)) for rpi, ite in rps]
        if s and np.mean(s) > best_s:
            best_s, best = np.mean(s), al
    return rp.predict(Xte, best), rp.predict(Xtr, best), f"alpha={best:g}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--act-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--layers", default="1,7,14,21,28")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--families", default="linear,kernel,mlp")
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--mlp-iters", type=int, default=50)
    p.add_argument("--mlp-l2", default="1e-4,1e-2,1,100",
                   help="comma-separated L2 grid for the MLP; a single value fixes it "
                        "(use 0 for the unregularised probe)")
    p.add_argument("--mlp-inner", type=int, default=2,
                   help="inner folds for the MLP's L2 selection")
    p.add_argument("--fractions", default="",
                   help="restrict to these checkpoints, e.g. 0.1 -- lets the five "
                        "checkpoints run as independent concurrent jobs")
    a = p.parse_args()

    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    mlp_alphas = [float(x) for x in a.mlp_l2.split(",")]
    X, M = load(Path(a.act_dir), Path(a.run_dir))
    if a.fractions:
        want = [round(float(x), 6) for x in a.fractions.split(",")]
        keep = M.fraction.round(6).isin(want).to_numpy()
        X, M = X[keep], M[keep].reset_index(drop=True)
        print(f"  restricted to checkpoints {want}: {len(M)} rows")
    layers = [int(x) for x in a.layers.split(",")]
    fams = a.families.split(",")
    rows, t0 = [], time.time()

    for frac in sorted(M.fraction.unique()):
        m = (M.fraction == frac).to_numpy()
        Mi, Xi = M[m].reset_index(drop=True), X[m]
        g = Mi.example_id.to_numpy()
        print(f"\n=== checkpoint {frac:.0%}  n={len(Mi)}  questions={len(set(g))} ===")

        for tgt in REG_TARGETS + BIN_TARGETS:
            binary = tgt in BIN_TARGETS
            y_all = Mi[tgt].to_numpy(dtype=float)
            ok = ~np.isnan(y_all)
            if ok.sum() < 50 or (binary and len(np.unique(y_all[ok])) < 2):
                print(f"  {tgt:<24} skipped (n={int(ok.sum())})"); continue
            y = y_all[ok].astype(int if binary else float)
            go, Lo = g[ok], Mi.prefix_tokens.to_numpy(float)[ok].reshape(-1, 1)
            folds = list(GroupKFold(n_splits=a.folds).split(np.zeros(ok.sum()), y, go))
            base = _acc(y, np.full_like(y, np.bincount(y).argmax())) if binary else 0.0

            # prefix_len baseline, on the same folds
            pl_te = np.zeros(len(y)); pl_tr_s = []
            for tr, te in folds:
                pr, ptr, _ = fit_linear(Lo[tr], y[tr], Lo[te], go[tr], binary)
                pl_te[te] = pr; pl_tr_s.append(_acc(y[tr], ptr) if binary else _r2(y[tr], ptr))
            s_len = _acc(y, pl_te.astype(int)) if binary else _r2(y, pl_te)

            for L1 in layers:
                Z = Xi[ok][:, L1 - 1, :].astype(np.float32)
                fam_scores = {}
                for fam in fams:
                    te_pred = np.zeros(len(y), dtype=float); tr_scores, notes = [], []
                    for tr, te in folds:
                        sc = StandardScaler().fit(Z[tr])
                        Ztr, Zte = sc.transform(Z[tr]), sc.transform(Z[te])
                        if fam == "linear":
                            pt, ptr, nt = fit_linear(Ztr, y[tr], Zte, go[tr], binary)
                        elif fam == "kernel":
                            pt, ptr, nt = fit_kernel(Ztr, y[tr], Zte, go[tr], binary)
                        else:
                            # "mlp_noreg" is the same network with L2 pinned to 0: the
                            # control showing the capacity/data mismatch directly
                            # (train R^2 ~0.99, test ~0.2), against which the L2 row
                            # shows whether penalising it buys anything back.
                            al_grid = [0.0] if fam == "mlp_noreg" else mlp_alphas
                            pt, ptr, nt = fit_mlp(Ztr, y[tr], Zte, go[tr], binary,
                                                  a.mlp_hidden, a.mlp_iters,
                                                  al_grid, a.mlp_inner)
                        te_pred[te] = pt
                        tr_scores.append(_acc(y[tr], ptr.astype(int)) if binary
                                         else _r2(y[tr], ptr))
                        notes.append(nt)
                    s_te = _acc(y, te_pred.astype(int)) if binary else _r2(y, te_pred)
                    s_tr = float(np.mean(tr_scores))
                    fam_scores[fam] = (s_te, s_tr)
                    rows.append(dict(fraction=frac, target=tgt, layer=L1, family=fam,
                                     metric="acc" if binary else "R2",
                                     test=round(s_te, 4), train=round(s_tr, 4),
                                     baseline=round(base, 4), prefix_len=round(s_len, 4),
                                     n=int(ok.sum()), hp=notes[0],
                                     elapsed_min=round((time.time() - t0) / 60, 1)))
                    pd.DataFrame(rows).to_csv(out / "probe_ablation_partial.csv", index=False)
                sm = "  ".join(f"{f}: test {v[0]:.3f} / train {v[1]:.3f}"
                               for f, v in fam_scores.items())
                print(f"  {tgt:<24} L{L1:<3} base {base:.3f}  len {s_len:.3f}  {sm}")

    tag = ("_f" + a.fractions.replace(".", "").replace(",", "_")) if a.fractions else ""
    pd.DataFrame(rows).to_csv(out / f"probe_ablation{tag}.csv", index=False)
    print(f"\nwrote {out/f'probe_ablation{tag}.csv'}  ({len(rows)} rows, "
          f"{(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
