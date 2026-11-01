#!/usr/bin/env python3
"""The improvement ladder: MLP variants that are meant to MATCH or BEAT ridge.

All variants share one trainer (`fit_net`). The two things that were broken before are fixed:
  * L2 is an explicit term IN THE LOSS, lambda/n_tr * ||W||^2, on the same logspace(0,6,13) grid
    ridge uses, so lambda has ridge's meaning (hidden=0 + lambda=alpha == Ridge(alpha), see B1).
  * nothing is selected on test: lambda (and anything else swept) is chosen on the inner 3
    grouped folds, exactly as ridge's alpha is.

VARIANTS (--variants)
  ridge                reference, refit here so rows/folds/alpha selection are identical
  mlp_h<H>             MLP(H) with L2-in-loss, lambda on the inner folds              [rung 2]
  skip_h<H>            y = w_ridge.x + g(x): the linear part is INITIALISED at ridge's closed-form
                       solution and penalised at ridge's own alpha*, g's output layer is zero-init,
                       and epoch 0 is a candidate for early stopping -- so the fit starts exactly
                       at ridge and can only be kept if it improves a held-out question split
                       [rung 1, the "MLP >= linear by construction" version]
  resid_h<H>           two-stage: g fitted on CROSS-FITTED ridge residuals (inner 3 folds), test
                       prediction = ridge(full train) + g                              [rung 1b]
  rskip_h<H>           same but the base is ridge's IN-SAMPLE prediction (the naive residual fit),
                       kept to show what cross-fitting is worth
  ens<S>_<variant>     seed-average of <variant> over S seeds                          [rung 3]
Add `_nolam` to use a single lambda (no inner sweep) for cheap exploration.

USAGE
  $PY src/probing/mlpgap_ladder.py --fractions 0.5 --variants ridge,mlp_h256,skip_h256 \
      --targets answer_entropy_bits_mm --tag dev
OUT runs/goal2/gpqa25/sweep/_mlpgap/{results,preds}.<tag>.csv
"""
import argparse, copy, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.mlpgap_core import (Data, OUT, ALPHAS, TARGETS, REG, score, task_of,  # noqa: E402
                                 write, pred_frame, print)
from probing.sweep import _group_folds  # noqa: E402

CS = np.logspace(-4, 1, 6)                 # logistic grid, as in ref_fixed
LAM = np.logspace(2, 6, 9)                 # MLP L2 grid (ridge picks 1e4-3.2e4; below 1e2 is hopeless)


# ---- the one trainer ----------------------------------------------------------

class Square(__import__("torch").nn.Module):
    """g(x) = sum_k c_k (v_k.x)^2 is a 1-hidden-layer net with a SQUARE activation, i.e. exactly
    a low-rank quadratic form -- the architecture that matches what the degree-2 polynomial kernel
    found on rem_len_mean. ReLU can only approximate that, and needs many more units to do it."""
    def forward(self, x):
        return x * x


def fit_net(Xtr, ytr, gtr, task="reg", hidden=256, lam=1e4, lr=1e-3, epochs=300, batch=256,
            patience=30, val_frac=0.15, seed=0, w0=None, b0=0.0, alpha_lin=None,
            train_linear=True, base_tr=None, lam_out=None, act="relu", sched="",
            opt="adam", rounds=12, Xg=None):
    """Fit  pred = [w.x + b] + g(x)  (+ base_tr during training).

    w0 not None   -> the linear part exists, starts at (w0, b0) and is penalised at alpha_lin/n.
                     g's output layer is zero-initialised, so epoch 0 reproduces (w0, b0) exactly
                     and epoch 0 is itself a candidate for early stopping.
    w0 is None    -> pure MLP (plus base_tr if given, which is a frozen offset, i.e. residual fit).
    hidden == 0   -> no MLP at all (linear only).
    patience <= 0 -> NO early stopping and NO validation hold-out: the net is trained on ALL the
                     training rows and regularised only by the explicit L2, which is exactly how
                     ridge is fitted. Early stopping is a second, uncontrolled regulariser that
                     competes with L2 -- with it on, `skip` stops at epoch 0 in every fold (the
                     start point is already a regularised optimum), so L2 never gets tested.
    Xg not None   -> the NONLINEAR head reads these columns instead of all 3,584 (in practice the
                     top-K PCA scores, fitted on the training rows), while the linear part keeps
                     the full feature set. A quadratic form over 3,584 inputs needs 3,584*h
                     parameters from 3,930 rows; over 64 inputs it needs 64*h.
    opt == "lbfgs" -> full-batch L-BFGS in `rounds` blocks of 50 iterations instead of minibatch
                     Adam, so the fit is converged rather than stopped (the B1 lesson applied to
                     the MLP). With patience > 0 the val loss is checked between rounds.
    Returns an object with .predict(X) and .info.
    """
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    reg = task == "reg"
    off = np.zeros(len(ytr)) if base_tr is None else np.asarray(base_tr, float)

    # target scaling: for reg, standardise (y - offset); for bin keep logits/labels as they are
    if reg:
        r = ytr - off
        ym, ys = float(r.mean()), float(r.std()) or 1.0
    else:
        ym, ys = 0.0, 1.0
    # reg: the offset is SUBTRACTED from the target (g predicts the residual).
    # bin: the offset is a frozen LOGIT added inside the forward pass (BCE needs total logits).
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Gt = None if Xg is None else torch.tensor(Xg, dtype=torch.float32)
    dg = d if Xg is None else Xg.shape[1]
    yt = torch.tensor((ytr - off - ym) / ys if reg else ytr, dtype=torch.float32)
    ot = torch.tensor(off, dtype=torch.float32)
    off_fwd = (not reg) and base_tr is not None

    if patience > 0:
        u = np.unique(gtr)
        va = np.isin(gtr, rng.choice(u, max(1, int(val_frac * len(u))), replace=False))
        tr, vi = np.where(~va)[0], np.where(va)[0]
    else:
        tr, vi = np.arange(len(ytr)), np.arange(0)

    mods = {}
    if w0 is not None:
        lin = torch.nn.Linear(d, 1)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor((w0 / ys).reshape(1, -1), dtype=torch.float32))
            lin.bias.fill_(float((b0 - ym) / ys) if reg else float(b0))
        for p in lin.parameters():
            p.requires_grad_(bool(train_linear))
        mods["lin"] = lin
    if hidden:
        g = torch.nn.Sequential(torch.nn.Linear(dg, hidden),
                                Square() if act == "square" else torch.nn.ReLU(),
                                torch.nn.Linear(hidden, 1))
        if w0 is not None or base_tr is not None:
            torch.nn.init.zeros_(g[2].weight); torch.nn.init.zeros_(g[2].bias)
        mods["g"] = g
    assert mods, "need a linear part or a hidden layer"
    net = torch.nn.ModuleDict(mods)
    # --- instrumentation: did g actually MOVE? (positive control for the skip rungs) ---
    init = {}
    if "g" in mods:
        init["w1"] = float(mods["g"][0].weight.detach().norm())
        init["w2"] = float(mods["g"][2].weight.detach().norm())
    if "lin" in mods:
        init["lin"] = float(mods["lin"].weight.detach().norm())

    def _fwd(Xb, ob, Gb=None):
        o = ob if ob is not None else torch.zeros(len(Xb))
        if "lin" in mods:
            o = o + mods["lin"](Xb).squeeze(1)
        if "g" in mods:
            o = o + mods["g"](Xb if Gb is None else Gb).squeeze(1)
        return o

    def fwd(idx):
        return _fwd(Xt[idx], ot[idx] if off_fwd else None,
                    None if Gt is None else Gt[idx])

    lo = lam if lam_out is None else lam_out
    n = len(tr)

    def penalty():
        p = torch.zeros(())
        if "lin" in mods and train_linear:
            p = p + (alpha_lin / n) * (mods["lin"].weight ** 2).sum()
        if "g" in mods:
            p = p + (lam / n) * (mods["g"][0].weight ** 2).sum() \
                  + (lo / n) * (mods["g"][2].weight ** 2).sum()
        return p

    lf = (lambda p, t: ((p - t) ** 2).mean()) if reg else \
         (lambda p, t: torch.nn.functional.binary_cross_entropy_with_logits(p, t))
    params = [p for p in net.parameters() if p.requires_grad]
    optr = torch.optim.Adam(params, lr=lr)
    sc = (torch.optim.lr_scheduler.CosineAnnealingLR(optr, T_max=epochs)
          if sched == "cos" else None)

    def val_loss():
        if not len(vi):
            return np.nan
        net.eval()
        with torch.no_grad():
            return float(lf(fwd(vi), yt[vi]))

    ep = 0
    if opt == "lbfgs":
        lb = torch.optim.LBFGS(params, lr=1.0, max_iter=50, history_size=50,
                               tolerance_grad=1e-11, tolerance_change=1e-13,
                               line_search_fn="strong_wolfe")
        best, bstate, bad, bep = val_loss(), copy.deepcopy(net.state_dict()), 0, 0
        # cache the full-batch tensors once: the strong-Wolfe line search calls the closure
        # several times per iteration, and Xt[tr] would copy the whole 3,930x3,584 block each time
        full = len(tr) == len(ytr)
        Xb = Xt if full else Xt[tr]
        yb_ = yt if full else yt[tr]
        ob_ = (ot if full else ot[tr]) if off_fwd else None
        Gb_ = None if Gt is None else (Gt if full else Gt[tr])
        for ep in range(1, rounds + 1):
            net.train()

            def closure():
                lb.zero_grad()
                l = lf(_fwd(Xb, ob_, Gb_), yb_) + penalty()
                l.backward()
                return l
            lb.step(closure)
            if patience > 0:
                v = val_loss()
                if v < best - 1e-9:
                    best, bstate, bad, bep = v, copy.deepcopy(net.state_dict()), 0, ep
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if patience > 0:
            net.load_state_dict(bstate)
        else:
            with torch.no_grad():
                best, bep = float(lf(_fwd(Xb, ob_, Gb_), yb_) + penalty()), ep
    else:
        best, bstate, bad, bep = val_loss(), copy.deepcopy(net.state_dict()), 0, 0
        bs = len(tr) if batch in (0, None) else batch
        for ep in range(1, epochs + 1):
            net.train()
            for b in np.array_split(rng.permutation(tr), max(1, len(tr) // bs)):
                optr.zero_grad()
                (lf(fwd(b), yt[b]) + penalty()).backward()
                optr.step()
            if sc is not None:
                sc.step()
            if patience > 0:
                v = val_loss()
                if v < best - 1e-9:
                    best, bstate, bad, bep = v, copy.deepcopy(net.state_dict()), 0, ep
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if patience > 0:
            net.load_state_dict(bstate)
        else:
            with torch.no_grad():
                best, bep = float(lf(fwd(tr), yt[tr]) + penalty()), ep
    net.eval()
    diag = dict(n_params_opt=sum(p.numel() for p in params),
                g_in_opt=bool("g" in mods and all(
                    any(p is q for q in params) for p in mods["g"].parameters())))
    with torch.no_grad():
        if "lin" in mods:
            diag["lin_norm_init"] = init.get("lin")
            diag["lin_norm_final"] = float(mods["lin"].weight.norm())
            if w0 is not None:
                w_now = mods["lin"].weight.detach().numpy().ravel() * ys
                diag["lin_rel_change"] = float(np.linalg.norm(w_now - w0)
                                               / max(np.linalg.norm(w0), 1e-30))
        if "g" in mods:
            Gin = Xt if Xg is None else Gt
            pre = mods["g"][0](Gin)                       # pre-activation
            hid = mods["g"][1](pre)                       # post-activation
            out = mods["g"][2](hid).squeeze(1)
            c = mods["g"][2].weight.detach().abs().ravel()
            contrib = (hid.std(0) * c)                    # per-unit contribution scale
            diag.update(
                g_w1_norm_init=init["w1"], g_w1_norm_final=float(mods["g"][0].weight.norm()),
                g_w2_norm_init=init["w2"], g_w2_norm_final=float(mods["g"][2].weight.norm()),
                g_w1_rel_change=float((mods["g"][0].weight.norm() - init["w1"]).abs()
                                      / max(init["w1"], 1e-30)),
                g_out_rms=float(out.pow(2).mean().sqrt()), g_out_max=float(out.abs().max()),
                g_out_rms_vs_y=float(out.pow(2).mean().sqrt() / max(float(yt.std()), 1e-30)),
                frac_units_live=float((contrib > 1e-8).float().mean()),
                frac_units_relu_active=(float((pre > 0).any(0).float().mean())
                                        if act == "relu" else float("nan")))

    class P:
        info = dict(best_epoch=bep, epochs_run=ep, val_loss=best, **diag)

        def predict(self, X, base=None, Xg_te=None):
            with torch.no_grad():
                Z = torch.tensor(X, dtype=torch.float32)
                o = torch.zeros(len(X))
                if "lin" in mods:
                    o = o + mods["lin"](Z).squeeze(1)
                if "g" in mods:
                    o = o + mods["g"](Z if Xg_te is None
                                      else torch.tensor(Xg_te, dtype=torch.float32)).squeeze(1)
                o = o.numpy()
            if reg:
                o = o * ys + ym + (0.0 if base is None else np.asarray(base, float))
                return o
            if base is not None:
                o = o + np.asarray(base, float)
            return 1 / (1 + np.exp(-o))
    return P()


# ---- bases --------------------------------------------------------------------

REF_C = {}


def ref_C(fraction, target, fold):
    """The logistic C that ref_fixed's inner 3-fold CV picked for this (fraction, target, fold).
    Reused rather than re-selected: identical rows, folds and grid, and re-selecting costs ~18
    dense 3,584-dim logistic fits per outer fold."""
    if not REF_C:
        r = pd.read_csv(Path("runs/goal2/gpqa25/sweep/ref_fixed/results.csv"))
        r = r[(r.feature == "act_L21") & (r.probe == "logistic")]
        for x in r.itertuples():
            for i, hp in enumerate(json.loads(x.hp)):
                REF_C[(round(x.fraction, 3), x.target, i)] = float(hp["C"])
    return REF_C[(round(fraction, 3), target, fold)]


def linear_base(sp, task):
    """ridge (reg) or L2 logistic (bin) on sp.Xtr, hyper-parameter chosen on inner folds.
    -> w, b, alpha_equivalent (L2 coefficient in the mean-loss parametrisation, times n)."""
    if task == "reg":
        al = sp.choose_alpha()
        w, b = sp.ridge_coef(al)
        return w, b, al
    from sklearn.linear_model import LogisticRegression
    c = sp.ref_c
    m = LogisticRegression(max_iter=3000, C=c).fit(sp.Xtr, sp.ytr.astype(int))
    # sklearn minimises sum(logloss) + 0.5/C ||w||^2  ==  mean + 0.5/(C n) ||w||^2
    return m.coef_.ravel(), float(m.intercept_[0]), 0.5 / c


def crossfit_base(sp, task, al):
    """Cross-fitted linear predictions on the TRAINING rows (inner 3 grouped folds), so residuals
    have out-of-sample magnitude. Ridge train R2 is 0.48 vs test 0.28, so in-sample residuals
    would mis-state the job left for g."""
    out = np.zeros(len(sp.ytr))
    for itr, ite in _group_folds(len(sp.ytr), sp.gtr, 3):
        Atr, Ate = sp.Xtr[itr], sp.Xtr[ite]
        mu, sd = Atr.mean(0), Atr.std(0); sd = np.where(sd < 1e-8, 1, sd)
        Ztr, Zte = (Atr - mu) / sd, (Ate - mu) / sd
        if task == "reg":
            U, s, Vt = np.linalg.svd(Ztr.astype(np.float64), full_matrices=False)
            yb = float(sp.ytr[itr].mean())
            out[ite] = Zte @ (Vt.T @ (s / (s ** 2 + al) * (U.T @ (sp.ytr[itr] - yb)))) + yb
        else:
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(max_iter=3000, C=0.5 / al).fit(Ztr, sp.ytr[itr].astype(int))
            out[ite] = m.decision_function(Zte)
    return out


# ---- variants -----------------------------------------------------------------

def parse(v):
    """'ens5_skip_h256_nolam' -> (seeds, kind, hidden, sweep_lambda). 'bagK_' = K members, each
    trained on a QUESTION bootstrap of the training rows (rung 6); 'ensK_' = K seeds, same rows."""
    seeds, nolam, bag = 1, False, False
    if v.startswith("bag"):
        seeds, bag, v = int(v[3:].split("_")[0]), True, v.split("_", 1)[1]
    elif v.startswith("ens"):
        seeds, v = int(v[3:].split("_")[0]), v.split("_", 1)[1]
    if v.endswith("_nolam"):
        nolam, v = True, v[:-6]
    kind, h = (v.split("_h") + ["0"])[:2]
    return seeds, kind, int(h), not nolam, bag


def run_variant(sp, task, v, lr, epochs, batch, patience, lam_grid, act="relu", sched="",
                opt="adam", rounds=12, lam_inf=True, g_pca=0):
    """-> (test prediction, train prediction, info dict)"""
    seeds, kind, h, sweep, bag = parse(v)
    w0, b0, al = linear_base(sp, task)
    lin_te, lin_tr = sp.Xte @ w0 + b0, sp.Xtr @ w0 + b0
    if kind == "ridge":
        p, q = lin_te, lin_tr
        if task == "bin":
            p, q = 1 / (1 + np.exp(-p)), 1 / (1 + np.exp(-q))
        return p, q, dict(alpha=al)

    kw = dict(task=task, hidden=h, lr=lr, epochs=epochs, batch=batch, patience=patience,
              act=act, sched=sched, opt=opt, rounds=rounds)
    base_te = base_tr = None
    if kind == "skip":                      # joint: starts exactly at the linear solution
        kw.update(w0=w0, b0=b0, alpha_lin=al, train_linear=True)
    elif kind == "resid":                   # frozen cross-fitted base, g fits the residual
        base_tr, base_te = crossfit_base(sp, task, al), lin_te
    elif kind == "rskip":                   # frozen IN-SAMPLE base (the naive version)
        base_tr, base_te = lin_tr, lin_te
    elif kind != "mlp":
        raise ValueError(v)

    # lam = inf means "no hidden unit at all", i.e. EXACTLY the linear probe. Putting it in the
    # grid is what makes "a linear probe is a special case of an MLP" true operationally: inner-fold
    # selection can fall back to ridge, so the MLP cannot lose to it except through selection noise.
    def gproj(sprt):
        """top-g_pca PCA scores of the (already train-standardised) inputs, train-fitted."""
        if not g_pca:
            return None, None
        Vt = sprt.svd[2][:g_pca]
        A, B = sprt.Xtr @ Vt.T, sprt.Xte @ Vt.T
        sd = A.std(0); sd[sd < 1e-8] = 1.0
        return (A / sd).astype(np.float32), (B / sd).astype(np.float32)

    Gtr, Gte = gproj(sp)
    lams = list(lam_grid) if sweep else [float(np.median(lam_grid))]
    if lam_inf:
        lams = lams + [np.inf]
    if len(lams) > 1:
        sc = np.zeros(len(lams))
        for inn in sp.inners:
            ikw, ib_te = dict(kw), None
            iw, ib, _ = linear_base_fast(inn, task, al)      # one fit, reused below
            if kind in ("resid", "rskip"):
                ikw["base_tr"] = (crossfit_base(inn, task, al) if kind == "resid"
                                  else inn.Xtr @ iw + ib)
                ib_te = inn.Xte @ iw + ib
            elif kind == "skip":
                ikw.update(w0=iw, b0=ib, alpha_lin=al)
            lin_i = inn.Xte @ iw + ib
            if task == "bin":
                lin_i = 1 / (1 + np.exp(-lin_i))
            iGtr, iGte = gproj(inn)
            for j, L in enumerate(lams):
                if not np.isfinite(L):
                    sc[j] += score(task, inn.yte, lin_i)
                    continue
                m = fit_net(inn.Xtr, inn.ytr, inn.gtr, lam=L, seed=0, Xg=iGtr, **ikw)
                sc[j] += score(task, inn.yte, m.predict(inn.Xte, ib_te, iGte))
        lam = float(lams[int(np.argmax(sc))])
    else:
        lam = float(lams[0])

    if not np.isfinite(lam):                 # inner folds chose the linear special case
        p, q = lin_te, lin_tr
        if task == "bin":
            p, q = 1 / (1 + np.exp(-p)), 1 / (1 + np.exp(-q))
        return p, q, dict(alpha=al, lam="inf(linear)", best_epochs=[0], seeds=0)

    ps, qs, eps = [], [], []
    for s_ in range(seeds):
        if bag:                                  # question bootstrap of the TRAINING rows
            rng = np.random.default_rng(1000 + s_)
            u = np.unique(sp.gtr)
            pick = rng.choice(u, len(u), replace=True)
            idx = np.concatenate([np.where(sp.gtr == q)[0] for q in pick])
            kb = dict(kw)
            if "w0" in kb:                       # linear part stays ridge-on-the-full-train
                pass
            m = fit_net(sp.Xtr[idx], sp.ytr[idx], sp.gtr[idx], lam=lam, seed=s_,
                        base_tr=None if base_tr is None else base_tr[idx],
                        Xg=None if Gtr is None else Gtr[idx], **kb)
        else:
            m = fit_net(sp.Xtr, sp.ytr, sp.gtr, lam=lam, seed=s_, base_tr=base_tr,
                        Xg=Gtr, **kw)
        ps.append(m.predict(sp.Xte, base_te, Gte))
        qs.append(m.predict(sp.Xtr, base_tr, Gtr))
        eps.append(m.info["best_epoch"])
        last = m.info
    return (np.mean(ps, 0), np.mean(qs, 0),
            dict(alpha=al, lam=lam, best_epochs=eps, seeds=seeds,
                 **{k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in last.items() if k != "best_epoch"}))


def linear_base_fast(sp, task, al):
    """Linear base at a GIVEN hyper-parameter (used inside inner folds, where re-selecting would
    need a third level of nesting)."""
    if task == "reg":
        w, b = sp.ridge_coef(al)
        return w, b, al
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=3000, C=0.5 / al).fit(sp.Xtr, sp.ytr.astype(int))
    return m.coef_.ravel(), float(m.intercept_[0]), al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.5")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--variants", default="ridge,mlp_h256,skip_h256,resid_h256,rskip_h256")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--act", default="relu", choices=["relu", "square"])
    ap.add_argument("--opt", default="adam", choices=["adam", "lbfgs"])
    ap.add_argument("--rounds", type=int, default=12, help="L-BFGS blocks of 50 iterations")
    ap.add_argument("--g-pca", type=int, default=0,
                    help="the NONLINEAR head reads the top-K PCA scores (linear part keeps 3,584)")
    ap.add_argument("--no-lam-inf", action="store_true",
                    help="drop the lam=inf (pure-linear) candidate from the L2 grid")
    ap.add_argument("--sched", default="", choices=["", "cos"])
    ap.add_argument("--lam-grid", default="2,6,9", help="logspace a,b,n for the MLP L2 grid")
    ap.add_argument("--max-questions", type=int)
    ap.add_argument("--train-questions", type=int, default=0,
                    help="cap the TRAINING questions per outer fold (rung 8 curve)")
    ap.add_argument("--tag", default="ladder")
    a = ap.parse_args()
    lg = [float(x) for x in a.lam_grid.split(",")]
    lam_grid = np.logspace(lg[0], lg[1], int(lg[2]))

    D = Data(a.max_questions)
    rows, preds = [], []
    for frac in [float(x) for x in a.fractions.split(",")]:
        for t in a.targets.split(","):
            task = task_of(t)
            sps = D.splits(frac, t, train_questions=a.train_questions)
            rws = D.last_rows
            y = D.M[t].to_numpy(float)[rws]
            yi = y.astype(int) if task == "bin" else y
            for v in a.variants.split(","):
                vn = (v + ("_sq" if a.act == "square" else "")
                      + ("_cos" if a.sched else "") + ("_lb" if a.opt == "lbfgs" else "")
                      + ("_nostop" if a.patience <= 0 else "")
                      + (f"_p{a.g_pca}" if a.g_pca else ""))
                t0 = time.time()
                oof, trs, infos = np.full(len(y), np.nan), [], []
                for fi, sp in enumerate(sps):
                    sp.ref_c = ref_C(frac, t, fi) if task == "bin" else None
                    for ii in sp.inners:
                        ii.ref_c = sp.ref_c
                    p, q, info = run_variant(sp, task, v, a.lr, a.epochs, a.batch,
                                             a.patience, lam_grid, a.act, a.sched,
                                             a.opt, a.rounds, not a.no_lam_inf, a.g_pca)
                    oof[sp.te] = p
                    trs.append(score(task, sp.ytr, q))
                    infos.append(info)
                    print(f"   {t:<22}{vn:<18}f{fi} {info}  {time.time()-t0:.0f}s")
                rows.append(dict(variant=vn, fraction=round(frac, 3), target=t, n=len(y),
                                 train_questions=a.train_questions,
                                 metric="acc" if task == "bin" else "R2",
                                 test=score(task, yi, oof), train=float(np.mean(trs)),
                                 hp=json.dumps(infos, default=str),
                                 runtime_s=round(time.time() - t0, 1)))
                preds.append(pred_frame(D, rws, vn, t, frac, y, oof))
                print(f"  == {t:<22}{vn:<18}{rows[-1]['metric']} test {rows[-1]['test']:.4f} "
                      f"train {rows[-1]['train']:.4f}  {rows[-1]['runtime_s']}s")
                write(rows, preds, a.tag)


if __name__ == "__main__":
    main()
