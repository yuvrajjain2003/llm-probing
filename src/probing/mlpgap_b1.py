#!/usr/bin/env python3
"""B1 -- CORRECTNESS CONTROL. Can our SGD-head machinery find the LINEAR optimum at all?

For each outer fold and target: alpha* is chosen exactly as ridge chooses it (inner 3 grouped
folds, logspace(0,6,13)); then a `hidden=0` head is trained on the SAME objective ridge solves,
    loss(w, b) = mean((y - Xw - b)^2) + (alpha*/n_tr) * ||w||^2
which is sklearn's Ridge(alpha*) objective divided by n_tr. Three optimisers:
    lbfgs    full batch, float64, strong-Wolfe line search        -- the control that must pass
    adam     full batch, 1e4 steps, cosine-decayed lr             -- "a real optimiser"
    recipe4a step-4a minibatch AdamW with DECOUPLED wd (lr 1e-4, batch 256, wd 10, patience 30),
             i.e. the recipe every previous MLP/LoRA-head result used -- diagnostic, not a control
The weights are compared with ridge's closed form directly: cosine similarity and relative norm
difference, plus test R2. PASS = cosine > 0.999 and |R2 - R2_ridge| < 1e-3 for lbfgs.

USAGE  $PY src/probing/mlpgap_b1.py --fractions 0.5 [--targets ...] [--max-questions N]
OUT    runs/goal2/gpqa25/sweep/_mlpgap/results.b1<tag>.csv  (+ preds)
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.mlpgap_core import (Data, OUT, REG, score, write, pred_frame, print)  # noqa: E402


def _cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def fit_linear(X, y, alpha, how, seed=0, groups=None, steps=10000, verbose=False):
    """Minimise mean((y - Xw - b)^2) + (alpha/n)||w||^2 (or the step-4a recipe). -> w, b, info"""
    import copy, torch
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", 8)))
    torch.manual_seed(seed)
    dt = torch.float64 if how == "lbfgs" else torch.float32
    ym, ys = float(y.mean()), float(y.std()) or 1.0
    Xt = torch.tensor(X, dtype=dt)
    yt = torch.tensor((y - ym) / ys, dtype=dt)
    n, d = X.shape
    lin = torch.nn.Linear(d, 1, dtype=dt)
    torch.nn.init.zeros_(lin.weight); torch.nn.init.zeros_(lin.bias)
    lam = alpha / n

    def pen():
        return lam * (lin.weight ** 2).sum()

    if how == "lbfgs":
        opt = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=400, history_size=50,
                                tolerance_grad=1e-14, tolerance_change=1e-16,
                                line_search_fn="strong_wolfe")
        last = None
        for it in range(6):
            def closure():
                opt.zero_grad()
                l = ((lin(Xt).squeeze(1) - yt) ** 2).mean() + pen()
                l.backward()
                return l
            last = float(opt.step(closure))
        info = dict(final_loss=last, steps=6 * 400)
    elif how == "adam":
        opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for it in range(steps):
            opt.zero_grad()
            l = ((lin(Xt).squeeze(1) - yt) ** 2).mean() + pen()
            l.backward(); opt.step(); sch.step()
        info = dict(final_loss=float(l), steps=steps)
    elif how == "recipe4a":
        # exactly the historical head recipe: minibatch AdamW, DECOUPLED wd, grouped val early stop
        rng = np.random.default_rng(seed)
        u = np.unique(groups)
        va = np.isin(groups, rng.choice(u, max(1, int(0.15 * len(u))), replace=False))
        tr = np.where(~va)[0]
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-4, weight_decay=10.0)
        best, bstate, bad, ep = np.inf, None, 0, 0
        for ep in range(300):
            for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 256)):
                opt.zero_grad()
                ((lin(Xt[b]).squeeze(1) - yt[b]) ** 2).mean().backward()
                opt.step()
            with torch.no_grad():
                v = float(((lin(Xt[va]).squeeze(1) - yt[va]) ** 2).mean())
            if v < best:
                best, bstate, bad = v, copy.deepcopy(lin.state_dict()), 0
            else:
                bad += 1
                if bad >= 30:
                    break
        lin.load_state_dict(bstate)
        info = dict(final_loss=best, steps=ep + 1)
    else:
        raise ValueError(how)
    w = lin.weight.detach().numpy().ravel().astype(np.float64) * ys
    b = float(lin.bias.detach()) * ys + ym
    return w, b, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.5")
    ap.add_argument("--targets", default=",".join(REG))
    ap.add_argument("--hows", default="lbfgs,adam,recipe4a")
    ap.add_argument("--max-questions", type=int)
    ap.add_argument("--adam-steps", type=int, default=10000)
    ap.add_argument("--ref-alpha", action="store_true",
                    help="take alpha from ref_fixed's stored per-fold inner-CV choice instead of "
                         "re-selecting it. Identical rows, folds and grid, so the same number -- "
                         "but it skips 15 of the 20 SVDs per (fraction, target).")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    D = Data(a.max_questions)
    REF = {}
    if a.ref_alpha:
        r = pd.read_csv("runs/goal2/gpqa25/sweep/ref_fixed/results.csv")
        r = r[(r.feature == "act_L21") & (r.probe == "ridge_act")]
        for x in r.itertuples():
            for i, hp in enumerate(json.loads(x.hp)):
                REF[(round(x.fraction, 3), x.target, i)] = float(hp["alpha"])
    rows, preds, diag = [], [], []
    for frac in [float(x) for x in a.fractions.split(",")]:
        for t in a.targets.split(","):
            sps = D.splits(frac, t, inner=not a.ref_alpha)
            rws = D.last_rows
            y = D.M[t].to_numpy(float)[rws]
            oof = {h: np.full(len(y), np.nan) for h in ["ridge"] + a.hows.split(",")}
            for fi, sp in enumerate(sps):
                al = (REF[(round(frac, 3), t, fi)] if a.ref_alpha else sp.choose_alpha())
                w0, b0 = sp.ridge_coef(al)
                oof["ridge"][sp.te] = sp.Xte @ w0 + b0
                for how in a.hows.split(","):
                    t0 = time.time()
                    w, b, info = fit_linear(sp.Xtr, sp.ytr, al, how, groups=sp.gtr,
                                            steps=a.adam_steps)
                    oof[how][sp.te] = sp.Xte @ w + b
                    diag.append(dict(fraction=round(frac, 3), target=t, fold=fi, how=how,
                                     alpha=al, cos_w=_cos(w, w0),
                                     relnorm=float(abs(np.linalg.norm(w) - np.linalg.norm(w0))
                                                   / np.linalg.norm(w0)),
                                     rel_l2_err=float(np.linalg.norm(w - w0) / np.linalg.norm(w0)),
                                     norm_ridge=float(np.linalg.norm(w0)),
                                     norm_fit=float(np.linalg.norm(w)),
                                     b_ridge=b0, b_fit=b,
                                     r2_te_ridge=score("reg", sp.yte, sp.Xte @ w0 + b0),
                                     r2_te_fit=score("reg", sp.yte, sp.Xte @ w + b),
                                     steps=info["steps"], secs=round(time.time() - t0, 1)))
                    print(f"  {t:<24}f{fi} {how:<9}cos {diag[-1]['cos_w']:.6f} "
                          f"relnorm {diag[-1]['relnorm']:.4f} R2 fit {diag[-1]['r2_te_fit']:.4f} "
                          f"ridge {diag[-1]['r2_te_ridge']:.4f}  {diag[-1]['secs']}s")
                    pd.DataFrame(diag).to_csv(
                        OUT / f"b1_diag{a.tag}.csv", index=False)
            for h, o in oof.items():
                rows.append(dict(variant=f"b1_{h}", fraction=round(frac, 3), target=t, n=len(y),
                                 metric="R2", test=score("reg", y, o), train=np.nan, hp=""))
                preds.append(pred_frame(D, rws, f"b1_{h}", t, frac, y, o))
                print(f"  == {t:<24}{h:<10}pooled test R2 {rows[-1]['test']:.4f}")
            write(rows, preds, "b1" + a.tag)


if __name__ == "__main__":
    main()
