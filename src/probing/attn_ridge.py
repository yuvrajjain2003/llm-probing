#!/usr/bin/env python3
"""Ridge / logistic on the ATTENTION-POOLED layer-21 states (round 7, item 7).

Same move the project already made for LoRA in R8 (`lora_ridge.py`, which was
checked to reproduce ref_fixed's ridge exactly on the unadapted states): train
the pooling with Adam, then read it out with the project's own closed-form
probe, using the ref_fixed grids and inner CV.  That removes the optimiser from
the comparison, which matters because round 4 established that an Adam-trained
linear head on these features sits far below ridge.

Fold i's states come from the attention weights learned on fold i's TRAINING
questions only, so there is no leakage; the 5 seeds' out-of-fold predictions are
averaged, as round 4 recommends.

    $PY src/probing/attn_ridge.py --tag gpqa25 --states <dir> --out <dir>
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.sweep import (KEYS, Feature, Ridge, Logistic, Views, run_combo,   # noqa: E402
                           fixed_fold_of)
from probing.attn_probe import rows_frame, paths, SPECS                        # noqa: E402

BIN = "committed_by_here"
GRID = {"reg": (Ridge, [dict(alpha=a) for a in np.logspace(0, 6, 13)]),
        "bin": (Logistic, [dict(C=c) for c in np.logspace(-4, 1, 6)])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, choices=list(SPECS))
    p.add_argument("--tok-sub", default="activations_tokens")
    p.add_argument("--states", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()

    M = rows_frame(a.tag, a.tok_sub)
    fold_of = fixed_fold_of(paths(a.tag, a.tok_sub)["run_dir"], a.folds, 0)
    src, out = Path(a.states), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("states.frac*.npz"))
    if not files:
        raise SystemExit(f"no states under {src}")

    acc = {}
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        rows, X = z["rows"], z["X"].astype(np.float32)
        t = str(z["target"]); fold = int(z["fold"]); seed = int(z["seed"])
        m = M.loc[rows].reset_index(drop=True)
        g = m.example_id.to_numpy()
        te = np.array([fold_of[int(q)] == fold for q in g])
        ok = m[t].notna().to_numpy()
        task = "bin" if t == BIN else "reg"
        y = m[t].to_numpy()[ok]
        y = y.astype(int) if task == "bin" else y.astype(float)
        cls, hps = GRID[task]
        v = Views(Feature(), X[ok], g[ok],
                  [(np.where(~te[ok])[0], np.where(te[ok])[0])])
        oof, _, chosen = run_combo(v, cls, hps, y, g[ok], task)
        q = m.loc[ok, KEYS][te[ok]].copy()
        q["target"] = t; q["y_true"] = y[te[ok]]; q["y_pred"] = oof[te[ok]]
        q["seed"] = seed; q["hp"] = str(chosen[0])
        acc.setdefault(t, []).append(q)
        print(f"  [{i+1}/{len(files)}] {f.name}: {int(te.sum())} test rows, hp {chosen[0]}",
              flush=True)

    P = []
    for t, parts in acc.items():
        d = pd.concat(parts, ignore_index=True)
        e = (d.groupby(KEYS + ["target"], as_index=False)
             .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
                  n_seeds=("seed", "nunique")))
        assert e.n_seeds.nunique() == 1, "seeds do not cover the same rows"
        P.append(e.drop(columns="n_seeds"))
    P = pd.concat(P, ignore_index=True)
    P["feature"] = "tok_L21_attn"; P["probe"] = "attn_ridge"
    P.to_csv(out / "preds_attn_ridge.csv", index=False)

    R = []
    for (t, fr), g in P.groupby(["target", "fraction"]):
        if t == BIN:
            sc = float(((g.y_pred >= .5) == g.y_true).mean()); mt = "acc"
        else:
            sc = float(1 - ((g.y_pred - g.y_true) ** 2).sum() /
                       ((g.y_true - g.y_true.mean()) ** 2).sum()); mt = "r2"
        R.append(dict(feature="tok_L21_attn", probe="attn_ridge", fraction=fr,
                      target=t, n=len(g), metric=mt, test=sc))
    R = pd.DataFrame(R); R.to_csv(out / "results_attn_ridge.csv", index=False)
    print(R.pivot(index="fraction", columns="target", values="test").round(4).to_string())
    print("wrote", out)


if __name__ == "__main__":
    main()
