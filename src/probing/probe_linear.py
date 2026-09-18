#!/usr/bin/env python3
"""Linear probes on GPQA activations, as a function of depth through the CoT.

Fixes three problems with the first smoke attempt:

  1. UNIT OF ANALYSIS. Treating each token position as a sample gave ~36k rows
     from only 576 CoTs / 197 questions -- massive duplication of very few
     independent labels. Here activations are mean-pooled within a depth bin, so
     one CoT contributes exactly one vector per (bin, layer).
  2. REGULARISATION. C was fixed at 0.01 with no justification. With n~576 and
     d=3584 the fit is badly underdetermined, so C is swept.
  3. NO CONTROL. A null result is uninterpretable without something that must
     succeed. `depth` is that control: early vs late activations are trivially
     separable, so if it fails the probe code is broken, not the hypothesis.

Targets:
  answer  -- the model's OWN final answer (4-class). The early-exit target: can
             we tell where it is heading before it gets there?
  correct -- will this CoT be right (binary)
  depth   -- positive control (early vs late half of the CoT)

CV is grouped by example_id throughout: the 3 CoTs of a question share its text,
so a random split would leak.
"""
from __future__ import annotations

import argparse, functools, builtins, glob, json
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(builtins.print, flush=True)
LAYERS = {0: 14, 1: 21, 2: 28}


def pooled(act_dir: Path, n_bins: int):
    """Mean-pool activations within each (CoT, depth-bin). Streams shard by shard."""
    vecs, keys = [], []
    for ip in sorted(glob.glob(str(act_dir / "index.shard*.parquet"))):
        sfx = ip.split("index.")[1].replace(".parquet", "")
        arr = np.load(act_dir / f"activations.{sfx}.npy", mmap_mode="r")
        idx = pd.read_parquet(ip)
        # clip so frac==1.0 lands in the last bin, not one past it
        idx["bin"] = np.minimum((idx.frac_through_think * n_bins).astype(int), n_bins - 1)
        for (eid, cid, b), g in idx.groupby(["example_id", "cot_id", "bin"]):
            rows = np.asarray(arr[g.row_idx.to_numpy()], dtype=np.float32)
            vecs.append(rows.mean(axis=0))
            keys.append((int(eid), int(cid), int(b)))
    X = np.stack(vecs)
    K = pd.DataFrame(keys, columns=["example_id", "cot_id", "bin"])
    return X, K


def labels(run_dir: Path):
    recs = [json.loads(l) for l in open(run_dir / "raw" / "sample_records.jsonl")]
    return {(int(r["example_id"]), int(r["sample_id"])):
            (r.get("final_answer_norm"), bool(r.get("is_correct"))) for r in recs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--act-dir", default="runs/goal2/activations/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--run-dir", default="runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--out-dir", default="runs/goal2/probes/gpqa_diamond")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--cs", type=float, nargs="+",
                   default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    a = p.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    cache = out / f"pooled_{a.bins}bins.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        X, K = z["X"], pd.DataFrame(z["K"], columns=["example_id", "cot_id", "bin"])
        print(f"  loaded cached pooled features {X.shape}")
    else:
        print(f"  pooling activations into {a.bins} depth bins ...")
        X, K = pooled(Path(a.act_dir), a.bins)
        np.savez(cache, X=X, K=K.to_numpy())
        print(f"  pooled {X.shape}  ({X.nbytes/2**20:.0f} MB)")

    lab = labels(Path(a.run_dir))
    K["answer"] = [lab.get((e, c), (None, None))[0] for e, c in zip(K.example_id, K.cot_id)]
    K["correct"] = [lab.get((e, c), (None, None))[1] for e, c in zip(K.example_id, K.cot_id)]
    K["depth"] = (K["bin"] >= a.bins // 2).astype(int)      # positive control

    rows = []
    for target in ("answer", "correct", "depth"):
        print(f"\n=== target: {target} ===")
        for b in range(a.bins):
            m = (K["bin"] == b).to_numpy() if target != "depth" else np.ones(len(K), bool)
            sub = K[m]
            keep = sub[target].notna().to_numpy()
            Xi_all, y_all = X[m][keep], sub[target].to_numpy()[keep]
            g_all = sub.example_id.to_numpy()[keep]
            if len(np.unique(y_all)) < 2:
                continue
            base = pd.Series(y_all).value_counts(normalize=True).max()
            for col in range(X.shape[1]):
                Xi = Xi_all[:, col, :]
                best = None
                for C in a.cs:
                    accs = []
                    for tr, te in GroupKFold(n_splits=a.folds).split(Xi, y_all, g_all):
                        clf = make_pipeline(StandardScaler(),
                                            LogisticRegression(max_iter=3000, C=C))
                        clf.fit(Xi[tr], y_all[tr])
                        accs.append(float((clf.predict(Xi[te]) == y_all[te]).mean()))
                    m_ = float(np.mean(accs))
                    if best is None or m_ > best[0]:
                        best = (m_, float(np.std(accs)), C)
                rows.append(dict(target=target, bin=b, layer=LAYERS[col], n=len(y_all),
                                 baseline=round(base, 3), acc=round(best[0], 4),
                                 sd=round(best[1], 4), best_C=best[2]))
                print(f"   bin {b} ({b/a.bins:.0%}-{(b+1)/a.bins:.0%})  layer {LAYERS[col]:>2}  "
                      f"n={len(y_all):>4}  base {base:.3f}  acc {best[0]:.3f} "
                      f"+-{best[1]:.3f}  (C={best[2]})")
            if target == "depth":
                break          # control uses all bins at once
    df = pd.DataFrame(rows)
    df.to_csv(out / "probe_results.csv", index=False)
    print(f"\n  wrote {out/'probe_results.csv'}")


if __name__ == "__main__":
    main()
