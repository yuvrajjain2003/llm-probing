#!/usr/bin/env python3
"""Smoke probe: can a linear read-out of the activations predict the model's own
final answer, above the 25% chance floor?

This is a pipeline check, not a result. It answers one question -- is there any
decodable signal, i.e. did extraction produce something meaningful -- before any
serious probing work is built on top.

Two things it does NOT cut corners on:

  Grouping. There are ~800k rows but only ~198 questions, and rows within a CoT
  (and the 3 CoTs of a question) are heavily dependent. Splitting rows at random
  would put the same question in train and test and report near-perfect accuracy
  that is pure leakage. CV is grouped by example_id.

  Target. We predict the model's OWN final answer (what it will conclude), not
  the gold answer. That is the early-exit target: if we can tell where it is
  heading, we can stop generating.
"""
from __future__ import annotations

import argparse, glob, json, functools, builtins
print = functools.partial(builtins.print, flush=True)
from pathlib import Path

import numpy as np
import pandas as pd

LAYER_NAMES = {0: 14, 1: 21, 2: 28}          # column index -> decoder layer


def load(act_dir: Path, lo: float, hi: float):
    """Rows whose frac_through_think falls in [lo, hi], with activations."""
    idx_files = sorted(glob.glob(str(act_dir / "index.shard*.parquet")))
    X, meta = [], []
    for ip in idx_files:
        sfx = ip.split("index.")[1].replace(".parquet", "")
        ap = act_dir / f"activations.{sfx}.npy"
        idx = pd.read_parquet(ip)
        keep = idx[(idx.frac_through_think >= lo) & (idx.frac_through_think <= hi)]
        if keep.empty:
            continue
        arr = np.load(ap, mmap_mode="r")
        X.append(np.asarray(arr[keep.row_idx.to_numpy()], dtype=np.float32))
        meta.append(keep)
    return np.concatenate(X), pd.concat(meta, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--act-dir", default="runs/goal2/activations/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--run-dir", default="runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--depth", type=float, nargs=2, default=[0.45, 0.55],
                   help="frac_through_think window to probe")
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X, meta = load(Path(a.act_dir), *a.depth)
    print(f"  rows {X.shape[0]:,}  shape {X.shape}  depth {a.depth[0]}-{a.depth[1]}")

    recs = [json.loads(l) for l in open(Path(a.run_dir) / "raw" / "sample_records.jsonl")]
    lab = {(int(r["example_id"]), int(r["sample_id"])): r.get("final_answer_norm")
           for r in recs}
    meta["y"] = [lab.get((e, c)) for e, c in zip(meta.example_id, meta.cot_id)]
    ok = meta.y.isin(list("ABCD")).to_numpy()
    X, meta = X[ok], meta[ok].reset_index(drop=True)
    y = meta.y.to_numpy()
    groups = meta.example_id.to_numpy()
    print(f"  usable rows {len(y):,}  questions {meta.example_id.nunique()}  "
          f"CoTs {meta.groupby(['example_id','cot_id']).ngroups}")
    print(f"  class balance: {dict(pd.Series(y).value_counts().sort_index())}")
    maj = pd.Series(y).value_counts(normalize=True).max()
    print(f"  chance 0.250 | majority-class {maj:.3f}\n")

    for col in range(X.shape[1]):
        Xi = X[:, col, :]
        accs = []
        for tr, te in GroupKFold(n_splits=a.folds).split(Xi, y, groups):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=0.01,
                                                   multi_class="multinomial"))
            clf.fit(Xi[tr], y[tr])
            accs.append(float((clf.predict(Xi[te]) == y[te]).mean()))
        print(f"  decoder layer {LAYER_NAMES[col]:>2}: "
              f"grouped {a.folds}-fold acc {np.mean(accs):.3f} +- {np.std(accs):.3f}   "
              f"folds {[round(x,3) for x in accs]}")


if __name__ == "__main__":
    main()
