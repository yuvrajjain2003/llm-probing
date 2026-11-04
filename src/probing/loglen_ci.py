#!/usr/bin/env python3
"""CIs for the log/winsorised length refits, computed from an existing _loglen/preds.csv.

Exists because `loglen_refit.py`'s 90 fits (~1 CPU-hour) completed and were saved, but its CI block
then crashed on `P.transform == tr`: `transform` is a DataFrame METHOD, so that comparison is the
scalar False, every filter came back empty, and `multinomial` got a zero-length pvals. The bug is
fixed in `loglen_refit.py` too; this script recovers the CIs without refitting.

Imports boot / r2 / TARGETS / TRANSFORMS / FRACS / SPEC / B / SEED from loglen_refit, so the
bootstrap algebra is literally the same code, not a copy.

    ./envs/goal1/bin/python src/probing/loglen_ci.py --dir runs/goal2/globalop_v1/sweep/_loglen
Output: <dir>/ci.csv
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.loglen_refit import boot, r2, TARGETS, TRANSFORMS, FRACS, SPEC, B, SEED  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/goal2/globalop_v1/sweep/_loglen")
    ap.add_argument("--fracs", nargs="+", type=float, default=FRACS)
    a = ap.parse_args()
    out = Path(a.dir)
    P = pd.read_csv(out / "preds.csv")
    print(f"{len(P):,} pred rows, {P.example_id.nunique()} questions")

    ci, draws, point = [], {}, {}
    for t in TARGETS:
        for tr in TRANSFORMS:
            for frac in a.fracs:
                x = P[(P["target"] == t) & (P["transform"] == tr)
                      & np.isclose(P["fraction"], frac)]
                eid = x[x["feature"] == "act_L21"].example_id.to_numpy()
                assert len(eid), f"no rows for {t}/{tr}/{frac}"
                qs = np.sort(np.unique(eid)); qi = {int(v): i for i, v in enumerate(qs)}
                W = np.random.default_rng(SEED).multinomial(
                    len(qs), np.ones(len(qs)) / len(qs), size=B).astype(float)
                for fname, _, _ in SPEC:
                    d = x[x["feature"] == fname]
                    assert (d.example_id.to_numpy() == eid).all(), f"row order {fname}"
                    y, p = d.y_true.to_numpy(float), d.y_pred.to_numpy(float)
                    draws[(fname, t, tr, frac)] = boot(y, p, eid, W, qi)
                    point[(fname, t, tr, frac)] = r2(y, p)
            for fname, _, _ in SPEC:
                d = np.mean([draws[(fname, t, tr, f)] for f in a.fracs], 0)
                v = float(np.mean([point[(fname, t, tr, f)] for f in a.fracs]))
                row = dict(feature=fname, target=t, transform=tr, score=v,
                           ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5))
                if fname != "act_L21":
                    b = np.mean([draws[("act_L21", t, tr, f)] for f in a.fracs], 0)
                    lo, hi = np.percentile(d - b, [2.5, 97.5])
                    row.update(diff_vs_ours=v - float(np.mean(
                        [point[("act_L21", t, tr, f)] for f in a.fracs])),
                        diff_lo=lo, diff_hi=hi, significant=bool(lo > 0 or hi < 0))
                ci.append(row)
    C = pd.DataFrame(ci)
    C.to_csv(out / "ci.csv", index=False)
    print("\n== mean over cuts, R2 by scale (95% question-bootstrap CI; diff is vs our probe) ==")
    print(C.round(4).to_string(index=False))
    print("\nwrote", out / "ci.csv")


if __name__ == "__main__":
    main()
