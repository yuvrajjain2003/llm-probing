#!/usr/bin/env python3
"""Round 7, item 4: per-cut confidence intervals for the remaining-length probes.

`_ref_ci/per_checkpoint.csv` and `_loglen/results.csv` already hold the per-cut scores, but the only
stored bootstrap covers the MEAN over cuts (that gap is flagged on figure m6). This fills it in by
resampling questions from the saved `_loglen/preds.csv`, which holds out-of-fold predictions for all
three features, both length targets and all three target scales (raw, log1p, winsor99).

Nothing is refitted. The bootstrap algebra is imported from `src/probing/loglen_refit.py` (`boot`,
`r2`), the same code `src/probing/loglen_ci.py` uses for the mean-over-cuts intervals, so the per-cut
numbers here and the published mean-over-cuts numbers come from one implementation.

Control: the `raw` rows of `_loglen/preds.csv` are a refit of the published probes on the published
target, so their per-cut R2 must reproduce `ref_fixed/results.csv`. That check is printed.

    ./envs/goal1/bin/python src/analysis/r7_depth_ci.py
Outputs: runs/goal2/<tag>/sweep/_r7depth/per_cut_ci.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.loglen_refit import boot, r2, TARGETS, TRANSFORMS, FRACS, SPEC, B, SEED  # noqa: E402

R = Path("runs/goal2")
TAGS = ["gpqa25", "mmlupro_v1", "globalop_v1"]
OURS = "act_L21"


def per_cut(tag, fracs):
    out = R / tag / "sweep/_r7depth"
    out.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(R / tag / "sweep/_loglen/preds.csv")
    print(f"\n== {tag}: {len(P):,} pred rows, {P.example_id.nunique()} questions")

    rows, draws, point = [], {}, {}
    for t in TARGETS:
        for tr in TRANSFORMS:
            for frac in fracs:
                x = P[(P["target"] == t) & (P["transform"] == tr)
                      & np.isclose(P["fraction"], frac)]
                eid = x[x["feature"] == OURS].example_id.to_numpy()
                assert len(eid), f"no rows for {t}/{tr}/{frac}"
                qs = np.sort(np.unique(eid))
                qi = {int(v): i for i, v in enumerate(qs)}
                W = np.random.default_rng(SEED).multinomial(
                    len(qs), np.ones(len(qs)) / len(qs), size=B).astype(float)
                for fname, _, _ in SPEC:
                    d = x[x["feature"] == fname]
                    assert (d.example_id.to_numpy() == eid).all(), f"row order {fname}"
                    y, p = d.y_true.to_numpy(float), d.y_pred.to_numpy(float)
                    draws[(fname, t, tr, frac)] = boot(y, p, eid, W, qi)
                    point[(fname, t, tr, frac)] = r2(y, p)
                for fname, _, _ in SPEC:
                    dd = draws[(fname, t, tr, frac)]
                    row = dict(tag=tag, feature=fname, target=t, transform=tr, fraction=frac,
                               n_rows=int((x["feature"] == fname).sum()), n_questions=len(qs),
                               score=point[(fname, t, tr, frac)],
                               ci_lo=float(np.percentile(dd, 2.5)),
                               ci_hi=float(np.percentile(dd, 97.5)))
                    if fname != OURS:
                        b = draws[(OURS, t, tr, frac)]
                        lo, hi = np.percentile(dd - b, [2.5, 97.5])
                        row.update(diff_vs_ours=point[(fname, t, tr, frac)]
                                   - point[(OURS, t, tr, frac)],
                                   diff_lo=float(lo), diff_hi=float(hi),
                                   significant=bool(lo > 0 or hi < 0))
                    rows.append(row)
    C = pd.DataFrame(rows)
    C.to_csv(out / "per_cut_ci.csv", index=False)

    # control: the raw refit must reproduce the published per-cut reference
    ref = pd.read_csv(R / tag / "sweep/ref_fixed/results.csv")
    ref["fraction"] = ref.fraction.round(3)
    m = (C[C["transform"] == "raw"].merge(
        ref[["feature", "target", "fraction", "test"]], on=["feature", "target", "fraction"]))
    print(f"   raw refit vs published ref_fixed, {len(m)} cells: "
          f"max abs diff {np.abs(m.score - m.test).max():.2e}")
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=TAGS)
    ap.add_argument("--fracs", nargs="+", type=float, default=FRACS)
    a = ap.parse_args()
    allc = []
    for tag in a.tags:
        allc.append(per_cut(tag, a.fracs))
    C = pd.concat(allc, ignore_index=True)
    for t in TARGETS:
        for tr in ["raw", "log1p"]:
            print(f"\n== {t}, {tr} scale: R2 per cut (CI half-width in brackets) ==")
            x = C[(C.target == t) & (C["transform"] == tr)].copy()
            x["cell"] = x.apply(
                lambda r: f"{r.score:+.3f} ({(r.ci_hi - r.ci_lo) / 2:.3f})", axis=1)
            print(x.pivot_table(index=["tag", "feature"], columns="fraction", values="cell",
                                aggfunc="first").to_string())


if __name__ == "__main__":
    main()
