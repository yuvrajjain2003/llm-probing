#!/usr/bin/env python3
"""Round 7, item 1: band scores with MEAN SQUARED ERROR beside the R2 columns, all three datasets.

The professor asked for MSE per difficulty band instead of R2, because R2 inside a band divides by
that band's own target variance: GPQA `v_easy` reads -1.090 and MMLU-Pro `easy` -1.049 only because
the truth is nearly constant there. MSE has the same units in every band, and two reference lines
make it readable: the MSE of predicting that band's own mean (what R2 = 0 means) and the MSE of
predicting the full-set mean (how much work band membership alone does).

`src/analysis/icl_bands.py` gained a `wmse` helper and the columns `mse`, `mse_ci_lo`, `mse_ci_hi`,
`rmse`, `mse_band_mean`, `mse_global_mean` (and `mse_diff` / `mse_ci_*` / `mse_significant` in the
paired file). Nothing was removed or renamed. This wrapper repoints that script's output at a NEW
directory, `_r7bands`, so the published `_iclsplit` / `_iclbands` files are not touched, and the
existing columns can be diffed against them (see --verify).

Row set: the ICL row set in every case (1 chain per question per cut), so our probe, the frontier
model and the baselines are on identical rows. GPQA uses icl_decompose's own label set; the new
datasets use newds_bands_icl's.

Band measure, which differs by dataset and must be named whenever these numbers are quoted:
  gpqa25, mmlupro_v1 : quintiles of p_gold (share of the question's rollouts that are correct)
  globalop_v1        : no gold answer, so quintiles of the out-of-fold question-text measure

    ./envs/goal1/bin/python src/analysis/r7_bands_mse.py
    ./envs/goal1/bin/python src/analysis/r7_bands_mse.py --verify
Outputs: runs/goal2/<tag>/sweep/_r7bands/p5_bands_{scores,paired}[.<diff>].csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import icl_decompose as dc          # noqa: E402
import icl_bands as ib              # noqa: E402
import newds_bands_icl as nb        # noqa: E402

R = Path("runs/goal2")
# tag, diff suffix (None = GPQA's unsuffixed naming), published dir to diff against
JOBS = [("gpqa25", None, "_iclsplit"),
        ("mmlupro_v1", "rollout", "_iclbands"),
        ("globalop_v1", "qtext", "_iclbands")]
KEEP_SCORES = ["band", "band_col", "n_questions", "label", "target", "score", "ci_lo", "ci_hi",
               "target_sd", "sd_vs_all", "within_cut_corr", "corr_ci_lo", "corr_ci_hi",
               "score_recentred", "recentred_ci_lo", "recentred_ci_hi"]


def run_one(tag, suffix):
    """Run icl_bands.main() for one dataset, writing into <tag>/sweep/_r7bands."""
    S = R / tag / "sweep"
    out = S / "_r7bands"
    out.mkdir(parents=True, exist_ok=True)
    if tag == "gpqa25":
        diff_path = S / "_difficulty/question_difficulty.csv"
        dc.OUT, dc.DIFF = out, diff_path
        ib.OUT = out
        sys.argv = ["icl_bands.py", "--band-col", "band_quintile"]
        ib.main()
    else:
        diff_path = (S / "_difficulty/question_difficulty.csv") if suffix == "rollout" \
            else (S / "_qtext/question_difficulty.csv")
        P, reg, has_bin = nb.build(tag)
        targets = reg + ([nb.BIN] if has_bin else [])
        dc.OUT, dc.DIFF = out, diff_path
        dc.load = lambda: P
        ib.OUT, ib.LABELS, ib.TARGETS, ib.REG = out, nb.LABELS, targets, reg
        sys.argv = ["icl_bands.py", "--band-col", "band_quintile"]
        ib.main()
    if suffix:
        for nm in ["p5_bands_scores", "p5_bands_paired"]:
            (out / f"{nm}.csv").rename(out / f"{nm}.{suffix}.csv")
    print(f"\n### {tag}: wrote {out}\n")


def verify():
    """Every column the published file already had must be reproduced exactly."""
    ok = True
    for tag, suffix, pub in JOBS:
        sfx = f".{suffix}" if suffix else ""
        new = pd.read_csv(R / tag / "sweep/_r7bands" / f"p5_bands_scores{sfx}.csv")
        old = pd.read_csv(R / tag / "sweep" / pub / f"p5_bands_scores{sfx}.csv")
        cols = [c for c in KEEP_SCORES if c in old.columns]
        k = ["band", "label", "target"]
        a = old[cols].sort_values(k).reset_index(drop=True)
        b = new[cols].sort_values(k).reset_index(drop=True)
        assert list(a.columns) == list(b.columns) and len(a) == len(b), f"{tag}: shape"
        num = [c for c in cols if pd.api.types.is_numeric_dtype(a[c])]
        # the published files already hold NaN in a few CI cells (degenerate bootstrap draws);
        # require the NaN pattern to match and the finite values to be identical
        same_nan = bool((a[num].isna() == b[num].isna()).to_numpy().all())
        d = float(np.nanmax(np.abs((a[num] - b[num]).to_numpy())))
        same_str = all((a[c] == b[c]).all() for c in cols if c not in num)
        print(f"{tag:12s} {len(a):4d} rows, {len(cols)} published columns: "
              f"max abs diff {d:.3e}, NaN pattern identical={same_nan}, "
              f"text columns identical={same_str}")
        ok = ok and d == 0.0 and same_str and same_nan
    print("\nVERIFY:", "PASS, published columns byte-equal" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="only diff against the published files")
    ap.add_argument("--tags", nargs="+", default=[j[0] for j in JOBS])
    a = ap.parse_args()
    if not a.verify:
        for tag, suffix, _ in JOBS:
            if tag in a.tags:
                run_one(tag, suffix)
    verify()
