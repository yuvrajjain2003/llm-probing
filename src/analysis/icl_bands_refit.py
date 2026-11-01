#!/usr/bin/env python3
"""Does the band result survive refitting the probe's OUTPUT MAP inside each band?

fig9 subsets the stored out-of-fold predictions. Those come from a ridge fitted on all 198 questions
(minus the held-out fold), so inside a band they carry a population-level intercept and scale that
nobody ever recentred. That is what makes the very-easy entropy R2 so negative. This asks how much of
the band collapse is that artefact: refit an intercept, then an intercept and slope, inside each band
(in-sample, so generous to the probe) and see whether the conclusion moves.

Not done here, and a different question: refitting the probe's 3,584 WEIGHTS inside a band. With
about 39 questions per band that fit would be dominated by noise.

    ./envs/goal1/bin/python src/analysis/icl_bands_refit.py
Output: runs/goal2/gpqa25/sweep/_iclsplit/p6_band_refit.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import icl_decompose as dc

ORDER = ["all questions", "v_hard", "hard", "medium", "easy", "v_easy"]


def main():
    dif = pd.read_csv(dc.DIFF).set_index("example_id")
    P = dc.load()
    rows = []
    for lab in [dc.OURS, "Luna k=0"]:
        for t in ["answer_entropy_bits_mm", "rem_len_mean"]:
            for b in ORDER:
                keep = set(dif.index) if b == "all questions" else set(dif.index[dif.band_quintile == b])
                r = {"as_stored": [], "refit_intercept": [], "refit_intercept_and_slope": []}
                n = 0
                for f in dc.FRACS:
                    x = P[(P.target == t) & (P.fraction.round(3) == f)
                          & (P.example_id.isin(keep)) & (P.label == lab)]
                    y, p = x.y_true.to_numpy(float), x.y_pred.to_numpy(float)
                    sst = ((y - y.mean()) ** 2).sum(); n += len(y)
                    r["as_stored"].append(1 - ((y - p) ** 2).sum() / sst)
                    r["refit_intercept"].append(1 - ((y - (p - p.mean() + y.mean())) ** 2).sum() / sst)
                    r["refit_intercept_and_slope"].append(np.corrcoef(y, p)[0, 1] ** 2)
                rows.append(dict(label=lab, target=t, band=b, rows_per_cut=n // len(dc.FRACS),
                                 **{k: float(np.mean(v)) for k, v in r.items()}))
    R = pd.DataFrame(rows)
    R.to_csv(dc.OUT / "p6_band_refit.csv", index=False)
    for lab in R.label.unique():
        for t in R.target.unique():
            print(f"\n== {lab}, {t} (mean over the 5 cuts) ==")
            print(R[(R.label == lab) & (R.target == t)]
                  .set_index("band")[["rows_per_cut", "as_stored", "refit_intercept",
                                      "refit_intercept_and_slope"]].round(3).to_string())
    print("\nwrote", dc.OUT / "p6_band_refit.csv")


if __name__ == "__main__":
    main()
