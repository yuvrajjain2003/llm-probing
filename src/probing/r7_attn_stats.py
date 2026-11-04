#!/usr/bin/env python3
"""What did the learned attention actually learn? (round 7, item 7)

`attn_probe.py train --pooling attn --save-states` writes, per cut, one row per
(row, fold, seed, target) with the entropy-based effective number of attended
tokens, the largest single weight, and the weight on the LAST token (the one
position the published probe reads).  This summarises them.

The reference points that make the numbers mean something:
    eff_tokens / n_tok  ->  1.0 is exactly uniform, i.e. mean pooling
    max_weight * n_tok  ->  1.0 is exactly uniform
    last_weight * n_tok ->  1.0 means the last token gets its uniform share

    $PY src/probing/r7_attn_stats.py --states DIR [--out CSV]
"""
import argparse, glob
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fs = sorted(glob.glob(str(Path(a.states) / "attn_stats.frac*.csv")))
    if not fs:
        raise SystemExit(f"no attn_stats.frac*.csv under {a.states}")
    D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    D["uniformity"] = D.eff_tokens / D.n_tok            # 1.0 = exactly uniform
    D["max_rel"] = D.max_weight * D.n_tok               # 1.0 = exactly uniform
    D["last_rel"] = D.last_weight * D.n_tok             # 1.0 = uniform share
    g = (D.groupby(["fraction", "target"])
         .agg(n_rows=("n_tok", "size"), n_tok_med=("n_tok", "median"),
              eff_tokens_med=("eff_tokens", "median"),
              uniformity_med=("uniformity", "median"),
              uniformity_p10=("uniformity", lambda x: x.quantile(.10)),
              max_rel_med=("max_rel", "median"),
              last_rel_med=("last_rel", "median"))
         .reset_index())
    pd.set_option("display.width", 220)
    print(g.round(4).to_string(index=False))
    print("\nuniformity = effective attended tokens / tokens available."
          "\n  1.0 means the learned attention IS mean pooling."
          "\n  values near 1 mean the learned weighting adds nothing over uniform."
          "\nlast_rel = weight on the last token, relative to its uniform share"
          " (1.0 = no special treatment).")
    out = Path(a.out) if a.out else Path(a.states) / "attn_stats_summary.csv"
    g.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
