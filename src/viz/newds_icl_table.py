"""Frontier in-context LLM (GPT-6 Luna) vs our probe on a NEW dataset, same rows, same method.

`src/viz/icl_table.py` is GPQA-hardcoded (its OUT / ICL / RUNS and its "980 scored rows" threshold).
Rather than fork it, this repoints `master_table`'s module constants at runtime and calls the same
`master_table.main()`, so the scoring, the question bootstrap and the paired CIs are literally the
published code. `master_table.py` gained one repointable constant (`ICL_DIR`) for this; the GPQA
`_icl` table was regenerated after that change and matches the published one (mean_score diff 0.0 on
all 105 rows, CI columns within 4e-15).

    ./envs/goal1/bin/python src/viz/newds_icl_table.py --tag globalop_v1
Outputs: runs/goal2/<tag>/sweep/_icl/{ci_scores.csv, ci_paired.csv, table.md}
"""
import argparse, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import master_table as mt

# the three configs runs/newds_icl_luna.sh runs, in its priority order
RUNS = [("GPT-6 Luna, k=0", "gpt6_luna_high_k0"),
        ("GPT-6 Luna, k=10", "gpt6_luna_high_k10"),
        ("GPT-6 Luna, k=0, depth hidden everywhere", "gpt6_luna_high_blind_k0")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="globalop_v1")
    ap.add_argument("--min-rows", type=int, default=2000,
                    help="skip a config with fewer scored 10-90%% rows (500 q x 5 cuts = 2500)")
    a = ap.parse_args()

    mt.S = Path(f"runs/goal2/{a.tag}/sweep")
    mt.ICL_DIR = Path(f"runs/goal2/{a.tag}/icl")
    OUT = mt.S / "_icl"; OUT.mkdir(parents=True, exist_ok=True)

    SPEC = {"Our model L21 (ridge)": ("ref_fixed", "act_L21", None, "probe"),
            "TF-IDF on prefix":      ("ref_fixed", "tfidf", None, "text baseline"),
            "Prefix length":         ("ref_fixed", "prefix_len", None, "text baseline")}
    base_label, N = None, 0
    for lab, tag in RUNS:
        f = mt.ICL_DIR / f"preds_{tag}.csv"
        if not f.exists():
            print(f"skip {lab}: no predictions"); continue
        d = pd.read_csv(f)
        n = len(d[(d.probe == "icl_cal") & (d.target == mt.TARGETS[0]) & (d.fraction > 0)])
        if n < a.min_rows:
            print(f"skip {lab}: only {n} scored 10-90% rows"); continue
        SPEC[lab] = (f"icl:{tag}", tag, "icl_cal", "in-context LLM")
        if n > N:
            base_label, N = lab, n
    assert base_label, "no complete in-context run found"
    print(f"{len(SPEC) - 3} in-context runs; common rows from {base_label} ({N})")

    # pair the depth-hidden run against the matching run that shows depth (the professor's ask)
    REFS = ["Our model L21 (ridge)"] + [l for l in ("GPT-6 Luna, k=0",) if l in SPEC]
    mt.main(SPEC=SPEC, REFS=REFS, OUT=OUT, base_label=base_label,
            title=f"In-context LLMs vs the probe on {a.tag} (calibrated, same rows)")


if __name__ == "__main__":
    main()
