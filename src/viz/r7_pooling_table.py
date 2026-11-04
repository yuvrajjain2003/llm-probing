#!/usr/bin/env python3
"""Score the four poolings identically (round 7, item 7).

A thin wrapper: it repoints ``master_table``'s module constants and calls the
PUBLISHED ``master_table.main()``, so the scoring, the 2000-draw shared question
bootstrap and the paired intervals are the same code that produced every other
table in this project.  Nothing here fits a model.

    $PY src/viz/r7_pooling_table.py --tag gpqa25 --variant pilot5

Outputs runs/goal2/<tag>/sweep/r7_pooling/_table_<variant>/{ci_scores,ci_paired,
per_checkpoint}.csv and table.md.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import master_table as MT                                    # noqa: E402  (never edited)

NORM = "answer_entropy_bits_mm_norm"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--variant", required=True, help="suffix of the ridge_/adam_ dirs")
    p.add_argument("--norm-entropy", action="store_true",
                   help="also score answer_entropy_bits_mm_norm (the new datasets)")
    p.add_argument("--targets", nargs="+", default=None,
                   help="restrict the target list (needed when an approach covers only some)")
    p.add_argument("--suffix", default="")
    a = p.parse_args()

    S = Path(f"runs/goal2/{a.tag}/sweep")
    MT.S = S
    out = S / "r7_pooling" / f"_table_{a.variant}{a.suffix}"
    out.mkdir(parents=True, exist_ok=True)
    if a.norm_entropy:
        MT.REG = MT.REG + [NORM]
        MT.TARGETS = MT.REG + ["committed_by_here"]
    if a.targets:
        MT.TARGETS = list(a.targets)
        MT.REG = [t for t in MT.TARGETS if t != "committed_by_here"]

    R, A = f"r7_pooling/ridge_{a.variant}", f"r7_pooling/adam_{a.variant}"
    SPEC = {
        "Last token (ridge, published probe)": (R, "act_L21", None, "closed form"),
        "Mean pool (ridge)":                   (R, "mean_L21", None, "closed form"),
        "Max pool (ridge)":                    (R, "max_L21", None, "closed form"),
        "Last token (Adam)":                   (A, "tok_L21_last", "last_adam", "trained"),
        "Mean pool (Adam)":                    (A, "tok_L21_mean", "mean_adam", "trained"),
        "Max pool (Adam)":                     (A, "tok_L21_max", "max_adam", "trained"),
        "Mean over tokens (Adam, exact control)": (A, "tok_L21_meantok", "meantok_adam", "trained"),
        "Attention pool (Adam)":               (A, "tok_L21_attn", "attn_adam", "trained"),
        "Attention pool (ridge readout)":      (A, "tok_L21_attn", "attn_ridge", "closed form"),
    }
    # keep only the (feature, probe) pairs that actually have predictions on disk
    import pandas as pd
    have = {}
    for sw in {v[0] for v in SPEC.values()}:
        fs = sorted((S / sw).glob("preds*.csv"))
        if fs:
            d = pd.concat([pd.read_csv(x, usecols=["feature", "probe"]) for x in fs])
            have[sw] = set(map(tuple, d.drop_duplicates().to_numpy()))
    SPEC = {k: v for k, v in SPEC.items()
            if v[0] in have and any(f == v[1] and (v[2] is None or pr == v[2])
                                    for f, pr in have[v[0]])}
    REFS = [k for k in ("Last token (ridge, published probe)", "Mean pool (ridge)",
                        "Mean over tokens (Adam, exact control)", "Mean pool (Adam)")
            if k in SPEC]
    print("scoring:", list(SPEC))
    MT.main(SPEC=SPEC, REFS=REFS, OUT=out,
            base_label=list(SPEC)[0],
            title=f"Pooling comparison, {a.tag} ({a.variant})")


if __name__ == "__main__":
    main()
