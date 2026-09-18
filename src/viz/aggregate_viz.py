#!/usr/bin/env python3
"""Roll per-dataset Goal-1 tables into the single JSON that ``plot_goal1.py`` reads.

This is the step that was missing: ``plot_goal1.py`` expects a ``goal1_viz.json``
that nothing in the repo produced, so the figures in ``runs/goal1/figures/`` could
not be regenerated from checked-in code.  This script closes that gap by walking a
sweep dir and reducing each run's ``labels/checkpoint_labels.csv`` and
``robustness/surface_leakage.csv`` down to per-fraction means.

Every value is aggregated **across CoTs within a fraction**, so one row per
checkpoint per dataset.  Reductions are NaN-safe: ``acc_maj10`` is undefined for a
checkpoint with fewer than 10 valid answers (and ``rem_len_mean`` for one where no
continuation terminated), and those checkpoints are skipped rather than poisoning
the mean.

Pure reduction over CSVs -- no GPU, no model.

Usage
-----
python src/viz/aggregate_viz.py                                  # sweep=runs/goal1/base
python src/viz/aggregate_viz.py --sweep-dir runs/goal1/base --out goal1_viz.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Fields pulled straight from surface_leakage.csv. The letter/choice pair is only
# meaningful for multiple-choice datasets; answer_text is the dataset-agnostic one
# (the only signal available for free-form math sets, which have no choices).
LEAK_FIELDS = [
    "mode_letter_declared", "gold_letter_declared",
    "mode_choice_text_present", "gold_choice_text_present",
    "mode_answer_text_present", "gold_answer_text_present",
]


def _sem(s: pd.Series) -> float:
    """Standard error of the mean, NaN-safe. NaN when fewer than 2 observations."""
    v = s.dropna().to_numpy(dtype=float)
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def _clean(x: float) -> float | None:
    """JSON has no NaN literal -- emit null so the plot can skip the point."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def summarise_run(run_dir: Path) -> dict | None:
    """Per-fraction aggregates for one dataset x model dir, or None if unfinished."""
    ck_path = run_dir / "labels" / "checkpoint_labels.csv"
    if not ck_path.exists():
        return None
    ck = pd.read_csv(ck_path)
    if ck.empty:
        return None

    by_fraction = []
    for frac, grp in ck.groupby("fraction", sort=True):
        by_fraction.append(dict(
            fraction=float(frac),
            n_cots=int(len(grp)),
            entropy_mean=_clean(grp["answer_entropy_bits"].mean()),
            entropy_se=_clean(_sem(grp["answer_entropy_bits"])),
            p_mode=_clean(grp["p_mode"].mean()),
            p_gold=_clean(grp["p_gold"].mean()),
            rem_len=_clean(grp["rem_len_mean"].mean()),
            rem_len_se=_clean(_sem(grp["rem_len_mean"])),
            acc_single=_clean(grp["acc_single"].mean()),
            acc_maj5=_clean(grp["acc_maj5"].mean()),
            acc_maj10=_clean(grp["acc_maj10"].mean()),
            # Share of this fraction's checkpoints that actually have 10 parsed
            # answers. acc_maj10 is averaged over only those, so when coverage is
            # low the majority@10 line describes an easier subset than acc_single
            # does -- the two are not comparable until this is ~1.0. Carried into
            # the viz file so a plot can show the confound instead of hiding it.
            maj10_coverage=_clean((grp["n_valid"] >= 10).mean()),
            acc_single_maj10sub=_clean(grp.loc[grp["n_valid"] >= 10, "acc_single"].mean()),
        ))

    leakage = []
    leak_path = run_dir / "robustness" / "surface_leakage.csv"
    if leak_path.exists():
        for r in pd.read_csv(leak_path).sort_values("fraction").itertuples():
            row = dict(fraction=float(r.fraction), n=int(r.n))
            for f in LEAK_FIELDS:
                row[f] = _clean(getattr(r, f, float("nan")))
            leakage.append(row)

    out = dict(by_fraction=by_fraction, leakage=leakage)

    summary_path = run_dir / "overall_summary.json"
    if summary_path.exists():
        out["summary"] = json.loads(summary_path.read_text())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    p.add_argument("--sweep-dir", default=str(repo / "runs" / "goal1" / "base"),
                   help="dir holding <dataset>/<model>/ run folders")
    p.add_argument("--model", default=None,
                   help="model slug to aggregate; defaults to the only one present "
                        "(the viz JSON is keyed by dataset, so one model per file)")
    p.add_argument("--out", default=str(repo / "runs" / "goal1" / "goal1_viz.json"))
    args = p.parse_args()

    sweep = Path(args.sweep_dir)
    runs = sorted(d for d in sweep.glob("*/*") if d.is_dir())
    if not runs:
        raise SystemExit(f"no <dataset>/<model> dirs under {sweep}")

    models = sorted({d.name for d in runs})
    if args.model:
        model = args.model
    elif len(models) == 1:
        model = models[0]
    else:
        raise SystemExit(f"multiple models present ({', '.join(models)}); pass --model")

    data, skipped = {}, []
    for run_dir in runs:
        if run_dir.name != model:
            continue
        dataset = run_dir.parent.name
        summary = summarise_run(run_dir)
        if summary is None:
            skipped.append(dataset)
            continue
        data[dataset] = summary

    if not data:
        raise SystemExit(f"no finished runs for model {model} under {sweep}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"\n{'─'*60}")
    print(f"  model     : {model}")
    print(f"  datasets  : {len(data)}  ({', '.join(sorted(data))})")
    if skipped:
        print(f"  skipped   : {', '.join(skipped)}  (no labels/ yet -- still running?)")
    print(f"  written to: {out_path}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
