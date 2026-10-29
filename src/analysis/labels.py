#!/usr/bin/env python3
"""Goal 1 step 4: compute behavioral labels from continuation records.

Consumes ``continuations.jsonl`` (from ``continuations.py``) and emits two tables:

  checkpoint_labels.{parquet,csv}
      one row per (dataset, model, example_id, cot_id, fraction).  Answer
      distribution, remaining-reasoning length, and value-of-added-compute.

  commitment.{parquet,csv}
      one row per (dataset, model, example_id, cot_id): the derived commitment
      juncture -- the earliest checkpoint whose modal answer stays modal (and
      concentrated) through every later checkpoint of that CoT.

These are the labels a Goal 2 probe will try to predict.  Nothing here touches a
model; it is a pure reduction over the continuation JSONL, so it is cheap to
rerun as K grows (the K-sufficiency analysis reruns it on subsamples).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

# Running this file as a script puts src/<package>/ on sys.path, not src/, so the
# cross-package imports below would not resolve.  Put src/ on the path explicitly;
# this also lets the module be imported as `<package>.<module>`.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from generation.run_eval import _entropy, _load_jsonl

LOG2 = math.log(2.0)


def _entropy_mm(vals: list[str]) -> float:
    """Miller-Madow bias-corrected entropy, in bits.

    The plug-in estimator ``_entropy`` is biased LOW at small K by roughly
    (m-1)/(2N) nats, m = number of observed answers.  Miller-Madow adds that term
    back.  This matters here because the GPQA run measured K* on a K=100 slice
    and then generated the remaining 80% of questions at K=25 -- a choice that was
    justified by the Miller-Madow criterion (``k_bias_mm``) but which the stored
    plug-in labels never actually applied.  At K=25 with 3 distinct answers the
    plug-in label is ~0.06 bits low, versus the 0.02-bit tolerance the K choice
    was made against.  Both estimators are stored so the difference is auditable
    rather than hidden.
    """
    if not vals:
        return float("nan")
    n = len(vals)
    m = len(Counter(vals))
    return _entropy(vals) + (m - 1) / (2.0 * n * LOG2)

# A checkpoint is "concentrated" when its modal answer holds at least this share
# of valid continuations.  Used only for the derived commitment juncture.
COMMIT_THRESHOLD = 0.8


def _valid_answers(grp: pd.DataFrame) -> list[str]:
    return [a for a in grp["final_answer_norm"].dropna().tolist() if str(a).strip()]


def _majority_correct(answers: list[str], gold: str | None, k: int) -> float | None:
    """Accuracy of a k-continuation majority vote, averaged over which k we pick.

    Deterministic and unbiased for small k: rather than pick an arbitrary subset,
    average the vote outcome over every size-k combination via a light bootstrap
    when exact enumeration is large.  Returns None if fewer than k answers exist.
    """
    n = len(answers)
    if n < k or gold is None:
        return None
    rng = np.random.default_rng(0)
    trials = 400
    hits = 0
    idx = np.arange(n)
    for _ in range(trials):
        pick = [answers[i] for i in rng.choice(idx, size=k, replace=False)]
        mode = Counter(pick).most_common(1)[0][0]
        hits += int(mode == gold)
    return hits / trials


def _checkpoint_row(grp: pd.DataFrame) -> pd.Series:
    valid = _valid_answers(grp)
    n_total = len(grp)
    n_valid = len(valid)
    gold = grp["gold_answer_norm"].iloc[0]
    gold = gold if (isinstance(gold, str) and gold.strip()) else None

    counts = Counter(valid)
    mode_ans, mode_n = (counts.most_common(1)[0] if counts else (None, 0))

    # Remaining reasoning length: only continuations that actually finished.
    term = grp[grp["terminated"] == True]  # noqa: E712
    rem = term["remaining_reasoning_tokens"].to_numpy(dtype=float)

    single_acc = float(grp["is_correct"].mean()) if n_total else float("nan")

    return pd.Series(dict(
        n_continuations=n_total,
        n_valid=n_valid,
        n_terminated=int(len(term)),
        parse_success_rate=float(grp["parse_success"].mean()),
        termination_rate=float((grp["terminated"] == True).mean()),  # noqa: E712
        # 1. answer distribution
        answer_entropy_bits=_entropy(valid),
        answer_entropy_bits_mm=_entropy_mm(valid),
        n_unique_answers=len(counts),
        mode_answer=mode_ans,
        p_mode=(mode_n / n_valid) if n_valid else float("nan"),
        p_gold=(counts.get(gold, 0) / n_valid) if (n_valid and gold) else float("nan"),
        answer_dist=json.dumps(dict(counts)),
        # 2. remaining reasoning length
        rem_len_mean=float(np.mean(rem)) if rem.size else float("nan"),
        rem_len_median=float(np.median(rem)) if rem.size else float("nan"),
        rem_len_std=float(np.std(rem)) if rem.size else float("nan"),
        # 3. value of added compute
        acc_single=single_acc,
        acc_maj5=_majority_correct(valid, gold, 5),
        acc_maj10=_majority_correct(valid, gold, 10),
    ))


def _commitment(ck: pd.DataFrame) -> pd.Series:
    """Earliest fraction whose modal answer stays modal+concentrated afterwards."""
    ck = ck.sort_values("fraction")
    fracs = ck["fraction"].tolist()
    modes = ck["mode_answer"].tolist()
    pmodes = ck["p_mode"].tolist()
    final = modes[-1] if modes else None

    commit_frac = None
    for i, f in enumerate(fracs):
        tail_modes = modes[i:]
        tail_p = pmodes[i:]
        same = all(m == modes[i] and m is not None for m in tail_modes)
        concentrated = all((p is not None and p >= COMMIT_THRESHOLD) for p in tail_p)
        if same and concentrated:
            commit_frac = f
            break

    return pd.Series(dict(
        final_mode_answer=final,
        commitment_fraction=commit_frac,
        committed=commit_frac is not None,
        n_checkpoints=len(fracs),
    ))


def add_norm_entropy(ck: pd.DataFrame, examples_csv: Path) -> pd.DataFrame:
    """Entropy divided by log2(number of options), so questions with different
    option counts are on one 0-1 scale (Miller-Madow can exceed 1 slightly).
    NaN for free-form datasets or when examples.csv is missing."""
    n_opt = {}
    if examples_csv.exists():
        ex = pd.read_csv(examples_csv)
        n_opt = {int(e): len(json.loads(c)) for e, c in zip(ex["example_id"], ex["choices"])
                 if isinstance(c, str)}
    ck["n_options"] = ck["example_id"].map(n_opt).astype(float)
    denom = np.log2(ck["n_options"].where(ck["n_options"] > 1))
    ck["answer_entropy_bits_norm"] = ck["answer_entropy_bits"] / denom
    ck["answer_entropy_bits_mm_norm"] = ck["answer_entropy_bits_mm"] / denom
    return ck


def build_labels(cont_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _load_jsonl(cont_path)
    if df.empty:
        raise ValueError(f"no continuations in {cont_path}")

    keys = ["dataset", "model", "example_id", "cot_id", "fraction"]
    ck = (df.groupby(keys, sort=True).apply(_checkpoint_row, include_groups=False)
          .reset_index())

    cot_keys = ["dataset", "model", "example_id", "cot_id"]
    commit = (ck.groupby(cot_keys, sort=True).apply(_commitment, include_groups=False)
              .reset_index())
    return ck, commit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--continuations", required=True,
                   help="path to continuations.jsonl")
    p.add_argument("--out-dir", default=None,
                   help="defaults to a 'labels' dir alongside continuations.jsonl's "
                        "'raw' dir, i.e. --continuations .../raw/continuations.jsonl "
                        "-> .../labels/")
    args = p.parse_args()

    cont_path = Path(args.continuations)
    out_dir = (Path(args.out_dir) if args.out_dir
               else cont_path.parent.parent / "labels")
    out_dir.mkdir(parents=True, exist_ok=True)

    ck, commit = build_labels(cont_path)
    ck = add_norm_entropy(ck, cont_path.parent / "examples.csv")
    for name, tbl in [("checkpoint_labels", ck), ("commitment", commit)]:
        tbl.to_csv(out_dir / f"{name}.csv", index=False)
        try:
            tbl.to_parquet(out_dir / f"{name}.parquet", index=False)
        except Exception as e:
            print(f"  (parquet skipped for {name}: {e})")

    print(f"\n{'─'*60}")
    print(f"  checkpoints labeled : {len(ck)}")
    print(f"  CoTs (commitment)   : {len(commit)}  "
          f"committed={int(commit['committed'].sum())}")
    print(f"  written to          : {out_dir}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
