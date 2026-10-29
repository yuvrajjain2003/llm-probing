#!/usr/bin/env python3
"""Re-apply ``parse_output`` to already-generated continuations, in place.

Parsing happens at generation time, so ``parse_success`` / ``final_answer`` /
``is_correct`` are baked into ``continuations.jsonl``. When a parser rule is added
(see the ``ANSWER: X`` fallback in ``run_eval.parse_output``), the fix only reaches
existing runs if those records are re-parsed -- and re-parsing is far cheaper than
re-generating on GPU, since the raw text is already stored.

Replicates ``continuations.py`` exactly: ``parse_output(prefix_text + continuation)``
then ``grade(...)``. Nothing model-derived is touched -- ``continuation_text``,
``terminated`` and ``remaining_reasoning_tokens`` are carried through untouched.

Writes to a temp file and renames, so an interrupted run cannot truncate the
original; ``--backup`` keeps a copy first. ``--dry-run`` reports the delta only.

Usage
-----
    python src/generation/reparse.py --run-dir runs/goal1/base/gpqa_diamond/<model> --dry-run
    python src/generation/reparse.py --run-dir runs/goal1/base/gpqa_diamond/<model> --backup
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# src/ on the path, so `generation.run_eval` resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from generation.run_eval import (  # noqa: E402
    DATASETS, _ANSWER_LINE_RE, build_example_metrics, grade, parse_output,
)


def _load(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]



def _reparse_base(run_dir: Path, *, dry_run: bool, backup: bool) -> None:
    """Re-parse the base CoTs and rebuild the metrics derived from them.

    ``pass@1`` / majority-vote / entropy in ``example_metrics.csv`` and
    ``overall_summary.json`` are all computed from ``final_answer_norm``, so a parser
    change makes them stale until these are rebuilt. Reuses ``run_eval``'s own
    ``build_example_metrics`` so the definitions cannot drift.
    """
    rec_path = run_dir / "raw" / "sample_records.jsonl"
    summ_path = run_dir / "overall_summary.json"
    if not rec_path.exists():
        sys.exit(f"missing: {rec_path}")

    rows = _load(rec_path)
    dataset = rows[0]["dataset"]
    cfg = DATASETS[dataset]

    n_fixed = n_broke = 0
    out = []
    for r in rows:
        parsed = parse_output(r["raw_completion"], cfg.get("answer_line_re", _ANSWER_LINE_RE))
        pred_norm, _, is_correct = grade(cfg, parsed["final_answer"], r["gold_answer"])
        n_fixed += bool(parsed["parse_success"]) and not bool(r["parse_success"])
        n_broke += bool(r["parse_success"]) and not bool(parsed["parse_success"])
        out.append({**r,
                    "final_answer": parsed["final_answer"],
                    "final_answer_norm": pred_norm,
                    "is_correct": is_correct,
                    "parse_success": parsed["parse_success"],
                    "parse_error": parsed["parse_error"]})

    df = pd.DataFrame(out)
    ex = build_example_metrics(df)
    old_summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    new_summ = {**old_summ,
                "pass_at_1": float(df["is_correct"].mean()),
                "mean_fraction_correct": float(ex["fraction_correct"].mean()),
                "majority_vote_accuracy": float(ex["mode_correct"].mean()),
                "mean_answer_entropy_bits": float(ex["answer_entropy_bits"].mean()),
                "mean_reasoning_tokens": float(ex["mean_reasoning_tokens"].mean()),
                "parse_success_rate": float(df["parse_success"].mean())}

    n = len(rows)
    print(f"\n{'─'*60}")
    print(f"  {dataset}  BASE CoTs ({n})")
    print(f"  parse-fail   : {1-old_summ.get('parse_success_rate', float('nan')):.1%}"
          f"  ->  {1-new_summ['parse_success_rate']:.1%}")
    print(f"  newly parsed : {n_fixed}   regressed: {n_broke}   <-- regressed must be 0")
    for k in ("pass_at_1", "majority_vote_accuracy", "mean_answer_entropy_bits"):
        print(f"  {k:24}: {old_summ.get(k, float('nan')):.4f} -> {new_summ[k]:.4f}")
    if dry_run:
        print("  DRY RUN -- nothing written")
        print(f"{'─'*60}\n")
        return
    if backup:
        (rec_path.with_suffix(".jsonl.bak")).write_bytes(rec_path.read_bytes())
    tmp = rec_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, rec_path)
    ex.to_csv(run_dir / "labels" / "example_metrics.csv", index=False)
    summ_path.write_text(json.dumps(new_summ, indent=2), encoding="utf-8")
    print(f"  rewrote {rec_path}, example_metrics.csv, overall_summary.json")
    print(f"{'─'*60}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="dataset x model dir holding raw/continuations.jsonl")
    ap.add_argument("--base", action="store_true",
                    help="re-parse raw/sample_records.jsonl (base CoTs) instead of the "
                         "continuations, and rebuild example_metrics.csv + overall_summary.json")
    ap.add_argument("--dry-run", action="store_true", help="report the delta, write nothing")
    ap.add_argument("--backup", action="store_true",
                    help="copy continuations.jsonl to .bak before overwriting")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if args.base:
        _reparse_base(run_dir, dry_run=args.dry_run, backup=args.backup)
        return

    cont_path = run_dir / "raw" / "continuations.jsonl"
    ckpt_path = run_dir / "raw" / "checkpoints.jsonl"
    for p in (cont_path, ckpt_path):
        if not p.exists():
            sys.exit(f"missing: {p}")

    conts = _load(cont_path)
    # parse_output sees prefix + continuation, exactly as continuations.py fed it
    prefix = {(c["example_id"], c["cot_id"], c["fraction"]): c["prefix_text"]
              for c in _load(ckpt_path)}

    dataset = conts[0]["dataset"]
    cfg = DATASETS[dataset]

    n_fixed = n_broke = n_changed_ans = n_changed_correct = 0
    before_err, after_err = Counter(), Counter()
    out: list[dict] = []

    for r in conts:
        key = (r["example_id"], r["cot_id"], r["fraction"])
        if key not in prefix:
            sys.exit(f"no checkpoint for {key} -- checkpoints.jsonl does not match")
        parsed = parse_output(prefix[key] + r["continuation_text"], cfg.get("answer_line_re", _ANSWER_LINE_RE))
        pred_norm, _, is_correct = grade(cfg, parsed["final_answer"], r["gold_answer"])

        was_ok, now_ok = bool(r["parse_success"]), bool(parsed["parse_success"])
        n_fixed += now_ok and not was_ok
        n_broke += was_ok and not now_ok
        n_changed_ans += was_ok and now_ok and r["final_answer_norm"] != pred_norm
        n_changed_correct += bool(r["is_correct"]) != bool(is_correct)
        before_err[r["parse_error"] or ("ok" if was_ok else "?")] += 1
        after_err[parsed["parse_error"] or ("ok" if now_ok else "?")] += 1

        out.append({**r,
                    "final_answer": parsed["final_answer"],
                    "final_answer_norm": pred_norm,
                    "is_correct": is_correct,
                    "parse_success": parsed["parse_success"],
                    "parse_error": parsed["parse_error"]})

    ok_b = sum(bool(r["parse_success"]) for r in conts)
    ok_a = sum(bool(r["parse_success"]) for r in out)
    n = len(conts)
    print(f"\n{'─'*60}")
    print(f"  {dataset}  ({n} continuations)")
    print(f"  parse-fail : {(n-ok_b)/n:.1%}  ->  {(n-ok_a)/n:.1%}")
    print(f"  newly parsed        : {n_fixed}")
    print(f"  regressed to failure: {n_broke}   <-- must be 0")
    print(f"  answer changed on an already-parsing record: {n_changed_ans}   <-- must be 0")
    print(f"  is_correct changed  : {n_changed_correct}")
    print(f"  parse_error before  : {dict(before_err.most_common(6))}")
    print(f"  parse_error after   : {dict(after_err.most_common(6))}")

    if args.dry_run:
        print("  DRY RUN -- nothing written")
        print(f"{'─'*60}\n")
        return

    if args.backup:
        bak = cont_path.with_suffix(".jsonl.bak")
        bak.write_bytes(cont_path.read_bytes())
        print(f"  backup -> {bak}")

    tmp = cont_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, cont_path)          # atomic: no truncated file on interrupt
    print(f"  rewrote {cont_path}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
