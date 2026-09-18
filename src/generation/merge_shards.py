#!/usr/bin/env python3
"""Concatenate per-shard JSONL into the unsuffixed names the rest of Goal 1 reads.

Sharded workers each append to ``<name>.shard{NN}.jsonl`` so no two processes ever
write one file. This merges them back to ``<name>.jsonl`` -- the paths that
``continuations.py``, ``labels.py``, ``robustness.py`` and ``plot_goal1.py``
already expect -- and rebuilds the metrics that a sharded ``run_eval.py`` skips
(each shard sees only its slice, so only the merged view is meaningful).

Idempotent: rewrites the merged files from the shards every time, so it is safe to
run mid-flight to inspect partial progress and again at the end.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Running this file as a script puts src/<package>/ on sys.path, not src/, so the
# cross-package imports below would not resolve.  Put src/ on the path explicitly;
# this also lets the module be imported as `<package>.<module>`.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation.run_eval import _load_jsonl, build_example_metrics

MERGE_NAMES = ("sample_records", "checkpoints", "continuations")


def merge_one(raw_dir: Path, stem: str, exclude: set[int] | None = None) -> int:
    shards = sorted(raw_dir.glob(f"{stem}.shard*.jsonl"))
    if exclude:
        # A shard still running (or known bad) can be held out: its rows are complete
        # per prefix, so dropping it loses prefixes but never partial ones.
        keep = []
        for sh in shards:
            idx = int(sh.name.split(".shard")[1].split(".")[0])
            (keep.append(sh) if idx not in exclude
             else print(f"  {stem:16s} EXCLUDING {sh.name}"))
        shards = keep
    if not shards:
        return 0
    out = raw_dir / f"{stem}.jsonl"
    n = 0
    # Stream rather than load: continuations.jsonl runs to several GB at K=100.
    with out.open("w", encoding="utf-8") as fh:
        for sh in shards:
            with sh.open("r", encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        fh.write(line)
                        n += 1
    print(f"  {stem:16s} {len(shards)} shards -> {n:,} rows  ({out.name})")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="a <output-dir>/<dataset>/<model_slug> directory")
    p.add_argument("--exclude-shards", default="",
                   help="comma-separated shard indices to leave out of the merge "
                        "(e.g. one still running)")
    p.add_argument("--no-metrics", action="store_true",
                   help="only concatenate; skip example_metrics/overall_summary")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    raw_dir = run_dir / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"no raw/ under {run_dir}")

    print(f"\nmerging shards in {raw_dir}")
    exclude = {int(x) for x in args.exclude_shards.split(",") if x.strip()}
    counts = {stem: merge_one(raw_dir, stem, exclude) for stem in MERGE_NAMES}
    if not any(counts.values()):
        print("  nothing to merge (no .shardNN.jsonl files found)")
        return

    if args.no_metrics or not counts["sample_records"]:
        return

    all_df = _load_jsonl(raw_dir / "sample_records.jsonl")
    labels_dir = run_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    ex_metrics = build_example_metrics(all_df)
    ex_metrics.to_csv(labels_dir / "example_metrics.csv", index=False)

    cfg_path = run_dir / "config" / "run_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    summary = dict(
        dataset=cfg.get("dataset"),
        model=cfg.get("model"),
        model_slug=run_dir.name,
        num_examples=int(ex_metrics["example_id"].nunique()),
        samples_per_example=cfg.get("samples_per_example"),
        total_samples=len(all_df),
        pass_at_1=float(all_df["is_correct"].mean()),
        mean_fraction_correct=float(ex_metrics["fraction_correct"].mean()),
        majority_vote_accuracy=float(ex_metrics["mode_correct"].mean()),
        mean_answer_entropy_bits=float(ex_metrics["answer_entropy_bits"].mean()),
        mean_reasoning_tokens=float(ex_metrics["mean_reasoning_tokens"].mean()),
        parse_success_rate=float(all_df["parse_success"].mean()),
    )
    (run_dir / "overall_summary.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
    print(f"\n  examples         : {summary['num_examples']}")
    print(f"  pass@1           : {summary['pass_at_1']:.1%}")
    print(f"  parse success    : {summary['parse_success_rate']:.1%}")
    print(f"  wrote            : {run_dir}/overall_summary.json\n")


if __name__ == "__main__":
    main()
