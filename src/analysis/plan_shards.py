#!/usr/bin/env python3
"""Pick a safe concurrency, and a shard order that keeps the tail short.

Two failures on the 2026-09-26 GPQA run motivate this, and both were avoidable:

  CONCURRENCY WAS A CONSTANT.  `max_num_seqs=256` was tuned on a benchmark whose
  sequences were ~1,200 tokens.  The slowest shard's contexts averaged ~5,700, so its
  working set was 256 x 5,700 = 1.46M tokens against a 923k-token KV cache -- 158%.
  vLLM preempted continuously: the shard ran 3.5x slower than its peers AND tripped
  `assert len(seqs) == 1` (vLLM's n>1-group-preemption bug), failing whole shards.
  `concurrency` below sizes the running set from the ACTUAL prefix lengths so demand
  stays under a target fraction of the cache.

  SHARDS RAN IN INDEX ORDER.  With 16 shards over 198 questions, the unluckiest shard
  held the entire run hostage: 15 of 16 finished in a day, the last took another day
  on its own.  `order` sorts shards by estimated cost, longest first -- the standard
  fix for this (longest-processing-time-first): the expensive work starts while there
  are still workers to overlap it with, so the tail is a short shard, not a long one.

    plan_shards.py concurrency --run-dir R --num-shards 16 --shard-index 3
    plan_shards.py order       --run-dir R --num-shards 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Measured on an H100 80GB at gpu_memory_utilization=0.85 for Qwen-7B: vLLM reported
# 57,717 GPU blocks x 16 tokens.  Override for another GPU/model/utilisation.
DEFAULT_KV_TOKENS = 923_000


def _prefix_tokens(run_dir: Path, num_shards: int, shard_index: int | None) -> np.ndarray:
    """Prefix lengths (tokens) for a shard, from checkpoints.jsonl if it exists."""
    f = run_dir / "raw" / "checkpoints.jsonl"
    shards = sorted((run_dir / "raw").glob("checkpoints.shard*.jsonl"))
    out = []
    for path in ([f] if f.exists() else shards):
        with path.open() as fh:
            for line in fh:
                r = json.loads(line)
                if shard_index is None or int(r["example_id"]) % num_shards == shard_index:
                    out.append(int(r["token_offset"]))
    return np.asarray(out, dtype=float)


def _base_cost(run_dir: Path) -> dict[int, float]:
    """Per-example generation cost, from the base CoTs (available before continuations)."""
    cost: dict[int, float] = {}
    f = run_dir / "raw" / "sample_records.jsonl"
    paths = [f] if f.exists() else sorted((run_dir / "raw").glob("sample_records.shard*.jsonl"))
    for path in paths:
        with path.open() as fh:
            for line in fh:
                r = json.loads(line)
                cost[int(r["example_id"])] = cost.get(int(r["example_id"]), 0.0) + \
                    float(r.get("reasoning_tokens") or 0)
    return cost


def cmd_concurrency(a: argparse.Namespace) -> int:
    pre = _prefix_tokens(Path(a.run_dir), a.num_shards, a.shard_index)
    if pre.size == 0:
        print(a.fallback)                      # nothing to measure yet: stay conservative
        return 0
    ctx = pre.mean() + a.gen_tokens            # a running sequence holds prefix + what it generates
    n = int(a.target * a.kv_tokens / max(ctx, 1.0))
    n = max(a.min_seqs, min(a.max_seqs, n))
    print(n)
    if a.explain:
        print(f"  prefixes: {pre.size:,}  mean {pre.mean():,.0f}  p90 {np.quantile(pre,0.9):,.0f} tokens",
              file=sys.stderr)
        print(f"  context/seq ~{ctx:,.0f}; {n} seqs => {n*ctx/1000:,.0f}k tokens "
              f"= {n*ctx/a.kv_tokens:.0%} of a {a.kv_tokens/1000:,.0f}k cache", file=sys.stderr)
    return 0


def cmd_order(a: argparse.Namespace) -> int:
    cost = _base_cost(Path(a.run_dir))
    if not cost:
        print(" ".join(str(i) for i in range(a.num_shards)))   # no data: natural order
        return 0
    per = np.zeros(a.num_shards)
    for eid, c in cost.items():
        per[eid % a.num_shards] += c
    order = np.argsort(-per)                                   # most expensive first
    print(" ".join(str(int(i)) for i in order))
    if a.explain:
        print(f"  shard cost spread: max {per.max()/1e6:.1f}M vs min {per.min()/1e6:.1f}M "
              f"base-CoT tokens ({per.max()/max(per.min(),1):.1f}x)", file=sys.stderr)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("concurrency", "order"):
        q = sub.add_parser(name)
        q.add_argument("--run-dir", required=True)
        q.add_argument("--num-shards", type=int, required=True)
        q.add_argument("--explain", action="store_true")
    c = sub.choices["concurrency"]
    c.add_argument("--shard-index", type=int, default=None)
    c.add_argument("--kv-tokens", type=int, default=DEFAULT_KV_TOKENS)
    c.add_argument("--target", type=float, default=0.75,
                   help="fraction of the KV cache the running set may occupy")
    c.add_argument("--gen-tokens", type=int, default=2800,
                   help="tokens a continuation is expected to generate")
    c.add_argument("--min-seqs", type=int, default=32)
    c.add_argument("--max-seqs", type=int, default=256)
    c.add_argument("--fallback", type=int, default=128)
    a = p.parse_args()
    sys.exit({"concurrency": cmd_concurrency, "order": cmd_order}[a.cmd](a))


if __name__ == "__main__":
    main()
