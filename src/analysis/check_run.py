#!/usr/bin/env python3
"""Completeness checks between pipeline stages, so a dependency chain stops on a bad
or partial stage instead of feeding it downstream.  Pure CPU, no model.

    check_run.py base        --run-dir R --n-cots N
    check_run.py labels      --run-dir R --k K
    check_run.py activations --run-dir R --act-dir A --n-shards S

Each prints what it found and exits 0 (ok) or 1 (fail), so a job script can gate on it.
`activations` is the check on the actual deliverable: the extracted vectors must be
finite, non-degenerate, the right shape, and 1:1 with checkpoint_labels.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_LAYERS = 28
KEY = ["example_id", "cot_id", "fraction"]


def cmd_base(a: argparse.Namespace) -> int:
    run = Path(a.run_dir)
    n_ex = len(pd.read_csv(run / "raw" / "examples.csv"))
    with open(run / "raw" / "sample_records.jsonl") as fh:
        n = sum(1 for _ in fh)
    print(f"base CoTs: {n} merged; expected {n_ex} questions x {a.n_cots} = {n_ex * a.n_cots}")
    if n != n_ex * a.n_cots:
        print("FAIL: base run is incomplete or over-full")
        return 1
    return 0


def cmd_labels(a: argparse.Namespace) -> int:
    run = Path(a.run_dir)
    lab = pd.read_csv(run / "labels" / "checkpoint_labels.csv")
    term_ex, n_term = set(), 0
    with open(run / "raw" / "sample_records.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if "</think>" in r["raw_completion"]:
                term_ex.add(int(r["example_id"]))
                n_term += 1
    lab_ex = set(lab.example_id.astype(int))
    n_cots = lab.groupby(["example_id", "cot_id"]).ngroups
    print(f"label rows (prefixes): {len(lab)}   CoTs labelled: {n_cots} of {n_term} terminated "
          f"base CoTs   questions: {len(lab_ex)}")
    print("n_continuations per prefix:", lab.n_continuations.value_counts().sort_index().to_dict())
    ok = True
    bad = int((lab.n_continuations != a.k).sum())
    if bad:
        print(f"FAIL: {bad} prefixes do not have exactly K={a.k} continuations")
        ok = False
    if lab_ex != term_ex:
        print(f"FAIL: questions missing from labels: {sorted(term_ex - lab_ex)[:10]}  "
              f"extra: {sorted(lab_ex - term_ex)[:10]}")
        ok = False
    if n_cots != n_term:
        print(f"note: {n_term - n_cots} terminated CoTs produced no labelled prefix "
              "(empty or one-sentence CoTs?) -- inspect")
    return 0 if ok else 1


def cmd_activations(a: argparse.Namespace) -> int:
    act_dir, run = Path(a.act_dir), Path(a.run_dir)
    idx_files = sorted(act_dir.glob("index.shard*.parquet"))
    act_files = sorted(act_dir.glob("activations.shard*.npy"))
    ok = True
    if len(idx_files) != a.n_shards or len(act_files) != a.n_shards:
        print(f"FAIL: found {len(idx_files)} index / {len(act_files)} activation files, "
              f"expected {a.n_shards} shards")
        return 1

    frames, hidden, n_bad_rows = [], set(), 0
    for fi, fa in zip(idx_files, act_files):
        idx = pd.read_parquet(fi)
        arr = np.load(fa, mmap_mode="r")
        if arr.ndim != 3 or arr.shape[0] != len(idx) or arr.shape[1] != N_LAYERS:
            print(f"FAIL: {fa.name} shape {arr.shape} vs {len(idx)} index rows / {N_LAYERS} layers")
            ok = False
            continue
        if arr.dtype != np.float16:
            print(f"FAIL: {fa.name} dtype {arr.dtype}, expected float16")
            ok = False
        hidden.add(arr.shape[2])
        for s in range(0, len(arr), 128):                     # block-wise: never the whole store
            blk = np.asarray(arr[s:s + 128], dtype=np.float32)
            if not np.isfinite(blk).all():
                print(f"FAIL: {fa.name} rows {s}..{s + len(blk)} contain NaN/Inf")
                ok = False
            n_bad_rows += int((np.abs(blk).max(axis=(1, 2)) == 0).sum())
        frames.append(idx)
    if n_bad_rows:
        print(f"FAIL: {n_bad_rows} checkpoint vectors are all zeros")
        ok = False
    if not frames:
        return 1
    idx = pd.concat(frames, ignore_index=True)
    idx["fraction"] = idx["fraction"].round(6)

    lab = pd.read_csv(run / "labels" / "checkpoint_labels.csv")
    lab["fraction"] = lab["fraction"].round(6)
    dup = int(idx.duplicated(KEY).sum())
    if dup:
        print(f"FAIL: {dup} duplicate (example_id, cot_id, fraction) rows in the activation index")
        ok = False
    ik = set(map(tuple, idx[KEY].to_numpy()))
    lk = set(map(tuple, lab[KEY].to_numpy()))
    if lk - ik:
        print(f"FAIL: {len(lk - ik)} labelled checkpoints have no activation, e.g. {sorted(lk - ik)[:3]}")
        ok = False
    if ik - lk:
        print(f"note: {len(ik - lk)} activation rows have no label (not fatal), e.g. {sorted(ik - lk)[:3]}")

    # Token positions must grow with depth within a CoT and lie inside the sequence.
    if (idx.token_pos < idx.n_prompt_tokens).any():
        print("FAIL: some checkpoint positions fall inside the prompt")
        ok = False
    for (_, _), g in idx.sort_values("fraction").groupby(["example_id", "cot_id"]):
        if not g.prefix_tokens.is_monotonic_increasing or g.prefix_tokens.duplicated().any():
            print(f"FAIL: prefix_tokens not strictly increasing for CoT {g.iloc[0][['example_id','cot_id']].tolist()}")
            ok = False
            break

    # Different checkpoints of one CoT are different prefixes, so their vectors must differ
    # (identical ones would mean the capture read the same position repeatedly).
    same = tested = 0
    for fi, fa in zip(idx_files, act_files):
        sub = pd.read_parquet(fi)
        arr = np.load(fa, mmap_mode="r")
        for _, g in sub.groupby(["example_id", "cot_id"]):
            if len(g) < 2 or tested >= 200:
                continue
            r = g.sort_values("fraction").row_idx.to_list()
            tested += 1
            same += int(np.array_equal(arr[r[0], 14], arr[r[1], 14]))
    if same:
        print(f"FAIL: {same}/{tested} CoTs have identical layer-14 vectors at their first two checkpoints")
        ok = False

    print(f"activations: {len(idx)} vectors over {idx.groupby(['example_id','cot_id']).ngroups} CoTs, "
          f"{N_LAYERS} layers x hidden {sorted(hidden)}, fp16, {sum(f.stat().st_size for f in act_files)/2**30:.2f} GiB; "
          f"labels: {len(lab)} rows; layer-14 distinctness tested on {tested} CoTs")
    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("base");        b.add_argument("--run-dir", required=True); b.add_argument("--n-cots", type=int, required=True)
    l = sub.add_parser("labels");      l.add_argument("--run-dir", required=True); l.add_argument("--k", type=int, required=True)
    c = sub.add_parser("activations"); c.add_argument("--run-dir", required=True)
    c.add_argument("--act-dir", required=True); c.add_argument("--n-shards", type=int, required=True)
    a = p.parse_args()
    sys.exit({"base": cmd_base, "labels": cmd_labels, "activations": cmd_activations}[a.cmd](a))


if __name__ == "__main__":
    main()
