#!/usr/bin/env python3
"""0% (empty-prefix) labels from the existing base samples -- no new generation.

Continuing from an empty prefix = sampling full answers from the prompt, which the
base run already did (same prompt, temperature, top_p, max_tokens).  So the K
"continuations" of the 0% checkpoint are the base samples of that question.
Reuses labels._checkpoint_row and continuations._remaining_reasoning_tokens.

One row per question; cot_id = -1 (the prefix is shared by all CoTs).
    python src/analysis/labels_f0.py --run-dir <goal1 run dir>
-> <run-dir>/labels/checkpoint_labels_f0.csv  (refuses to overwrite)
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from transformers import AutoTokenizer
from generation.run_eval import _load_jsonl
from generation.continuations import _remaining_reasoning_tokens
from analysis.labels import _checkpoint_row

p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True)
a = p.parse_args()
run = Path(a.run_dir)
out = run / "labels" / "checkpoint_labels_f0.csv"
assert not out.exists(), f"{out} exists"
df = _load_jsonl(run / "raw" / "sample_records.jsonl")
tok = AutoTokenizer.from_pretrained(df["model"].iloc[0])
df["terminated"] = df["raw_completion"].str.contains("</think>", regex=False)
df["remaining_reasoning_tokens"] = [_remaining_reasoning_tokens(c, tok) for c in df["raw_completion"]]
keys = ["dataset", "model", "example_id"]
ck = df.groupby(keys, sort=True).apply(_checkpoint_row, include_groups=False).reset_index()
ck.insert(3, "cot_id", -1)
ck.insert(4, "fraction", 0.0)
ck.to_csv(out, index=False)
print(f"wrote {out} ({len(ck)} rows)")
