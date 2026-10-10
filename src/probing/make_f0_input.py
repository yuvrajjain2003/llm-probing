#!/usr/bin/env python3
"""Write a fraction-0.0 (empty prefix) input dir for extract_checkpoints.py.

One row per question, cot_id -1 (the 0% prefix is shared by all CoTs of a question).
    python src/probing/make_f0_input.py --run-dir <goal1 run dir> --out-dir <new dir>
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from generation.checkpoint import make_checkpoints

p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True); p.add_argument("--out-dir", required=True)
a = p.parse_args()
raw = Path(a.out_dir) / "raw"; raw.mkdir(parents=True, exist_ok=True)
ex = pd.read_csv(Path(a.run_dir) / "raw" / "examples.csv")
ex.to_csv(raw / "examples.csv", index=False)
(ck,) = make_checkpoints("x.", None, (0.0,))   # any text; fraction 0.0 -> empty prefix
with (raw / "checkpoints.jsonl").open("w") as fh:
    for eid in ex["example_id"]:
        fh.write(json.dumps(dict(example_id=int(eid), cot_id=-1, **ck.to_dict())) + "\n")
print(f"wrote {len(ex)} rows to {raw}")
