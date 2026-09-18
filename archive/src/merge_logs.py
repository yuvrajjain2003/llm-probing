"""
merge_logs_v3.py
────────────────
Merges all worker CSVs for gsm8k_v3 and logiqa_v3 into a single
master_log.csv per dataset.

For examples that have .pt files but no log entry (killed mid-batch),
we simply note them as missing — we do NOT delete the .pt files.
The probing scripts only use examples with log entries anyway
since they need metadata (think_end_pos, is_correct, etc).
"""

import os
import glob
import pandas as pd

BASE = "/playpen-ssd/yuvraj/llm-probing-project/activations"

for dataset in ["gsm8k_v3", "logiqa_v3"]:
    print(f"\n{'='*50}")
    print(f"Merging {dataset}...")
    print(f"{'='*50}")

    csv_dir = os.path.join(BASE, dataset)
    sampled_dir = os.path.join(csv_dir, "sampled")

    # Load and merge all worker CSVs
    dfs = []
    for f in sorted(glob.glob(os.path.join(csv_dir, "master_log_worker*.csv"))):
        df = pd.read_csv(f)
        print(f"  {os.path.basename(f)}: {len(df)} rows")
        dfs.append(df)

    if not dfs:
        print(f"  No CSVs found for {dataset}, skipping.")
        continue

    merged = pd.concat(dfs, ignore_index=True)

    # Drop resumed placeholders
    if 'resumed' in merged.columns:
        before = len(merged)
        merged = merged[merged['resumed'] == False]
        print(f"\n  Dropped {before - len(merged)} resumed placeholders")

    # Deduplicate by example_id (keep first occurrence)
    before = len(merged)
    merged = merged.drop_duplicates(subset='example_id', keep='first')
    print(f"  Dropped {before - len(merged)} duplicates")

    # Sort by example_id
    merged = merged.sort_values('example_id').reset_index(drop=True)

    # Report coverage
    pt_files = set(
        int(f.replace('example_', '').replace('.pt', ''))
        for f in os.listdir(sampled_dir) if f.endswith('.pt')
    )
    logged_ids = set(merged['example_id'].astype(int).tolist())
    unlogged   = pt_files - logged_ids

    print(f"\n  Total logged examples:  {len(merged)}")
    print(f"  Total .pt files:        {len(pt_files)}")
    print(f"  Unlogged .pt files:     {len(unlogged)}")
    print(f"  (These .pt files are kept but won't be used in probing)")

    if 'is_correct' in merged.columns:
        print(f"\n  is_correct distribution: {merged['is_correct'].value_counts().to_dict()}")

    if 'ground_truth' in merged.columns:
        print(f"  ground_truth distribution: {merged['ground_truth'].value_counts().to_dict()}")

    # Save master log
    out_path = os.path.join(csv_dir, "master_log.csv")
    merged.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")
    print(f"  Shape: {merged.shape}")

print("\nDone!")