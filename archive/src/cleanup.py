"""
cleanup_orphaned_pts.py
───────────────────────
Deletes .pt files (both sampled and averaged) that don't have
a corresponding log entry in any worker CSV.
Run this AFTER killing workers and BEFORE relaunching.
"""

import os
import pandas as pd

DRY_RUN = False  # Set to False to actually delete

for dataset in ['gsm8k', 'logiqa']:
    print(f"\n{'='*50}")
    print(f"{dataset}_v3")
    print(f"{'='*50}")

    base         = f'/playpen-ssd/yuvraj/llm-probing-project/activations/{dataset}_v3'
    sampled_dir  = os.path.join(base, 'sampled')
    averaged_dir = os.path.join(base, 'averaged')

    # Collect all logged example IDs from worker CSVs
    logged_ids = set()
    csvs = [f for f in os.listdir(base) if f.endswith('.csv')]
    for csv in csvs:
        df = pd.read_csv(os.path.join(base, csv))
        # Only count entries with real metadata (not resumed placeholders)
        if 'resumed' in df.columns:
            df = df[df['resumed'] == False]
        logged_ids.update(df['example_id'].astype(int).tolist())

    print(f"Logged example IDs with full metadata: {len(logged_ids)}")

    # Find orphaned .pt files in sampled/
    sampled_pts  = set(f for f in os.listdir(sampled_dir)  if f.endswith('.pt'))
    averaged_pts = set(f for f in os.listdir(averaged_dir) if f.endswith('.pt'))

    orphaned = []
    for fname in sampled_pts:
        example_id = int(fname.replace('example_', '').replace('.pt', ''))
        if example_id not in logged_ids:
            orphaned.append(example_id)

    print(f"Total .pt files:   {len(sampled_pts)}")
    print(f"Orphaned (no log): {len(orphaned)}")
    print(f"Clean (have log):  {len(sampled_pts) - len(orphaned)}")

    if orphaned:
        print(f"\nOrphaned example IDs (first 20): {sorted(orphaned)[:20]}")
        if DRY_RUN:
            print(f"\n[DRY RUN] Would delete {len(orphaned)} sampled + {len(orphaned)} averaged .pt files")
            print("Set DRY_RUN = False to actually delete.")
        else:
            deleted = 0
            for example_id in orphaned:
                fname = f"example_{example_id}.pt"
                for d in [sampled_dir, averaged_dir]:
                    fpath = os.path.join(d, fname)
                    if os.path.exists(fpath):
                        os.remove(fpath)
                        deleted += 1
            print(f"Deleted {deleted} .pt files.")
    else:
        print("No orphaned files — all .pt files have log entries!")

print("\nDone. Now safe to relaunch workers.")