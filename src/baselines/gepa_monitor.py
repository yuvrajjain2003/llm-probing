#!/usr/bin/env python3
"""Watch running --pct GEPA runs: per candidate prompt, its GEPA-val score and within-checkpoint Spearman
correlation (predicted vs true percentile, mean over 10-90%) on the val questions, vs the seed prompt.
Reads <run>/metric_log.jsonl (written by gepa_luna.make_metric_pct). Only candidates with a full val
evaluation (>= 90% of val rows) are shown; the first one is the seed.

    python src/baselines/gepa_monitor.py                 # all *_pct* / *_rank* runs
    python src/baselines/gepa_monitor.py --glob '*gemma*'  # some runs; prints a RULE line per run

Stop rule (fixed 2026-11-01, before any data): a run is BAD if a candidate beats the seed on the GEPA
score (rank_val for --rank runs, val otherwise) while its entropy or rem_len_mean rank corr is more than
0.05 below the seed's; a run is STALE if >= 5 new candidates were fully evaluated and none beat the seed.
"""
import argparse, json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

OUT = Path("runs/goal2/gpqa25/gepa")
PCT = ["entropy_pct", "p_mode_pct", "rem_len_mean_pct", "rem_len_std_pct"]

ap = argparse.ArgumentParser(); ap.add_argument("--glob", default=None); args = ap.parse_args()
runs = sorted(OUT.glob(args.glob)) if args.glob else sorted({*OUT.glob("*_pct*"), *OUT.glob("*_rank*")})
for run in [r for r in runs if r.is_dir()]:
    log = run / "metric_log.jsonl"
    if not log.exists():
        continue
    val_q = set(json.loads((run / "split.json").read_text())["val_q"])
    d = pd.read_json(log, lines=True)
    d = d[d.example_id.isin(val_q)]
    key = [c for c in ["example_id", "cot_id", "fraction"] if c in d]   # cot_id logged from 2026-11-03
    n_val = d.drop_duplicates(key).shape[0]
    order = d.drop_duplicates("instr").instr.tolist()        # first appearance = seed first
    rows = []
    for i, h in enumerate(order):
        g = d[d.instr == h].drop_duplicates(key)
        if len(g) < 0.9 * n_val:
            continue
        r = dict(cand=i, n=len(g), val=g.score.mean(),   # val = per-row pct score; rank_val = --rank's batch score
                 rank_val=(pd.Series({o: spearmanr([x[0] for x in g[o].dropna()], [x[1] for x in g[o].dropna()])[0]
                                      for o in PCT}).mean() + 1) / 2)
        for o in PCT:
            gg = g[(g.fraction > 0) & g[o].notna()]
            r[o.replace("_pct", "")] = pd.Series(
                {fr: spearmanr([x[0] for x in q[o]], [x[1] for x in q[o]])[0] for fr, q in gg.groupby("fraction")}).mean()
        rows.append(r)
    print(f"== {run.name}: {len(order)} prompts seen, {len(rows)} fully evaluated on val ({n_val} rows)")
    if rows:
        t = pd.DataFrame(rows)
        print(t.round(3).to_string(index=False))
        k = "rank_val" if "_rank" in run.name else "val"
        s0, new = t.iloc[0], t.iloc[1:]
        beat = new[new[k] > s0[k]]
        bad = ((beat.entropy < s0.entropy - 0.05) | (beat.rem_len_mean < s0.rem_len_mean - 0.05)).any()
        stale = len(new) >= 5 and beat.empty
        print(f"RULE {run.name}: new={len(new)} beat_seed={len(beat)} BAD={bool(bad)} STALE={bool(stale)}")
