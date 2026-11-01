#!/usr/bin/env python3
"""Noise floor of the within-question labels (idea 1 follow-up). USAGE: envs/goal1/bin/python src/analysis/redundancy_noise.py
Per row, estimation variance of the label from its K rollouts:
  rem_len_mean : (rem_len_std)^2 / n_valid
  entropy_mm, p_mode : parametric bootstrap, DRAWS resamples of n_valid answers from the row's answer_dist,
                       labels._entropy_mm (project helper) / mode count, variance over draws.
Writes _redundancy/noise_floor.csv."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.labels import _entropy_mm

OUT = Path("runs/goal2/gpqa25/sweep/_redundancy"); DRAWS = 200
RUN = Path("runs/goal1/gpqa25/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")


def boot(args):
    dist, n, seed = args
    keys = list(dist); p = np.array([dist[k] for k in keys], float); p /= p.sum()
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(n, p, size=DRAWS)
    ent = [_entropy_mm([k for k, c in zip(keys, row) for _ in range(c)]) for row in cnt]
    return float(np.var(ent, ddof=1)), float(np.var(cnt.max(1) / n, ddof=1))


if __name__ == "__main__":
    lab = pd.read_csv(RUN / "labels/checkpoint_labels.csv"); lab["fraction"] = lab.fraction.round(3)
    lab = lab[lab.fraction.isin([.1, .3, .5, .7, .9]) & lab.answer_entropy_bits_mm.notna()].reset_index(drop=True)
    with Pool(32) as pool:
        r = pool.map(boot, [(json.loads(d), int(n), i) for i, (d, n) in enumerate(zip(lab.answer_dist, lab.n_valid))], chunksize=200)
    lab["nv_answer_entropy_bits_mm"] = [x[0] for x in r]; lab["nv_p_mode"] = [x[1] for x in r]
    lab["nv_rem_len_mean"] = lab.rem_len_std ** 2 / lab.n_valid
    dec = pd.read_csv(OUT / "decomposition.csv"); dec = dec[dec.feature == "act_L21"]
    rows = []
    for t in ["answer_entropy_bits_mm", "p_mode", "rem_len_mean"]:
        for f in [.1, .3, .5, .7, .9]:
            d = lab[(lab.fraction == f) & lab[t].notna()]
            g = d.groupby("example_id")[t]
            ssw = g.apply(lambda v: ((v - v.mean()) ** 2).sum()).sum(); dfw = len(d) - g.ngroups
            obs = ssw / dfw; noise = d["nv_" + t].mean()
            corr = max(obs - noise, 0.0); ceil = corr / obs
            tot = d[t].var(ddof=1)        # overall variance across all rows at this checkpoint
            m = dec[(dec.target == t) & (dec.fraction == f)].iloc[0]
            rows.append(dict(target=t, fraction=f, n_rows=len(d), obs_within_var=obs, noise_var=noise,
                             corrected_within_var=corr, ceiling_r2=ceil, total_var=tot, overall_ceiling_vs_25mean_target=1 - noise / tot, measured_within_r2_rescaled=m.within_r2_rescaled,
                             measured_within_pearson=m.within_pearson, frac_of_ceiling=m.within_r2_rescaled / ceil if ceil > 0 else np.nan))
    df = pd.DataFrame(rows); df.to_csv(OUT / "noise_floor.csv", index=False)
    print(df.round(4).to_string())
