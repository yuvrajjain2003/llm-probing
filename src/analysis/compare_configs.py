#!/usr/bin/env python3
"""Is data generated under a new vLLM engine config distributionally the same?

    compare_configs.py --baseline <run>/raw/continuations.jsonl \
                       --candidate <bench>/raw/continuations.shard00.jsonl

Engine knobs like BlockSpaceManagerV2 and multi-step scheduling change only HOW
work is batched, so the continuations they produce should be samples from the same
distribution.  But a bug in block reuse would corrupt text, and a bug in multi-step
stop handling would shift `remaining_reasoning_tokens` -- which is one of the Goal 1
labels.  So this does not take that on faith.

Sampling is stochastic (temperature 0.7) and batch composition differs, so outputs
are never bit-identical; the question is whether the DISTRIBUTIONS differ.  Two views:

  UNPAIRED   over all shared prefixes: KS test on the remaining-length distribution,
             two-proportion tests on termination / parse / correctness rates.
  PAIRED     per prefix (same example_id, cot_id, fraction): the difference in mean
             remaining length, bootstrapped over prefixes.  This removes
             prefix-to-prefix variation, which is far larger than any config effect.

A non-significant result is NOT proof of equivalence, so every comparison reports a
confidence interval and the smallest shift the sample could have detected.  Read the
CI, not the p-value.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

KEY = ["example_id", "cot_id", "fraction"]


def load(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df["fraction"] = df["fraction"].round(6)
    print(f"  {label:9s} {len(df):>7,} continuations over "
          f"{df.groupby(KEY).ngroups:>5,} prefixes  ({path})")
    return df


def prop_test(name: str, a: pd.Series, b: pd.Series) -> dict:
    """Two-proportion comparison with a CI on the difference (candidate - baseline)."""
    na, nb = len(a), len(b)
    pa, pb = a.mean(), b.mean()
    se = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    diff = pb - pa
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    z = diff / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return dict(metric=name, baseline=pa, candidate=pb, diff=diff, lo=lo, hi=hi,
                p=p, mde=1.96 * se)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="significance threshold; default 0.01 because several "
                         "metrics are tested at once")
    a = ap.parse_args()

    print("Loading:")
    base = load(Path(a.baseline), "baseline")
    cand = load(Path(a.candidate), "candidate")

    # Compare only prefixes present in both, so neither side is judged on work the
    # other never attempted.
    shared = (set(map(tuple, base[KEY].to_numpy()))
              & set(map(tuple, cand[KEY].to_numpy())))
    if not shared:
        sys.exit("FAIL: the two files share no prefixes -- nothing to compare")
    bk = base.set_index(KEY).loc[sorted(shared)].reset_index()
    ck = cand.set_index(KEY).loc[sorted(shared)].reset_index()
    print(f"\n  shared prefixes: {len(shared):,}  "
          f"({len(bk):,} baseline vs {len(ck):,} candidate continuations)\n")

    rows, verdict_bad = [], []

    # ── Rates ────────────────────────────────────────────────────────────────
    for name, col in [("terminated", "terminated"),
                      ("parse_success", "parse_success"),
                      ("is_correct", "is_correct")]:
        r = prop_test(name, bk[col].astype(float), ck[col].astype(float))
        rows.append(r)

    # ── Remaining reasoning length: the label multi-step could plausibly break ──
    tb = bk.loc[bk.terminated, "remaining_reasoning_tokens"].to_numpy(float)
    tc = ck.loc[ck.terminated, "remaining_reasoning_tokens"].to_numpy(float)
    ks = stats.ks_2samp(tb, tc)
    se = np.sqrt(tb.var(ddof=1) / len(tb) + tc.var(ddof=1) / len(tc))
    diff = tc.mean() - tb.mean()
    rows.append(dict(metric="rem_len_mean", baseline=tb.mean(), candidate=tc.mean(),
                     diff=diff, lo=diff - 1.96 * se, hi=diff + 1.96 * se,
                     p=float(ks.pvalue), mde=1.96 * se))

    print("UNPAIRED (candidate - baseline; CI is 95%):")
    print(f"  {'metric':<16}{'baseline':>11}{'candidate':>11}{'diff':>11}"
          f"{'95% CI':>22}{'p':>9}")
    for r in rows:
        flag = "  <-- DIFFERS" if r["p"] < a.alpha else ""
        if r["p"] < a.alpha:
            verdict_bad.append(r["metric"])
        print(f"  {r['metric']:<16}{r['baseline']:>11.4f}{r['candidate']:>11.4f}"
              f"{r['diff']:>+11.4f}  [{r['lo']:>+8.4f},{r['hi']:>+8.4f}]"
              f"{r['p']:>9.3g}{flag}")
    print(f"\n  KS statistic on the remaining-length distribution: {ks.statistic:.4f}")

    # ── Paired over prefixes ─────────────────────────────────────────────────
    gb = bk[bk.terminated].groupby(KEY).remaining_reasoning_tokens.mean()
    gc = ck[ck.terminated].groupby(KEY).remaining_reasoning_tokens.mean()
    j = pd.concat([gb.rename("b"), gc.rename("c")], axis=1).dropna()
    if len(j) >= 20:
        d = (j.c - j.b).to_numpy()
        rng = np.random.default_rng(0)
        boot = rng.choice(d, size=(4000, len(d)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        w = stats.wilcoxon(j.b, j.c)
        print(f"\nPAIRED over {len(j):,} prefixes (mean remaining length per prefix):")
        print(f"  mean difference {d.mean():+.1f} tokens, bootstrap 95% CI "
              f"[{lo:+.1f}, {hi:+.1f}], Wilcoxon p={w.pvalue:.3g}")
        print(f"  baseline mean {j.b.mean():,.0f} tokens -> the CI rules out shifts "
              f"beyond {max(abs(lo), abs(hi)) / j.b.mean():.1%} of it")
        if w.pvalue < a.alpha:
            verdict_bad.append("rem_len_paired")
    else:
        print(f"\nPAIRED: only {len(j)} shared prefixes with terminated "
              "continuations -- too few to test; generate more before trusting this.")

    print("\n" + "=" * 72)
    if verdict_bad:
        print(f"VERDICT: DIFFERS from baseline on {', '.join(sorted(set(verdict_bad)))}.")
        print("Do not use this config for the real run without understanding why.")
        sys.exit(1)
    print("VERDICT: no detectable difference at alpha="
          f"{a.alpha} on any metric.")
    print("This is not proof of equivalence -- read the CIs above for the size of")
    print("shift this sample could actually have ruled out.")


if __name__ == "__main__":
    main()
