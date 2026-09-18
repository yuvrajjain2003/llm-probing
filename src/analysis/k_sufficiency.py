#!/usr/bin/env python3
"""How many continuations per prefix (K) do the Goal 1 labels actually need?

We already sampled K=100 per checkpoint, so this asks "what would K'<100 have
given us?" by SUBSAMPLING those 100 -- no GPU, no new generation.

Three criteria, because they answer different questions and disagree:

  bias       Plug-in Shannon entropy is systematically LOW at small K, by about
             (A-1)/(2K ln2) bits for A distinct answers. That shrinks like 1/K.
             Mean reasoning length is unbiased, so it has no bias curve -- a
             useful contrast when reading the plots.
  precision  Spread of the estimate across subsamples. Shrinks like 1/sqrt(K),
             i.e. slower than bias, so it always elbows later.
  decision   Does the derived COMMITMENT label still come out the same? This is
             the one that matters: the labels are the deliverable, not the
             moments. Reported as agreement with the full-K=100 answer.

Subsampling is WITHOUT replacement (we ask "what if we had drawn only K'"),
unlike a bootstrap. Draws are nested via shared permutations (common random
numbers) so the curves are smooth rather than jittering independently per K'.

Outputs under <run-dir>/k_sufficiency/:
  curves.csv     per (fraction, k): bias / precision / decision metrics
  summary.csv    smallest K meeting each criterion, overall and per fraction
  *.png          elbow plots
"""
from __future__ import annotations

import argparse
import json
import math
import collections
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Dense where the curve bends, sparse where it does not: uniform steps waste
# resolution above ~40 and give too few points below 10, where the elbow sits.
KGRID = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 100)
# Extended grid for datasets whose K* is censored at 100 (AIME: answers are
# integers 0-999, ~12 distinct per prefix, so the curve is still climbing at 100).
KGRID_EXT = KGRID + (125, 150, 200, 250, 300)
COMMIT_THRESHOLD = 0.8      # must match labels.py
LOG2 = math.log(2.0)


def _entropy_from_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plug-in and Miller-Madow entropies (bits) for each row of a count matrix.

    Miller-Madow adds (m-1)/(2N) nats, m = observed non-empty categories. It is a
    first-order bias correction: worth showing, because correcting the estimator
    is an alternative to paying for a larger K.
    """
    n = counts.sum(axis=1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = counts / n[:, None]
        terms = np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0)
    plug = -terms.sum(axis=1)
    m = (counts > 0).sum(axis=1)
    mm = plug + (m - 1) / (2.0 * n * LOG2)
    return plug, mm


def load_groups(cont_path: Path, limit_examples: int | None):
    """Per (example_id, cot_id, fraction): answer codes, terminated flags, lengths.

    Answers are coded to ints PER COT (not per checkpoint) so that a modal answer
    is comparable across the checkpoints of one CoT -- the commitment rule needs
    to ask whether the mode is the *same* answer at later fractions.
    """
    raw = defaultdict(list)
    keep: set[int] | None = None
    if limit_examples is not None:
        with cont_path.open() as f:
            ids = {json.loads(line)["example_id"] for line in f}
        keep = set(sorted(ids)[:limit_examples])
        print(f"  limiting to {len(keep)} examples (smoke test)")

    with cont_path.open() as f:
        for line in f:
            r = json.loads(line)
            if keep is not None and r["example_id"] not in keep:
                continue
            raw[(r["example_id"], r["cot_id"], r["fraction"])].append(
                (r.get("final_answer_norm"), bool(r.get("terminated")),
                 r.get("remaining_reasoning_tokens"))
            )

    by_cot = defaultdict(dict)
    for (eid, cid, frac), rows in raw.items():
        by_cot[(eid, cid)][frac] = rows

    groups = []
    for (eid, cid), per_frac in by_cot.items():
        vocab: dict[str, int] = {}
        for rows in per_frac.values():
            for a, _, _ in rows:
                if isinstance(a, str) and a.strip() and a not in vocab:
                    vocab[a] = len(vocab)
        for frac, rows in per_frac.items():
            codes = np.array([vocab.get(a, -1) if isinstance(a, str) else -1
                              for a, _, _ in rows], dtype=np.int32)
            term = np.array([t for _, t, _ in rows], dtype=bool)
            rem = np.array([(x if x is not None else np.nan) for _, _, x in rows],
                           dtype=float)
            groups.append(dict(example_id=eid, cot_id=cid, fraction=frac,
                               codes=codes, term=term, rem=rem, n_vocab=len(vocab)))
    return groups


def analyse(groups, kgrid, n_draws: int, seed: int):
    """Subsample each checkpoint's 100 continuations down to each K' in kgrid."""
    rng = np.random.default_rng(seed)
    # Groups need NOT all hold the same number of continuations: a run that
    # measured K on a slice at K=100 and then generated the rest at a smaller K
    # has mixed sizes. Permutations are cached per size, and each group is only
    # evaluated at grid points it can actually support.
    _perm_cache: dict[int, np.ndarray] = {}

    def perms_for(size: int) -> np.ndarray:
        # Common random numbers within a size: K'=5 is nested in K'=10, so the
        # curves move smoothly instead of jittering independently per K'.
        if size not in _perm_cache:
            _perm_cache[size] = np.argsort(rng.random((n_draws, size)), axis=1)
        return _perm_cache[size]

    rows = []
    # (example_id, cot_id) -> fraction -> per-draw (mode_code, p_mode) at each K'
    commit_store = defaultdict(lambda: defaultdict(dict))

    skipped = 0
    for g in groups:
        codes, term, rem, A = g["codes"], g["term"], g["rem"], max(g["n_vocab"], 1)
        N = len(codes)
        perms = perms_for(N)
        for k in kgrid:
            if k > N:            # cannot subsample more than this group holds
                skipped += 1
                continue
            sel = codes[perms[:, :k]]                       # (draws, k)
            valid = sel >= 0
            nvalid = valid.sum(axis=1)
            # counts per draw via a single bincount over draw*A + code
            rr = np.repeat(np.arange(n_draws), k).reshape(n_draws, k)[valid]
            comb = rr * A + sel[valid]
            counts = np.bincount(comb, minlength=n_draws * A).reshape(n_draws, A)

            ok = nvalid > 0
            plug = np.full(n_draws, np.nan)
            mm = np.full(n_draws, np.nan)
            pmode = np.full(n_draws, np.nan)
            modec = np.full(n_draws, -1, dtype=np.int32)
            if ok.any():
                pl, mmv = _entropy_from_counts(counts[ok])
                plug[ok], mm[ok] = pl, mmv
                pmode[ok] = counts[ok].max(axis=1) / nvalid[ok]
                modec[ok] = counts[ok].argmax(axis=1)

            # remaining length: only continuations that finished reasoning
            st = term[perms[:, :k]]
            sr = rem[perms[:, :k]]
            cnt = st.sum(axis=1)
            with np.errstate(invalid="ignore"):
                rmean = np.where(cnt > 0,
                                 np.nansum(np.where(st, sr, 0.0), axis=1) / np.maximum(cnt, 1),
                                 np.nan)

            # Finite-population correction. Subsampling k of N=100 WITHOUT
            # replacement makes the spread collapse to 0 at k=N by construction --
            # the same sample every draw -- which reads as infinite precision.
            # The SD we would see drawing k fresh continuations is larger by
            # 1/sqrt(1 - k/N). Reported alongside the raw value, not instead of it.
            fpc = math.sqrt(1.0 - k / N) if k < N else float("nan")
            rows.append(dict(
                example_id=g["example_id"], cot_id=g["cot_id"], fraction=g["fraction"],
                k=k,
                ent_mean=np.nanmean(plug), ent_sd_raw=np.nanstd(plug),
                ent_sd=(np.nanstd(plug) / fpc) if fpc and fpc > 0 else np.nan,
                mm_mean=np.nanmean(mm),
                rem_mean=np.nanmean(rmean),
                rem_sd=(np.nanstd(rmean) / fpc) if fpc and fpc > 0 else np.nan,
                pmode_mean=np.nanmean(pmode),
            ))
            commit_store[(g["example_id"], g["cot_id"])][g["fraction"]][k] = (modec, pmode)

    if skipped:
        print(f"  note: {skipped} (group, K) cells skipped -- K exceeded the "
              f"continuations that group holds (mixed-K run)")
    return pd.DataFrame(rows), commit_store


def commitment_agreement(commit_store, kgrid, n_draws: int) -> pd.DataFrame:
    """Fraction of subsamples whose commitment juncture matches the K=100 answer.

    Mirrors labels.py::_commitment -- earliest fraction whose modal answer stays
    modal AND >= COMMIT_THRESHOLD concentrated through every later checkpoint.
    """
    def juncture(mode_mat, pmode_mat, fracs):
        """mode_mat/pmode_mat: (draws, n_fracs). Returns commit fraction per draw."""
        d, F = mode_mat.shape
        out = np.full(d, np.nan)
        for i in range(F):
            same = (mode_mat[:, i:] == mode_mat[:, [i]]).all(axis=1) & (mode_mat[:, i] >= 0)
            conc = (pmode_mat[:, i:] >= COMMIT_THRESHOLD).all(axis=1)
            hit = same & conc & np.isnan(out)
            out[hit] = fracs[i]
        return out

    rows = []
    for k in kgrid:
        agree = tot = 0
        for _cot, per_frac in commit_store.items():
            fracs = sorted(per_frac)
            if not fracs or not all(k in per_frac[f] for f in fracs):
                continue
            ref_k = min(max(per_frac[f]) for f in fracs)
            mode_mat = np.stack([per_frac[f][k][0] for f in fracs], axis=1)
            pmode_mat = np.stack([per_frac[f][k][1] for f in fracs], axis=1)
            ref_mode = np.stack([per_frac[f][ref_k][0] for f in fracs], axis=1)
            ref_pmode = np.stack([per_frac[f][ref_k][1] for f in fracs], axis=1)
            got = juncture(mode_mat, pmode_mat, fracs)
            # reference: the full-K answer (draw 0 uses all 100, so any draw works)
            ref = juncture(ref_mode, ref_pmode, fracs)
            agree += int(np.sum((got == ref) | (np.isnan(got) & np.isnan(ref))))
            tot += got.size
        rows.append(dict(k=k, commit_agreement=agree / tot if tot else np.nan))
    return pd.DataFrame(rows)


def estimate_reference(curve: pd.DataFrame, kmin: int = 20) -> dict:
    """Estimate the K->infinity entropy by extrapolation, not by trusting max K.

    Taking the deepest estimate as "truth" is circular: it forces the error at that
    K to be zero, so the analysis can never report that your deepest data is itself
    insufficient. Instead fit the known bias shape and read off the intercept.

      plug-in bias is O(1/K)   -> regress ent_mean on 1/K,  intercept = H_inf
      Miller-Madow residual is O(1/K^2) -> regress mm_mean on 1/K^2, intercept = H_inf

    The two rest on different assumptions, so their disagreement is a usable error
    bar on the reference itself. When it exceeds the tolerance, K* is not
    trustworthy at any K in range -- which is exactly the "censored" case.
    """
    fit = curve[curve.k >= kmin]
    if len(fit) < 3:
        return dict(reference=float(curve.mm_mean.iloc[-1]), uncertainty=float("nan"),
                    plugin_extrap=float("nan"), mm_extrap=float("nan"), method="max-K fallback")
    k = fit.k.to_numpy(float)
    plug_int = float(np.polyfit(1.0 / k, fit.ent_mean.to_numpy(float), 1)[1])
    mm_int = float(np.polyfit(1.0 / k**2, fit.mm_mean.to_numpy(float), 1)[1])
    # plug-in carries the dominant 1/K term and is the better-conditioned fit,
    # so it is the primary; MM's extrapolation is the independent cross-check.
    return dict(reference=plug_int, uncertainty=abs(plug_int - mm_int),
                plugin_extrap=plug_int, mm_extrap=mm_int, method=f"extrapolated (k>={kmin})")


def pick_k(curve: pd.DataFrame, col: str, ref_k: int, tol: float,
           mode: str, ref_col: str | None = None,
           ref_value: float | None = None) -> int | float:
    """Smallest k in the curve meeting a tolerance (and holding for all larger k).

    For bias, ref_col lets the reference come from a different column: plug-in at
    K=100 is still biased low, so the Miller-Madow value at K=100 is the better
    available estimate of the K->infinity limit.
    """
    c = curve.sort_values("k")
    ref = (float(ref_value) if ref_value is not None
           else float(c.loc[c.k == ref_k, ref_col or col].iloc[0]))
    for i, k in enumerate(c.k.values):
        tail = c.iloc[i:]
        if mode == "bias":
            good = (tail[col] - ref).abs().le(tol).all()
        elif mode == "below":
            good = tail[col].le(tol).all()
        else:  # "above"
            good = tail[col].ge(tol).all()
        if good:
            return int(k)
    return float("nan")


def make_plots(curve: pd.DataFrame, agree: pd.DataFrame, out: Path, kstar: dict):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    fr = sorted(curve.fraction.unique())
    cmap = plt.get_cmap("viridis")
    cols = {f: cmap(i / max(len(fr) - 1, 1)) for i, f in enumerate(fr)}

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    for f in fr:
        d = curve[curve.fraction == f].sort_values("k")
        ax[0, 0].plot(d.k, d.ent_mean, "o-", ms=3.5, color=cols[f], label=f"{f:.0%}")
        ax[0, 0].plot(d.k, d.mm_mean, "--", lw=1, color=cols[f], alpha=.55)
        ax[0, 1].plot(d.k, d.ent_sd, "o-", ms=3.5, color=cols[f], label=f"{f:.0%}")
        ax[1, 0].plot(d.k, d.rem_mean, "o-", ms=3.5, color=cols[f], label=f"{f:.0%}")

    ax[0, 0].set(title="Entropy vs K  (bias: solid=plug-in, dashed=Miller-Madow)",
                 xlabel="K", ylabel="mean entropy (bits)")
    ax[0, 1].set(title="Entropy precision vs K  (spread across subsamples)",
                 xlabel="K", ylabel="SD of entropy (bits)")
    ax[1, 0].set(title="Mean remaining reasoning length vs K  (unbiased: flat)",
                 xlabel="K", ylabel="tokens")
    ax[0, 1].set_yscale("log"); ax[0, 1].set_xscale("log")
    ax[0, 0].set_xscale("log")
    ax[1, 0].set_xscale("log")

    ax[1, 1].plot(agree.k, agree.commit_agreement * 100, "o-", color="#8a3324")
    ax[1, 1].axhline(99, ls=":", c="gray", lw=1)
    ax[1, 1].set(title="Commitment label agreement with K=100",
                 xlabel="K", ylabel="% of CoTs with same juncture")
    ax[1, 1].set_xscale("log")

    for a in ax.ravel():
        a.grid(alpha=.25, which="both")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.legend(title="checkpoint", fontsize=8, ncol=2)
    if not math.isnan(kstar.get("overall", float("nan"))):
        for a in ax.ravel():
            a.axvline(kstar["overall"], color="#9c2b1f", ls="--", lw=1.2, alpha=.8)

    fig.suptitle("K-sufficiency: subsampling the existing K=100 continuations", y=.995)
    fig.tight_layout()
    fig.savefig(out / "k_sufficiency.png", dpi=140)
    print(f"  wrote {out/'k_sufficiency.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--limit-examples", type=int, default=None,
                   help="use only the first N example_ids (smoke test)")
    p.add_argument("--kgrid", default="default", choices=["default", "extended"],
                   help="'extended' adds 125-300, for datasets whose K* is "
                        "censored at 100")
    p.add_argument("--group-size", default="max",
                   help="use groups with >= this many continuations "
                        "('max' = the deepest size present)")
    p.add_argument("--draws", type=int, default=200, help="subsamples per K")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bias-tol", type=float, default=0.02,
                   help="bits of entropy bias considered negligible")
    p.add_argument("--sd-tol", type=float, default=0.05,
                   help="bits of entropy SD considered acceptable")
    p.add_argument("--agree-tol", type=float, default=0.99,
                   help="required commitment-label agreement with K=100")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cont = run_dir / "raw" / "continuations.jsonl"
    out = run_dir / "k_sufficiency"; out.mkdir(parents=True, exist_ok=True)

    print(f"\nloading {cont}")
    groups = load_groups(cont, args.limit_examples)

    # Every point on the curve must come from the SAME set of groups, or the
    # comparison is between populations rather than between K values. A mixed-K
    # run (slice at K=100, remainder at a smaller K) would otherwise put 2,844
    # groups at K=25 and only the 600 deep ones at K=100. Keep the deep groups.
    sizes = collections.Counter(len(g["codes"]) for g in groups)
    target = max(sizes) if args.group_size == "max" else int(args.group_size)
    deep = [g for g in groups if len(g["codes"]) >= target]
    if len(sizes) > 1:
        print(f"  mixed group sizes {dict(sizes)} -> using the {len(deep)} groups "
              f"with >= {target} continuations, so all K share one population")
    groups = deep
    grid = KGRID_EXT if args.kgrid == "extended" else KGRID
    kgrid = tuple(k for k in grid if k <= target)
    if not groups:
        raise SystemExit(f"no groups with >= {target} continuations")
    print(f"  {len(groups)} checkpoint groups, {args.draws} subsamples per K")

    per_group, store = analyse(groups, kgrid, args.draws, args.seed)
    curve = (per_group.groupby(["fraction", "k"], as_index=False)
             .agg(ent_mean=("ent_mean", "mean"), ent_sd=("ent_sd", "mean"),
                  ent_sd_raw=("ent_sd_raw", "mean"),
                  mm_mean=("mm_mean", "mean"), rem_mean=("rem_mean", "mean"),
                  rem_sd=("rem_sd", "mean"), pmode_mean=("pmode_mean", "mean")))
    overall = (per_group.groupby("k", as_index=False)
               .agg(ent_mean=("ent_mean", "mean"), ent_sd=("ent_sd", "mean"),
                    ent_sd_raw=("ent_sd_raw", "mean"),
                    mm_mean=("mm_mean", "mean"), rem_mean=("rem_mean", "mean"),
                    rem_sd=("rem_sd", "mean")))
    overall.insert(0, "fraction", "ALL")
    agree = commitment_agreement(store, kgrid, args.draws)

    pd.concat([curve, overall]).to_csv(out / "curves.csv", index=False)
    agree.to_csv(out / "commitment_agreement.csv", index=False)

    refinfo = estimate_reference(overall)
    REF = refinfo["reference"]
    print(f"\n  reference H(K->inf) = {REF:.4f} bits  [{refinfo['method']}]")
    print(f"    plug-in extrapolation {refinfo['plugin_extrap']:.4f} | "
          f"MM extrapolation {refinfo['mm_extrap']:.4f} | "
          f"MM at max K {overall.mm_mean.iloc[-1]:.4f}")
    print(f"    reference uncertainty {refinfo['uncertainty']:.4f} bits", end="")
    if refinfo["uncertainty"] > args.bias_tol:
        print(f"  *** EXCEEDS the {args.bias_tol} tolerance: K* is NOT reliable at "
              f"any K in this range -- generate deeper data ***")
    else:
        print("  (within tolerance -- reference is trustworthy)")

    rows = [dict(scope="ALL",
                 k_bias=pick_k(overall, "ent_mean", kgrid[-1], args.bias_tol, "bias",
                               ref_col="mm_mean"),
                 k_bias_mm=pick_k(overall, "mm_mean", kgrid[-1], args.bias_tol, "bias",
                                  ref_col="mm_mean"),
                 k_precision=pick_k(overall, "ent_sd", 100, args.sd_tol, "below"),
                 k_decision=pick_k(agree.assign(fraction="ALL"), "commit_agreement",
                                   kgrid[-1], args.agree_tol, "above"))]
    for f in sorted(curve.fraction.unique()):
        d = curve[curve.fraction == f]
        rows.append(dict(scope=f"frac={f}",
                         k_bias=pick_k(d, "ent_mean", kgrid[-1], args.bias_tol, "bias",
                                       ref_value=estimate_reference(d)["reference"]),
                         k_bias_mm=pick_k(d, "mm_mean", kgrid[-1], args.bias_tol, "bias",
                                          ref_value=estimate_reference(d)["reference"]),
                         k_precision=pick_k(d, "ent_sd", 100, args.sd_tol, "below"),
                         k_decision=float("nan")))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)

    kstar = {"overall": max([v for v in summary.iloc[0][["k_bias", "k_precision",
                                                         "k_decision"]]
                             if isinstance(v, (int, float)) and not math.isnan(v)]
                            or [float("nan")])}
    make_plots(curve, agree, out, kstar)

    print(f"\n{'-'*64}")
    print(summary.to_string(index=False))
    print(f"\n  entropy bias tol {args.bias_tol} bits | SD tol {args.sd_tol} bits"
          f" | commitment agreement {args.agree_tol:.0%}")
    print(f"  K* (max of the three criteria) = {kstar['overall']}")
    print(f"  written to {out}")
    print(f"{'-'*64}\n")


if __name__ == "__main__":
    main()
