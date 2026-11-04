#!/usr/bin/env python3
"""Slide figures for the professor meeting (2026-11-03).

Reads ONLY already-published CSVs and refits nothing. Every number here was produced by
`icl_table.py` / `newds_icl_table.py` (`_icl/`), `newds_difficulty_scoring.py` (`_iclsplit/`)
and `loglen_refit.py` + `loglen_ci.py` (`_loglen/`), and is read straight out of their outputs.

    ./envs/goal1/bin/python src/viz/meeting_figures.py
Outputs: docs/figures/meeting/m{1..7}_*.png

House style follows src/viz/newds_figures.py: same rcParams, same palette slots, significance
carried by filled-vs-hollow markers (never by colour alone), every point that has an interval in
the source CSV shows it, and any point that has no interval available says so on the figure.
No em dashes anywhere.

Rerunnable and tolerant of missing inputs: GlobalOpinionQA's `_iclsplit/p2_*` files were being
computed when this was written, so m5 and m6 include whichever datasets are present and print a
warning for the rest. Rerun the module once they land.

SIGN CONVENTION, the one thing in here that is easy to get wrong. In `_icl/ci_paired.csv` the
column `diff` is (label minus `minus`), so with `minus` = our probe a NEGATIVE value means our
probe is ahead. m4 flips that sign so the plotted quantity is always "our probe minus the other
approach" and says so on the axis. In `_loglen/ci.csv` the column `diff_vs_ours` is
(feature minus act_L21), so a NEGATIVE value means our probe is ahead; m7 plots absolute scores
and uses that column only for the significance flag, so no flip is needed there.

TWO SOURCE CHOICES worth knowing, both deliberate:
  * m5/m6 take GPQA from `_newds_reports/_p6_gpqa_ref/`, not from `gpqa25/sweep/_iclsplit/`.
    The `_iclsplit` copy is the round-6 run on the 1,178-row ICL SUBSAMPLE (published entropy
    0.309, partialled 0.092); `_p6_gpqa_ref` is the same script on the FULL `ref_fixed` rows
    (0.289, partialled 0.025), which is what MMLU-Pro's files cover, so only that pairing is
    comparable across datasets.
  * m1 to m3 take GPQA from `gpqa25/sweep/_icl/`, which IS the ICL subsample, because the whole
    point there is to score the frontier model and the probe on identical rows.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path("runs/goal2")
OUT = Path("docs/figures/meeting"); OUT.mkdir(parents=True, exist_ok=True)

# palette slots 1-3 of the dataviz skill's default categorical palette, unchanged, as in
# slide_figures.py / round6_figures.py / newds_figures.py
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#e3e2dd", "#8a8884"

ENT, LEN, LSD = "answer_entropy_bits_mm", "rem_len_mean", "rem_len_std"
NICE = {ENT: "Answer entropy", LEN: "Remaining length", LSD: "Length spread"}

OURS, TFIDF, PLEN = "Our model L21 (ridge)", "TF-IDF on prefix", "Prefix length"
LUNA0 = "GPT-6 Luna, k=0"
LUNA10 = "GPT-6 Luna, k=10"
LUNAB = "GPT-6 Luna, k=0, depth hidden everywhere"

DSSHORT = {"gpqa25": "GPQA", "mmlupro_v1": "MMLU-Pro", "globalop_v1": "GlobalOpinionQA"}
DSQ = {"gpqa25": 198, "mmlupro_v1": 2499, "globalop_v1": 2555}
TAGS = ["gpqa25", "mmlupro_v1", "globalop_v1"]

# Arial is not installed in this environment (checked with matplotlib.font_manager); Liberation
# Sans is metric-compatible with Arial and is what actually renders, with DejaVu Sans last.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14.5, "axes.labelcolor": MUTED,
    "axes.edgecolor": MUTED, "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
    "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False, "legend.fontsize": 13,
})


# ---------------------------------------------------------------- helpers ----
def get(d, **kw):
    for k, v in kw.items():
        d = d[d[k] == v]
    assert len(d) == 1, f"{kw} matched {len(d)} rows"
    return d.iloc[0]


def hbar(ax, y, v, lo, hi, col, mk, filled=True, ms=13, label=None, lw=3.0):
    """One horizontal point with its 95% interval. Hollow marker = not significant."""
    ax.plot([lo, hi], [y, y], color=col, lw=lw, solid_capstyle="butt", zorder=2)
    for e in (lo, hi):
        ax.plot([e, e], [y - .15, y + .15], color=col, lw=1.8, zorder=2)
    ax.plot([v], [y], mk, ms=ms, color=col if filled else "white", markeredgecolor=col,
            markeredgewidth=2.4, zorder=4, label=label)


def caption(fig, text, y=-0.02, size=10.5):
    fig.text(0.012, y, text, fontsize=size, color=MUTED, ha="left", va="top")


def icl_scores(tag):
    return pd.read_csv(R / tag / "sweep/_icl/ci_scores.csv")


def icl_paired(tag):
    return pd.read_csv(R / tag / "sweep/_icl/ci_paired.csv")


def icl_n(tag):
    """Rows per cut point in the ICL comparison (one chain per question)."""
    d = pd.read_csv(R / tag / "sweep/_icl/per_checkpoint.csv")
    d = d[(d.label == OURS) & (d.target == ENT)]
    return int(d.n.min()), int(d.n.sum())


def p2(tag, kind, name):
    """P6 difficulty files. `kind` is 'rollout' (label-derived) or 'qtext' (text-derived).

    Returns None when the file does not exist yet, so the caller can skip that dataset.
    """
    p = (R / f"_newds_reports/_p6_gpqa_ref/{kind}/{name}.{kind}.csv" if tag == "gpqa25"
         else R / tag / f"sweep/_iclsplit/{name}.{kind}.csv")
    return pd.read_csv(p) if p.exists() else None


def have_p2(tag, name):
    return all(p2(tag, k, name) is not None for k in ("rollout", "qtext"))


# Plain-English names for the difficulty constructs, read from the files rather than assumed.
MEASURE_NICE = {
    "p_gold": "difficulty = share of the question's rollouts that are correct",
    "qtext_pgold": "difficulty = share correct, predicted from the question text",
    "ent0_norm": "difficulty = the question's OWN entropy before any reasoning",
    "qtext_ent_norm_mean": "difficulty = that entropy, predicted from the question text",
}


def measure_of(tag, kind):
    """Which difficulty construct this dataset's run actually used. Read, never assumed."""
    sub = "_difficulty/question_difficulty.csv" if kind == "rollout" \
        else "_qtext/question_difficulty.csv"
    f = R / tag / "sweep" / sub
    if not f.exists():
        return "p_gold"                       # GPQA's round-6 file predates the column
    d = pd.read_csv(f, usecols=lambda c: c in ("difficulty_measure",))
    if "difficulty_measure" not in d.columns:
        return "p_gold"
    return str(d.difficulty_measure.dropna().unique()[0])


def comparable(tag):
    """True when this dataset's difficulty is the accuracy-based measure the other two use.

    GlobalOpinionQA has no accuracy labels (p_gold is null on all 63,684 rows), so its run
    substitutes the question's own 0%-cut normalised entropy. That is a different experiment
    and must not share an axis with the p_gold datasets without saying so.
    """
    return measure_of(tag, "rollout") == "p_gold"


def loglen(tag):
    return pd.read_csv(R / tag / "sweep/_loglen/ci.csv")


# ============================================== m1 / m2 / m3 ================
# The professor's explicit ask: the frontier model against our probe, with intervals.
FOREST = [
    (OURS, "Our probe: activations, layer 21", BLUE, "o"),
    (TFIDF, "TF-IDF on the same prefix text", ORANGE, "s"),
    (PLEN, "Prefix length alone", ORANGE, "^"),
    (LUNA0, "GPT-6 Luna, zero shot (k=0)", AQUA, "D"),
    (LUNA10, "GPT-6 Luna, 10 examples (k=10)", AQUA, "v"),
    (LUNAB, "GPT-6 Luna, k=0, cut depth hidden", AQUA, "<"),
]
SHORT = {OURS: "Our probe\n(activations L21)", TFIDF: "TF-IDF on prefix",
         PLEN: "Prefix length", LUNA0: "GPT-6 Luna, k=0",
         LUNA10: "GPT-6 Luna, k=10", LUNAB: "GPT-6 Luna, k=0,\ndepth hidden"}


def frontier_fig(tag, fname, number):
    sc, pr = icl_scores(tag), icl_paired(tag)
    nq, nrow = icl_n(tag)
    fig, axs = plt.subplots(1, 2, figsize=(17.4, 6.4))
    for ax, t in zip(axs, (ENT, LEN)):
        vals = [get(sc, label=lab, target=t) for lab, _, _, _ in FOREST]
        # build the paired-comparison column first, so the right margin can be sized to fit it
        texts = []
        for lab, _, _, _ in FOREST:
            if lab == OURS:
                texts.append(("reference", BLUE)); continue
            pp = get(pr, label=lab, minus=OURS, target=t)
            # pp.diff is (this approach) minus (our probe): negative means we are ahead
            if not pp.significant:
                flag = "tie"
            else:
                # how close the interval comes to zero, relative to the gap itself
                edge = min(abs(pp.ci_lo), abs(pp.ci_hi))
                flag = "ours ahead, marginal" if edge < 0.1 * abs(pp["diff"]) else "ours ahead"
            texts.append((f"{-pp['diff']:+.3f} vs ours, {flag}",
                          INK if pp.significant else MUTED))
        lo = min(min(r.ci_lo for r in vals), 0.0)
        hi = max(r.ci_hi for r in vals)
        span = hi - lo
        xlo = lo - 0.06 * span
        xhi = hi + (0.10 + max(len(x[0]) for x in texts) / 58.0) * span
        for i, ((lab, _, col, mk), r, (txt, tcol)) in enumerate(zip(FOREST, vals, texts)):
            if lab == OURS:  # distinguish our probe from every other row
                ax.axhspan(i - 0.45, i + 0.45, color=BLUE, alpha=0.085, zorder=0)
            hbar(ax, i, r.mean_score, r.ci_lo, r.ci_hi, col, mk, filled=bool(r.ci_lo > 0))
            ax.annotate(txt, (xhi - 0.01 * span, i), ha="right", va="center",
                        fontsize=12.0, color=tcol)
        ax.axvline(0, color=MUTED, lw=1.2, zorder=1)
        ax.set_yticks(range(len(FOREST)))
        ax.set_yticklabels([SHORT[lab] for lab, _, _, _ in FOREST], fontsize=13.0, color=INK)
        for tick, (lab, _, _, _) in zip(ax.get_yticklabels(), FOREST):
            if lab == OURS:
                tick.set_color(BLUE); tick.set_fontweight("bold")
        ax.invert_yaxis(); ax.set_ylim(len(FOREST) - 0.4, -0.6)
        ax.set_xlim(xlo, xhi); ax.grid(axis="y", visible=False)
        ax.set_xlabel("R$^2$, mean over the five cut points")
        ax.set_title(NICE[t], loc="left", pad=10)
    fig.suptitle(f"{DSSHORT[tag]}: a frontier model with no activations, against our probe",
                 fontsize=19.5, y=1.105, x=0.012, ha="left", va="bottom")
    fig.text(0.012, 1.015, f"{nq} questions per cut point, {nrow:,} rows in all. "
             f"Bars are 95% question-bootstrap intervals (2,000 draws). "
             f"Filled marker: interval excludes zero.",
             fontsize=13.0, color=MUTED, ha="left", va="bottom")
    # NOTE 2026-11-03: the fine-print caption that used to sit under m1-m3 was REMOVED at the
    # user's instruction (it rendered too small to read on a slide and crowded the figure). Its
    # content now lives in the deck's speaker notes for the three frontier slides. Do not put it
    # back on the figure.
    fig.tight_layout()
    fig.savefig(OUT / fname); plt.close(fig)
    print(f"m{number}: {fname}")


# ======================================================== m4 ================
def m4():
    """Does the frontier model's tie survive on more data? Plotted as ours minus Luna."""
    rows = [t for t in TAGS if (R / t / "sweep/_icl/ci_paired.csv").exists()]
    fig, axs = plt.subplots(1, 2, figsize=(16.2, 5.0))
    for ax, t in zip(axs, (ENT, LEN)):
        pts = []
        for tag in rows:
            p = get(icl_paired(tag), label=LUNA0, minus=OURS, target=t)
            # flip: stored diff is Luna minus ours, we plot ours minus Luna
            pts.append((tag, -p["diff"], -p.ci_hi, -p.ci_lo, bool(p.significant)))
        hi = max(p[3] for p in pts); lo = min(min(p[2] for p in pts), 0.0)
        span = hi - lo
        xlo, xhi = lo - 0.22 * span, hi + 0.16 * span
        ax.axvspan(0, xhi, color=BLUE, alpha=0.06, zorder=0)
        ax.axvspan(xlo, 0, color=ORANGE, alpha=0.06, zorder=0)
        for i, (tag, v, cl, ch, sig) in enumerate(pts):
            hbar(ax, i, v, cl, ch, BLUE if sig else GREY, "o", filled=sig, ms=14)
            ax.annotate(f"{v:+.3f}" + ("" if sig else "  (not significant)"),
                        (v, i - 0.30), ha="center", va="bottom", fontsize=13.5,
                        color=INK if sig else MUTED)
        ax.axvline(0, color=MUTED, lw=1.4, zorder=1)
        ax.set_yticks(range(len(pts)))
        ax.set_yticklabels([f"{DSSHORT[p[0]]}\n{DSQ[p[0]]:,} questions" for p in pts],
                           fontsize=13.5, color=INK)
        ax.invert_yaxis(); ax.set_ylim(len(pts) - 0.42, -0.72)
        ax.set_xlim(xlo, xhi); ax.grid(axis="y", visible=False)
        ax.set_xlabel("our probe minus GPT-6 Luna k=0, R$^2$\n(right of zero: our probe ahead)")
        ax.set_title(NICE[t], loc="left", pad=10)
        ax.annotate("our probe ahead", (0.98, 0.04), xycoords="axes fraction", ha="right",
                    va="bottom", fontsize=12.5, color=BLUE)
    fig.suptitle("The frontier model tied our probe on GPQA. "
                 "On both larger datasets our probe is significantly ahead.",
                 fontsize=19.0, y=1.045, x=0.012, ha="left")
    # fine-print caption removed 2026-11-03 at the user's instruction: it rendered
    # too small to read on a slide. The text now lives in the deck's speaker notes.
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "m4_frontier_gap_across_datasets.png"); plt.close(fig)
    print("m4: m4_frontier_gap_across_datasets.png  datasets:", [DSSHORT[t] for t in rows])


# ======================================================== m5 ================
ENTN = "answer_entropy_bits_mm_norm"
NICE[ENTN] = "Answer entropy, normalised for option count"


def m5():
    """The difficulty bracket. Two panels, because the two halves are not the same experiment."""
    have = [t for t in TAGS if have_p2(t, "p2_ci_scores")]
    missing = [DSSHORT[t] for t in TAGS if t not in have]
    if missing:
        print(f"  WARNING m5: skipping {', '.join(missing)}, "
              f"_iclsplit/p2_ci_scores.{{rollout,qtext}}.csv not present yet. Rerun when it lands.")
    left = [t for t in have if comparable(t)]
    right = [t for t in have if not comparable(t)]
    if not left:
        print("  m5 SKIPPED: no p_gold difficulty files at all."); return
    ncol = 2 if right else 1
    fig, axs = plt.subplots(1, ncol, figsize=(9.9 * ncol, 6.2), squeeze=False)
    axs = axs[0]

    # ---- panel A: the datasets that have a gold answer, so difficulty is accuracy based
    ax = axs[0]
    for i, tag in enumerate(left):
        pub = get(p2(tag, "rollout", "p2_ci_scores"), label=OURS, target=ENT, metric="r2")
        low = get(p2(tag, "rollout", "p2_ci_scores"), label=OURS, target=ENT,
                  metric="r2_partialled")
        upp = get(p2(tag, "qtext", "p2_ci_scores"), label=OURS, target=ENT, metric="r2_partialled")
        ax.add_patch(plt.Rectangle((low.value, i - 0.20), upp.value - low.value, 0.40,
                                   facecolor=AQUA, alpha=0.18, edgecolor="none", zorder=1))
        ax.annotate(f"honest answer: {low.value:.2f} to {upp.value:.2f}",
                    ((low.value + upp.value) / 2, i + 0.24), ha="center", va="top",
                    fontsize=12.5, color="#0f7d57")
        for r, col, mk, lab in [
                (pub, GREY, "o", "as published, nothing removed"),
                (low, ORANGE, "v", "accuracy-based difficulty removed (LOWER bound)"),
                (upp, AQUA, "^", "same, predicted from question text (UPPER bound)")]:
            hbar(ax, i, r.value, r.ci_lo, r.ci_hi, col, mk, filled=bool(r.ci_lo > 0), ms=13,
                 label=lab if i == 0 else None)
    ax.set_yticks(range(len(left)))
    ax.set_yticklabels([f"{DSSHORT[t]}\n{DSQ[t]:,} questions" for t in left],
                       fontsize=14, color=INK)
    ax.set_ylim(len(left) + 0.75, -0.62)
    ax.set_title("Datasets WITH a right answer\ndifficulty = share of the question's rollouts "
                 "that are correct", loc="left", pad=10, fontsize=14.5)
    ax.plot([], [], "o", ms=11, color="white", markeredgecolor=MUTED, markeredgewidth=2.2,
            label="hollow marker: interval includes zero")
    ax.legend(loc="lower left", fontsize=11.8, ncol=1, labelspacing=0.3,
              borderaxespad=0.6)

    # ---- panel B: no gold answer, so a different construct entirely
    if right:
        ax = axs[1]
        tag = right[0]
        mr, mq = measure_of(tag, "rollout"), measure_of(tag, "qtext")
        rows = [(ENTN, "normalised\n(quote this one)"), (ENT, "raw\n(confounded)")]
        for i, (t, nice) in enumerate(rows):
            try:
                pub = get(p2(tag, "rollout", "p2_ci_scores"), label=OURS, target=t, metric="r2")
            except AssertionError:
                continue
            m1 = get(p2(tag, "rollout", "p2_ci_scores"), label=OURS, target=t,
                     metric="r2_partialled")
            m2 = get(p2(tag, "qtext", "p2_ci_scores"), label=OURS, target=t,
                     metric="r2_partialled")
            lo, hi = min(m1.value, m2.value), max(m1.value, m2.value)
            ax.add_patch(plt.Rectangle((lo, i - 0.20), hi - lo, 0.40, facecolor=ORANGE,
                                       alpha=0.16, edgecolor="none", zorder=1))
            ax.annotate(f"range of the two: {lo:.2f} to {hi:.2f}", ((lo + hi) / 2, i + 0.24),
                        ha="center", va="top", fontsize=12.5, color="#b14415")
            for r, col, mk, lab in [
                    (pub, GREY, "o", "as published, nothing removed"),
                    (m1, ORANGE, "v", "own 0% entropy removed"),
                    (m2, AQUA, "^", "that entropy from question text, removed")]:
                hbar(ax, i, r.value, r.ci_lo, r.ci_hi, col, mk, filled=bool(r.ci_lo > 0), ms=13,
                     label=lab if i == 0 else None)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([n for _, n in rows], fontsize=13.5, color=INK)
        ax.set_ylim(len(rows) + 0.75, -0.62)
        ax.set_title(f"{DSSHORT[tag]}, {DSQ[tag]:,} questions, NO right answer\n"
                     f"so a different measure: {mr} and {mq}",
                     loc="left", pad=10, fontsize=14.5, color="#b14415")
        ax.annotate("NOT the same experiment as the left panel, and NOT a directed bracket.\n"
                    "Neither measure is accuracy-based, and which of the two removes more\n"
                    "FLIPS between the raw and the normalised target.",
                    (0.985, 0.30), xycoords="axes fraction", ha="right", va="top",
                    fontsize=11.8, color="#b14415")
        ax.legend(loc="lower left", fontsize=11.8, ncol=1, labelspacing=0.3,
                  borderaxespad=0.6)

    for ax in axs:
        ax.axvline(0, color=MUTED, lw=1.4, zorder=1)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("R$^2$ on answer entropy, our probe, mean over the five cut points")
    fig.suptitle("How much of the entropy signal is just question difficulty? "
                 "Report the range, never one number.",
                 fontsize=19.0, y=1.03, x=0.012, ha="left")
    # fine-print caption removed 2026-11-03 at the user's instruction: it rendered
    # too small to read on a slide. The text now lives in the deck's speaker notes.
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "m5_difficulty_bracket.png"); plt.close(fig)
    print("m5: m5_difficulty_bracket.png  comparable:", [DSSHORT[t] for t in left],
          "| separate panel:", [DSSHORT[t] for t in right])


# ======================================================== m6 ================
def m6():
    """The depth effect: shallow cuts are nearly all difficulty, deep cuts are mostly not."""
    rows = [t for t in TAGS if have_p2(t, "p2_per_checkpoint")]
    missing = [DSSHORT[t] for t in TAGS if t not in rows]
    if missing:
        print(f"  WARNING m6: skipping {', '.join(missing)}, "
              f"_iclsplit/p2_per_checkpoint.{{rollout,qtext}}.csv not present yet. "
              f"Rerun when it lands.")
    if not rows:
        print("  m6 SKIPPED: no difficulty files at all."); return
    fig, axs = plt.subplots(1, len(rows), figsize=(6.1 * len(rows) + 0.4, 6.6), squeeze=False)
    axs = axs[0]
    for ax, tag in zip(axs, rows):
        # GlobalOpinionQA's raw entropy is option-count confounded, so use the normalised target
        t = ENT if comparable(tag) else ENTN
        dr, dq = p2(tag, "rollout", "p2_per_checkpoint"), p2(tag, "qtext", "p2_per_checkpoint")
        dr = dr[(dr.label == OURS) & (dr.target == t)].sort_values("fraction")
        dq = dq[(dq.label == OURS) & (dq.target == t)].sort_values("fraction")
        x = dr.fraction.to_numpy() * 100
        band = "range between the two difficulty measures" if comparable(tag) else \
               "range between the two measures (not a bracket)"
        ax.fill_between(x, dr.r2_partialled, dq.r2_partialled, color=AQUA, alpha=0.16,
                        zorder=1, label=band)
        for d, key, col, mk, ls, lab in [
                (dr, "r2", BLUE, "o", "-", "as published"),
                (dr, "r2_of_difficulty_alone", ORANGE, "s", "-",
                 "question difficulty alone (in sample)"),
                (dq, "r2_partialled", AQUA, "^", "--", "survives removal, text-based measure"),
                (dr, "r2_partialled", AQUA, "v", "-",
                 "survives removal, label-based measure")]:
            ax.plot(x, d[key], color=col, lw=2.4, ls=ls, zorder=2)
            ax.plot(x, d[key], mk, ms=11, color=col, markeredgecolor="white",
                    markeredgewidth=1.6, zorder=4, label=lab)
        ax.axhline(0, color=MUTED, lw=1.2, zorder=1)
        ax.set_xticks(x); ax.set_xticklabels([f"{int(v)}%" for v in x])
        ax.grid(axis="x", visible=False)
        sub = "entropy" if t == ENT else "entropy, NORMALISED"
        ax.set_title(f"{DSSHORT[tag]}, {DSQ[tag]:,} questions\n{sub}", loc="left", pad=10,
                     fontsize=15)
        ax.set_xlabel("cut point: fraction of the chain written")
        if not comparable(tag):
            ax.annotate("different difficulty measure:\nno gold answer on this dataset",
                        (0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                        fontsize=11.5, color="#b14415")
    axs[0].set_ylabel("R$^2$ on answer entropy")
    ylo = min(ax.get_ylim()[0] for ax in axs); yhi = max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:
        ax.set_ylim(ylo, yhi + 0.06 * (yhi - ylo))
    # the MMLU-Pro panel has empty space below its lines; GPQA's and GlobalOpinionQA's do not
    axs[min(1, len(axs) - 1)].legend(loc="lower left", fontsize=11.5, ncol=1,
                                     bbox_to_anchor=(0.0, 0.0), labelspacing=0.35)
    fig.suptitle("The difficulty explanation is a SHALLOW-CUT story. "
                 "Deeper in the chain, most of the entropy signal survives.",
                 fontsize=18.5, y=1.035, x=0.012, ha="left")
    # fine-print caption removed 2026-11-03 at the user's instruction: it rendered
    # too small to read on a slide. The text now lives in the deck's speaker notes.
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT / "m6_depth_partialled.png"); plt.close(fig)
    print("m6: m6_depth_partialled.png  datasets:", [DSSHORT[t] for t in rows])


# ======================================================== m8 ================
def m8():
    """Does our advantage over the text baseline survive difficulty removal? It grows."""
    rows = [t for t in TAGS if have_p2(t, "p2_ci_paired")]
    missing = [DSSHORT[t] for t in TAGS if t not in rows]
    if missing:
        print(f"  WARNING m8: skipping {', '.join(missing)}, "
              f"_iclsplit/p2_ci_paired.{{rollout,qtext}}.csv not present yet.")
    if not rows:
        print("  m8 SKIPPED: no paired difficulty files at all."); return
    fig, ax = plt.subplots(figsize=(14.8, 1.55 * len(rows) + 2.3))
    series = [("r2", "rollout", GREY, "o", "as published, nothing removed"),
              ("r2_partialled", "rollout", ORANGE, "v",
               "label-derived difficulty removed (harsher)"),
              ("r2_partialled", "qtext", AQUA, "^",
               "text-derived difficulty removed (gentler)")]
    for i, tag in enumerate(rows):
        for j, (metric, kind, col, mk, lab) in enumerate(series):
            d = p2(tag, kind, "p2_ci_paired")
            r = get(d, label=OURS, **{"minus": TFIDF}, target=ENT, metric=metric)
            y = i + (j - 1) * 0.24
            hbar(ax, y, r["diff"], r.ci_lo, r.ci_hi, col, mk, filled=bool(r.significant),
                 ms=12, label=lab if i == 0 else None)
        if i:
            ax.axhline(i - 0.5, color=GRID, lw=1.2, zorder=0)
        if not comparable(tag):
            ax.annotate("different difficulty measure\n(no gold answer here)", (0.012, i),
                        xycoords=("axes fraction", "data"), ha="left", va="center",
                        fontsize=11.5, color="#b14415")
    ax.axvline(0, color=MUTED, lw=1.4, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{DSSHORT[t]}\n{DSQ[t]:,} questions" for t in rows],
                       fontsize=14, color=INK)
    ax.set_ylim(len(rows) + 0.42, -0.60)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("our probe minus TF-IDF on the same prefix text, R$^2$ on answer entropy\n"
                  "(right of zero: activations beat the words)")
    ax.plot([], [], "o", ms=12, color="white", markeredgecolor=MUTED, markeredgewidth=2.2,
            label="hollow marker: interval includes zero")
    ax.legend(loc="lower left", ncol=2, fontsize=11.8, labelspacing=0.3, borderaxespad=0.6)
    fig.suptitle("Take question difficulty away and the activations-beat-text gap GROWS on both "
                 "large datasets.", fontsize=18.5, y=1.03, x=0.012, ha="left")
    # fine-print caption removed 2026-11-03 at the user's instruction: it rendered
    # too small to read on a slide. The text now lives in the deck's speaker notes.
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "m8_vs_text_after_difficulty.png"); plt.close(fig)
    print("m8: m8_vs_text_after_difficulty.png  datasets:", [DSSHORT[t] for t in rows])


# ======================================================== m7 ================
FEATS = [("act_L21", "Our probe: activations L21", BLUE, "o"),
         ("tfidf", "TF-IDF on prefix", ORANGE, "s"),
         ("prefix_len", "Prefix length alone", AQUA, "^")]
TRANS = [("raw", "raw tokens"), ("log1p", "log tokens")]


def m7():
    """Raw against log scale on the two length targets. Token counting is a raw-scale artefact."""
    rows = [t for t in TAGS if (R / t / "sweep/_loglen/ci.csv").exists()]
    fig, axs = plt.subplots(2, len(rows), figsize=(5.4 * len(rows) + 0.6, 10.2), squeeze=False)
    for ri, t in enumerate((LEN, LSD)):
        for ci, tag in enumerate(rows):
            ax = axs[ri, ci]
            d = loglen(tag)
            for fi, (feat, lab, col, mk) in enumerate(FEATS):
                xs, ys = [], []
                for xi0, (tr, _) in enumerate(TRANS):
                    xi = xi0 + (fi - 1) * 0.07  # keep coincident points apart
                    r = get(d, feature=feat, target=t, **{"transform": tr})
                    # diff_vs_ours is (feature minus act_L21); NaN on our own row
                    sig = True if feat == "act_L21" else bool(r.significant)
                    ax.errorbar([xi], [r.score], yerr=[[r.score - r.ci_lo], [r.ci_hi - r.score]],
                                fmt="none", ecolor=col, elinewidth=2.4, capsize=5, capthick=2.0,
                                zorder=3)
                    ax.plot([xi], [r.score], mk, ms=14, color=col if sig else "white",
                            markeredgecolor=col, markeredgewidth=2.4, zorder=4)
                    xs.append(xi); ys.append(r.score)
                ax.plot(xs, ys, color=col, lw=2.4, zorder=2, label=lab if (ri == 0 and ci == 0)
                        else None)
            ax.set_xlim(-0.45, 1.45); ax.set_xticks([0, 1])
            ax.set_xticklabels([n for _, n in TRANS], fontsize=14)
            ax.grid(axis="x", visible=False)
            ax.set_ylim(-0.05, 0.86)
            if ci == 0:
                ax.set_ylabel(f"R$^2$, {NICE[t].lower()}")
            else:
                ax.set_yticklabels([])
            if ri == 0:
                ax.set_title(f"{DSSHORT[tag]}, {DSQ[tag]:,} questions", loc="left", pad=10)
    axs[0, 0].legend(loc="lower left", fontsize=12.5, bbox_to_anchor=(0.0, 0.0))
    for ci, tag in enumerate(rows):
        axs[1, ci].set_xlabel("target scale")
    fig.suptitle("\"It is just counting tokens\" is an artefact of the raw scale. "
                 "On the log scale our probe wins everywhere.",
                 fontsize=18.5, y=1.015, x=0.012, ha="left")
    # fine-print caption removed 2026-11-03 at the user's instruction: it rendered
    # too small to read on a slide. The text now lives in the deck's speaker notes.
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "m7_loglen.png"); plt.close(fig)
    print("m7: m7_loglen.png  datasets:", [DSSHORT[t] for t in rows])


def main():
    frontier_fig("gpqa25", "m1_frontier_ci_gpqa.png", 1)
    frontier_fig("mmlupro_v1", "m2_frontier_ci_mmlupro.png", 2)
    frontier_fig("globalop_v1", "m3_frontier_ci_globalop.png", 3)
    m4()
    m5()
    m6()
    m7()
    m8()
    print("\nwrote to", OUT.resolve())


if __name__ == "__main__":
    sys.exit(main())
