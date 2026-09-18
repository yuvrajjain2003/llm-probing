#!/usr/bin/env python3
"""Generate docs/goal1_timing_estimate.pdf -- the compute-time estimate writeup.

Page 1 : one-page overview (the page to show someone who asked "how long?")
Page 2 : how the estimate was derived
Page 3 : bottlenecks and issues found while scaling

No pandoc/LaTeX/reportlab on this box, so pages are composed directly with
matplotlib. Text is placed on a 0-1 axes in figure coordinates.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
# matplotlib defaults to pdf.fonttype 3 (Type 3), where each glyph is a drawing
# program rather than real text. Many viewers (Preview, VS Code, some browser
# viewers) render Type 3 pages BLANK, and the text is not selectable/searchable.
# 42 = embed real TrueType fonts.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INK   = "#1a1a1a"
MUTED = "#5b6169"
ACCENT= "#8a3324"
RULE  = "#c8ccd2"
GOOD  = "#1d6b3f"
BAD   = "#9c2b1f"

MONO = {"family": "DejaVu Sans Mono"}
SANS = {"family": "DejaVu Sans"}


class Page:
    """Simple top-down text cursor on a letter-size figure."""

    def __init__(self, pdf):
        self.fig = plt.figure(figsize=(8.5, 11))
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.axis("off")
        # Pin the coordinate system: ax.plot() for the rules would otherwise
        # autoscale the axes and shift already-placed text (it was pushing the
        # page title off the top edge).
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1)
        self.ax.set_autoscale_on(False)
        self.pdf = pdf
        self.L, self.R = 0.078, 0.922
        self.y = 0.945

    def text(self, s, size=9.2, dy=0.020, color=INK, weight="normal",
             mono=False, x=None, style="normal"):
        self.ax.text(self.L if x is None else x, self.y, s, size=size, color=color,
                     weight=weight, va="top", ha="left", style=style,
                     **(MONO if mono else SANS))
        self.y -= dy

    def title(self, s, sub=None):
        self.text(s, size=17.5, dy=0.028, weight="bold")
        if sub:
            self.text(sub, size=9.6, dy=0.020, color=MUTED)
        self.rule(); self.gap(0.008)

    def head(self, s):
        self.gap(0.012)
        self.text(s.upper(), size=9.6, dy=0.018, weight="bold", color=ACCENT)

    def rule(self, pad=0.008):
        self.y -= pad
        self.ax.plot([self.L, self.R], [self.y, self.y], color=RULE, lw=0.8)
        self.y -= pad

    def gap(self, dy=0.014):
        self.y -= dy

    def bullet(self, s, size=9.0, dy=0.0175, color=INK):
        self.ax.text(self.L + 0.012, self.y, "•", size=size, color=MUTED,
                     va="top", **SANS)
        self.ax.text(self.L + 0.030, self.y, s, size=size, color=color, va="top",
                     **SANS)
        self.y -= dy

    def table(self, rows, size=8.5, dy=0.0172, header=True, colors=None):
        for i, r in enumerate(rows):
            w = "bold" if (header and i == 0) else "normal"
            c = INK if colors is None or colors[i] is None else colors[i]
            self.text(r, size=size, dy=dy, mono=True, weight=w, color=c)
            if header and i == 0:
                self.y += dy - 0.004
                self.ax.plot([self.L, self.R], [self.y, self.y], color=RULE, lw=0.6)
                self.y -= 0.010

    def note(self, s, size=8.4):
        self.text(s, size=size, dy=0.0165, color=MUTED, style="italic")

    def footer(self, n):
        self.ax.text(0.5, 0.032, f"{n}", size=8.5, color=MUTED, ha="center", **SANS)
        self.ax.plot([self.L, self.R], [0.052, 0.052], color=RULE, lw=0.6)

    def done(self, n):
        self.footer(n); self.pdf.savefig(self.fig); plt.close(self.fig)


def build(path="docs/goal1_timing_estimate.pdf"):
    with PdfPages(path) as pdf:
        page1(pdf); page2(pdf); page3(pdf)
    print(f"wrote {path}")


# ── Page 1 ────────────────────────────────────────────────────────────────────
def page1(pdf):
    p = Page(pdf)
    p.title("Goal 1: How Long the Data Generation Takes",
            "DeepSeek-R1-Distill-Qwen-7B  ·  8 GPUs (RTX A6000, 48 GB each)")

    p.head("What we are generating")
    p.text("We want to know whether a model has already decided its answer part-way through",
           size=9.6, dy=0.0185)
    p.text("its reasoning. So we let it reason, cut that reasoning off at five points (10%,",
           size=9.6, dy=0.0185)
    p.text("30%, 50%, 70%, 90%), and restart it 100 times from each point to see where it",
           size=9.6, dy=0.0185)
    p.text("ends up. If the 100 restarts agree, it had committed; if they scatter, it had not.",
           size=9.6, dy=0.0185)
    p.gap(0.006)
    p.text("Those restarts are the cost. We do this for 3 reasoning attempts per question.",
           size=9.6, dy=0.0185)

    p.head("Time required")
    p.table([
        "dataset          questions    restarts    GPU-hours    real time    status",
        "MATH-500               500     691,900          ~28      ~5 hours    running",
        "AIME-2025               30      36,500           ~7      ~1 hour     next",
        "GPQA-Diamond           198     264,000          ~43      ~5 hours    deferred",
    ], size=8.8, colors=[None, GOOD, GOOD, MUTED])
    p.gap(0.010)
    p.note("8 GPUs run at once, so ~28 GPU-hours of work finishes in about 5 hours of real time.")

    p.gap(0.014)
    p.text("Currently running (MATH-500 + AIME):   about 6 hours.",
           size=11.5, weight="bold", dy=0.026)
    p.text("Adding GPQA-Diamond would make it about 10 hours.", size=9.8, dy=0.020, color=MUTED)

    p.head("Why it is much faster than we first thought")
    p.text("Our first estimate, scaled up from small trial runs, was about 48 hours. Before",
           size=9.6, dy=0.0185)
    p.text("committing the machines we ran a short test, which found three fixable problems —",
           size=9.6, dy=0.0185)
    p.text("in one of them the GPUs were reserving memory but doing no work at all. Fixing",
           size=9.6, dy=0.0185)
    p.text("them made the job roughly 5x cheaper.", size=9.6, dy=0.0185)

    p.gap(0.016); p.rule()
    p.note("MATH-500 figures are measured on the run in progress; the others are projected.")
    p.done(1)


# ── Page 2 ────────────────────────────────────────────────────────────────────
def page2(pdf):
    p = Page(pdf)
    p.title("How the estimate was calculated",
            "Time a small trial run, scale it up, then re-time the real thing")

    p.head("Step 1 — time a small trial")
    p.text("We ran 33 questions per dataset with 10 restarts each on a single GPU and timed it.",
           size=9.6, dy=0.020)
    p.table([
        "dataset            restarts     time taken     restarts per second",
        "MATH-500              4,470        53m 25s                    1.39",
        "GPQA-Diamond          4,400         3h 30m                    0.35",
        "AIME-2025             2,050         1h 55m                    0.30",
    ], size=8.8)
    p.gap(0.006)
    p.note("GPQA and AIME are ~4x slower simply because they write much more text per restart.")

    p.head("Step 2 — scale up to all questions and 100 restarts")
    p.text("Not every reasoning attempt is usable: some run past the length limit before they",
           size=9.6, dy=0.0185)
    p.text("finish, and those cannot be cut up. The usable ones give us the restart counts.",
           size=9.6, dy=0.020)
    p.table([
        "dataset          questions   attempts   usable   restarts to write",
        "MATH-500               500      1,500    1,442             691,900",
        "AIME-2025               30         90       73              36,500",
        "GPQA-Diamond           198       ~594     ~528             264,000",
    ], size=8.8)
    p.gap(0.008)
    p.text("Restarts divided by the trial speed gave our first estimate:", size=9.6, dy=0.020)
    p.table([
        "MATH-500        691,900 / 1.39 per sec   =   138 GPU-hours",
        "GPQA-Diamond    264,000 / 0.35 per sec   =   210 GPU-hours",
        "AIME-2025        36,500 / 0.30 per sec   =    34 GPU-hours",
        "TOTAL                                        382 GPU-hours   (~48 hours)",
    ], size=8.8, header=False, colors=[None, BAD, None, BAD])

    p.head("Step 3 — test first, then re-time")
    p.text("48 hours on 8 shared machines is a large commitment, so we ran a 15-minute test",
           size=9.6, dy=0.0185)
    p.text("first. It found the problems on page 3. After fixing them the measured speed went",
           size=9.6, dy=0.0185)
    p.text("from 1.39 to about 9.7 restarts per second per GPU — roughly 7x faster.",
           size=9.6, dy=0.020)
    p.table([
        "dataset          first estimate     after fixes     real time",
        "MATH-500            138 GPU-h         ~28 GPU-h      ~5 hours",
        "AIME-2025            34 GPU-h          ~7 GPU-h      ~1 hour",
        "GPQA-Diamond        210 GPU-h         ~43 GPU-h      ~5 hours",
    ], size=8.8)

    p.gap(0.012)
    p.note("The 8 GPUs do not finish together, because some questions are far longer than")
    p.note("others. The job is done only when the slowest one is, so we quote ~5 hours")
    p.note("rather than the ~3.5 hour average.")
    p.done(2)


# ── Page 3 ────────────────────────────────────────────────────────────────────
def page3(pdf):
    p = Page(pdf)
    p.title("What we found and fixed",
            "Problems that only appeared at the larger scale")

    p.head("1. Asking for 100 restarts at once did not fit in GPU memory")
    p.text("The GPU has to hold every in-progress restart in memory at the same time. Ten",
           size=9.6, dy=0.0185)
    p.text("fitted comfortably; a hundred needed more than twice the space available, so the",
           size=9.6, dy=0.0185)
    p.text("GPU spent its time shuffling data instead of generating — memory reserved, but",
           size=9.6, dy=0.0185)
    p.text("0% of its compute in use. Fixed by requesting the restarts in batches of 20.",
           size=9.6, dy=0.0185)
    p.gap(0.004)
    p.text("This one mattered most: unfixed, all 8 GPUs would have run overnight and produced",
           size=9.4, dy=0.0185, color=BAD)
    p.text("nothing. The trial run never showed it, because 10 restarts fitted easily.",
           size=9.4, dy=0.0185, color=BAD)

    p.head("2. The first stage kept the GPUs almost idle")
    p.text("The initial reasoning attempts were generated one question at a time, giving the",
           size=9.6, dy=0.0185)
    p.text("GPU only 3 pieces of work at once — a small fraction of what it can handle. Doing",
           size=9.6, dy=0.0185)
    p.text("all questions together made it ~5x faster; with 8 GPUs, this stage for MATH-500",
           size=9.6, dy=0.0185)
    p.text("went from an estimated 3.5 hours to about 5 minutes.", size=9.6, dy=0.0185)

    p.head("3. A setting was left over from the small trial")
    p.text("A limit on how much the GPU handles at once was still set for the 10-restart",
           size=9.6, dy=0.0185)
    p.text("trial. Raising it to match the new workload gave a further 1.9x speed-up.",
           size=9.6, dy=0.0185)

    p.head("4. AIME's length limit was distorting the results, not just the cost")
    p.text("More than half of AIME's reasoning attempts were still unfinished when they hit",
           size=9.6, dy=0.0185)
    p.text("the length limit, and had to be discarded. Doubling the limit recovered most:",
           size=9.6, dy=0.020)
    p.table([
        "                          old limit      new limit",
        "usable attempts            41 (46%)       73 (81%)",
        "accuracy                      30.0%          42.2%",
    ], size=8.8)
    p.gap(0.006)
    p.text("The accuracy rise is not the model improving. The old limit was cutting off exactly",
           size=9.6, dy=0.0185)
    p.text("the hard problems that need long reasoning, so our earlier AIME score of 30% was",
           size=9.6, dy=0.0185)
    p.text("too low — an artifact of our setup rather than a property of the model.",
           size=9.6, dy=0.0185)

    p.head("5. Why GPQA-Diamond was deferred")
    p.text("GPQA costs about 4x more per restart than MATH-500, because the model writes far",
           size=9.6, dy=0.0185)
    p.text("more text on it. That is a difficulty effect: the model is trained mainly on maths",
           size=9.6, dy=0.0185)
    p.text("and code, while GPQA is graduate biology, chemistry and physics, so it deliberates",
           size=9.6, dy=0.0185)
    p.text("instead of settling. It was set aside when it looked like 210 GPU-hours; at ~43 it",
           size=9.6, dy=0.0185)
    p.text("is now affordable and could be added back.", size=9.6, dy=0.0185)
    p.done(3)


if __name__ == "__main__":
    build()
