#!/usr/bin/env python
"""Generate summary tables and plots from a completed eval sweep.

Usage
-----
python src/viz/report.py --sweep-dir runs/sweep
# Outputs → runs/sweep/report/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Load results ──────────────────────────────────────────────────────────────

DATASET_LABELS = {
    "gsm8k":        "GSM8K",
    "aqua_rat":     "AQuA-RAT",
    "mmlu_college_math": "MMLU (college math)",
    "mmlu_pro_math":     "MMLU-Pro (math)",
    "arc_challenge":"ARC-Chall.",
    "openbookqa":   "OpenBookQA",
}
MODEL_LABELS = {
    "DeepSeek-R1-Distill-Qwen-7B":  "Qwen-7B",
    "DeepSeek-R1-Distill-Llama-8B": "Llama-8B",
}
DATASET_ORDER = ["gsm8k", "aqua_rat", "mmlu_college_math", "mmlu_pro_math", "arc_challenge", "openbookqa"]
MODEL_ORDER   = list(MODEL_LABELS)


def load_summaries(sweep_dir: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sweep_dir.rglob("overall_summary.json"):
        data = json.loads(summary_path.read_text())
        rows.append(data)
    if not rows:
        raise FileNotFoundError(f"No overall_summary.json files found under {sweep_dir}")
    df = pd.DataFrame(rows)
    df["dataset_label"] = df["dataset"].map(DATASET_LABELS).fillna(df["dataset"])
    df["model_label"]   = df["model_slug"].map(MODEL_LABELS).fillna(df["model_slug"])
    return df


def load_example_metrics(sweep_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sweep_dir.rglob("example_metrics.csv"):
        df = pd.read_csv(path)
        # infer dataset and model from path: sweep/dataset/model_slug/example_metrics.csv
        parts = path.parts
        model_slug = parts[-2]
        dataset    = parts[-3]
        df["dataset"]    = dataset
        df["model_slug"] = model_slug
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _heatmap(pivot: pd.DataFrame, title: str, fmt: str, vmin: float | None,
             vmax: float | None, cmap: str, out: Path, annot_fmt: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns) * 1.6), max(3, len(pivot.index) * 0.9)))
    mask = pivot.isna()
    annot = pivot.applymap(lambda v: annot_fmt.format(v) if annot_fmt and not np.isnan(v) else "")
    sns.heatmap(pivot, ax=ax, annot=annot if annot_fmt else True, fmt="" if annot_fmt else fmt,
                cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0.5, linecolor="white",
                mask=mask, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def _bar_grouped(df: pd.DataFrame, metric: str, ylabel: str, title: str, out: Path,
                 pct: bool = False) -> None:
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].values]
    models   = [m for m in MODEL_ORDER   if m in df["model_slug"].values]

    x     = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 1.4), 4))

    colors = ["#4C72B0", "#DD8452"]
    for i, model in enumerate(models):
        sub = df[df["model_slug"] == model].set_index("dataset")
        vals = [sub.loc[d, metric] if d in sub.index else np.nan for d in datasets]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=MODEL_LABELS.get(model, model),
                      color=colors[i], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                label = f"{val:.0%}" if pct else f"{val:.1f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.01 if pct else bar.get_height() * 0.03),
                        label, ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.legend(framealpha=0.9)
    if pct:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, min(1.05, ax.get_ylim()[1] * 1.15))
    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", default="/playpen-ssd/yuvraj/llm-probing-project/runs/sweep")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    report_dir = sweep_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {sweep_dir} …")
    df = load_summaries(sweep_dir)
    ex = load_example_metrics(sweep_dir)

    print(f"Found {len(df)} completed runs.\n")

    # ── Console table ─────────────────────────────────────────────────────────
    table_cols = ["dataset", "model_slug", "pass_at_1", "majority_vote_accuracy",
                  "mean_answer_entropy_bits", "mean_reasoning_tokens", "parse_success_rate"]
    available = [c for c in table_cols if c in df.columns]
    pretty = df[available].copy()
    pretty["model_slug"] = pretty["model_slug"].map(MODEL_LABELS).fillna(pretty["model_slug"])
    pretty["dataset"]    = pretty["dataset"].map(DATASET_LABELS).fillna(pretty["dataset"])
    for col in ["pass_at_1", "majority_vote_accuracy", "parse_success_rate"]:
        if col in pretty.columns:
            pretty[col] = pretty[col].map(lambda v: f"{v:.1%}")
    if "mean_reasoning_tokens" in pretty.columns:
        pretty["mean_reasoning_tokens"] = pretty["mean_reasoning_tokens"].map(lambda v: f"{v:.0f}")
    if "mean_answer_entropy_bits" in pretty.columns:
        pretty["mean_answer_entropy_bits"] = pretty["mean_answer_entropy_bits"].map(lambda v: f"{v:.3f}")
    pretty.columns = ["Dataset", "Model", "pass@1", "Maj. vote", "Entropy (bits)",
                       "Avg reasoning toks", "Parse %"][: len(pretty.columns)]
    print(pretty.to_string(index=False))
    print()

    # ── Heatmaps ─────────────────────────────────────────────────────────────
    sns.set_theme(style="white", font_scale=1.05)

    for metric, title, fmt, vmin, vmax, cmap, annot_fmt in [
        ("pass_at_1",               "pass@1 Accuracy",           ".0%", 0, 1,   "YlGn",  "{:.0%}"),
        ("majority_vote_accuracy",  "Majority-Vote Accuracy",    ".0%", 0, 1,   "YlGn",  "{:.0%}"),
        ("mean_answer_entropy_bits","Mean Answer Entropy (bits)", ".2f", 0, None,"OrRd",  "{:.2f}"),
        ("mean_reasoning_tokens",   "Mean Reasoning Tokens",      ".0f", 0, None,"Blues", "{:.0f}"),
    ]:
        if metric not in df.columns:
            continue
        # rows = datasets, cols = models
        ds_order = [d for d in DATASET_ORDER if d in df["dataset"].values]
        mo_order = [m for m in MODEL_ORDER   if m in df["model_slug"].values]
        pivot = df.pivot_table(index="dataset", columns="model_slug",
                               values=metric, aggfunc="mean")
        pivot = pivot.reindex(index=ds_order, columns=mo_order)
        pivot.index   = [DATASET_LABELS.get(d, d) for d in pivot.index]
        pivot.columns = [MODEL_LABELS.get(m, m) for m in pivot.columns]

        _heatmap(pivot, title, fmt, vmin, vmax, cmap,
                 report_dir / f"heatmap_{metric}.png", annot_fmt)

    # ── Bar charts ────────────────────────────────────────────────────────────
    _bar_grouped(df, "pass_at_1", "Accuracy", "pass@1 Accuracy by Dataset & Model",
                 report_dir / "bar_accuracy.png", pct=True)
    _bar_grouped(df, "mean_reasoning_tokens", "Avg tokens",
                 "Mean Reasoning Tokens by Dataset & Model",
                 report_dir / "bar_reasoning_tokens.png", pct=False)
    if "mean_answer_entropy_bits" in df.columns:
        _bar_grouped(df, "mean_answer_entropy_bits", "Entropy (bits)",
                     "Mean Answer Entropy by Dataset & Model",
                     report_dir / "bar_entropy.png", pct=False)

    # ── Scatter: reasoning tokens vs accuracy ─────────────────────────────────
    if not ex.empty and {"mean_reasoning_tokens", "fraction_correct"}.issubset(ex.columns):
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {"DeepSeek-R1-Distill-Qwen-7B": "#4C72B0",
                  "DeepSeek-R1-Distill-Llama-8B": "#DD8452"}
        for model_slug, grp in ex.groupby("model_slug"):
            ax.scatter(grp["mean_reasoning_tokens"], grp["fraction_correct"],
                       alpha=0.4, s=20,
                       color=colors.get(model_slug, "grey"),
                       label=MODEL_LABELS.get(model_slug, model_slug))
        ax.set_xlabel("Mean reasoning tokens (per example)")
        ax.set_ylabel("Fraction correct")
        ax.set_title("Reasoning length vs. accuracy (per example)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.legend(framealpha=0.9, fontsize=8)
        sns.despine(ax=ax)
        plt.tight_layout()
        fig.savefig(report_dir / "scatter_tokens_vs_accuracy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved scatter_tokens_vs_accuracy.png")

    # ── Markdown summary ──────────────────────────────────────────────────────
    if "pass_at_1" in df.columns and "model_slug" in df.columns:
        pivot_md = df.pivot_table(index="dataset", columns="model_slug",
                                  values="pass_at_1", aggfunc="mean")
        ds_order = [d for d in DATASET_ORDER if d in pivot_md.index]
        mo_order = [m for m in MODEL_ORDER   if m in pivot_md.columns]
        pivot_md = pivot_md.reindex(index=ds_order, columns=mo_order)
        pivot_md.index   = [DATASET_LABELS.get(d, d) for d in pivot_md.index]
        pivot_md.columns = [MODEL_LABELS.get(m, m) for m in pivot_md.columns]

        lines = ["# Eval Sweep — pass@1 Accuracy\n"]
        lines.append("| Dataset | " + " | ".join(pivot_md.columns) + " |")
        lines.append("|" + "---|" * (len(pivot_md.columns) + 1))
        for idx, row in pivot_md.iterrows():
            vals = " | ".join(f"{v:.1%}" if not pd.isna(v) else "—" for v in row)
            lines.append(f"| {idx} | {vals} |")
        md_path = report_dir / "summary.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  saved summary.md")

    print(f"\nAll outputs written to {report_dir}/")

    # ── PDF ───────────────────────────────────────────────────────────────────
    build_pdf(df, pretty, report_dir)


def build_pdf(df: pd.DataFrame, summary_table: pd.DataFrame, report_dir: Path) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mpimg

    pdf_path = report_dir / "sweep_report.pdf"

    # ordered list of (image_path, exists?)
    wide_plots = [
        report_dir / "bar_accuracy.png",
        report_dir / "bar_reasoning_tokens.png",
        report_dir / "bar_entropy.png",
    ]
    heatmaps = [
        report_dir / "heatmap_pass_at_1.png",
        report_dir / "heatmap_majority_vote_accuracy.png",
        report_dir / "heatmap_mean_answer_entropy_bits.png",
        report_dir / "heatmap_mean_reasoning_tokens.png",
    ]
    scatter = report_dir / "scatter_tokens_vs_accuracy.png"

    with PdfPages(pdf_path) as pdf:
        # ── Page 1: title + summary table ────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("white")

        fig.text(0.5, 0.93, "DeepSeek-R1 Distill — Eval Sweep Results",
                 ha="center", va="top", fontsize=18, fontweight="bold")
        fig.text(0.5, 0.88, "(partial sweep — mmlu_pro_math & arc_challenge/Llama-8B pending)",
                 ha="center", va="top", fontsize=10, color="#666666", style="italic")

        ax_tbl = fig.add_axes([0.05, 0.12, 0.90, 0.68])
        ax_tbl.axis("off")

        col_labels = list(summary_table.columns)
        cell_text  = summary_table.values.tolist()
        tbl = ax_tbl.table(cellText=cell_text, colLabels=col_labels,
                           loc="upper center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#cccccc")
            if row == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f5f5f5")
            else:
                cell.set_facecolor("white")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Pages 2–4: wide bar charts (one per page) ────────────────────────
        for img_path in wide_plots:
            if not img_path.exists():
                continue
            fig, ax = plt.subplots(figsize=(11, 5.5))
            ax.imshow(mpimg.imread(img_path))
            ax.axis("off")
            plt.tight_layout(pad=0.5)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Page 5: 2×2 heatmap grid ─────────────────────────────────────────
        existing_hm = [p for p in heatmaps if p.exists()]
        if existing_hm:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            for ax, img_path in zip(axes.flat, existing_hm):
                ax.imshow(mpimg.imread(img_path))
                ax.axis("off")
            for ax in axes.flat[len(existing_hm):]:
                ax.set_visible(False)
            plt.suptitle("Heatmaps", fontsize=13, fontweight="bold", y=1.01)
            plt.tight_layout(pad=1.0)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Page 6: scatter ───────────────────────────────────────────────────
        if scatter.exists():
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.imshow(mpimg.imread(scatter))
            ax.axis("off")
            plt.tight_layout(pad=0.5)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        d = pdf.infodict()
        d["Title"]   = "DeepSeek-R1 Distill Eval Sweep"
        d["Subject"] = "LLM probing sweep results"

    print(f"  saved sweep_report.pdf")


if __name__ == "__main__":
    main()
