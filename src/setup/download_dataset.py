#!/usr/bin/env python3
"""Download the full HF datasets (all splits) into <repo>/data/.

Usage:
    python download_dataset.py                       # download all datasets below
    python download_dataset.py aqua_rat openbookqa    # download a subset
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# hf_name, hf_config, filter_fn — kept in sync with DATASETS in run_eval.py,
# minus `split`: we now pull every split HF provides (train/validation/test/...).
DOWNLOADS: dict[str, dict] = {
    "aqua_rat": dict(
        hf_name="aqua_rat", hf_config="raw", filter_fn=None,
    ),
    "mmlu_college_math": dict(
        hf_name="cais/mmlu", hf_config="college_mathematics", filter_fn=None,
    ),
    "mmlu_pro_math": dict(
        hf_name="TIGER-Lab/MMLU-Pro", hf_config=None,
        filter_fn=lambda ex: ex["category"] == "math",
    ),
    "openbookqa": dict(
        hf_name="allenai/openbookqa", hf_config="main", filter_fn=None,
    ),
    "math500": dict(
        hf_name="HuggingFaceH4/MATH-500", hf_config=None, filter_fn=None,
    ),
    "aime2025": dict(
        hf_name="math-ai/aime25", hf_config=None, filter_fn=None,
    ),
    # 2022-2025, 30 problems per year, with a `year` field. Year matters: this
    # model's training data predates 2025, so 2022-24 are plausibly memorised
    # while 2025 is not -- keep year so that can be tested rather than assumed.
    "aime_2022_2025": dict(
        hf_name="allenai/aime-2022-2025", hf_config=None, filter_fn=None,
    ),
    "gpqa_diamond": dict(
        hf_name="Idavidrein/gpqa", hf_config="gpqa_diamond", filter_fn=None,
    ),
}


def download(name: str) -> None:
    cfg = DOWNLOADS[name]
    print(f"[{name}] downloading {cfg['hf_name']} ({cfg['hf_config'] or 'default'}, all splits) ...")

    if cfg["hf_config"]:
        dsd = load_dataset(cfg["hf_name"], cfg["hf_config"])
    else:
        dsd = load_dataset(cfg["hf_name"])

    if cfg["filter_fn"]:
        dsd = dsd.filter(cfg["filter_fn"])

    out_dir = DATA_DIR / name
    dsd.save_to_disk(str(out_dir))

    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for split, ds in dsd.items():
        ds.to_csv(str(csv_dir / f"{split}.csv"), index=False)

    counts = ", ".join(f"{split}={len(ds)}" for split, ds in dsd.items())
    print(f"[{name}] saved [{counts}] -> {out_dir} (+ csv/)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets", nargs="*",
        help=f"Dataset keys to download (default: all). Choices: {', '.join(DOWNLOADS)}",
    )
    args = parser.parse_args()

    names = args.datasets or list(DOWNLOADS)
    unknown = [n for n in names if n not in DOWNLOADS]
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)} (choices: {', '.join(DOWNLOADS)})")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        download(name)


if __name__ == "__main__":
    main()
