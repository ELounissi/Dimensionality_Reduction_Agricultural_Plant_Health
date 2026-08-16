#!/usr/bin/env python3
"""Compare the benchmark results with the originally published scores.

The published per-cell means (six models, twelve tables) are shipped as
published_baseline.csv; this script aligns them with results/runs.csv and
reports per-model deltas."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    pub = pd.read_csv(HERE.parent / "data" / "published_baseline.csv")
    runs = pd.read_csv(Path(args.results) / "runs.csv")
    new = runs.groupby(["task", "crop", "dr", "model"], as_index=False).agg(
        new=("score", "mean"))
    cmp = pub.merge(new, on=["task", "crop", "dr", "model"], how="left")
    cmp["delta"] = (cmp.new - cmp.published).round(4)
    for task in ("classification", "regression"):
        sub = cmp[cmp.task == task]
        print(f"\n=== {task}: new vs published (mean over 18 cells/model) ===")
        agg = sub.groupby("model").agg(
            published=("published", "mean"), new=("new", "mean"),
            mean_delta=("delta", "mean"),
            cells_up=("delta", lambda s: int((s > 0.01).sum())),
            cells_close=("delta", lambda s: int(((s >= -0.01) & (s <= 0.01)).sum())),
            cells_down=("delta", lambda s: int((s < -0.01).sum())))
        print(agg.round(4).to_string())
    cmp.to_csv(Path(args.results) / "vs_published.csv", index=False)


if __name__ == "__main__":
    main()
