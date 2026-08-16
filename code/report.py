#!/usr/bin/env python3
"""Rankings and combined win/loss tables from runs.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DRO = ["No Reduction", "PCA", "KernelPCA", "Isomap", "MDS", "BOTCAST"]
MODELS = ["Decision tree", "Random forest", "k-NN", "MLP", "LSTM", "RNN GRU",
          "Mamba SSM", "TabPFN"]


def win_loss(cells: pd.DataFrame, task: str) -> tuple[dict, dict]:
    counts = {(m, d): [0, 0] for m in MODELS for d in DRO}
    for crop in ("Carrot", "Lettuce", "Onion"):
        p = (cells[(cells.task == task) & (cells.crop == crop)]
             .pivot(index="model", columns="dr", values="m")
             .reindex(index=MODELS, columns=DRO))
        for m in MODELS:
            row = p.loc[m]
            for d, v in row.items():
                counts[(m, d)][0] += int(v >= row.max() - 0.01)
                counts[(m, d)][1] += int(v <= row.min() + 0.01)
        for d in DRO:
            col = p[d]
            for m, v in col.items():
                counts[(m, d)][0] += int(v >= col.max() - 0.01)
                counts[(m, d)][1] += int(v <= col.min() + 0.01)
    col_mean = {(m, d): float(cells[(cells.task == task) & (cells.model == m)
                                    & (cells.dr == d)].m.mean())
                for m in MODELS for d in DRO}
    best = {d: max(MODELS, key=lambda m: (counts[(m, d)][0] - counts[(m, d)][1],
                                          counts[(m, d)][0], col_mean[(m, d)]))
            for d in DRO}
    return counts, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    root = Path(args.results)
    runs = pd.read_csv(root / "runs.csv")
    cells = runs.groupby(["task", "crop", "dr", "model"], as_index=False).agg(
        m=("score", "mean"))
    lines = ["# All-models comparison: rankings and win/loss tables", ""]
    for task in ("classification", "regression"):
        r = cells[cells.task == task].groupby("model").m.mean().sort_values(
            ascending=False)
        lines += [f"## {task}", "",
                  "Overall means: " + "  >  ".join(f"{m} {v:.4f}"
                                                   for m, v in r.items()), ""]
        counts, best = win_loss(cells, task)
        lines += ["| Model | " + " | ".join(DRO) + " |",
                  "|" + "---|" * (len(DRO) + 1)]
        for m in MODELS:
            row = []
            for d in DRO:
                w, l = counts[(m, d)]
                cell = f"{w}/{-l}" if l else f"{w}/0"
                row.append(f"**{cell}**" if best[d] == m else cell)
            lines.append(f"| {m} | " + " | ".join(row) + " |")
        lines += ["", "Column bests: " + ", ".join(f"{d}: {best[d]}"
                                                   for d in DRO), ""]
    (root / "win_loss.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
