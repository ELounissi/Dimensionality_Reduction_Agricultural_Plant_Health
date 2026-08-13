#!/usr/bin/env python3
"""Generate Wilcoxon and pairwise statistical results."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

MODEL_ORDER = ["Decision tree", "Random forest", "k-NN", "MLP", "LSTM", "RNN GRU"]
DR_ORDER = ["No Reduction", "PCA", "MDS", "Isomap", "KernelPCA", "BOTCAST"]


def paired(df, column, a, b, keys):
    left = df[df[column] == a].set_index(keys)["score"]
    right = df[df[column] == b].set_index(keys)["score"]
    joined = pd.concat([left, right], axis=1, join="inner").dropna()
    x, y = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
    if len(joined) < 5:
        return np.nan, np.nan, len(joined)
    p = 1.0 if np.allclose(x, y) else float(wilcoxon(x, y).pvalue)
    return p, float(np.median(x) - np.median(y)), len(joined)


def holm(values):
    p = np.asarray(values, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    adjusted = np.minimum(1.0, np.maximum.accumulate(
        [(len(order) - rank) * p[idx] for rank, idx in enumerate(order)]))
    for idx, value in zip(order, adjusted):
        out[idx] = value
    return out.tolist()


def pair_matrix(df, column, order, keys, family, records):
    pairs = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            p, difference, n = paired(df, column, a, b, keys)
            pairs.append((a, b, p, difference, n))
    adjusted = holm([x[2] for x in pairs])
    lookup = {}
    for (a, b, p, difference, n), p_holm in zip(pairs, adjusted):
        lookup[(a, b)] = (p, p_holm, difference)
        lookup[(b, a)] = (p, p_holm, -difference)
        records.append(dict(task="pooled", family=family, a=a, b=b,
                            p_raw=p, p_holm=p_holm,
                            median_difference=difference, pairs=n))
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/cluster")
    parser.add_argument("--split", default="random", choices=["random"])
    args = parser.parse_args()
    root = Path(args.results)
    runs = pd.read_csv(root / "runs.csv")
    runs = runs[runs.split == args.split]
    runs = runs[((runs.seq_len == 1) & ~((runs.dr == "BOTCAST") & (runs.model == "RNN GRU"))) |
                ((runs.seq_len == 7) & (runs.dr == "BOTCAST") & (runs.model == "RNN GRU"))]
    cells = (runs.groupby(["task", "crop", "dr", "model"], as_index=False)
                  .agg(score=("score", "mean")))

    records = []
    baseline = {}
    for task in ("classification", "regression"):
        sub = cells[cells.task == task]
        values = []
        for dr in DR_ORDER[1:]:
            p, difference, n = paired(sub, "dr", dr, "No Reduction", ["crop", "model"])
            values.append((dr, p, difference, n))
        baseline[task] = values
        for (dr, p, difference, n), p_holm in zip(values, holm([x[1] for x in values])):
            records.append(dict(task=task, family="dr_vs_baseline", a=dr,
                                b="No Reduction", p_raw=p, p_holm=p_holm,
                                median_difference=difference, pairs=n))

    dr_lookup = pair_matrix(cells, "dr", DR_ORDER, ["task", "crop", "model"],
                            "dr_pairwise", records)
    model_lookup = pair_matrix(cells, "model", MODEL_ORDER, ["task", "crop", "dr"],
                               "model_pairwise", records)

    lines = ["# Wilcoxon signed-rank tests", "",
             "Seeds are averaged before testing. Raw p-values are shown in the readable tables. "
             "Holm-adjusted p-values are saved in statistical_tests.csv.", "",
             "## Dimensionality reduction against No Reduction", "",
             "| Task | " + " | ".join(DR_ORDER[1:]) + " |",
             "|" + "---|" * len(DR_ORDER)]
    for task in ("classification", "regression"):
        lines.append("| " + task + " | " + " | ".join(
            f"{p:.4f}{'*' if p < .05 else ''} ({difference:+.4f})"
            for _, p, difference, _ in baseline[task]) + " |")
    for title, lookup, order in (("DR pairwise comparison", dr_lookup, DR_ORDER),
                                 ("Model pairwise comparison", model_lookup, MODEL_ORDER)):
        lines += ["", f"## {title}", "", "| | " + " | ".join(order) + " |",
                  "|" + "---|" * (len(order) + 1)]
        for a in order:
            row = []
            for b in order:
                if a == b:
                    row.append("---")
                else:
                    p, _, difference = lookup[(a, b)]
                    row.append(f"{p:.4f}{'*' if p < .05 else ''} ({'+' if difference >= 0 else '-'})")
            lines.append(f"| {a} | " + " | ".join(row) + " |")
    (root / "statistical_tests.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(records).to_csv(root / "statistical_tests.csv", index=False)

    print("STATISTICAL_RESULTS_OK")


if __name__ == "__main__":
    main()
