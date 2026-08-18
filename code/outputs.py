#!/usr/bin/env python3
"""Generate every benchmark artifact from results/runs.csv.

Writes clean, ready-to-use CSV tables (per-representation score tables,
win/loss tables, runtime summary, overall rankings) under outputs/tables
and publication-quality figures under outputs/figures. Run it after
run_all.py, or directly on the shipped results to regenerate everything.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DRO = ["No Reduction", "PCA", "KernelPCA", "Isomap", "MDS", "BOTCAST"]
MODELS = ["Decision tree", "Random forest", "k-NN", "MLP", "LSTM", "RNN GRU",
          "Mamba SSM", "TabPFN"]
CROPS = ["Carrot", "Lettuce", "Onion"]
ci = lambda s: 1.96 * s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0


def score_tables(runs: pd.DataFrame, out: Path) -> None:
    """One tidy CSV per task and representation: mean, 95% CI half-width,
    and a best-in-crop marker for every model and crop."""
    g = runs.groupby(["task", "crop", "dr", "model"]).score
    mean, half = g.mean(), g.apply(lambda s: ci(s))
    for task in sorted(runs.task.unique()):
        for dr in DRO:
            rows = []
            present = [m for m in MODELS if (task, CROPS[0], dr, m) in mean.index]
            best = {c: max(present, key=lambda m: mean.get((task, c, dr, m), -9))
                    for c in CROPS}
            for m in present:
                row = {"model": m}
                for c in CROPS:
                    key = (task, c, dr, m)
                    row[f"{c}_mean"] = round(float(mean[key]), 4)
                    row[f"{c}_ci95"] = round(float(half[key]), 4)
                    row[f"{c}_best"] = best[c] == m
                rows.append(row)
            name = f"scores_{task}_{dr.replace(' ', '_')}.csv"
            pd.DataFrame(rows).to_csv(out / name, index=False)


def win_loss_tables(runs: pd.DataFrame, out: Path) -> None:
    """Combined wins/losses per model-representation pairing, computed with
    the 0.01 two-axis rule, with the best pairing flagged per column."""
    cells = runs.groupby(["task", "crop", "dr", "model"], as_index=False).agg(
        m=("score", "mean"))
    for task in sorted(runs.task.unique()):
        present = [m for m in MODELS
                   if not cells[(cells.task == task) & (cells.model == m)].empty]
        counts = {(m, d): [0, 0] for m in present for d in DRO}
        for crop in CROPS:
            p = (cells[(cells.task == task) & (cells.crop == crop)]
                 .pivot(index="model", columns="dr", values="m")
                 .reindex(index=present, columns=DRO))
            for m in present:
                row = p.loc[m]
                for d, v in row.items():
                    counts[(m, d)][0] += int(v >= row.max() - 0.01)
                    counts[(m, d)][1] += int(v <= row.min() + 0.01)
            for d in DRO:
                col = p[d]
                for m, v in col.items():
                    counts[(m, d)][0] += int(v >= col.max() - 0.01)
                    counts[(m, d)][1] += int(v <= col.min() + 0.01)
        col_mean = {(m, d): float(cells[(cells.task == task)
                                        & (cells.model == m)
                                        & (cells.dr == d)].m.mean())
                    for m in present for d in DRO}
        best = {d: max(present,
                       key=lambda m: (counts[(m, d)][0] - counts[(m, d)][1],
                                      counts[(m, d)][0], col_mean[(m, d)]))
                for d in DRO}
        rows = []
        for m in present:
            row = {"model": m}
            for d in DRO:
                w, l = counts[(m, d)]
                row[d] = f"{w}/{-l if l else 0}"
                row[f"{d}_best"] = best[d] == m
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / f"win_loss_{task}.csv", index=False)


def rankings(runs: pd.DataFrame, out: Path) -> None:
    """Overall model rankings per task (mean over crops and representations)."""
    cells = runs.groupby(["task", "crop", "dr", "model"], as_index=False).agg(
        m=("score", "mean"))
    rows = []
    for task in sorted(runs.task.unique()):
        r = (cells[cells.task == task].groupby("model").m.mean()
             .sort_values(ascending=False))
        for rank, (m, v) in enumerate(r.items(), start=1):
            rows.append(dict(task=task, rank=rank, model=m,
                             overall_score=round(float(v), 4)))
    pd.DataFrame(rows).to_csv(out / "rankings.csv", index=False)


def timing_table(runs: pd.DataFrame, out: Path) -> None:
    """Mean end-to-end experimental unit time per representation, with the
    speedup relative to No Reduction."""
    unit = runs.groupby(["task", "crop", "dr", "seed"]).fit_seconds.sum()
    t = unit.groupby(["task", "dr"]).mean().unstack("task")
    rows = []
    for d in DRO:
        row = {"method": d}
        for task in t.columns:
            base = t.loc["No Reduction", task]
            row[f"{task}_seconds"] = round(float(t.loc[d, task]), 3)
            row[f"{task}_speedup"] = round(float(base / t.loc[d, task]), 3)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "timing.csv", index=False)


def figures(runs: pd.DataFrame, out: Path) -> None:
    """Boxplot comparisons per crop: models grouped by representation and
    representations grouped by model, for each task."""
    # Type sized for reading at column width in print.
    plt.rcParams.update({
        "font.size": 17, "axes.titlesize": 23, "axes.labelsize": 20,
        "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 17,
    })
    colors = list(plt.get_cmap("tab10").colors[:8])
    cells = runs.groupby(["task", "crop", "dr", "model"], as_index=False).agg(
        m=("score", "mean"))
    panels = [(t, by) for t in sorted(runs.task.unique())
              for by in ("dr", "model")]
    for fig_no, (task, by) in enumerate(panels, start=2):
        metric = "F1" if task == "classification" else r"$R^2$"
        order = DRO if by == "dr" else MODELS
        fig, axes = plt.subplots(3, 1, figsize=(12, 20))
        for panel, (ax, crop) in enumerate(zip(axes, CROPS)):
            block = cells[(cells.task == task) & (cells.crop == crop)]
            vals = [block[block[by] == name]["m"].dropna().to_numpy()
                    for name in order]
            vals = [v if len(v) else np.array([np.nan]) for v in vals]
            bp = ax.boxplot(vals, patch_artist=True, widths=0.5)
            for box, color in zip(bp["boxes"], colors):
                box.set_facecolor(color)
                box.set_edgecolor("black")
            for key in ("whiskers", "caps", "medians"):
                for line in bp[key]:
                    line.set_color("black")
            # Scale each panel to its boxes and whiskers rather than to its
            # extreme outliers. A single weak seed would otherwise stretch the
            # axis and squeeze every box into a narrow band at the top, leaving
            # most of the panel empty. Outliers outside the resulting view are
            # not drawn; the boxes, whiskers and medians are unchanged.
            wy = np.concatenate([w.get_ydata() for w in bp["whiskers"]])
            wy = wy[np.isfinite(wy)]
            if len(wy):
                lo, hi = float(wy.min()), float(wy.max())
                pad = 0.10 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.05
                ylo, yhi = lo - pad, hi + pad
                for fl in bp["fliers"]:
                    xd, yd = fl.get_xdata(), fl.get_ydata()
                    keep = (yd >= ylo) & (yd <= yhi)
                    fl.set_data(xd[keep], yd[keep])
                ax.set_ylim(ylo, yhi)
            ax.grid(axis="y", alpha=0.25, linewidth=0.8)
            ax.set_axisbelow(True)
            labels = [("kNN" if x == "k-NN" else "Botcast" if x == "BOTCAST"
                       else x) for x in order]
            ax.set_xticks(range(1, len(order) + 1), labels, rotation=35,
                          ha="right")
            ax.set_ylabel(f"{metric} Score")
            ax.set_title(f"{task.title()} - {crop}")
            fig.text(.012, .985 - panel * .333, f"({chr(97 + panel)})",
                     fontsize=26, fontweight="bold")
        fig.tight_layout(rect=[0.02, 0, 1, 1])
        fig.savefig(out / f"Figure_{fig_no}.png", dpi=170,
                    bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results",
                    help="directory containing runs.csv")
    args = ap.parse_args()
    runs = pd.read_csv(Path(args.results) / "runs.csv")
    tables = ROOT / "outputs" / "tables"
    figs = ROOT / "outputs" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    score_tables(runs, tables)
    win_loss_tables(runs, tables)
    rankings(runs, tables)
    timing_table(runs, tables)
    figures(runs, figs)
    print(f"wrote {len(list(tables.glob('*.csv')))} tables and "
          f"{len(list(figs.glob('*.png')))} figures under outputs/")


if __name__ == "__main__":
    main()
