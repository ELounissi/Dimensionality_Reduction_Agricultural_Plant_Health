#!/usr/bin/env python3
"""Regenerate figures, timing summaries, and win-loss tables from runs.csv."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DR_ORDER = ["No Reduction", "PCA", "KernelPCA", "Isomap", "MDS", "BOTCAST"]
MODEL_ORDER = ["Decision tree", "Random forest", "k-NN", "MLP", "LSTM", "RNN GRU"]
CROPS = ["Carrot", "Lettuce", "Onion"]
COLORS = list(plt.get_cmap("tab10").colors[:6])
DR_COLORS = [COLORS[0], COLORS[1], COLORS[2], COLORS[4], COLORS[5], COLORS[3]]


def _analysis_rows(runs: pd.DataFrame, split: str = "random") -> pd.DataFrame:
    sub = runs[runs["split"] == split].copy()
    sub = sub[((sub.seq_len == 1) & ~((sub.dr == "BOTCAST") & (sub.model == "RNN GRU"))) |
              ((sub.seq_len == 7) & (sub.dr == "BOTCAST") & (sub.model == "RNN GRU"))]
    return (sub.groupby(["task", "crop", "dr", "model"], as_index=False)
            .agg(score=("score", "mean")))


def _style_boxplot(bp, colors=COLORS) -> None:
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_edgecolor("black")
    for key in ("whiskers", "caps", "medians"):
        for line in bp[key]:
            line.set_color("black")


def comparison_figure(data: pd.DataFrame, task: str, by: str, output: Path) -> None:
    metric = "F1" if task == "classification" else r"$R^2$"
    if by == "dr":
        order, other = DR_ORDER, "model"
        title_base = "Comparison of Dimensionality Reduction Methods"
        xlabel = "Dimensionality Reduction Methods"
    else:
        order, other = MODEL_ORDER, "dr"
        title_base = "Comparison of ML and DL Algorithms"
        xlabel = "ML and DL Algorithms"

    fig, axes = plt.subplots(3, 1, figsize=(9, 18.5))
    for panel, (ax, crop) in enumerate(zip(axes, CROPS)):
        block = data[(data.task == task) & (data.crop == crop)]
        vals = [block[block[by] == name].set_index(other).reindex(
            MODEL_ORDER if by == "dr" else DR_ORDER)["score"].dropna().to_numpy()
                for name in order]
        bp = ax.boxplot(vals, patch_artist=True, widths=0.5)
        _style_boxplot(bp, DR_COLORS if by == "dr" else COLORS)
        labels = [("kNN" if x == "k-NN" else "Botcast" if x == "BOTCAST" else x)
                  for x in order]
        ax.set_xticks(range(1, len(order) + 1), labels, rotation=40)
        ax.set_ylabel(f"{metric} Score")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{title_base} - {task.title()} -  {crop}")
    fig.subplots_adjust(left=.12, right=.98, bottom=.08, top=.98, hspace=.38)
    for panel, ax in enumerate(axes):
        pos = ax.get_position()
        fig.text(.015, min(pos.y1 + .005, .985), f"({chr(97 + panel)})",
                 fontsize=20, fontweight="bold")
    fig.savefig(output, dpi=100)
    plt.close(fig)


def training_curve_figure(runs: pd.DataFrame, output: Path) -> None:
    chosen = runs[(runs.split == "random") & (runs.task == "classification") &
                  (runs.crop == "Onion") & (runs.dr == "No Reduction") &
                  (runs.model == "RNN GRU") & (runs.seed == 0)]
    if chosen.empty:
        raise RuntimeError("representative GRU run for Figure 1 is missing")
    row = chosen.sort_values("seq_len").iloc[-1]
    train_loss = np.asarray(json.loads(row.train_loss_history), dtype=float)
    val_loss = np.asarray(json.loads(row.val_loss_history), dtype=float)
    train_acc = np.asarray(json.loads(row.train_accuracy_history), dtype=float)
    val_acc = np.asarray(json.loads(row.val_accuracy_history), dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.8))
    axes[0].plot(train_loss, label="Train Loss")
    axes[0].plot(val_loss, label="Validation Loss")
    axes[0].set(title="Train and Validation Loss", xlabel="Epochs", ylabel="Loss")
    axes[0].legend(); axes[0].grid(True, alpha=.5)
    axes[1].plot(train_acc, label="Train Accuracy")
    axes[1].plot(val_acc, label="Validation Accuracy")
    axes[1].set(title="Train and Validation Accuracy", xlabel="Epochs", ylabel="Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=.5)
    axes[0].text(-0.06, 1.03, "(a)", transform=axes[0].transAxes, fontweight="bold", fontsize=12)
    axes[1].text(-0.06, 1.03, "(b)", transform=axes[1].transAxes, fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=100, bbox_inches="tight")
    plt.close(fig)


def pairing_win_loss(data: pd.DataFrame, root: Path) -> None:
    """Tables 17-18: combined near-best/near-worst counts along both axes.

    For every crop, a pairing receives a win when it lies within 0.01 of the
    best DR method for that model and separately when it lies within 0.01 of the
    best model for that DR method. Losses are defined analogously against the
    worst value. Thus each cell has at most six wins and six losses across the
    three crops, matching the reported combined-count definition.
    """
    lines = ["# Combined pairing wins/losses", "",
             "Threshold: within 0.01 of the best/worst score; both the within-model "
             "DR comparison and within-DR model comparison are counted."]
    records = []
    for task in ("classification", "regression"):
        counts = {(m, d): [0, 0] for m in MODEL_ORDER for d in DR_ORDER}
        for crop in CROPS:
            block = data[(data.task == task) & (data.crop == crop)]
            pivot = block.pivot(index="model", columns="dr", values="score").reindex(
                index=MODEL_ORDER, columns=DR_ORDER)
            for model in MODEL_ORDER:
                row = pivot.loc[model]
                for dr, value in row.items():
                    counts[(model, dr)][0] += int(value >= row.max() - 0.01)
                    counts[(model, dr)][1] += int(value <= row.min() + 0.01)
            for dr in DR_ORDER:
                col = pivot[dr]
                for model, value in col.items():
                    counts[(model, dr)][0] += int(value >= col.max() - 0.01)
                    counts[(model, dr)][1] += int(value <= col.min() + 0.01)
        lines += ["", f"## {task}", "", "| Model | " + " | ".join(DR_ORDER) + " |",
                  "|" + "---|" * (len(DR_ORDER) + 1)]
        for model in MODEL_ORDER:
            cells = []
            for dr in DR_ORDER:
                wins, losses = counts[(model, dr)]
                cells.append(f"{wins}/-{losses}")
                records.append(dict(task=task, model=model, dr=dr,
                                    wins=wins, losses=losses, threshold=0.01))
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
    (root / "pairing_win_loss.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(records).to_csv(root / "pairing_win_loss.csv", index=False)

def timing_artifacts(root: Path) -> None:
    timing = pd.read_csv(root / "timing_comparison.csv")
    timing = timing[timing.split == "random"].copy()
    timing["dr"] = pd.Categorical(timing.dr, DR_ORDER, ordered=True)
    timing = timing.sort_values(["task", "dr"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, task in zip(axes, ("classification", "regression")):
        block = timing[timing.task == task]
        ax.bar(range(len(block)), block.mean_seconds, color=DR_COLORS, edgecolor="black")
        ax.set_xticks(range(len(block)),
                      ["Botcast" if str(x) == "BOTCAST" else str(x) for x in block.dr],
                      rotation=40, ha="right")
        ax.set_title(task.title())
        ax.set_xlabel("Dimensionality Reduction Methods")
        ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("Mean Experimental Unit Time (seconds)")
    fig.suptitle("Runtime Comparison With and Without Dimensionality Reduction")
    fig.tight_layout()
    fig.savefig(root / "figures" / "Timing_Comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/cluster")
    ap.add_argument("--split", default="random", choices=["random"])
    args = ap.parse_args()
    root = Path(args.results)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(root / "runs.csv")
    if "split" not in runs:
        runs["split"] = "random"
    analysis = _analysis_rows(runs, args.split)
    training_curve_figure(runs, figures / "Figure_1.png")
    comparison_figure(analysis, "classification", "dr", figures / "Figure_2.png")
    comparison_figure(analysis, "regression", "dr", figures / "Figure_3.png")
    comparison_figure(analysis, "classification", "model", figures / "Figure_4.png")
    comparison_figure(analysis, "regression", "model", figures / "Figure_5.png")
    pairing_win_loss(analysis, root)
    timing_artifacts(root)
    print(f"wrote five study figures, the timing figure, and win-loss tables under {root}")


if __name__ == "__main__":
    main()
