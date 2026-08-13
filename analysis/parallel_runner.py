#!/usr/bin/env python3
"""Parallel driver for the full benchmark.

The experiment grid is embarrassingly parallel over the unit
``(task, crop, reduction, seed)``: each unit owns its own train/test split and
shares nothing with the others. This driver farms the units out over the cores
of the current machine, keeps every worker single-threaded so the cores are not
oversubscribed, and tolerates the failure of an individual unit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

# every worker must stay single-threaded: the parallelism is at the unit level
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dr_agri import (CROPS, DR_METHODS, MODEL_ORDER, build_sequences,  # noqa: E402
                     confidence_interval, dataset_summary, load_crop,
                     run_classical_all, run_neural)
from dr_agri.evaluate import prepare_neural_view  # noqa: E402

NEURAL = ["MLP", "LSTM", "RNN GRU"]


def _limit_threads() -> None:
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def run_unit(task: str, crop: str, dr: str, seed: int, seq_lens: list[int],
             skip_classical: bool) -> dict:
    """One independent cell of the grid. Never raises: failures are reported."""
    _limit_threads()
    t0 = time.perf_counter()
    rows: list[dict] = []
    try:
        data = load_crop(crop, task)
        if not skip_classical:
            rows.extend(r.as_dict() for r in
                        run_classical_all(data, dr, seed, n_jobs=1))
        for sl in seq_lens:
            seqs = build_sequences(data, sl)
            view, dr_s, sizes = prepare_neural_view(data, seqs, dr, seed)
            # The released study uses one time step for MLP, LSTM and the
            # standard GRU runs. Its BOTCAST GRU uses the seven-day window.
            models = (["MLP", "LSTM"] + ([] if dr == "BOTCAST" else ["RNN GRU"])) if sl == 1 else (
                     ["RNN GRU"] if dr == "BOTCAST" and sl == 7 else [])
            for model in models:
                rows.append(run_neural(data, view, dr_s, sizes, dr, model, seed, sl).as_dict())
        for r in rows:
            r["split"] = "random"
        return dict(ok=True, rows=rows, unit=(task, crop, dr, seed),
                    seconds=time.perf_counter() - t0, error="")
    except Exception:
        # Never mix a partially completed unit into aggregate tables.
        return dict(ok=False, rows=[], unit=(task, crop, dr, seed),
                    seconds=time.perf_counter() - t0, error=traceback.format_exc())


def source_manifest(root: Path) -> dict:
    """SHA-256 of every source and data file, so a run can be traced back."""
    digests = {}
    for pattern in ("dr_agri/*.py", "analysis/*.py", "*.py", "requirements.txt",
                    "data/*/crop_no_sensitive_data.csv", "data/*/combined_daily_meteo.csv",
                    "data/*/datasets_vars/Botcast.txt"):
        for path in sorted(root.glob(pattern)):
            digests[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed-values", type=int, nargs="+", default=None,
                    help="explicit seeds for a distributed shard; overrides --seeds")
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[1, 7])
    ap.add_argument("--crops", nargs="*", default=list(CROPS))
    ap.add_argument("--dr", nargs="*", default=list(DR_METHODS))
    ap.add_argument("--tasks", nargs="*", default=["classification", "regression"])
    ap.add_argument("--n-jobs", type=int,
                    default=os.cpu_count() or 1)
    ap.add_argument("--skip-classical", action="store_true")
    ap.add_argument("--out", default="results/reproduced")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out = Path(args.out)
    (out / "manifests").mkdir(parents=True, exist_ok=True)

    seed_values = args.seed_values if args.seed_values is not None else list(range(args.seeds))
    units = [(task, crop, dr, seed)
             for task in args.tasks for crop in args.crops
             for dr in args.dr for seed in seed_values]
    print(f"workers={args.n_jobs} units={len(units)} "
          f"seeds={seed_values} seq_lens={args.seq_lens}", flush=True)

    started = time.perf_counter()
    outcomes = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10, batch_size=1)(
        delayed(run_unit)(t, c, d, s, args.seq_lens, args.skip_classical)
        for t, c, d, s in units)
    wall = time.perf_counter() - started

    failures = [o for o in outcomes if not o["ok"]]
    rows = [r for o in outcomes for r in o["rows"]]
    if failures:
        failure_log = out / "failures.json"
        failure_log.write_text(json.dumps(failures, indent=2, default=str), encoding="utf-8")
        print(f"FAILED_UNITS {len(failures)}; details saved to {failure_log}",
              file=sys.stderr, flush=True)
        print(failures[0]["error"], file=sys.stderr, flush=True)
    if not rows:
        raise SystemExit("every unit failed; see failures.json")

    df = pd.DataFrame(rows)
    df["seq_len"] = df["seq_len"].astype(int)
    if "split" not in df.columns:
        df["split"] = "random"
    df.to_csv(out / "runs.csv", index=False)

    unit_rows = []
    for o in outcomes:
        task, crop, dr, seed = o["unit"]
        unit_rows.append(dict(task=task, crop=crop, dr=dr, seed=seed, split="random",
                              seconds=o["seconds"], ok=o["ok"]))
    unit_times = pd.DataFrame(unit_rows)
    unit_times.to_csv(out / "unit_timings.csv", index=False)
    good = unit_times[unit_times.ok].copy()
    baseline = (good[good.dr == "No Reduction"]
                .rename(columns={"seconds": "baseline_seconds"})
                [["task", "crop", "seed", "split", "baseline_seconds"]])
    paired_times = good.merge(baseline, on=["task", "crop", "seed", "split"], how="inner")
    paired_times["speedup_vs_no_reduction"] = (
        paired_times["baseline_seconds"] / paired_times["seconds"])
    timing_comparison = (paired_times.groupby(["split", "task", "dr"], as_index=False)
                         .agg(mean_seconds=("seconds", "mean"),
                              median_seconds=("seconds", "median"),
                              mean_speedup_vs_no_reduction=("speedup_vs_no_reduction", "mean"),
                              median_speedup_vs_no_reduction=("speedup_vs_no_reduction", "median"),
                              paired_runs=("seconds", "size")))
    timing_comparison.to_csv(out / "timing_comparison.csv", index=False)

    agg = (df.groupby(["split", "task", "crop", "dr", "model", "seq_len"])
             .agg(score_mean=("score", "mean"),
                  score_ci=("score", lambda s: confidence_interval(s.to_numpy())),
                  score_std=("score", "std"),
                  fit_seconds=("fit_seconds", "mean"),
                  predict_seconds=("predict_seconds", "mean"),
                  n_components=("n_components", "first"),
                  epochs=("epochs_run", "mean"),
                  n_runs=("score", "size"))
             .reset_index())
    agg.to_csv(out / "aggregated.csv", index=False)

    with open(out / "tables.md", "w") as fh:
        fh.write("# Result tables (leakage-free protocol)\n\n")
        fh.write(f"Seeds: {args.seeds}. Recurrent input window: {max(args.seq_lens)} time steps. "
                 f"Score: F1 for classification, R2 for regression, on the held-out test "
                 f"partition, mean +/- 95% confidence interval over seeds.\n")
        for task in args.tasks:
            metric = "F1" if task == "classification" else "R2"
            for dr in args.dr:
                sub = agg[(agg.task == task) & (agg.dr == dr) & (agg.split == "random")]
                sub = sub[((sub.seq_len == 1) & ~((sub.dr == "BOTCAST") & (sub.model == "RNN GRU"))) |
                          ((sub.seq_len == 7) & (sub.dr == "BOTCAST") & (sub.model == "RNN GRU"))]
                if sub.empty:
                    continue
                m = sub.pivot(index="model", columns="crop", values="score_mean").reindex(MODEL_ORDER)
                c = sub.pivot(index="model", columns="crop", values="score_ci").reindex(MODEL_ORDER)
                cols = [x for x in ["Carrot", "Lettuce", "Onion"] if x in m.columns]
                fh.write(f"\n## {metric} scores, {dr}\n\n")
                fh.write("| Models | " + " | ".join(cols) + " |\n")
                fh.write("|" + "---|" * (len(cols) + 1) + "\n")
                for model in MODEL_ORDER:
                    if model not in m.index:
                        continue
                    fh.write(f"| {model} | " + " | ".join(
                        f"{m.loc[model, x]:.4f} ± {c.loc[model, x]:.4f}" for x in cols) + " |\n")

    timings = (df.groupby(["dr", "model"])[["fit_seconds", "predict_seconds"]]
                 .mean().reset_index())
    timings.to_csv(out / "timings.csv", index=False)

    manifest = dict(
        workers=args.n_jobs,
        seeds=seed_values,
        seq_lens=args.seq_lens,
        crops=args.crops,
        dr=args.dr,
        tasks=args.tasks,
        splits=["random"],
        primary_split="random",
        units_total=len(units),
        units_failed=len(failures),
        rows=len(df),
        wall_clock_seconds=round(wall, 1),
        core_seconds=round(sum(o["seconds"] for o in outcomes), 1),
        slowest_unit=max(outcomes, key=lambda o: o["seconds"])["unit"],
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        scipy=scipy.__version__,
        sklearn=sklearn.__version__,
        torch=torch.__version__,
        platform=platform.platform(),
        source_sha256=source_manifest(root),
    )
    (out / "manifests" / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    dataset_summary().to_csv(out / "dataset_summary.csv", index=False)

    if failures:
        with open(out / "manifests" / "failures.log", "w") as fh:
            for f in failures:
                fh.write(f"UNIT_FAILED {f['unit']}\n{f['error']}\n\n")

    status = out / "manifests" / f"job_{manifest['job_id'] or 'local'}.status"
    status.write_text(json.dumps(dict(state="COMPLETED" if not failures else "COMPLETED_WITH_FAILURES",
                                      units_failed=len(failures), rows=len(df),
                                      wall_clock_seconds=manifest["wall_clock_seconds"]), indent=2))

    print(f"\n{len(df)} rows from {len(units) - len(failures)}/{len(units)} units "
          f"in {wall:.0f}s wall ({manifest['core_seconds']:.0f}s core) -> {out}", flush=True)
    if failures:
        print(f"WARNING {len(failures)} units failed, see manifests/failures.log", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
