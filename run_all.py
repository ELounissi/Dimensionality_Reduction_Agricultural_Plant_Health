#!/usr/bin/env python3
"""Eight-model comparison of ML/DL approaches x dimensionality reduction.

Models and their natural inputs:
  Decision tree, Random forest, k-NN, MLP, TabPFN - the sampling-day design
  row (projected by the cell's DR transform, fitted on training rows only).
  LSTM, RNN GRU, Mamba SSM - the plant's last K visits as sequence steps
  (strictly backward-looking; see pipeline.py).

Protocol per (task, crop, reduction, seed): a shared deterministic 80/20
split. Classical models select hyperparameters by five-fold CV inside the
80% and refit on the full 80%. TabPFN uses its pretrained prior as-is (no
hyperparameter search - the prior is the method). Neural models carve a 10%
validation share out of the 80% for early stopping (400 epochs, patience
20, best epoch restored); their architecture is selected once per cell by
two repeats of five-fold CV on the selection split. The untouched 20% test
partition is scored exactly once per run. One model per run; no ensembling.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, r2_score  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from nets import make_net, predict_net, train_net  # noqa: E402
from pipeline import DRO, Cell  # noqa: E402

REPO = HERE / "code"
sys.path.insert(0, str(REPO))
from dr_agri.evaluate import VAL_FRACTION_OF_TRAIN, _outer_split  # noqa: E402
from dr_agri.models import CLASSICAL_GRIDS, make_classical  # noqa: E402

SELECTION_SEED = 0
CLASSICAL = ("Decision tree", "Random forest", "k-NN")
SEQ_MODELS = ("LSTM", "RNN GRU", "Mamba SSM")
RNN_GRID = [dict(K=k, units=u) for k, u in product((2, 4, 6), (32, 64))]
MLP_GRID = [dict(hidden=h) for h in ((64, 32), (32, 16), (128, 64))]


def selection_score(task, y, y_hat):
    return accuracy_score(y, y_hat) if task == "classification" else r2_score(y, y_hat)


def final_score(task, y, y_hat):
    return (f1_score(y, y_hat, zero_division=0) if task == "classification"
            else r2_score(y, y_hat))


_CELLS: dict[tuple, Cell] = {}


def CELL(crop, task):
    key = (crop, task)
    if key not in _CELLS:
        _CELLS[key] = Cell(crop, task)
    return _CELLS[key]


# ------------------------------------------------------------------ classical
def _grid(name):
    keys = [k.replace("model__", "") for k in CLASSICAL_GRIDS[name]]
    return [dict(zip(keys, v)) for v in product(*CLASSICAL_GRIDS[name].values())]


def run_classical(crop, task, dr, name, seed):
    cell = CELL(crop, task)
    y = cell.y
    tr, te = _outer_split(y, task, seed)
    t0 = time.perf_counter()
    cv = (StratifiedKFold(5, shuffle=True, random_state=seed)
          if task == "classification" else KFold(5, shuffle=True, random_state=seed))
    grid = _grid(name)
    base = make_classical(name, task, random_state=seed)
    scores = np.zeros(len(grid))
    for a, b in cv.split(tr, y[tr]):
        ia, ib = tr[a], tr[b]
        Z = cell.project(ia, dr, seed)
        for gi, params in enumerate(grid):
            est = clone(base).set_params(**params).fit(Z[ia], y[ia])
            scores[gi] += selection_score(task, y[ib], est.predict(Z[ib]))
    best = grid[int(np.argmax(scores))]
    Z = cell.project(tr, dr, seed)
    final = clone(base).set_params(**best).fit(Z[tr], y[tr])
    y_hat = final.predict(Z[te])
    return dict(task=task, crop=crop, dr=dr, model=name, seed=seed,
                score=float(final_score(task, y[te], y_hat)),
                fit_seconds=time.perf_counter() - t0,
                n_train=len(tr), n_test=len(te), best_params=json.dumps(best))


# ------------------------------------------------------------------ TabPFN
def run_tabpfn(crop, task, dr, seed):
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    device = os.environ.get("TABPFN_DEVICE", "cpu")
    cell = CELL(crop, task)
    y = cell.y
    tr, te = _outer_split(y, task, seed)
    t0 = time.perf_counter()
    Z = cell.project(tr, dr, seed)
    model = (TabPFNClassifier(device=device, random_state=seed)
             if task == "classification"
             else TabPFNRegressor(device=device, random_state=seed))
    model.fit(Z[tr], y[tr])
    y_hat = np.asarray(model.predict(Z[te]))
    if task == "regression":
        y_hat = np.clip(y_hat, 0.0, 1.0)
    return dict(task=task, crop=crop, dr=dr, model="TabPFN", seed=seed,
                score=float(final_score(task, y[te], y_hat)),
                fit_seconds=time.perf_counter() - t0,
                n_train=len(tr), n_test=len(te),
                best_params=json.dumps({"model": "tabpfn-v3-default"}))


# ------------------------------------------------------------------ neural
def _views(cell, dr, seed, model, cfg, fit_idx, idx_sets):
    Z = cell.project(fit_idx, dr, seed)
    if model == "MLP":
        return [Z[ix] for ix in idx_sets]
    return [cell.sequences(ix, Z, cfg["K"]) for ix in idx_sets]


def cv_neural(crop, task, dr, model, cfg):
    cell = CELL(crop, task)
    y = cell.y
    tr_all, _ = _outer_split(y, task, SELECTION_SEED)
    scores = []
    for repeat in (0, 1):
        cv = (StratifiedKFold(5, shuffle=True, random_state=repeat)
              if task == "classification"
              else KFold(5, shuffle=True, random_state=repeat))
        for a, b in cv.split(tr_all, y[tr_all]):
            ia, ib = tr_all[a], tr_all[b]
            Xa, Xb = _views(cell, dr, SELECTION_SEED, model, cfg, ia, [ia, ib])
            net = make_net(model, task, dict(cfg, seed=SELECTION_SEED),
                           Xa.shape[-1])
            net, _ = train_net(net, task, Xa, y[ia], Xb, y[ib], SELECTION_SEED)
            scores.append(selection_score(task, y[ib], predict_net(net, task, Xb)))
    return dict(task=task, crop=crop, dr=dr, model=model, **cfg,
                mean_cv_score=float(np.mean(scores)))


def run_neural(crop, task, dr, model, cfg, seed):
    cell = CELL(crop, task)
    y = cell.y
    tr_all, te = _outer_split(y, task, seed)
    strat = y[tr_all] if task == "classification" else None
    tr, va = train_test_split(tr_all, test_size=VAL_FRACTION_OF_TRAIN,
                              random_state=seed, shuffle=True, stratify=strat)
    t0 = time.perf_counter()
    X_tr, X_va, X_te = _views(cell, dr, seed, model, cfg, tr, [tr, va, te])
    net = make_net(model, task, dict(cfg, seed=seed), X_tr.shape[-1])
    net, epochs = train_net(net, task, X_tr, y[tr], X_va, y[va], seed)
    y_hat = predict_net(net, task, X_te)
    return dict(task=task, crop=crop, dr=dr, model=model, seed=seed,
                score=float(final_score(task, y[te], y_hat)),
                fit_seconds=time.perf_counter() - t0, epochs_run=epochs,
                n_train=len(tr), n_test=len(te), best_params=json.dumps(cfg))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    all_models = ["Decision tree", "Random forest", "k-NN", "MLP",
                  "LSTM", "RNN GRU", "Mamba SSM", "TabPFN"]
    ap.add_argument("--tasks", nargs="+",
                    default=["classification", "regression"],
                    help="tasks to run (classification, regression)")
    ap.add_argument("--crops", nargs="+",
                    default=["Carrot", "Lettuce", "Onion"],
                    help="crops to include")
    ap.add_argument("--dr", nargs="+", default=DRO,
                    help="dimensionality-reduction methods to include")
    ap.add_argument("--models", nargs="+", default=all_models,
                    help="models to include")
    ap.add_argument("--exclude-models", nargs="+", default=[],
                    help="models to leave out of the selected set")
    ap.add_argument("--exclude-dr", nargs="+", default=[],
                    help="reductions to leave out of the selected set")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                    help="explicit seed values (default: 0-9)")
    ap.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1,
                    help="parallel worker processes")
    ap.add_argument("--out", default="results",
                    help="output directory for runs.csv")
    args = ap.parse_args()
    args.models = [m for m in args.models if m not in args.exclude_models]
    args.dr = [d for d in args.dr if d not in args.exclude_dr]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = [(c, t, d) for t in args.tasks for c in args.crops for d in args.dr]
    t0 = time.perf_counter()

    cv_jobs = [(c, t, d, m, cfg) for c, t, d in cells
               for m, grid in (("MLP", MLP_GRID), ("LSTM", RNN_GRID),
                               ("RNN GRU", RNN_GRID), ("Mamba SSM", RNN_GRID))
               if m in args.models
               for cfg in grid]
    print(f"cells={len(cells)} neural-cv jobs={len(cv_jobs)}", flush=True)
    if cv_jobs:
        sel = pd.DataFrame(Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5,
                                    batch_size=1)(
            delayed(cv_neural)(c, t, d, m, cfg) for c, t, d, m, cfg in cv_jobs))
        sel.to_csv(out / "neural_selection.csv", index=False)
        best_rows = (sel.sort_values("mean_cv_score", ascending=False,
                                     kind="mergesort")
                     .groupby(["task", "crop", "dr", "model"], as_index=False)
                     .first())
        best_rows.to_csv(out / "neural_best.csv", index=False)
    else:
        best_rows = pd.DataFrame(
            columns=["task", "crop", "dr", "model", "hidden", "K", "units"])

    def cfg_of(t, c, d, m):
        r = best_rows[(best_rows.task == t) & (best_rows.crop == c)
                      & (best_rows.dr == d) & (best_rows.model == m)].iloc[0]
        if m == "MLP":
            h = r["hidden"]
            return dict(hidden=tuple(int(x) for x in
                                     (h if isinstance(h, tuple) else eval(str(h)))))
        return dict(K=int(r["K"]), units=int(r["units"]))
    print(f"selection done in {time.perf_counter()-t0:.0f}s", flush=True)

    t1 = time.perf_counter()
    jobs = ([delayed(run_classical)(c, t, d, m, s) for c, t, d in cells
             for m in CLASSICAL if m in args.models for s in args.seeds]
            + ([delayed(run_tabpfn)(c, t, d, s) for c, t, d in cells
                for s in args.seeds] if "TabPFN" in args.models else [])
            + [delayed(run_neural)(c, t, d, m, cfg_of(t, c, d, m), s)
               for c, t, d in cells for m in ("MLP",) + SEQ_MODELS
               if m in args.models for s in args.seeds])
    rows = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5, batch_size=1)(jobs)
    pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
    print(f"finals: {len(rows)} runs in {time.perf_counter()-t1:.0f}s -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
