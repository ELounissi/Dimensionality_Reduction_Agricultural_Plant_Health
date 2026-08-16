"""Leakage-free evaluation protocol.

Two complementary resampling schemes are used, each where it is appropriate.

Classical models (DT, RF, k-NN)
    A stratified 80/20 train-test split. Inside the 80% training partition the
    hyperparameters are selected by a 5-fold cross-validated grid search. The
    scaler and the dimensionality reduction step live *inside* the
    cross-validation loop: at every split they are re-fitted on the four
    training folds and applied unchanged to the held-out fold. The winning
    configuration is refitted on the whole training partition and scored once on
    the untouched 20%.

Neural models (MLP, LSTM, GRU)
    A 70/10/20 train / validation / test partition, obtained by holding out 20%
    for testing and splitting the remaining 80% into 87.5% training and 12.5%
    validation. The scaler and the reducer are fitted on the 70% only. The
    validation partition drives early stopping and selects, per model family,
    the architecture, the dropout rate, the learning rate, and (for the
    recurrent models) the input window, from the explicit search spaces in
    ``models.NEURAL_GRIDS``. The test partition is used exactly once, to score
    the single selected candidate.

In both schemes no held-out observation takes part in any fitting or selection
decision, and the whole procedure is repeated over several random seeds.

The grid search is written explicitly rather than delegated to
``GridSearchCV`` so that the fit order (scaler, reducer, model, all on training
folds only) is visible in the source and can be audited line by line. It is also
substantially faster here, because the reducer is fitted once per fold instead
of once per candidate configuration.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .data import CropData
from .dr import make_reducer
from .models import CLASSICAL_GRIDS, NeuralModel, make_classical

TEST_SIZE = 0.20
VAL_FRACTION_OF_TRAIN = 0.125      # 0.8 * 0.125 = 0.10 -> a 70/10/20 partition
N_FOLDS = 5


@dataclass
class Result:
    crop: str
    task: str
    dr: str
    model: str
    seed: int
    score: float
    n_components: int
    fit_seconds: float
    predict_seconds: float
    n_train: int
    n_test: int
    seq_len: int = 1
    epochs_run: int = 0
    best_params: str = ""
    train_loss_history: str = ""
    val_loss_history: str = ""
    train_accuracy_history: str = ""
    val_accuracy_history: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _score(task: str, y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, zero_division=0) if task == "classification" else r2_score(y_true, y_pred)


def _selection_score(task: str, y_true, y_pred) -> float:
    """Metric used only for hyperparameter selection (as stated in the paper)."""
    return accuracy_score(y_true, y_pred) if task == "classification" else r2_score(y_true, y_pred)


def _outer_split(y, task: str, seed: int):
    """Reproducible plant-level 80/20 split used by every model."""
    idx = np.arange(len(y))
    strat = y if task == "classification" else None
    return train_test_split(idx, test_size=TEST_SIZE, random_state=seed,
                            shuffle=True, stratify=strat)


def _grid_points(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def _fit_view(X_fit: np.ndarray, others: tuple[np.ndarray, ...], dr_name: str,
              data: CropData, seed: int):
    """Fit imputer, scaler and reducer on ``X_fit`` only, then project partitions."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_fit)
    imputed = [imputer.transform(X_fit)] + [imputer.transform(o) for o in others]
    scaler = StandardScaler().fit(imputed[0])
    parts = [scaler.transform(p) for p in imputed]
    reducer = make_reducer(dr_name, data.feature_names, data.crop, random_state=seed)
    if reducer is not None:
        reducer.fit(parts[0])
        parts = [reducer.transform(p) for p in parts]
    return parts[0], parts[1:]


# --------------------------------------------------------------------------
# classical models
# --------------------------------------------------------------------------
def _score_candidates(model_name: str, base, candidates: list[dict], task: str,
                      Z_a, y_a, Z_b, y_b) -> np.ndarray:
    """Score every grid candidate of one model on one fold.

    For the Random Forest the ``n_estimators`` axis is traversed with
    ``warm_start``: a single ensemble is grown to the largest size and scored
    each time it reaches one of the requested sizes. This is mathematically
    identical to fitting one forest per size with the same seed, and it removes
    the largest single cost of the benchmark.
    """
    out = np.zeros(len(candidates))
    if model_name == "Random forest":
        sizes = sorted({c["n_estimators"] for c in candidates})
        rest_keys = [k for k in candidates[0] if k != "n_estimators"]
        groups: dict[tuple, list[int]] = {}
        for i, c in enumerate(candidates):
            groups.setdefault(tuple(c[k] for k in rest_keys), []).append(i)
        for combo, idx in groups.items():
            est = clone(base).set_params(warm_start=True, n_estimators=0,
                                         **dict(zip(rest_keys, combo)))
            by_size = {}
            for n in sizes:
                est.set_params(n_estimators=n)
                est.fit(Z_a, y_a)
                by_size[n] = _selection_score(task, y_b, est.predict(Z_b))
            for i in idx:
                out[i] = by_size[candidates[i]["n_estimators"]]
        return out
    for i, params in enumerate(candidates):
        est = clone(base).set_params(**params)
        est.fit(Z_a, y_a)
        out[i] = _selection_score(task, y_b, est.predict(Z_b))
    return out


def run_classical_all(data: CropData, dr_name: str, seed: int,
                      model_names: tuple[str, ...] = ("Decision tree", "Random forest", "k-NN"),
                      n_jobs: int = -1) -> list[Result]:
    """Cross-validated grid search for every classical model on one split.

    The five folds are built once and the scaler and reducer are fitted once per
    fold, then shared by all three models. This changes nothing statistically,
    since each model would otherwise receive exactly the same projection, and it
    removes a threefold duplication of the reduction step.
    """
    task = data.task
    tr, te = _outer_split(data.y, task, seed)
    X_tr, X_te, y_tr, y_te = data.X[tr], data.X[te], data.y[tr], data.y[te]
    cv = (StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
          if task == "classification" else KFold(N_FOLDS, shuffle=True, random_state=seed))
    grids = {m: _grid_points({k.replace("model__", ""): v for k, v in CLASSICAL_GRIDS[m].items()})
             for m in model_names}
    bases = {m: make_classical(m, task, random_state=seed) for m in model_names}

    def one_fold(a, b):
        Z_a, (Z_b,) = _fit_view(X_tr[a], (X_tr[b],), dr_name, data, seed)
        return {m: _score_candidates(m, bases[m], grids[m], task, Z_a, y_tr[a], Z_b, y_tr[b])
                for m in model_names}

    t0 = time.perf_counter()
    folds = list(cv.split(X_tr, y_tr))
    per_fold = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(one_fold)(a, b) for a, b in folds)
    n_used = len(folds)
    Z_tr, (Z_te,) = _fit_view(X_tr, (X_te,), dr_name, data, seed)
    shared_seconds = time.perf_counter() - t0

    results = []
    for m in model_names:
        mean_scores = np.mean([f[m] for f in per_fold[:n_used]], axis=0)
        best = grids[m][int(np.argmax(mean_scores))]
        t0 = time.perf_counter()
        final = clone(bases[m]).set_params(**best).fit(Z_tr, y_tr)
        fit_s = time.perf_counter() - t0 + shared_seconds / len(model_names)
        t0 = time.perf_counter()
        y_hat = final.predict(Z_te)
        predict_s = time.perf_counter() - t0
        results.append(Result(crop=data.crop, task=task, dr=dr_name, model=m, seed=seed,
                              score=float(_score(task, y_te, y_hat)),
                              n_components=int(Z_tr.shape[1]), fit_seconds=fit_s,
                              predict_seconds=predict_s, n_train=len(tr), n_test=len(te),
                              best_params=str(best)))
    return results


def run_classical(data: CropData, dr_name: str, model_name: str, seed: int) -> Result:
    """Single-model convenience wrapper around :func:`run_classical_all`."""
    return run_classical_all(data, dr_name, seed, model_names=(model_name,))[0]


# --------------------------------------------------------------------------
# neural models
# --------------------------------------------------------------------------
def prepare_neural_view(data: CropData, sequences: np.ndarray, dr_name: str, seed: int):
    """Split 70/10/20 and project the sequences, fitting on the 70% only.

    The reducer is estimated from the sampling-day rows of the training
    partition, which are independent observations, and is then applied to every
    time step of every partition. Fitting it on the flattened window instead
    would weight the estimate by seven highly autocorrelated copies of each
    observation, and would cost seven times more.
    """
    task = data.task
    tr_all, te = _outer_split(data.y, task, seed)
    strat = data.y[tr_all] if task == "classification" else None
    tr, va = train_test_split(tr_all, test_size=VAL_FRACTION_OF_TRAIN,
                              random_state=seed, shuffle=True, stratify=strat)

    n_feat = sequences.shape[2]
    seq_len = sequences.shape[1]
    t0 = time.perf_counter()
    fit_rows = sequences[tr][:, -1, :]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(fit_rows)
    scaler = StandardScaler().fit(imputer.transform(fit_rows))
    reducer = make_reducer(dr_name, data.feature_names, data.crop, random_state=seed)
    if reducer is not None:
        reducer.fit(scaler.transform(imputer.transform(fit_rows)))

    def project(block: np.ndarray) -> np.ndarray:
        n = len(block)
        flat = scaler.transform(imputer.transform(block.reshape(-1, n_feat)))
        if reducer is not None:
            flat = reducer.transform(flat)
        return flat.reshape(n, seq_len, flat.shape[1])

    views = {"train": (project(sequences[tr]), data.y[tr]),
             "val": (project(sequences[va]), data.y[va]),
             "test": (project(sequences[te]), data.y[te]),
             "indices": {"train": tr.copy(), "val": va.copy(), "test": te.copy()}}
    return views, time.perf_counter() - t0, (len(tr), len(va), len(te))


def run_neural(data: CropData, views_by_window: dict, dr_seconds: float, sizes,
               dr_name: str, model_name: str, seed: int) -> Result:
    """Validation-selected neural cell.

    Every candidate in the family's search space is trained on the training
    partition of every window the family examines; the candidate with the best
    validation score is the single model scored on the test partition. Ties are
    broken toward the earlier candidate, which makes the selection
    deterministic.
    """
    from .models import NEURAL_GRIDS, NEURAL_WINDOWS

    best = None
    t0 = time.perf_counter()
    for window in NEURAL_WINDOWS[model_name]:
        if window not in views_by_window:
            continue
        view = views_by_window[window]
        (X_tr, y_tr), (X_va, y_va) = view["train"], view["val"]
        for params in NEURAL_GRIDS[model_name]:
            model = NeuralModel(model_name, data.task, input_dim=X_tr.shape[2],
                                seed=seed, **params)
            model.fit(X_tr, y_tr, X_va, y_va)
            val_score = _selection_score(data.task, y_va, model.predict(X_va))
            if best is None or val_score > best[0]:
                best = (val_score, window, params, model)
    fit_s = time.perf_counter() - t0 + dr_seconds

    _, window, params, model = best
    X_te, y_te = views_by_window[window]["test"]
    t0 = time.perf_counter()
    y_hat = model.predict(X_te)
    predict_s = time.perf_counter() - t0

    return Result(crop=data.crop, task=data.task, dr=dr_name, model=model_name, seed=seed,
                  score=float(_score(data.task, y_te, y_hat)),
                  n_components=int(X_te.shape[2]),
                  fit_seconds=fit_s, predict_seconds=predict_s,
                  n_train=sizes[0], n_test=sizes[2],
                  seq_len=window,
                  epochs_run=model.epochs_run_,
                  best_params=json.dumps(dict(window=window, **params)),
                  train_loss_history=json.dumps(model.train_loss_history_),
                  val_loss_history=json.dumps(model.val_loss_history_),
                  train_accuracy_history=json.dumps(model.train_accuracy_history_),
                  val_accuracy_history=json.dumps(model.val_accuracy_history_))


def confidence_interval(values: np.ndarray, level: float = 0.95) -> float:
    """Half-width of the normal-approximation confidence interval of the mean."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
