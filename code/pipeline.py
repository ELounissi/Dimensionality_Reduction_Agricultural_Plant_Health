"""Shared data pipeline: every model family consumes its natural
representation of the released data.

* Single-day models (Decision tree, Random forest, k-NN, MLP, TabPFN)
  receive the sampling-day design row, projected by the cell's
  dimensionality-reduction transform (fitted on training rows only and
  re-standardized).
* Sequence models (LSTM, RNN GRU, Mamba SSM) receive the plant's last K
  visits as steps. Each step carries that visit day's projected design row
  plus its observed context: the severity and other-disease scores recorded
  at that (past) visit, the inverse-distance neighbor severity of that
  (past) sweep from the released plant-distance files, the time gap, a
  step mask, and an observed flag. The sampling-day step carries the last
  observed values of the same quantities. No model receives any same-day
  cross-plant information: the sequence models' advantage is exactly the
  temporal structure the dataset documents, nothing else.

All preprocessing (imputation, standardization, reduction) is fitted on
training rows only. Plant-pair distances are aligned for the 0/1-based id
offset between the distance files and the observation tables.

Evaluation convention: one-step-ahead prediction with observed history.
Every prediction at plant p, day t uses only observations dated strictly
before t, and a row's own target never appears in its own feature vector,
directly or indirectly.

Because the outer split holds out whole plants (see dr_agri/splits.py),
each plant's visit history lies entirely on one side of the partition.
Every sequence is therefore complete: a training row's history consists of
earlier visits to the same, training-side plant, and a scored row keeps the
full history a deployed model would hold for that plant. No held-out
observation reaches a fitting input, so the fitted weights are a function
of the training partition alone, and no sequence has to be truncated to
achieve it.

Only the cross-plant channel needs scoping: the inverse-distance neighbor
severity of a sweep aggregates the other plants observed in that sweep, so
when inputs are built for fitting or selection the aggregate is restricted
to the fitting partition (``allowed`` in :meth:`Cell.sequences`, which also
filters history steps should a caller pass a partition that cuts across
plants).

The pipeline's remaining leakage-free guarantees concern fitting:
imputation, standardization, dimensionality reduction, hyperparameter
selection, and early stopping use training-side data only, and each test
partition is scored exactly once per run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = HERE                                # dr_agri ships in this package
sys.path.insert(0, str(REPO))
from dr_agri.data import CROPS, load_crop  # noqa: E402
from dr_agri.dr import make_reducer  # noqa: E402

DRO = ["No Reduction", "PCA", "KernelPCA", "Isomap", "MDS", "BOTCAST"]
MODELS = ["Decision tree", "Random forest", "k-NN", "MLP", "LSTM",
          "RNN GRU", "Mamba SSM", "TabPFN"]
N_STEP = 8       # per-step context channels of the sequence input


class Cell:
    """Raw structures of one (crop, task), shared by every model."""

    def __init__(self, crop: str, task: str):
        self.data = load_crop(crop, task)
        self.task = task
        frame = self.data.frame
        cfg = CROPS[crop]
        self.n = len(frame)
        self.y = self.data.y
        self.sev = (frame[cfg["target"]].astype(float) / cfg["max_score"]).to_numpy()
        # ordinal scales are known a priori: cote_* run 0-4, incidence_* are binary
        self.others = np.stack(
            [frame[c].astype(float).to_numpy() / (4.0 if c.startswith("cote") else 1.0)
             for c in cfg["other_targets"]], axis=1)
        self.farm = frame["FarmID"].to_numpy()
        self.plant = frame["Plant_ID"].to_numpy()
        self.date = frame["Date"].to_numpy()
        # one group per monitored plant: the unit the outer split holds out,
        # which keeps each visit history whole (see dr_agri/splits.py)
        self.group = np.unique(
            np.stack([self.farm, self.plant]).T.astype(str), axis=0,
            return_inverse=True)[1]

        # chronological prior-visit lists per plant
        order = np.lexsort((self.date, self.plant, self.farm))
        self.past: list[list[int]] = [[] for _ in range(self.n)]
        seen: dict[tuple, list[int]] = {}
        for idx in order:
            key = (self.farm[idx], self.plant[idx])
            hist = seen.setdefault(key, [])
            self.past[idx] = hist.copy()
            hist.append(idx)
        self.prev = np.array([p[-1] if p else -1 for p in self.past])

        # plant-pair distances from the released files, with the plant
        # identifier conventions aligned to the observation tables
        dist = pd.read_csv(ROOT / "data" / crop / "field_distance.txt",
                   header=None)
        obs_ids = set(int(p) for p in self.plant)
        raw_ids = set(int(v) for v in pd.concat([dist[1], dist[2]]))
        shift = max((0, 1), key=lambda s: len({i + s for i in raw_ids} & obs_ids))
        self.w: dict[tuple, dict[int, float]] = {}
        for f, a, b, d in dist.itertuples(index=False):
            wt = 1.0 / (1.0 + float(d))
            a, b = int(a) + shift, int(b) + shift
            self.w.setdefault((f, a), {})[b] = wt
            self.w.setdefault((f, b), {})[a] = wt

        self.by_sweep: dict[tuple, list[int]] = {}
        for i in range(self.n):
            self.by_sweep.setdefault((self.farm[i], self.date[i]), []).append(i)

        # neighbor severity of each row's own sweep: used only as PAST
        # information (each history step's sweep predates the prediction day)
        self.press = np.array([self._idw(i) for i in range(self.n)])
        self._press_cache: dict[bytes, np.ndarray] = {}

    def _idw(self, i: int, allowed: np.ndarray | None = None) -> float:
        rows = self.by_sweep[(self.farm[i], self.date[i])]
        weights = self.w.get((self.farm[i], int(self.plant[i])), {})
        num = den = tot = cnt = 0.0
        for j in rows:
            if j == i or (allowed is not None and not allowed[j]):
                continue
            tot += self.sev[j]
            cnt += 1
            wj = weights.get(int(self.plant[j]))
            if wj:
                num += wj * self.sev[j]
                den += wj
        if den > 0:
            return num / den
        return tot / cnt if cnt else 0.0

    def _pressure(self, allowed: np.ndarray | None) -> np.ndarray:
        """Per-row IDW neighbor severity drawing only on allowed rows."""
        if allowed is None:
            return self.press
        key = allowed.tobytes()
        if key not in self._press_cache:
            self._press_cache[key] = np.array(
                [self._idw(i, allowed) for i in range(self.n)])
        return self._press_cache[key]

    def project(self, train_idx: np.ndarray, dr_name: str, seed: int) -> np.ndarray:
        """Impute, standardize, reduce, re-standardize; fits on train only."""
        X = self.data.X
        imputer = SimpleImputer(strategy="median",
                                keep_empty_features=True).fit(X[train_idx])
        Xi = imputer.transform(X)
        Z = StandardScaler().fit(Xi[train_idx]).transform(Xi)
        reducer = make_reducer(dr_name, self.data.feature_names,
                               self.data.crop, random_state=seed)
        if reducer is not None:
            reducer.fit(Z[train_idx])
            Z = reducer.transform(Z)
            Z = StandardScaler().fit(Z[train_idx]).transform(Z)
        return Z.astype(np.float32)

    def sequences(self, idxs: np.ndarray, Z: np.ndarray, K: int,
                  allowed: np.ndarray | None = None) -> np.ndarray:
        """(len(idxs), K+1, Z-dim + 8) sequence input for the recurrent and
        state-space models; strictly backward-looking."""
        # Past observations only: every value written here is dated strictly
        # before the prediction day t (the current step carries the last
        # *observed* context, never the row's own target). See the module
        # docstring for the one-step-ahead evaluation convention.
        # `allowed` restricts the history sources. Inputs built for fitting
        # or selection pass their fitting partition, so held-out rows are
        # simply absent from every training-side sequence - no channel of a
        # training input, and hence no gradient, depends on a held-out
        # observation. Scored rows pass None: they condition on the full
        # past observed before their own day, as in deployment.
        press = self._pressure(allowed)
        d = Z.shape[1]
        out = np.zeros((len(idxs), K + 1, d + N_STEP), dtype=np.float32)
        for r, i in enumerate(idxs):
            hist = (self.past[i] if allowed is None
                    else [j for j in self.past[i] if allowed[j]])
            steps = hist[-K:] + [i]
            offset = K + 1 - len(steps)
            prev = hist[-1] if hist else None
            for s, j in enumerate(steps):
                row = out[r, offset + s]
                row[:d] = Z[j]
                current = j == i
                src = prev if current else j
                if src is not None:
                    row[d + 0] = self.sev[src]
                    row[d + 1] = self.others[src, 0]
                    row[d + 2] = self.others[src, 1]
                    row[d + 3] = press[src]
                row[d + 4] = (self.date[i] - self.date[j]) / 30.0
                row[d + 5] = 0.5 if current else 1.0
                row[d + 6] = 1.0 if src is not None else 0.0
                row[d + 7] = ((self.date[i] - self.date[prev]) / 30.0
                              if (current and prev is not None) else 0.0)
        return out
