"""Dimensionality reduction methods, all usable inside a leakage-free pipeline.

Every transformer here exposes ``fit`` / ``transform`` separately, so it can be
fitted on a training partition and applied unchanged to held-out data. This is
what makes the evaluation protocol leakage free.

``MDS`` in scikit-learn is transductive: it exposes ``fit_transform`` but no
``transform``, so it cannot by itself embed new points. :class:`OutOfSampleMDS`
adds the standard out-of-sample extension: the embedding is computed on the
training partition, and a regressor learns the mapping from the input space to
the embedding so that held-out points can be projected without refitting.
See Bengio et al., "Out-of-sample extensions for LLE, Isomap, MDS, Eigenmaps and
Spectral Clustering", NeurIPS 2004.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, Isomap
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_is_fitted

N_COMPONENTS = 22          # KernelPCA, Isomap, MDS (value used in the original study)
PCA_VARIANCE = 0.95        # PCA keeps the components explaining 95% of the variance


class OutOfSampleMDS(BaseEstimator, TransformerMixin):
    """Metric MDS with an out-of-sample extension.

    Parameters
    ----------
    n_components : embedding dimension
    n_neighbors : neighbourhood size of the regressor used for the extension
    n_init : number of SMACOF restarts (1 is enough with a fixed seed and is
        markedly faster; the stress differences across restarts are negligible
        on datasets of this size)
    """

    def __init__(self, n_components: int = N_COMPONENTS, n_neighbors: int = 5,
                 n_init: int = 1, max_iter: int = 300, random_state: int | None = 0):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n_comp = min(self.n_components, X.shape[1], max(X.shape[0] - 1, 1))
        mds = MDS(n_components=n_comp, n_init=self.n_init, max_iter=self.max_iter,
                  random_state=self.random_state, normalized_stress="auto")
        embedding = mds.fit_transform(X)
        self.stress_ = float(mds.stress_)
        self.n_components_ = n_comp
        k = min(self.n_neighbors, len(X))
        self.extension_ = KNeighborsRegressor(n_neighbors=k, weights="distance").fit(X, embedding)
        return self

    def transform(self, X):
        check_is_fitted(self, "extension_")
        return self.extension_.predict(np.asarray(X, dtype=np.float64))


class ColumnSubset(BaseEstimator, TransformerMixin):
    """Select a fixed, expert-defined subset of columns (used for BOTCAST).

    This is a pure column selection: it has nothing to fit, so it can never leak
    information from held-out data.
    """

    def __init__(self, indices: tuple[int, ...] = ()):
        self.indices = indices

    def fit(self, X, y=None):
        self.indices_ = np.asarray(self.indices, dtype=int)
        return self

    def transform(self, X):
        check_is_fitted(self, "indices_")
        return np.asarray(X)[:, self.indices_]


def botcast_indices(feature_names: list[str], crop: str, data_root=None) -> np.ndarray:
    """Indices of the BOTCAST expert variables inside ``feature_names``."""
    from pathlib import Path

    from .data import DATA_ROOT

    root = Path(data_root) if data_root is not None else DATA_ROOT
    wanted: list[str] = []
    with open(root / crop / "datasets_vars" / "Botcast.txt") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("->"):
                wanted.append(line)
    lookup = {name: i for i, name in enumerate(feature_names)}
    idx = [lookup[w] for w in wanted if w in lookup]
    if not idx:
        raise ValueError(f"no BOTCAST variable found among the features of {crop}")
    return np.asarray(idx, dtype=int)


def make_reducer(name: str, feature_names: list[str] | None = None, crop: str | None = None,
                 random_state: int = 0):
    """Return an unfitted transformer, or ``None`` for the no-reduction baseline."""
    if name == "No Reduction":
        return None
    if name == "PCA":
        return PCA(n_components=PCA_VARIANCE, random_state=random_state)
    if name == "KernelPCA":
        # Nonlinear kernel PCA: RBF kernel with gamma = 1/n_features on
        # the standardized inputs.
        return KernelPCA(n_components=N_COMPONENTS, kernel="rbf", random_state=random_state)
    if name == "Isomap":
        return Isomap(n_components=N_COMPONENTS)
    if name == "MDS":
        return OutOfSampleMDS(n_components=N_COMPONENTS, random_state=random_state)
    if name == "BOTCAST":
        if feature_names is None or crop is None:
            raise ValueError("BOTCAST needs feature_names and crop")
        return ColumnSubset(indices=tuple(botcast_indices(feature_names, crop)))
    raise KeyError(f"unknown reducer {name!r}")


DR_METHODS = ("No Reduction", "PCA", "KernelPCA", "Isomap", "MDS", "BOTCAST")
