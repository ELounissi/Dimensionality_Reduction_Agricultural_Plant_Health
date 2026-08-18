"""Classical models and their cross-validated search grids.

The grids are compact and default-centred: the decision tree searches depth
and split size; the random forest keeps the standard 100-tree ensemble
(forest quality is monotone in tree count, so size is a compute budget
rather than a selection axis) and searches depth, feature subsampling, and
split size; k-NN searches the neighborhood size with the standard euclidean
metric. The neural architectures of the benchmark are defined in
``code/nets.py``.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

CLASSICAL_GRIDS = {
    "Decision tree": {
        "model__max_depth": [1, 2, 3, 4, 5],
        "model__min_samples_split": [5, 10, 20],
    },
    "Random forest": {
        "model__n_estimators": [100],
        "model__max_depth": [None, 5, 10],
        "model__max_features": ["sqrt", "log2"],
        "model__min_samples_split": [2, 5, 10],
    },
    "k-NN": {
        "model__n_neighbors": [1, 3, 5, 7],
        "model__weights": ["uniform"],
        "model__metric": ["euclidean"],
    },
}


def make_classical(name: str, task: str, random_state: int = 0):
    clf = task == "classification"
    if name == "Decision tree":
        return (DecisionTreeClassifier(criterion="gini", splitter="best",
                                       random_state=random_state)
                if clf else DecisionTreeRegressor(criterion="poisson",
                                                  splitter="best",
                                                  random_state=random_state))
    if name == "Random forest":
        return (RandomForestClassifier(criterion="gini",
                                       random_state=random_state, n_jobs=1)
                if clf else RandomForestRegressor(criterion="squared_error",
                                                  random_state=random_state,
                                                  n_jobs=1))
    if name == "k-NN":
        return KNeighborsClassifier() if clf else KNeighborsRegressor()
    raise KeyError(name)


MODEL_ORDER = ["Decision tree", "Random forest", "k-NN", "MLP", "LSTM",
               "RNN GRU", "Mamba SSM", "TabPFN"]
