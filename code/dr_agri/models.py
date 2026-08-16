"""Model zoo: three classical learners and three neural architectures.

Classical models are selected by cross-validated grid search over compact,
default-centred grids. Neural models are selected on the validation partition
from an explicit search space over architecture, regularization, and learning
rate; the recurrent models additionally select their input window (one-day or
seven-day) on validation. Early stopping is always driven by data held out
from the training partition and never by the test partition.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------
# classical models and their search grids
# --------------------------------------------------------------------------
CLASSICAL_GRIDS = {
    "Decision tree": {
        "model__max_depth": [1, 2, 3, 4, 5],
        "model__min_samples_split": [5, 10, 20],
    },
    # Standard default-centred grid. The forest size is fixed at the
    # scikit-learn default of 100 trees: forest performance is monotone and
    # plateaus in the number of trees, so tree count is a compute budget
    # rather than a selection axis.
    "Random forest": {
        "model__n_estimators": [100],
        "model__max_depth": [None, 5, 10],
        "model__max_features": ["sqrt", "log2"],
        "model__min_samples_split": [2, 5, 10],
    },
    # leaf_size only affects neighbour-tree construction speed, never the
    # predictions, so it is not a selection axis.
    "k-NN": {
        "model__n_neighbors": [1, 3, 5, 7],
        "model__weights": ["uniform"],
        "model__metric": ["euclidean"],
    },
}


def make_classical(name: str, task: str, random_state: int = 0):
    clf = task == "classification"
    if name == "Decision tree":
        return (DecisionTreeClassifier(criterion="gini", splitter="best", random_state=random_state)
                if clf else DecisionTreeRegressor(criterion="poisson", splitter="best",
                                                  random_state=random_state))
    if name == "Random forest":
        return (RandomForestClassifier(criterion="gini", random_state=random_state, n_jobs=1)
                if clf else RandomForestRegressor(criterion="squared_error",
                                                  random_state=random_state, n_jobs=1))
    if name == "k-NN":
        return KNeighborsClassifier() if clf else KNeighborsRegressor()
    raise KeyError(name)


# --------------------------------------------------------------------------
# neural architectures
# --------------------------------------------------------------------------
class MLPNet(nn.Module):
    """Feed-forward network on the sampling-day feature vector."""

    def __init__(self, input_dim: int, task: str = "classification",
                 hidden: tuple[int, ...] = (64, 32), dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 1))
        if task == "classification":
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):                       # x: (batch, seq, feat)
        return self.net(x[:, -1, :])            # the MLP is not sequential


class LSTMNet(nn.Module):
    """Two stacked LSTM layers with a small dense head."""

    def __init__(self, input_dim: int, task: str = "classification",
                 units: tuple[int, int] = (32, 16), dropout: float = 0.0):
        super().__init__()
        self.rnn1 = nn.LSTM(input_dim, units[0], batch_first=True)
        self.rnn2 = nn.LSTM(units[0], units[1], batch_first=True)
        head: list[nn.Module] = [nn.Linear(units[1], 8), nn.ReLU()]
        if dropout:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(8, 1))
        if task == "classification":
            head.append(nn.Sigmoid())
        self.head = nn.Sequential(*head)

    def forward(self, x):
        o, _ = self.rnn1(x)
        _, (h, _) = self.rnn2(o)
        return self.head(h[-1])


class GRUNet(nn.Module):
    """GRU encoder with a dense head."""

    def __init__(self, input_dim: int, task: str = "classification",
                 hidden: int = 64, layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        head: list[nn.Module] = [nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden // 2, 1)]
        if task == "classification":
            head.append(nn.Sigmoid())
        self.head = nn.Sequential(*head)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.head(h[-1])


NEURAL_ARCHITECTURES = {"MLP": MLPNet, "LSTM": LSTMNet, "RNN GRU": GRUNet}


def _grid(**axes) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, values)) for values in product(*(axes[k] for k in keys))]


# Search spaces sized for these datasets (roughly 280-740 training rows and
# 22-131 predictors per crop): two to three hidden layers of moderate width,
# light dropout, and the two standard Adam learning rates. All candidates are
# scored on the validation partition; nothing is selected on test data.
NEURAL_GRIDS = {
    "MLP": _grid(hidden=[(64, 32), (128, 64, 32), (32, 16)],
                 dropout=[0.0, 0.2], lr=[1e-3, 3e-4]),
    "LSTM": _grid(units=[(32, 16), (64, 32)],
                  dropout=[0.0, 0.2], lr=[1e-3, 3e-4]),
    "RNN GRU": _grid(hidden=[32, 64], layers=[1, 2], lr=[1e-3, 3e-4]),
}

#: input windows examined per family; the recurrent models choose between the
#: single-day and the seven-day window on the validation partition
NEURAL_WINDOWS = {"MLP": (1,), "LSTM": (1, 7), "RNN GRU": (1, 7)}

WEIGHT_DECAY = 1e-4        # standard small L2 penalty for all neural candidates
BATCH_SIZE = 32


class NeuralModel:
    """Trainer with early stopping on an explicit validation split.

    Parameters
    ----------
    name : one of ``NEURAL_ARCHITECTURES``
    task : ``"classification"`` or ``"regression"``
    max_epochs, patience : early-stopping budget. Training stops when the
        validation loss has not improved for ``patience`` epochs and the weights
        of the best validation epoch are restored.
    arch : keyword arguments forwarded to the architecture constructor
    """

    def __init__(self, name: str, task: str, input_dim: int, *, lr: float = 1e-3,
                 max_epochs: int = 200, patience: int = 10, seed: int = 0, **arch):
        self.name, self.task, self.input_dim = name, task, input_dim
        self.lr, self.max_epochs, self.patience, self.seed = lr, max_epochs, patience, seed
        self.arch = arch
        self.epochs_run_ = 0

    def fit(self, X, y, X_val, y_val):
        torch.manual_seed(self.seed)
        self.net = NEURAL_ARCHITECTURES[self.name](self.input_dim, self.task, **self.arch)
        criterion = nn.BCELoss() if self.task == "classification" else nn.MSELoss()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=WEIGHT_DECAY)
        loader = DataLoader(
            TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                          torch.as_tensor(y, dtype=torch.float32).view(-1, 1)),
            batch_size=BATCH_SIZE, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed))
        xv = torch.as_tensor(X_val, dtype=torch.float32)
        yv = torch.as_tensor(y_val, dtype=torch.float32).view(-1, 1)

        best, best_state, bad = float("inf"), None, 0
        self.train_loss_history_, self.val_loss_history_ = [], []
        self.train_accuracy_history_, self.val_accuracy_history_ = [], []
        for epoch in range(self.max_epochs):
            self.net.train()
            epoch_losses, epoch_true, epoch_pred = [], [], []
            for xb, yb in loader:
                pred = self.net(xb)
                loss = criterion(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_losses.append(float(loss.detach()))
                if self.task == "classification":
                    with torch.no_grad():
                        epoch_true.extend(yb.numpy().ravel().tolist())
                        epoch_pred.extend((pred >= 0.5).numpy().ravel().tolist())
            self.net.eval()
            with torch.no_grad():
                vp = self.net(xv)
                vloss = float(criterion(vp, yv))
            self.train_loss_history_.append(float(np.mean(epoch_losses)))
            self.val_loss_history_.append(vloss)
            if self.task == "classification":
                self.train_accuracy_history_.append(
                    float(np.mean(np.asarray(epoch_true) == np.asarray(epoch_pred))))
                self.val_accuracy_history_.append(
                    float(np.mean((vp.numpy().ravel() >= 0.5) == yv.numpy().ravel())))
            if vloss < best - 1e-6:
                best, bad = vloss, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.epochs_run_ = epoch + 1
        self.best_val_loss_ = best
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.as_tensor(X, dtype=torch.float32)).numpy().ravel()

    def predict(self, X):
        p = self.predict_proba(X)
        return (p >= 0.5).astype(int) if self.task == "classification" else p


MODEL_ORDER = ["Decision tree", "Random forest", "k-NN", "MLP", "LSTM", "RNN GRU"]
