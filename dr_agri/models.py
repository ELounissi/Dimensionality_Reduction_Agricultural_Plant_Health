"""Model zoo: three classical learners and three neural architectures.

The neural networks reproduce the architectures of the original study. They are
wrapped in a small scikit-learn-like API (``fit`` / ``predict``) that takes an
explicit validation set, so that early stopping is driven by data held out from
the *training* partition and never by the test partition.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------
# classical models and their search grids (identical to Appendix B of the paper)
# --------------------------------------------------------------------------
CLASSICAL_GRIDS = {
    "Decision tree": {
        "model__max_depth": [1, 2, 3, 4, 5],
        "model__min_samples_split": [5, 10, 20],
    },
    "Random forest": {
        "model__n_estimators": [10, 20, 30, 40, 50],
        "model__max_depth": [1, 2, 3, 4, 5],
        "model__max_features": ["sqrt", "log2"],
        "model__min_samples_split": [5, 10, 20],
    },
    "k-NN": {
        "model__n_neighbors": [1, 3, 5, 7],
        "model__leaf_size": [1, 6, 11, 17, 22, 28, 33, 39, 44, 50],
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
    """Four-hidden-layer feed-forward network used by the released study."""

    def __init__(self, input_dim: int, task: str = "classification"):
        super().__init__()
        sizes = [input_dim, 64, 32, 16, 8]
        layers: list[nn.Module] = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers += [nn.Linear(sizes[-1], 1)]
        if task == "classification":
            layers += [nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):                       # x: (batch, seq, feat)
        return self.net(x[:, -1, :])            # the MLP is not sequential


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Batch normalization that also handles a final singleton mini-batch."""
    def forward(self, x):
        if self.training and x.shape[0] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(x)


class LSTMNet(nn.Module):
    def __init__(self, input_dim: int, task: str = "classification",
                 dr_name: str = "No Reduction"):
        super().__init__()
        if task == "classification":
            if dr_name == "No Reduction":
                first, second = 32, 16
                self.head = nn.Sequential(nn.ReLU(), nn.Linear(16, 8), nn.ReLU(),
                                          nn.Linear(8, 1), nn.Sigmoid())
            elif dr_name in {"PCA", "KernelPCA", "BOTCAST"}:
                first, second = 64, 32
                self.head = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 8),
                                          nn.Tanh(), nn.Linear(8, 1), nn.Sigmoid())
            else:
                first, second = 32, 32
                self.head = nn.Sequential(SafeBatchNorm1d(32), nn.Dropout(0.3),
                                          nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 8),
                                          nn.Tanh(), SafeBatchNorm1d(8), nn.Dropout(0.3),
                                          nn.Linear(8, 1), nn.Sigmoid())
        else:
            first, second = (64, 32) if dr_name == "BOTCAST" else (32, 32)
            if dr_name == "BOTCAST":
                self.head = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 8),
                                          nn.Tanh(), nn.Linear(8, 1))
            else:
                self.head = nn.Sequential(SafeBatchNorm1d(32), nn.Dropout(0.3),
                                          nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 8),
                                          nn.Tanh(), SafeBatchNorm1d(8), nn.Dropout(0.3),
                                          nn.Linear(8, 1))
        self.rnn1 = nn.LSTM(input_dim, first, batch_first=True)
        self.rnn2 = nn.LSTM(first, second, batch_first=True)

    def forward(self, x):
        o, _ = self.rnn1(x)
        _, (h, _) = self.rnn2(o)
        return self.head(h[-1])


class GRUNet(nn.Module):
    def __init__(self, input_dim: int, task: str = "classification",
                 dr_name: str = "No Reduction"):
        super().__init__()
        if task == "classification" and dr_name == "KernelPCA":
            hidden_dim, num_layers, drop = 16, 1, 0.3
            head = [nn.Linear(16, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 1), nn.Sigmoid()]
        elif task == "regression" and dr_name == "KernelPCA":
            hidden_dim, num_layers, drop = 32, 1, 0.3
            head = [nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)]
        else:
            hidden_dim, num_layers, drop = 64, 2, (0.3 if dr_name in {"No Reduction", "BOTCAST"} else 0.2)
            head = [nn.Linear(64, 32), nn.ReLU(), nn.Dropout(drop),
                    nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)]
            if task == "classification":
                head.append(nn.Sigmoid())
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=drop if num_layers > 1 else 0.0)
        self.head = nn.Sequential(*head)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.head(h[-1])


NEURAL_ARCHITECTURES = {"MLP": MLPNet, "LSTM": LSTMNet, "RNN GRU": GRUNet}
def batch_size(name: str, task: str, dr_name: str) -> int:
    if name == "MLP" or name == "LSTM":
        return 64 if name == "LSTM" and task == "classification" and dr_name == "No Reduction" else 16
    if task == "classification":
        return 64 if dr_name in {"KernelPCA", "BOTCAST"} else 16
    return 32 if dr_name in {"KernelPCA", "BOTCAST"} else 16


class NeuralModel:
    """Trainer with early stopping on an explicit validation split.

    Parameters
    ----------
    name : one of ``NEURAL_ARCHITECTURES``
    task : ``"classification"`` or ``"regression"``
    max_epochs, patience : early-stopping budget. Training stops when the
        validation loss has not improved for ``patience`` epochs and the weights
        of the best validation epoch are restored.
    """

    def __init__(self, name: str, task: str, input_dim: int, *, dr_name: str = "No Reduction",
                 max_epochs: int = 100, patience: int = 10, lr: float = 1e-3,
                 l1: float = 0.0, seed: int = 0):
        self.name, self.task, self.input_dim = name, task, input_dim
        self.dr_name = dr_name
        self.max_epochs, self.patience, self.lr, self.l1, self.seed = max_epochs, patience, lr, l1, seed
        self.epochs_run_ = 0

    def fit(self, X, y, X_val, y_val):
        if self.name in {"MLP", "LSTM"}:
            return self._fit_keras(X, y, X_val, y_val)
        torch.manual_seed(self.seed)
        if self.name == "MLP":
            self.net = MLPNet(self.input_dim, self.task)
        else:
            self.net = NEURAL_ARCHITECTURES[self.name](self.input_dim, self.task, self.dr_name)
        criterion = nn.BCELoss() if self.task == "classification" else nn.MSELoss()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loader = DataLoader(
            TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                          torch.as_tensor(y, dtype=torch.float32).view(-1, 1)),
            batch_size=batch_size(self.name, self.task, self.dr_name), shuffle=True,
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
                if self.l1:
                    loss = loss + self.l1 * sum(p.abs().sum() for p in self.net.parameters())
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
                self.train_accuracy_history_.append(float(np.mean(np.asarray(epoch_true) == np.asarray(epoch_pred))))
                self.val_accuracy_history_.append(float(np.mean((vp.numpy().ravel() >= 0.5) == yv.numpy().ravel())))
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

    def _fit_keras(self, X, y, X_val, y_val):
        """Train the released TensorFlow MLP or LSTM with an explicit validation set."""
        import tensorflow as tf
        from tensorflow.keras import callbacks, layers, regularizers

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            pass

        if self.name == "MLP":
            model = tf.keras.Sequential([
                layers.Input(shape=(self.input_dim,)),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(8, activation="relu"),
                layers.Dense(1, activation="sigmoid" if self.task == "classification" else None),
            ])
            x_train, x_val = X[:, -1, :], X_val[:, -1, :]
        else:
            reg = None
            stack: list = [layers.Input(shape=(X.shape[1], self.input_dim))]
            if self.task == "classification" and self.dr_name == "No Reduction":
                reg = regularizers.l1(0.0001)
                stack += [layers.LSTM(32, return_sequences=True, kernel_regularizer=reg),
                          layers.LSTM(16, activation="relu", kernel_regularizer=reg),
                          layers.Dense(8, activation="relu", kernel_regularizer=reg),
                          layers.Dense(1, activation="sigmoid")]
            elif self.task == "classification" and self.dr_name in {"PCA", "KernelPCA", "BOTCAST"}:
                reg = regularizers.l1(0.0002)
                stack += [layers.LSTM(64, return_sequences=True, kernel_regularizer=reg),
                          layers.LSTM(32, kernel_regularizer=reg),
                          layers.Dense(16, activation="tanh", kernel_regularizer=reg),
                          layers.Dense(8, activation="tanh", kernel_regularizer=reg),
                          layers.Dense(1, activation="sigmoid")]
            elif self.task == "classification":
                stack += [layers.LSTM(32, return_sequences=True), layers.LSTM(32),
                          layers.BatchNormalization(), layers.Dropout(0.3),
                          layers.Dense(16, activation="tanh"), layers.Dense(8, activation="tanh"),
                          layers.BatchNormalization(), layers.Dropout(0.3),
                          layers.Dense(1, activation="sigmoid")]
            elif self.dr_name == "BOTCAST":
                stack += [layers.LSTM(64, return_sequences=True), layers.LSTM(32),
                          layers.Dense(16, activation="tanh"), layers.Dense(8, activation="tanh"),
                          layers.Dense(1)]
            else:
                stack += [layers.LSTM(32, return_sequences=True), layers.LSTM(32),
                          layers.BatchNormalization(), layers.Dropout(0.3),
                          layers.Dense(16, activation="tanh"), layers.Dense(8, activation="tanh"),
                          layers.BatchNormalization(), layers.Dropout(0.3), layers.Dense(1)]
            model = tf.keras.Sequential(stack)
            x_train, x_val = X, X_val

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss="binary_crossentropy" if self.task == "classification" else "mse",
                      metrics=["accuracy"] if self.task == "classification" else ["mae"])
        early = callbacks.EarlyStopping(monitor="val_loss", patience=self.patience,
                                        restore_best_weights=True)
        history = model.fit(x_train, y, validation_data=(x_val, y_val),
                            epochs=self.max_epochs,
                            batch_size=batch_size(self.name, self.task, self.dr_name),
                            callbacks=[early], verbose=0, shuffle=True)
        self.net = model
        self.epochs_run_ = len(history.history["loss"])
        self.best_val_loss_ = float(np.min(history.history["val_loss"]))
        self.train_loss_history_ = [float(v) for v in history.history["loss"]]
        self.val_loss_history_ = [float(v) for v in history.history["val_loss"]]
        self.train_accuracy_history_ = [float(v) for v in history.history.get("accuracy", [])]
        self.val_accuracy_history_ = [float(v) for v in history.history.get("val_accuracy", [])]
        return self

    def predict_proba(self, X):
        if self.name in {"MLP", "LSTM"}:
            inputs = X[:, -1, :] if self.name == "MLP" else X
            return self.net.predict(inputs, verbose=0).reshape(-1)
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.as_tensor(X, dtype=torch.float32)).numpy().ravel()

    def predict(self, X):
        p = self.predict_proba(X)
        return (p >= 0.5).astype(int) if self.task == "classification" else p


MODEL_ORDER = ["Decision tree", "Random forest", "k-NN", "MLP", "LSTM", "RNN GRU"]
