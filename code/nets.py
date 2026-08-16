"""Neural architectures and training loop.

All networks share one training recipe: Adam (lr 1e-3, weight decay 1e-4),
batch 32, up to 400 epochs, early stopping on the validation loss with
patience 20 and best-epoch restore. Regression outputs are clipped to the
valid [0, 1] severity range at prediction time. One model per run.

The Mamba SSM is a faithful pure-PyTorch selective state-space block
(input-dependent discretization and state/output projections, causal
depthwise convolution, gated output - Gu & Dao 2023). The official CUDA
kernels are unnecessary here: the sequences are at most seven steps long,
so the selective scan runs as an explicit recurrence.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

LR, WEIGHT_DECAY, BATCH = 1e-3, 1e-4, 32
N_CONTEXT = 8            # final-step context block, skipped into the heads


class _Head(nn.Module):
    def __init__(self, in_dim: int, task: str, width: int = 16, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_dim, width), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(width, 1),
            *([nn.Sigmoid()] if task == "classification" else []))

    def forward(self, x):
        return self.net(x)


class RecurrentNet(nn.Module):
    """Projection -> GRU/LSTM -> head, with the final-step context block
    skipped directly into the head."""

    def __init__(self, input_dim: int, task: str, units: int, kind: str,
                 proj: int = 32, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, proj), nn.ReLU())
        self.rnn = (nn.GRU if kind == "RNN GRU" else nn.LSTM)(
            proj, units, 1, batch_first=True)
        self.kind = kind
        self.head = _Head(units + N_CONTEXT, task, dropout=dropout)

    def forward(self, x):
        z = self.proj(x)
        if self.kind == "RNN GRU":
            _, h = self.rnn(z)
            top = h[-1]
        else:
            _, (h, _) = self.rnn(z)
            top = h[-1]
        return self.head(torch.cat([top, x[:, -1, -N_CONTEXT:]], dim=1))


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 conv_kernel: int = 3):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, conv_kernel,
                              padding=conv_kernel - 1, groups=self.d_inner)
        self.bc_proj = nn.Linear(self.d_inner, 2 * d_state)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):                      # (B, L, d_model)
        res = x
        x = self.norm(x)
        L = x.shape[1]
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2)
        u = F.silu(u)
        delta = F.softplus(self.dt_proj(u))            # (B, L, d_inner)
        B_, C_ = self.bc_proj(u).chunk(2, dim=-1)      # (B, L, d_state) each
        A = -torch.exp(self.A_log)                     # (d_inner, d_state)
        h = u.new_zeros(u.shape[0], self.d_inner, self.d_state)
        ys = []
        for t in range(L):                             # selective scan
            dt = delta[:, t].unsqueeze(-1)             # (B, d_inner, 1)
            h = h * torch.exp(dt * A) + dt * B_[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
            ys.append(torch.einsum("bds,bs->bd", h, C_[:, t]))
        y = torch.stack(ys, dim=1) + self.D * u
        return self.out_proj(y * F.silu(z)) + res


class MambaNet(nn.Module):
    """Projection -> two Mamba blocks -> head, same skip as the RNNs."""

    def __init__(self, input_dim: int, task: str, units: int,
                 proj: int = 32, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, units), nn.ReLU())
        self.blocks = nn.Sequential(MambaBlock(units), MambaBlock(units))
        self.head = _Head(units + N_CONTEXT, task, dropout=dropout)

    def forward(self, x):
        z = self.blocks(self.proj(x))[:, -1]
        return self.head(torch.cat([z, x[:, -1, -N_CONTEXT:]], dim=1))


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, task: str, hidden: tuple, dropout: float = 0.2):
        super().__init__()
        layers, prev = [], input_dim
        for w in hidden:
            layers += [nn.Linear(prev, w), nn.ReLU(), nn.Dropout(dropout)]
            prev = w
        layers.append(nn.Linear(prev, 1))
        if task == "classification":
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_net(model: str, task: str, cfg: dict, input_dim: int) -> nn.Module:
    torch.manual_seed(cfg["seed"])
    if model == "MLP":
        return MLPNet(input_dim, task, cfg["hidden"])
    if model == "Mamba SSM":
        return MambaNet(input_dim, task, cfg["units"])
    return RecurrentNet(input_dim, task, cfg["units"], model)


def train_net(net, task, Xa, ya, Xb, yb, seed, max_epochs=400, patience=20):
    torch.set_num_threads(1)
    criterion = nn.BCELoss() if task == "classification" else nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(Xa, dtype=torch.float32),
                      torch.as_tensor(ya, dtype=torch.float32).view(-1, 1)),
        batch_size=BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    xb = torch.as_tensor(Xb, dtype=torch.float32)
    yb_t = torch.as_tensor(yb, dtype=torch.float32).view(-1, 1)
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(max_epochs):
        net.train()
        for xt, yt in loader:
            loss = criterion(net(xt), yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vloss = float(criterion(net(xb), yb_t))
        if vloss < best - 1e-6:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, epoch + 1


def predict_net(net, task, X):
    net.eval()
    with torch.no_grad():
        p = net(torch.as_tensor(X, dtype=torch.float32)).numpy().ravel()
    if task == "classification":
        return (p >= 0.5).astype(int)
    return np.clip(p, 0.0, 1.0)
