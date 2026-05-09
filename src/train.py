"""Training loop with Adam + MSE loss and early stopping on validation MAE."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    best_val_mae: float
    history: list[dict] = field(default_factory=list)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float = 1e-3,
    max_epochs: int = 200,
    patience: int = 20,
    device: str = "cpu",
    verbose: bool = False,
) -> TrainResult:
    """Train a model with Adam + MSE; early-stop on validation MAE.

    Best-checkpoint state is restored before returning.
    """
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state: dict | None = None
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

        val_mae = _eval_mae(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_mse": float(np.mean(train_losses)) if train_losses else float("nan"),
            "val_mae": val_mae,
        })
        if verbose:
            print(f"  epoch {epoch:3d}  train_mse={history[-1]['train_mse']:.4f}  val_mae={val_mae:.4f}")

        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"  early-stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(best_val_mae=best_val, history=history)


@torch.no_grad()
def _eval_mae(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    abs_errs: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        abs_errs.append((pred - yb).abs().detach().cpu().numpy())
    if not abs_errs:
        return float("inf")
    return float(np.concatenate(abs_errs).mean())
