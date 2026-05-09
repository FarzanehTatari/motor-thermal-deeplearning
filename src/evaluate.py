"""Evaluation metrics and inference latency benchmark."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run a model over a DataLoader and return (predictions, ground_truth)."""
    model.eval()
    model.to(device)
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        p = model(xb).detach().cpu().numpy()
        preds.append(p)
        targets.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """RMSE, MAE, MaxAE in the original units of `true`."""
    err = pred - true
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    maxae = float(np.abs(err).max())
    return {"rmse": rmse, "mae": mae, "maxae": maxae}


@torch.no_grad()
def inference_latency(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    *,
    repeats: int = 20,
    warmup: int = 1,
) -> dict:
    """Measure mean and std per-batch forward time (in seconds).

    A warm-up call is run and discarded so the first-batch JIT cost
    doesn't pollute the timing.
    """
    model.eval()
    model.to(device)

    # warm-up on the first batch
    for xb, _ in loader:
        xb = xb.to(device)
        for _ in range(warmup):
            _ = model(xb)
        if device == "cuda":
            torch.cuda.synchronize()
        break

    timings: list[float] = []
    for xb, _ in loader:
        xb = xb.to(device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model(xb)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) / repeats)

    arr = np.asarray(timings, dtype=float)
    return {
        "latency_mean_s": float(arr.mean()),
        "latency_std_s": float(arr.std()),
    }
