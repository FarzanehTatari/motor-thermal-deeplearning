"""Plotting helpers for trajectory and error figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_pred_vs_true(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    model_name: str,
    target_name: str = "pm",
    save_path: str | Path | None = None,
):
    """Two-panel plot: ground truth vs prediction overlay, plus absolute error."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    t = np.arange(len(true))

    axes[0].plot(t, true, "k", linewidth=1.2, label="ground truth")
    axes[0].plot(t, pred, "r--", linewidth=1.0, label=f"{model_name} prediction")
    axes[0].set_ylabel(f"{target_name} [°C]")
    axes[0].set_title(f"{model_name} — {target_name} prediction on held-out profile")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    axes[1].plot(t, np.abs(pred - true), ".", markersize=3)
    axes[1].set_xlabel("sample index (in held-out profile)")
    axes[1].set_ylabel("|error| [°C]")
    axes[1].set_title("Absolute prediction error")
    axes[1].grid(True)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(
    pred_dict: dict,
    true: np.ndarray,
    *,
    target_name: str = "pm",
    save_path: str | Path | None = None,
):
    """Overlay all models' predictions on one panel; absolute errors on another."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    t = np.arange(len(true))
    styles = ["b--", "r-.", "g:", "m-"]

    axes[0].plot(t, true, "k", linewidth=1.4, label="ground truth")
    for (name, pred), style in zip(pred_dict.items(), styles):
        axes[0].plot(t, pred, style, linewidth=1.0, label=name)
    axes[0].set_ylabel(f"{target_name} [°C]")
    axes[0].set_title("Model comparison on held-out profile")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    for (name, pred), style in zip(pred_dict.items(), styles):
        axes[1].plot(
            t, np.abs(pred - true),
            style[0] + ".", markersize=3, label=f"{name} |err|",
        )
    axes[1].set_xlabel("sample index")
    axes[1].set_ylabel("|error| [°C]")
    axes[1].set_title("Absolute prediction error")
    axes[1].legend(loc="best")
    axes[1].grid(True)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
