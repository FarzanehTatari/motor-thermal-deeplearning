"""Paderborn motor temperature dataset loader and sequence windowing.

Public dataset: Kaggle "Electric Motor Temperature" by Wilhelm Kirchgaessner.
Download via:
    kaggle datasets download -d wkirgsn/electric-motor-temperature -p data --unzip
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

@dataclass
class StandardScaler:
    """Tiny z-score scaler that's safe against zero-variance columns.

    Fit on train data only, transform validation/test with the same stats.
    """
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "StandardScaler":
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        return cls(mean=mu, std=sigma)

    def transform(self, X: np.ndarray) -> np.ndarray:
        std = np.where(self.std < 1e-8, 1.0, self.std)
        return (X - self.mean) / std


# ---------------------------------------------------------------------------
# Loading + profile selection
# ---------------------------------------------------------------------------

def load_paderborn(csv_path: str | Path) -> pd.DataFrame:
    """Load the Paderborn / Kaggle motor-temperature CSV.

    Expected columns:
      ambient, coolant, u_d, u_q, motor_speed, torque, i_d, i_q,
      pm, stator_yoke, stator_tooth, stator_winding, profile_id
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found.\n"
            "Download the dataset with:\n"
            "  kaggle datasets download -d wkirgsn/electric-motor-temperature "
            "-p data --unzip"
        )
    return pd.read_csv(csv_path)


def select_profiles(df: pd.DataFrame, n_profiles: int, seed: int = 42) -> list[int]:
    """Pick a representative subset of profile_ids by load coverage.

    Strategy: sort all profiles by mean absolute torque, then sample
    n_profiles evenly along the sorted list so the chosen set spans
    low-, mid-, and high-load regimes.
    """
    profile_ids = sorted(df["profile_id"].unique().tolist())
    if n_profiles >= len(profile_ids):
        return profile_ids

    stats = df.groupby("profile_id")["torque"].apply(lambda s: s.abs().mean())
    sorted_profiles = stats.sort_values().index.tolist()
    indices = np.linspace(0, len(sorted_profiles) - 1, n_profiles).astype(int)
    chosen = [int(sorted_profiles[i]) for i in indices]
    # de-duplicate while preserving order, then sort for determinism
    return sorted(set(chosen))


# ---------------------------------------------------------------------------
# Sequence windowing
# ---------------------------------------------------------------------------

def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    length: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a fixed-length window over (X, y) and emit (X_seq, y_target).

    X : (T, F)            -> X_seq of shape (N, length, F)
    y : (T,) or (T, K)    -> y_target of shape (N,) or (N, K), taken at
                             the LAST sample of each window.

    Returns empty arrays if the input is shorter than `length`.
    """
    T = X.shape[0]
    F = X.shape[1]
    if T < length:
        empty_X = np.empty((0, length, F), dtype=X.dtype)
        empty_y = (
            np.empty((0,), dtype=y.dtype)
            if y.ndim == 1 else np.empty((0, y.shape[1]), dtype=y.dtype)
        )
        return empty_X, empty_y

    starts = np.arange(0, T - length + 1, stride)
    X_seq = np.stack([X[s : s + length] for s in starts], axis=0)
    y_target = np.stack([y[s + length - 1] for s in starts], axis=0)
    return X_seq, y_target


# ---------------------------------------------------------------------------
# Dataset / DataLoader plumbing
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """Wraps numpy arrays into a PyTorch Dataset of (X_window, y_scalar)."""

    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X_seq.astype(np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def build_dataloaders(
    df: pd.DataFrame,
    train_profiles: list[int],
    test_profiles: list[int],
    *,
    features: list[str],
    target: str,
    length: int,
    stride: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, StandardScaler]:
    """Fit StandardScaler on train profiles ONLY, then build per-profile sequences.

    Returns (train_loader, test_loader, scaler) for a single scalar target.
    """
    train_df = df[df["profile_id"].isin(train_profiles)]
    scaler = StandardScaler.fit(train_df[features].values)

    X_train_list, y_train_list = [], []
    for pid in train_profiles:
        sub = df[df["profile_id"] == pid].sort_index()
        X = scaler.transform(sub[features].values)
        y = sub[target].values
        Xs, ys = make_sequences(X, y, length, stride)
        if len(Xs) > 0:
            X_train_list.append(Xs)
            y_train_list.append(ys)

    X_test_list, y_test_list = [], []
    for pid in test_profiles:
        sub = df[df["profile_id"] == pid].sort_index()
        X = scaler.transform(sub[features].values)
        y = sub[target].values
        Xs, ys = make_sequences(X, y, length, stride)
        if len(Xs) > 0:
            X_test_list.append(Xs)
            y_test_list.append(ys)

    if not X_train_list:
        raise RuntimeError("No training sequences produced. Check window length vs profile lengths.")
    if not X_test_list:
        raise RuntimeError("No test sequences produced. Check window length vs profile lengths.")

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    train_loader = DataLoader(
        SequenceDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        SequenceDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, test_loader, scaler
