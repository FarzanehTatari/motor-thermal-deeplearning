"""Unit tests for src.data — sequence windowing and standardization."""

from __future__ import annotations

import numpy as np

from src.data import StandardScaler, make_sequences


# ---------------------------------------------------------------------------
# make_sequences
# ---------------------------------------------------------------------------

def test_make_sequences_shapes_default_stride():
    X = np.arange(100 * 3).reshape(100, 3).astype(float)
    y = np.arange(100).astype(float)
    Xs, ys = make_sequences(X, y, length=10, stride=1)
    # 100 - 10 + 1 = 91 windows
    assert Xs.shape == (91, 10, 3)
    assert ys.shape == (91,)
    # last sample of first window is index 9
    assert ys[0] == 9.0
    # last sample of last window is index 99
    assert ys[-1] == 99.0


def test_make_sequences_stride():
    X = np.zeros((100, 2))
    y = np.zeros(100)
    Xs, _ = make_sequences(X, y, length=10, stride=5)
    expected = (100 - 10) // 5 + 1
    assert Xs.shape[0] == expected


def test_make_sequences_returns_empty_when_too_short():
    X = np.zeros((5, 2))
    y = np.zeros(5)
    Xs, ys = make_sequences(X, y, length=10, stride=1)
    assert Xs.shape == (0, 10, 2)
    assert ys.shape == (0,)


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

def test_standard_scaler_zero_mean_unit_std():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    s = StandardScaler.fit(X)
    Xt = s.transform(X)
    assert np.allclose(Xt.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(Xt.std(axis=0), 1.0, atol=1e-7)


def test_standard_scaler_constant_column_no_nan():
    """Should not divide by zero on a constant column."""
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    s = StandardScaler.fit(X)
    Xt = s.transform(X)
    assert not np.any(np.isnan(Xt))
    # constant column maps to (X - 5) / 1 = 0  (since mean is 5, std fallback to 1)
    assert np.allclose(Xt[:, 1], 0.0)


def test_standard_scaler_uses_train_stats_only():
    """Transform should use the FIT-time mean/std, not re-fit on each call."""
    X_train = np.array([[1.0], [2.0], [3.0]])
    X_test = np.array([[10.0], [20.0]])
    s = StandardScaler.fit(X_train)
    Xt = s.transform(X_test)
    expected_mean = X_train.mean()
    expected_std = X_train.std()
    expected = (X_test - expected_mean) / expected_std
    assert np.allclose(Xt, expected)


def test_standard_scaler_round_trip_2d():
    """inverse_transform(transform(X)) ≈ X for a 2-D feature matrix."""
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    s = StandardScaler.fit(X)
    X_back = s.inverse_transform(s.transform(X))
    assert np.allclose(X_back, X)


def test_standard_scaler_round_trip_1d_target():
    """inverse_transform(transform(y)) ≈ y for a 1-D target — needed for
    denormalizing model predictions back to original units (°C)."""
    y = np.array([20.0, 35.0, 50.0, 80.0, 95.0])
    s = StandardScaler.fit(y)
    y_back = s.inverse_transform(s.transform(y))
    assert np.allclose(y_back, y)
