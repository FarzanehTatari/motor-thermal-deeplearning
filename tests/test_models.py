"""Unit tests for src.models — forward-pass shapes and NaN safety."""

from __future__ import annotations

import torch

from src.models import (
    GRURegressor,
    LSTMRegressor,
    MLPRegressor,
    build_model,
    count_parameters,
)


def test_gru_forward_shape():
    m = GRURegressor(input_size=8, hidden_size=16)
    x = torch.randn(4, 32, 8)
    y = m(x)
    assert y.shape == (4,)


def test_lstm_forward_shape():
    m = LSTMRegressor(input_size=8, hidden_size=13)
    x = torch.randn(4, 32, 8)
    y = m(x)
    assert y.shape == (4,)


def test_mlp_forward_shape():
    m = MLPRegressor(input_size=8, seq_length=32)
    x = torch.randn(4, 32, 8)
    y = m(x)
    assert y.shape == (4,)


def test_no_nan_on_random_input():
    cases = [
        (GRURegressor, dict(input_size=8, hidden_size=16)),
        (LSTMRegressor, dict(input_size=8, hidden_size=13)),
        (MLPRegressor, dict(input_size=8, seq_length=32)),
    ]
    for cls, kwargs in cases:
        m = cls(**kwargs)
        x = torch.randn(2, 32, 8)
        y = m(x)
        assert not torch.isnan(y).any()


def test_build_model_factory():
    cfg = {
        "gru": {"hidden_size": 16},
        "lstm": {"hidden_size": 13},
        "mlp": {"hidden_sizes": [128, 64]},
    }
    for name in ("gru", "lstm", "mlp"):
        m = build_model(name, input_size=8, seq_length=32, cfg=cfg)
        x = torch.randn(2, 32, 8)
        y = m(x)
        assert y.shape == (2,)
        assert count_parameters(m) > 0


def test_param_counts_roughly_matched_recurrent():
    """GRU(16) and LSTM(13) should have parameter counts within ~25% of each other,
    matching the GVSETS-paper "isolate the gating mechanism, not capacity" choice."""
    g = GRURegressor(input_size=8, hidden_size=16)
    l = LSTMRegressor(input_size=8, hidden_size=13)
    pg = count_parameters(g)
    pl = count_parameters(l)
    ratio = max(pg, pl) / min(pg, pl)
    assert ratio < 1.25, f"GRU and LSTM param counts too different: {pg} vs {pl}"
