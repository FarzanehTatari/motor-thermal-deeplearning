"""GRU, LSTM, and MLP models for motor thermal estimation.

All three accept input of shape (batch, time, features) and emit a
single-value prediction at t = length, mirroring the GVSETS-paper
"sequence -> last-output -> regression" architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Recurrent models (GRU, LSTM): single-layer, last-timestep regression head
# ---------------------------------------------------------------------------

class GRURegressor(nn.Module):
    """Single-layer GRU + linear head.

    Hidden size 16 in the default config — chosen so trainable-parameter
    count is roughly matched to LSTM(13), per the GVSETS paper.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
        output_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout) if (dropout > 0 and num_layers == 1) else nn.Identity()
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)              # (B, L, H)
        last = out[:, -1, :]              # (B, H) -- last-timestep output
        last = self.dropout(last)
        return self.head(last).squeeze(-1)  # (B,) for single-target


class LSTMRegressor(nn.Module):
    """Single-layer LSTM + linear head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 13,
        output_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout) if (dropout > 0 and num_layers == 1) else nn.Identity()
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.head(last).squeeze(-1)


# ---------------------------------------------------------------------------
# MLP baseline: flatten the window, two hidden layers, scalar regression
# ---------------------------------------------------------------------------

class MLPRegressor(nn.Module):
    """Feed-forward baseline: flatten (B, L, F) -> hidden layers -> scalar.

    Kept as the "is the recurrent structure actually doing useful work?"
    sanity check.
    """

    def __init__(
        self,
        input_size: int,
        seq_length: int,
        hidden_sizes: tuple[int, ...] = (128, 64),
        output_size: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_dim = input_size * seq_length
        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(name: str, input_size: int, seq_length: int, cfg: dict) -> nn.Module:
    """Build a model by name. `cfg` is the `model:` block of the YAML config."""
    name = name.lower()
    if name == "gru":
        return GRURegressor(input_size=input_size, **cfg.get("gru", {}))
    if name == "lstm":
        return LSTMRegressor(input_size=input_size, **cfg.get("lstm", {}))
    if name == "mlp":
        return MLPRegressor(input_size=input_size, seq_length=seq_length, **cfg.get("mlp", {}))
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
