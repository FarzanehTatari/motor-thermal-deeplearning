"""Train GRU, LSTM, and MLP on N-1 chosen profiles, test on 1 — fast turnaround.

Run from the repo root:
    python -m scripts.run_single_test --config configs/default.yaml

Outputs:
    results/tables/single_test_summary.csv
    results/figures/pm_temperature_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.data import build_dataloaders, load_paderborn, select_profiles
from src.evaluate import inference_latency, predict, regression_metrics
from src.models import build_model, count_parameters
from src.plotting import plot_model_comparison
from src.train import train_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Single held-out-profile head-to-head.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--test-profile-idx", type=int, default=0,
        help="index into the chosen-profiles list (default 0).",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)

    df = load_paderborn(cfg["data"]["csv_path"])
    chosen = select_profiles(df, cfg["cv"]["n_profiles"], cfg["cv"]["random_seed"])

    if not 0 <= args.test_profile_idx < len(chosen):
        raise ValueError(
            f"--test-profile-idx must be in [0, {len(chosen) - 1}], got {args.test_profile_idx}."
        )
    test_profiles = [chosen[args.test_profile_idx]]
    train_profiles = [p for p in chosen if p not in test_profiles]

    print(f"Chosen profile pool: {chosen}")
    print(f"Train profiles    : {train_profiles}")
    print(f"Test profile      : {test_profiles}")

    train_loader, test_loader, _, target_scaler = build_dataloaders(
        df, train_profiles, test_profiles,
        features=cfg["data"]["features"],
        target=cfg["data"]["target"],
        length=cfg["sequence"]["length"],
        stride=cfg["sequence"]["stride"],
        batch_size=cfg["training"]["batch_size"],
    )

    rows: list[dict] = []
    pred_dict: dict[str, "np.ndarray"] = {}
    true = None

    for model_name in ("gru", "lstm", "mlp"):
        print(f"\n==================== {model_name.upper()} ====================")
        model = build_model(
            model_name,
            input_size=len(cfg["data"]["features"]),
            seq_length=cfg["sequence"]["length"],
            cfg=cfg["model"],
        )
        n_params = count_parameters(model)
        print(f"  Parameters: {n_params}")

        train_model(
            model, train_loader, test_loader,
            lr=cfg["training"]["lr"],
            max_epochs=cfg["training"]["max_epochs"],
            patience=cfg["training"]["patience"],
            device=cfg["training"]["device"],
        )
        pred_z, true_z = predict(model, test_loader, cfg["training"]["device"])
        # Denormalize predictions and targets back to °C before reporting metrics.
        pred = target_scaler.inverse_transform(pred_z)
        true = target_scaler.inverse_transform(true_z)
        m = regression_metrics(pred, true)
        m.update(inference_latency(
            model, test_loader, cfg["training"]["device"],
            repeats=cfg["evaluation"]["latency_repeats"],
        ))
        m["model"] = model_name
        m["params"] = n_params
        rows.append(m)
        pred_dict[model_name.upper()] = pred

        print(
            f"  RMSE = {m['rmse']:.4f}  MAE = {m['mae']:.4f}  "
            f"MaxAE = {m['maxae']:.4f}  latency = {m['latency_mean_s']*1000:.2f} ms/batch"
        )

    # ---- save table ----
    tables_dir = Path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)[
        ["model", "params", "rmse", "mae", "maxae", "latency_mean_s", "latency_std_s"]
    ]
    table_path = tables_dir / "single_test_summary.csv"
    table.to_csv(table_path, index=False)
    print(f"\nSaved metrics  -> {table_path}")

    # ---- save plot ----
    figures_dir = Path(cfg["paths"]["figures_dir"])
    fig_path = figures_dir / "pm_temperature_comparison.png"
    plot_model_comparison(
        pred_dict,
        true,
        target_name=cfg["data"]["target"],
        save_path=fig_path,
    )
    print(f"Saved figure   -> {fig_path}")


if __name__ == "__main__":
    main()
