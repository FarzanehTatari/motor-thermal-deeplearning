"""Leave-one-profile-out cross-validation across GRU, LSTM, and MLP.

Run from the repo root:
    python -m scripts.run_cv --config configs/default.yaml

Outputs:
    results/tables/cv_raw.csv      (every fold x model x seed)
    results/tables/cv_summary.csv  (mean +/- std per model, aggregated)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data import build_dataloaders, load_paderborn, select_profiles
from src.evaluate import inference_latency, predict, regression_metrics
from src.models import build_model, count_parameters
from src.train import train_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Leave-one-profile-out CV.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = load_paderborn(cfg["data"]["csv_path"])
    chosen = select_profiles(df, cfg["cv"]["n_profiles"], cfg["cv"]["random_seed"])
    print(f"Cross-validation across {len(chosen)} profiles: {chosen}")

    rows: list[dict] = []
    n_seeds = cfg["training"]["num_seeds"]
    n_folds = len(chosen)

    for fold_idx, test_pid in enumerate(chosen):
        train_profiles = [p for p in chosen if p != test_pid]
        test_profiles = [test_pid]
        print(f"\n========== Fold {fold_idx+1}/{n_folds}  test profile = {test_pid} ==========")

        train_loader, test_loader, _ = build_dataloaders(
            df, train_profiles, test_profiles,
            features=cfg["data"]["features"],
            target=cfg["data"]["target"],
            length=cfg["sequence"]["length"],
            stride=cfg["sequence"]["stride"],
            batch_size=cfg["training"]["batch_size"],
        )

        for model_name in ("gru", "lstm", "mlp"):
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = build_model(
                    model_name,
                    input_size=len(cfg["data"]["features"]),
                    seq_length=cfg["sequence"]["length"],
                    cfg=cfg["model"],
                )

                train_model(
                    model, train_loader, test_loader,
                    lr=cfg["training"]["lr"],
                    max_epochs=cfg["training"]["max_epochs"],
                    patience=cfg["training"]["patience"],
                    device=cfg["training"]["device"],
                )
                pred, true = predict(model, test_loader, cfg["training"]["device"])
                m = regression_metrics(pred, true)
                m.update(inference_latency(
                    model, test_loader, cfg["training"]["device"],
                    repeats=cfg["evaluation"]["latency_repeats"],
                ))
                m.update({
                    "model": model_name,
                    "fold": fold_idx,
                    "test_profile": int(test_pid),
                    "seed": seed,
                    "params": count_parameters(model),
                })
                rows.append(m)
                print(
                    f"  {model_name.upper():4s} seed={seed}  "
                    f"RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  MaxAE={m['maxae']:.4f}"
                )

    # ---- save raw + summary ----
    tables_dir = Path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.DataFrame(rows)[[
        "model", "fold", "test_profile", "seed", "params",
        "rmse", "mae", "maxae", "latency_mean_s", "latency_std_s",
    ]]
    raw_path = tables_dir / "cv_raw.csv"
    raw.to_csv(raw_path, index=False)

    summary = (
        raw.groupby("model")[["rmse", "mae", "maxae", "latency_mean_s"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary_path = tables_dir / "cv_summary.csv"
    summary.to_csv(summary_path)

    print(f"\nSaved raw     -> {raw_path}")
    print(f"Saved summary -> {summary_path}")
    print("\n=== CV summary ===")
    print(summary)


if __name__ == "__main__":
    main()
