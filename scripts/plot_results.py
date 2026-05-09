"""Regenerate summary figures from the saved cv_raw.csv.

Run from the repo root:
    python -m scripts.plot_results --config configs/default.yaml

Outputs:
    results/figures/cv_metric_comparison.png  (bar chart, mean +/- std)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_path = tables_dir / "cv_raw.csv"
    if not raw_path.exists():
        print(f"No CV raw at {raw_path}. Run `python -m scripts.run_cv` first.")
        return

    raw = pd.read_csv(raw_path)
    summary = raw.groupby("model")[["rmse", "mae", "maxae"]].agg(["mean", "std"])
    print("\n=== CV summary ===")
    print(summary)

    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = ["rmse", "mae", "maxae"]
    x = list(range(len(metrics)))
    width = 0.25

    for i, model in enumerate(("gru", "lstm", "mlp")):
        if model not in summary.index:
            continue
        means = [summary.loc[model, (m, "mean")] for m in metrics]
        stds = [summary.loc[model, (m, "std")] for m in metrics]
        ax.bar(
            [xi + i * width for xi in x],
            means, width=width, yerr=stds, capsize=3, label=model.upper(),
        )

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("error [°C]")
    ax.set_title("Cross-validation: mean ± std across folds × seeds")
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()

    out_path = figures_dir / "cv_metric_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure -> {out_path}")


if __name__ == "__main__":
    main()
