# Motor Thermal Modeling with Deep Learning

> Recurrent and feed-forward virtual sensors (LSTM, GRU, MLP) for online motor temperature estimation in permanent-magnet synchronous motor (PMSM) traction drives, on the publicly available Paderborn motor-temperature dataset. The methodology generalizes my published industrial work on rare-earth-free Electrically Excited Synchronous Motors (EESM) — see *Related publications* below — to an open benchmark, so anyone can reproduce.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

📊 **Dataset:** [Kaggle / Paderborn Electric Motor Temperature](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature)
👤 **Author:** Farzaneh Tatari · [GitHub](https://github.com/FarzanehTatari) · [Scholar](https://scholar.google.com/citations?user=kocqXnAAAAAJ) · [LinkedIn](https://www.linkedin.com/in/farzaneh-tatari-75296a115/)

---

## What this repo demonstrates

A clean, reproducible Python implementation of three model families for predicting hot-spot temperatures inside a running motor from its electrical and thermal sensor stream:

- **GRU** (recurrent, gated) — single-layer, last-output regression head
- **LSTM** (recurrent, gated) — single-layer, last-output regression head
- **MLP** (feed-forward baseline) — windowed-input regression

All three are evaluated on **leave-one-profile-out cross-validation** over the Paderborn dataset, using train-set-only standardization, and reported with RMSE / MAE / MaxAE plus per-sequence inference latency. The headline comparison echoes the GRU-vs-LSTM finding from my GVSETS 2026 paper (rare-earth-free EESM, industrial data) on a different motor topology, so the methodology can be inspected end-to-end by anyone with a Python install.

---

## Headline result

> *Numbers are filled in after the first full CV run. The structure is fixed.*

| Model | Hidden units | RMSE [°C] | MAE [°C] | MaxAE [°C] | Inference [ms / win] |
| --- | --- | --- | --- | --- | --- |
| MLP (baseline) | — | TBD | TBD | TBD | TBD |
| LSTM | 13 | TBD | TBD | TBD | TBD |
| **GRU** | **16** | **TBD** | **TBD** | **TBD** | **TBD** |

Target: **permanent-magnet (PM) temperature** — the rotor analog in the Paderborn PMSM, comparable to the rotor-temperature target in my GVSETS 2026 EESM paper.

![Predicted vs ground-truth PM temperature trajectories on a held-out profile](results/figures/pm_temperature_comparison.png)

---

## Repository layout

```
.
├── src/
│   ├── data.py               # Paderborn loader, profile splits, sequence windowing
│   ├── models.py             # GRU, LSTM, MLP definitions (PyTorch)
│   ├── train.py              # training loop, optimizer, early stopping
│   ├── evaluate.py           # RMSE / MAE / MaxAE / inference latency
│   └── plotting.py           # trajectory + error figures
├── scripts/
│   ├── run_cv.py             # leave-one-profile-out CV across the 3 models
│   ├── run_single_test.py    # GRU vs LSTM on one held-out profile (faster turnaround)
│   └── plot_results.py       # regenerate the figures from saved metrics
├── configs/
│   └── default.yaml          # hyperparameters, feature/target lists, CV settings
├── tests/
│   ├── test_data.py
│   └── test_models.py
├── results/
│   ├── tables/               # CSV metric summaries (mean ± std over folds)
│   ├── figures/              # PNG trajectory and error plots
│   └── models/               # trained checkpoints (git-ignored, regenerable)
├── requirements.txt
└── README.md
```

---

## Quick reproduction

```bash
# 1. clone
git clone https://github.com/FarzanehTatari/motor-thermal-deeplearning.git
cd motor-thermal-deeplearning

# 2. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. download the Paderborn dataset (CSV, ~280 MB) into data/
mkdir -p data
# Option A: kaggle CLI (https://www.kaggle.com/docs/api)
kaggle datasets download -d wkirgsn/electric-motor-temperature -p data --unzip
# Option B: download the zip from the Kaggle page above and unzip into data/
```

Once the data is in place:

```bash
# Quick smoke test — one held-out profile, GRU vs LSTM, ~5–10 min on a laptop CPU
python -m scripts.run_single_test --config configs/default.yaml

# Full leave-one-profile-out cross-validation (~1 hr on Apple Silicon CPU)
python -m scripts.run_cv --config configs/default.yaml

# Regenerate plots from the saved metrics
python -m scripts.plot_results
```

After the runs finish, summary CSVs land in `results/tables/`:

- `cv_summary.csv` — per-fold RMSE / MAE / MaxAE / inference time, plus aggregates
- `single_test_summary.csv` — GRU vs LSTM head-to-head on the chosen profile

---

## Methodology

### Plant and dataset

Paderborn / Kaggle "Electric Motor Temperature" — a public ~1.3 M-row dataset of sensor recordings from a permanent-magnet synchronous motor on a test bench, sampled at ~2 Hz, organized into 70+ independent **profile sessions** (`profile_id` field) of varying load patterns.

Eight input features:

| Feature | Description |
| --- | --- |
| `ambient` | Ambient temperature [°C] |
| `coolant` | Coolant temperature [°C] |
| `u_d`, `u_q` | Stator voltages, dq-frame [V] |
| `motor_speed` | Mechanical speed [rpm] |
| `torque` | Mechanical torque [Nm] |
| `i_d`, `i_q` | Stator currents, dq-frame [A] |

Four temperature targets: `pm` (permanent-magnet rotor temp), `stator_yoke`, `stator_tooth`, `stator_winding`. The headline experiment predicts **`pm`**; the code supports any subset of targets via the config.

### Models

All three sit on the same input signature and produce a single value at the last sample of an input window, mirroring the GVSETS architecture:

- **GRU** — `sequenceInput → GRU(hidden=16, output=last) → Linear(1)`
- **LSTM** — `sequenceInput → LSTM(hidden=13, output=last) → Linear(1)`
  Hidden-unit counts are chosen to roughly equalize trainable parameters between GRU and LSTM, so the comparison isolates the gating mechanism, not capacity.
- **MLP baseline** — flatten the input window, two hidden layers, ReLU, single-output head. Kept as the "is the recurrent structure actually doing anything?" sanity check.

### Sequence handling

- Window length **L** (default 256 samples, ≈ 2 minutes at the Paderborn sampling rate). The GVSETS paper used 5-minute windows on 10 Hz data; we keep the absolute time horizon comparable.
- Stride **s** (default 8) reduces memory by ~30× with negligible information loss for the slow thermal dynamics.
- Input shape into the recurrent layer: `(batch, time = L, features = 8)`. The model emits the prediction at `t = L`.

### Training

- Optimizer Adam, initial learning rate `1e-3`, mini-batch 64, max epochs 200, with patience-based early stopping on validation MAE.
- Z-score standardization computed on the **train profiles only** and applied to the validation/test profiles — no data leakage.
- Loss: mean squared error on the standardized target. Metrics computed in original °C units.
- Five random seeds per fold for reproducibility; mean ± std reported.

### Cross-validation

Leave-one-`profile_id`-out across a configurable subset of the Paderborn profiles (default: 10 representative profiles selected by load coverage; full 70+ available via config). Each fold trains on 9 profiles, tests on the 10th. Reports per-fold metrics plus aggregate mean ± std.

### Metrics

- **RMSE** (°C) — primary
- **MAE** (°C) — robust to outliers
- **MaxAE** (°C) — worst-case prediction error, important for thermal protection use cases
- **Inference latency** — mean ± std of per-window forward-pass time, with a warm-up call discarded; useful for real-time deployment claims

---

## Tests

```bash
python -m pytest tests/
```

Covers the data loader (correct profile splits, no train/test leakage in the standardization step) and the model forward passes (input/output shapes match expected, no NaNs on a synthetic batch).

---

## Limitations

- **Different motor topology than the related EESM work.** Paderborn is PMSM (permanent magnets in the rotor); my GVSETS / ITEC work is on EESM (electrically excited rotor with field winding `i_f`/`v_f`). The thermal-estimation problem class is identical; the rotor heat-source physics differs, so absolute numbers don't transfer between datasets.
- **No on-vehicle / dynamometer validation in this repo.** Paderborn is itself a test-bench dataset, but this repo doesn't validate on a real vehicle.
- **No real-time embedded benchmark.** Inference latency is reported on the development machine, not on an embedded ECU.

---

## Roadmap

- [ ] Multi-target version (predict all four temperatures jointly)
- [ ] Attention / Transformer baseline alongside the recurrent models
- [ ] Quantization-aware training and INT8 inference latency benchmark
- [ ] Compare against a lumped-parameter thermal network (LPTN) baseline

---

## Related publications

This repo's methodology generalizes the author's two industrial papers on motor thermal modeling — both done on proprietary data and not directly publishable as code:

- **F. Tatari, M. M. Aligoudarzi.** *Deep Learning-Based Rotor Temperature Estimation for Rare-Earth-Free Motors.* NDIA GVSETS 2026. *Proposes GRU and LSTM virtual sensors for rotor temperature estimation in a 190 kW EESM, with 10-fold CV showing GRU outperforming LSTM at parameter-matched capacity.*
- **F. Tatari, D. Trapp, J. Schneider, M. M. Aligoudarzi.** *Data-driven Thermal Modeling for Electrically Excited Synchronous Motors — A Supervised Machine Learning Approach.* IEEE Transportation Electrification Conference (ITEC), 2024. [Paper](https://ieeexplore.ieee.org/document/10595912)

The Paderborn / Kaggle dataset itself is published in:

- **W. Kirchgässner, O. Wallscheid, J. Böcker.** *Estimating Electric Motor Temperatures with Deep Residual Machine Learning.* IEEE Transactions on Power Electronics, 2021.

---

## Citation

If you find this implementation useful in your own work:

```bibtex
@misc{tatari_motor_thermal_2026,
  title  = {Motor Thermal Modeling with Deep Learning: An Open-Data Reference Implementation},
  author = {Tatari, Farzaneh},
  year   = {2026},
  url    = {https://github.com/FarzanehTatari/motor-thermal-deeplearning}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
