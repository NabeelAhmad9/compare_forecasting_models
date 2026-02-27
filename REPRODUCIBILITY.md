# Reproducibility Guide

This document provides step-by-step instructions to reproduce all results reported in:

> *A Controlled Comparison of Deep Learning Architectures for Multi-Horizon Financial Forecasting — Evidence from 918 Experiments*

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data](#2-data)
3. [Configuration Reference](#3-configuration-reference)
4. [Reproducing the Full Pipeline](#4-reproducing-the-full-pipeline)
   - [Stage 1 — Hyperparameter Optimisation](#stage-1--hyperparameter-optimisation)
   - [Stage 2 — Configuration Freeze](#stage-2--configuration-freeze)
   - [Stage 3 — Multi-Seed Final Training](#stage-3--multi-seed-final-training)
   - [Stage 4 — Metric Aggregation](#stage-4--metric-aggregation)
   - [Stage 5 — Benchmarking](#stage-5--benchmarking)
5. [Determinism Controls](#5-determinism-controls)
6. [Checkpoint Resume](#6-checkpoint-resume)
7. [Paper Compilation](#7-paper-compilation)
8. [Expected Outputs](#8-expected-outputs)
9. [Notes and Limitations](#9-notes-and-limitations)

---

## 1. Environment Setup

### Python version

All experiments were run with **Python 3.11**.

### Option A — Conda (recommended, exact environment)

```bash
conda env create -f environment.yml
conda activate forecasting
```

`environment.yml` pins:

| Package | Version |
|---|---|
| Python | 3.11 |
| PyTorch | ≥ 2.0 |
| NumPy | ≥ 1.24 |
| pandas | ≥ 2.0 |
| scikit-learn | ≥ 1.2 |
| scipy | ≥ 1.10 |
| Optuna | ≥ 3.3 |
| matplotlib | ≥ 3.7 |
| seaborn | ≥ 0.12 |
| PyYAML | ≥ 6.0 |

### Option B — pip

```bash
pip install -r requirements.txt
```

`requirements.txt` additionally pins `numpy==1.24.0` (exact) for numerical reproducibility.

### Verify GPU availability

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

> **Note:** All results were obtained on CUDA-enabled hardware. CPU execution is supported but will be significantly slower. Minor floating-point differences between GPU architectures are expected (see [Section 9](#9-notes-and-limitations)).

---

## 2. Data

### Raw data

Raw hourly OHLCV CSV files are stored in `data/raw/` and **must not be modified**. They are already present in the repository:

| File | Asset | Class |
|---|---|---|
| `BTCUSDT_H1.csv` | BTC/USDT | Cryptocurrency |
| `ETHUSDT_H1.csv` | ETH/USDT | Cryptocurrency |
| `BNBUSDT_H1.csv` | BNB/USDT | Cryptocurrency |
| `ADAUSDT_H1.csv` | ADA/USDT | Cryptocurrency |
| `EURUSD_H1.csv` | EUR/USD | Forex |
| `USDJPY_H1.csv` | USD/JPY | Forex |
| `GBPUSD_H1.csv` | GBP/USD | Forex |
| `AUDUSD_H1.csv` | AUD/USD | Forex |
| `USA30IDXUSD_H1.csv` | Dow Jones | Indices |
| `USA500IDXUSD_H1.csv` | S&P 500 | Indices |
| `USATECHIDXUSD_H1.csv` | NASDAQ 100 | Indices |
| `DEUIDXEUR_H1.csv` | DAX | Indices |

Each file contains columns: `datetime, open, high, low, close, volume` at H1 (1-hour) frequency.

### Preprocessing (automated)

Preprocessing runs automatically as part of the pipeline. The steps applied are:

1. **Truncation** — Retain the most recent 30,000 time steps per asset (`max_samples: 30000` in `configs/dataset.yaml`).
2. **Chronological split** — 70% train / 15% validation / 15% test. Splits are strictly time-ordered; no shuffling.
3. **Normalisation** — Standard z-score (`mean=0, std=1`) fitted **exclusively on training data**, then applied to validation and test. Scaler saved as `data/processed/<category>/<asset>/<horizon>/scaler.pkl`.
4. **Windowing** — Rolling windows of length `w + h`:
   - Short-term (h=4): lookback `w = 24`, horizon `h = 4`
   - Long-term (h=24): lookback `w = 96`, horizon `h = 24`
5. **Outputs** — Saved as `.npy` tensors: `train_x.npy`, `train_y.npy`, `val_x.npy`, `val_y.npy`, `test_x.npy`, `test_y.npy`, plus `split_metadata.json`.

> **Data leakage prevention:** The scaler is fit on training data only. Test data is used exclusively in Stage 3 final evaluation — never during HPO or model selection.

---

## 3. Configuration Reference

All experimental parameters are declared in `configs/`. No hyperparameters are hardcoded in source files.

| File | Purpose |
|---|---|
| `configs/base.yaml` | Device (`auto`/`cpu`/`cuda`), directory paths, logging level |
| `configs/dataset.yaml` | `window_size`, `horizons`, features, split ratios, `max_samples`, scaler type |
| `configs/training.yaml` | Epochs (100), batch size (128), learning rate (0.005), early stopping (patience 15), eval seeds [123, 456, 789], optimizer (Adam), gradient clipping (1.0) |
| `configs/hpo.yaml` | Fixed HPO seed (42), `n_trials` (5), sampler (TPE), objective metric (`val_mse`), HPO epochs (50) |
| `configs/asset.yaml` | Asset list per category, representative assets for HPO (BTC/USDT, EUR/USD, USA30IDXUSD) |
| `configs/models/<model>/search_space.yaml` | Hyperparameter search bounds per model |
| `configs/models/<model>/<category>/<horizon>/best_params.yaml` | Frozen best config (auto-generated after Stage 1) |

---

## 4. Reproducing the Full Pipeline

### Prerequisites

Ensure the conda/pip environment is activated and you are in the project root:

```bash
cd compare_forecasting_models
conda activate forecasting   # or activate your pip environment
```

---

### Stage 1 — Hyperparameter Optimisation

**Purpose:** Find optimal hyperparameters per (model × asset class × horizon) using Bayesian optimisation with a fixed seed.

- **Fixed seed:** 42 (applied to Python `random`, NumPy, PyTorch, and Optuna sampler)
- **Sampler:** Optuna TPE
- **Trials:** 5 per (model, category, horizon) combination
- **Tuned on representative assets only:** BTC/USDT (crypto), EUR/USD (forex), USA30IDXUSD (indices)

```bash
# All models
python src/main.py --mode hpo

# Or using Makefile
make hpo

# Specific models
python src/main.py --mode hpo --models ModernTCN PatchTST
make hpo-model MODELS="ModernTCN PatchTST"
```

**Outputs:**
```
models/hpo/<model>/<category>/<asset>/<horizon>/trial_XXX/model.pt
logs/hpo/<model>_<category>_<asset>_<horizon>.log
```

---

### Stage 2 — Configuration Freeze

After Stage 1 completes, best hyperparameters are automatically serialised per (model, category, horizon):

```
configs/models/<model>/<category>/<horizon>/best_params.yaml
```

These files are held **fixed for all subsequent stages**. To regenerate them from existing HPO trial results without re-running HPO:

```bash
python scripts/generate_best_params_from_hpo.py
```

---

### Stage 3 — Multi-Seed Final Training

**Purpose:** Train each model on every asset at every horizon using the frozen best config, replicated across three seeds.

- **Seeds:** 123, 456, 789 (set in `configs/training.yaml` under `eval_seeds`)
- **All 12 assets** × **2 horizons** × **9 models** × **3 seeds** = **648 training runs**
- Uses identical preprocessing, splits, and evaluation protocol across all models

```bash
# All models, all assets
python src/main.py --mode final

# Or using Makefile
make final

# Specific models
python src/main.py --mode final --models DLinear LSTM
make final-model MODELS="DLinear"
```

**Outputs:**
```
models/final/<model>/<category>/<asset>/<horizon>/<seed>/
    model_best.pt          # checkpoint with lowest val loss
    model_last.pt          # checkpoint at final epoch
    training_state.json    # epoch, best metric, random states

results/final/<model>/<category>/<asset>/<horizon>/<seed>/
    best/metrics.json      # RMSE, MAE, Directional Accuracy
    best/predictions.csv   # test-set predictions vs actuals
    last/metrics.json
    last/predictions.csv
```

---

### Stage 4 — Metric Aggregation

Aggregation is performed automatically as part of the benchmark stage. Metrics from three seeds are combined to produce `mean ± std` per (model, asset, horizon):

```
results/final/<model>/<category>/<asset>/<horizon>/
    aggregated_best.json
    aggregated_last.json
```

To run aggregation in isolation:

```bash
python src/evaluation/aggregate.py
```

---

### Stage 5 — Benchmarking

**Purpose:** Generate leaderboards, figures, and statistical tests from aggregated results.

```bash
python src/main.py --mode benchmark

# Or
make benchmark
```

Statistical tests applied:
- **Friedman–Iman–Davenport** — omnibus rank test
- **Holm-corrected Wilcoxon** — pairwise significance with family-wise error control
- **Diebold-Mariano** — pairwise forecast accuracy comparison
- **Variance decomposition** — two-factor SS decomposition (architecture vs. seed)
- **Spearman rank correlation** — cross-horizon ranking stability

**Outputs:**
```
results/benchmark/global_summary/tables/
results/benchmark/global_summary/figures/
results/benchmark/categories/<category>/
results/benchmark/categories/<category>/assets/<asset>/horizons/<horizon>/
```

---

## 5. Determinism Controls

All randomness sources are managed through `src/utils/seed.py`, invoked at the start of every experimental run.

```python
# Applied at every run start:
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**HPO seed:** 42 (fixed across all trials for fair comparison — `configs/hpo.yaml: fixed_seed: 42`)

**Final training seeds:** 123, 456, 789 (configured in `configs/training.yaml: eval_seeds`)

> `torch.use_deterministic_algorithms(True)` is **not** enabled. Minor floating-point variation remains possible across GPU hardware generations. This is mitigated by the three-seed replication protocol, which shows ≤ 0.01% seed-driven variance.

---

## 6. Checkpoint Resume

All training runs support automatic resume. If a checkpoint exists, training continues from the last completed epoch without restarting.

Checkpoints save:
- Model `state_dict`
- Optimizer `state_dict`
- Scheduler `state_dict`
- PyTorch, NumPy, and Python random states
- Current epoch number and best validation metric
- Full config used for training

Resume is enabled by default (`configs/training.yaml: checkpoint.resume: true`). To force a fresh start, delete the checkpoint directory:

```bash
# Remove a specific run
rm -rf models/final/<model>/<category>/<asset>/<horizon>/<seed>/
```

---

### Notes

- Figures are included as PDFs from `paper/figures/`. Pre-generated figures are committed to the repository.
- Tables are included as `.tex` fragments from `paper/tables/`.
- The compiled PDF will be `paper/main.pdf`.
- If figures are missing (e.g., after `make clean`), run the benchmark stage first to regenerate them, then recompile. Missing figure files display gracefully as placeholder boxes thanks to the `\maybeincludegraphics` macro.

---

## 8. Expected Outputs

After running the full pipeline (`make all`), the following key outputs should be present:

| Path | Description |
|---|---|
| `data/processed/` | Windowed numpy tensors for all 12 assets × 2 horizons |
| `configs/models/*/best_params.yaml` | Frozen HPO configs (3 categories × 2 horizons × 9 models) |
| `models/final/` | 648 trained model checkpoints (9 × 12 × 2 × 3) |
| `results/final/` | Per-seed and aggregated metrics for all runs |
| `results/benchmark/global_summary/tables/global_ranking_aggregated.csv` | Global leaderboard |
| `results/benchmark/global_summary/figures/global_metric_heatmap_no_lstm.pdf` | RMSE heatmap |
| `results/benchmark/global_summary/tables/variance_decomposition.csv` | Architecture vs. seed variance |
| `results/benchmark/global_summary/tables/holm_wilcoxon.csv` | Pairwise significance |
| `results/benchmark/categories/<category>/` | Per-category analysis |
| `results/plots/` | Actual-vs-predicted forecast plots |

---

## 9. Notes and Limitations

### Hardware variation
`torch.use_deterministic_algorithms(True)` is not enabled. Bit-exact reproduction requires the **same GPU architecture**, driver version, and CUDA version as the original runs. Across different hardware, small floating-point differences are expected but do not affect reported results meaningfully (seed variance contributes ≤ 0.01% of total RMSE variance).

### HPO trial budget
The HPO budget is set to **5 trials** per (model, category, horizon) for computational tractability. Increasing `n_trials` in `configs/hpo.yaml` would likely improve mid-tier model performance but was not part of the original protocol.

### Directional accuracy
All models are trained with MSE loss. Directional accuracy (fraction of correctly forecast up/down movements) is 50.08% on average across all 54 evaluation combinations. Reproducing directional skill requires explicit loss-function redesign and is outside the scope of the current protocol.

### Single asset class per HPO
HPO is performed on one representative asset per class (BTC/USDT, EUR/USD, USA30IDXUSD) to prevent asset-level overfitting. The frozen configuration is applied to all other assets in the class. This ensures fairness but may not be globally optimal for each individual asset.

### Data leakage check
Scalers are fit on training partitions only. Split boundaries are saved in `split_metadata.json` per processed asset/horizon directory. Validation and test data are never accessed during HPO.

### Missing figures on first run
If `data/processed/` or `results/` are absent (e.g., after `make clean`), run the full pipeline before attempting paper compilation.

### Dependency pinning
`requirements.txt` pins `numpy==1.24.0` exactly. Using a different NumPy version may produce minor numerical differences in preprocessing. The `environment.yml` specifies `numpy>=1.24` for flexibility — use `requirements.txt` for strictest reproducibility.
