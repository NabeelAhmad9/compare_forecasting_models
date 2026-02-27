# A Controlled Comparison of Deep Learning Architectures for Multi-Horizon Financial Forecasting

> **Evidence from 918 Experiments**

---

## Description

This repository contains the full benchmarking framework for a controlled, protocol-driven comparison of **nine deep learning architectures** for multi-horizon financial time-series forecasting. The study addresses five methodological shortcomings common in published model comparisons: uncontrolled hyperparameter budgets, single-seed evaluation, single-horizon analysis, absent pairwise statistical correction, and narrow asset-class coverage.

**Nine architectures** spanning four families are evaluated:

| Family | Models |
|---|---|
| Transformer | Autoformer, iTransformer, PatchTST, TimeXer |
| MLP / Linear | DLinear, N-HiTS |
| CNN | ModernTCN, TimesNet |
| RNN | LSTM |

**Asset universe:** 12 hourly instruments across three classes — Cryptocurrency (BTC/USDT, ETH/USDT, BNB/USDT, ADA/USDT), Forex (EUR/USD, USD/JPY, GBP/USD, AUD/USD), and Equity Indices (Dow Jones, S&P 500, NASDAQ 100, DAX).

**Forecasting horizons:** h = 4 hours (short-term) and h = 24 hours (long-term), treated as completely separate experiments.

**Key result:** ModernTCN ranks first (mean rank 1.333, 75% win rate); PatchTST ranks second (mean rank 2.000). Architecture choice explains 99.90% of total RMSE variance; seed variance is negligible (0.01%).

---

## Installation

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Python 3.11
- PyTorch ≥ 2.0 (CUDA-enabled GPU recommended)

### Option 1 — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate forecasting
```

### Option 2 — pip

```bash
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Usage

All execution goes through the unified CLI entry point `src/main.py`.

### Full pipeline (HPO → Final Training → Benchmark)

```bash
python src/main.py --mode hpo
python src/main.py --mode final
python src/main.py --mode benchmark
```

Or via Makefile shortcuts:

```bash
make hpo        # Hyperparameter optimisation
make final      # Multi-seed final training
make benchmark  # Leaderboard generation and plots
make all        # Full pipeline in sequence
```

### Run a specific model only

```bash
python src/main.py --mode hpo    --models ModernTCN PatchTST
python src/main.py --mode final  --models DLinear
make hpo-model   MODELS="LSTM DLinear"
make final-model MODELS="TimesNet"
```

### Other useful commands

```bash
make install    # Install dependencies via pip
make setup      # Create conda environment
make clean      # Remove all generated outputs (data/processed, models, results, logs)
make clean-results  # Remove only results and logs
```

---

## Project Structure

```
compare_forecasting_models/
├── configs/                  # All YAML configuration files
│   ├── asset.yaml            # Asset names, categories, representative assets
│   ├── base.yaml             # Global defaults (device, paths, logging)
│   ├── dataset.yaml          # Window sizes, horizons, features, split ratios
│   ├── hpo.yaml              # HPO seed, n_trials, sampler settings
│   ├── training.yaml         # Epochs, batch size, early stopping, eval seeds
│   └── models/               # Per-model search spaces and frozen best params
│       └── <model>/
│           ├── search_space.yaml
│           └── <category>/<horizon>/best_params.yaml
├── data/
│   ├── raw/                  # Original hourly OHLCV CSVs (never modified)
│   └── processed/            # Windowed numpy tensors + scaler per asset/horizon
├── logs/                     # HPO, final training, and system logs
├── models/
│   ├── hpo/                  # Per-trial checkpoints from hyperparameter search
│   └── final/                # Best and last checkpoints from multi-seed training
├── results/
│   ├── final/                # Per-seed metrics, predictions, and aggregated stats
│   ├── benchmark/            # Leaderboards, statistical tests, and figures
│   └── plots/                # Actual-vs-predicted forecast visualisations
├── notebooks/                # Exploratory and analysis notebooks (01–07)
├── paper/                    # LaTeX source for the companion paper
│   ├── main.tex
│   ├── bibliography.bib
│   ├── sections/             # Introduction, literature review, results, etc.
│   ├── figures/              # Generated PDF figures
│   └── tables/               # Generated LaTeX tables
├── scripts/                  # Standalone helper scripts (param counts, plots)
├── src/                      # All source code
│   ├── main.py               # CLI entry point
│   ├── data/                 # Loader, preprocessing, scaler, windowing
│   ├── models/               # Nine model implementations + BaseModel
│   ├── training/             # Trainer, callbacks
│   ├── evaluation/           # Metrics, aggregation, plot utilities
│   ├── experiments/          # HPO runner, final runner, benchmark
│   └── utils/                # Config, logger, seed management
├── environment.yml
├── requirements.txt
└── Makefile
```

---

## Experimental Protocol

The benchmark follows a five-stage protocol that isolates architectural merit from confounding factors:

| Stage | Description |
|---|---|
| 1. HPO | Fixed-seed (42) Bayesian optimisation via Optuna TPE, 5 trials per (model × asset class × horizon). Tuned on one representative asset per class only. |
| 2. Config freeze | Best hyperparameters serialised to `configs/models/<model>/<category>/<horizon>/best_params.yaml` and held fixed for all assets in that class. |
| 3. Final training | Multi-seed retraining (seeds: 123, 456, 789) per (model × asset × horizon). Identical chronological 70/15/15 splits, OHLCV inputs, MSE loss, early stopping (patience 15). |
| 4. Aggregation | Per-seed metrics (RMSE, MAE, Directional Accuracy) aggregated to mean ± std across seeds. |
| 5. Benchmark | Global leaderboards, category-level analysis, statistical tests (Friedman-Iman-Davenport, Holm-Wilcoxon, Diebold-Mariano, variance decomposition). |

---

## Key Results

| Rank | Model | Family | Mean Rank | 
|---|---|---|---|
| 1 | ModernTCN | CNN | 1.333 |
| 2 | PatchTST | Transformer | 2.000 | 
| 3 | iTransformer | Transformer | 3.667 |
| 4 | TimeXer | Transformer | 4.292 |
| 5 | DLinear | MLP/Linear | 4.958 |
| 6 | N-HiTS | MLP | 5.250 | 
| 7 | TimesNet | CNN | 7.708 |
| 8 | Autoformer | Transformer | 7.833 |
| 9 | LSTM | RNN | 7.958 |

- Architecture choice explains **99.90%** of total forecast variance; seed variation accounts for **0.01%**.
- Rankings are **stable across horizons** despite 2–2.5× error amplification from h=4 to h=24.
- Directional accuracy is **indistinguishable from 50%** across all 54 model–category–horizon combinations, indicating MSE-trained models lack directional skill at hourly resolution.

---

## Paper

The companion paper, *A Controlled Comparison of Deep Learning Architectures for Multi-Horizon Financial Forecasting*, is compiled from the LaTeX sources in `paper/`. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for build instructions.

**Abstract:** Multi-horizon price forecasting is central to portfolio allocation, risk management, and algorithmic trading. This study compares nine deep learning architectures from four families (Transformer, MLP, CNN, RNN) across three asset classes and two forecasting horizons under a protocol-controlled five-stage framework totalling 918 experimental runs. ModernTCN achieves the best mean rank (1.333) with a 75% first-place rate. A clear three-tier performance hierarchy emerges. Architecture explains 99.90% of raw RMSE variance versus 0.01% for seed randomness.

**JEL Classification:** C45, C52, C53, G17

---

## References

Selected key references from `paper/bibliography.bib`:

- **Autoformer:** Wu et al. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.* NeurIPS 34.
- **PatchTST:** Nie et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.* ICLR.
- **iTransformer:** Liu et al. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.* ICLR.
- **TimeXer:** Wang et al. (2024). *TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables.* NeurIPS.
- **DLinear:** Zeng et al. (2023). *Are Transformers Effective for Time Series Forecasting?* AAAI 37, 11121–11128.
- **N-HiTS:** Challu et al. (2023). *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.* AAAI 37, 6989–6997.
- **TimesNet:** Wu et al. (2023). *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.* ICLR.
- **ModernTCN:** Luo & Wang (2024). *ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis.* ICLR.
- **LSTM:** Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.* Neural Computation 9(8), 1735–1780.
- **Optuna:** Akiba et al. (2019). *Optuna: A Next-Generation Hyperparameter Optimization Framework.* KDD.
- **PyTorch:** Paszke et al. (2019). NeurIPS 32, 8024–8035.
- **Reproducibility:** Bouthillier et al. (2021). *Accounting for Variance in Machine Learning Benchmarks.* MLSys 3.
- **Statistical tests:** Demšar (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR 7, 1–30. | Diebold & Mariano (1995). *Comparing Predictive Accuracy.* JBES 13(3), 253–263.

---

## License

This project contains both the research paper and code for our experiments. They are licensed separately:

- **Paper** – The paper and all accompanying figures are licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. You are free to share and adapt the work, provided proper attribution is given.
- **Code** – The source code in this repository is licensed under the MIT License. You are free to use, modify, and distribute the code, with or without modifications, provided that the original copyright notice and license is included.
