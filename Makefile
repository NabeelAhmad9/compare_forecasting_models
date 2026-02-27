# =============================================================================
#  Makefile — Comparative Evaluation of Deep Learning Forecasting Models
#  Project: compare_forecasting_models
#  Entry point: src/main.py  (modes: hpo | final | benchmark)
# =============================================================================
#
#  QUICK REFERENCE
#  ---------------
#  make setup              Create the conda environment (forecasting)
#  make install            Install Python dependencies via pip
#  make check              Verify environment and key imports
#  make hpo                Run HPO for ALL models
#  make final              Run multi-seed final training for ALL models
#  make benchmark          Generate leaderboards and comparison plots
#  make all                Full pipeline: hpo → final → benchmark
#  make evaluate           Evaluate best/last checkpoints (scripts/)
#  make gen-best-params    Extract best params from HPO trials → best_params.yaml
#  make compute-params     Compute trainable parameter counts for all models
#  make plot-predictions   Plot actual vs predicted for all assets
#  make plot-complexity    Scatter plot of model complexity vs performance
#  make hpo-model          HPO for specific models  (MODELS="LSTM DLinear")
#  make final-model        Final training for specific models (MODELS="LSTM")
#  make hpo-horizon        HPO for a specific horizon (HORIZON=4 or 24)
#  make final-horizon      Final training for a specific horizon
#  make clean              Remove ALL generated outputs
#  make clean-results      Remove only results/ and logs/ (keep checkpoints)
#  make clean-processed    Remove only data/processed/ (keep models & results)
#  make clean-models       Remove only saved model checkpoints
#  make help               Print this help message
#
#  VARIABLES (override on command line)
#  -------------------------------------
#  MODELS   — space-separated list of models,  e.g. MODELS="LSTM DLinear"
#  HORIZON  — forecast horizon to target,       e.g. HORIZON=4
#  SEED     — global random seed,               e.g. SEED=42
#  CONFIG   — path to custom config yaml,       e.g. CONFIG=configs/base.yaml
# =============================================================================

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

PYTHON        := python
CONDA_ENV     := forecasting
SRC           := src/main.py
SCRIPTS_DIR   := scripts

# All supported models (mirrors src/models/ and configs/models/ structure)
ALL_MODELS    := Autoformer DLinear iTransformer LSTM ModernTCN N-HiTS PatchTST TimesNet TimeXer

# Default values for overridable parameters
MODELS        ?= $(ALL_MODELS)
HORIZON       ?=
SEED          ?= 42

# Build optional CLI flags from variables so callers can override freely
_MODELS_FLAG  := $(if $(filter-out $(ALL_MODELS),$(MODELS) $(ALL_MODELS)),--models $(MODELS),)
# If MODELS was explicitly set to something other than the full default list, pass it
_MODELS_ARG   := $(if $(and $(MODELS),$(filter-out $(strip $(ALL_MODELS)),$(strip $(MODELS)))),--models $(MODELS),$(if $(filter $(strip $(MODELS)),$(strip $(ALL_MODELS))),,--models $(MODELS)))
_HORIZON_ARG  := $(if $(HORIZON),--horizon $(HORIZON),)
_SEED_ARG     := $(if $(SEED),--seed $(SEED),)
_CONFIG_ARG   := $(if $(CONFIG),--config $(CONFIG),)

# Compose a single "extra args" string that is appended to every main.py call
EXTRA_ARGS    := $(_HORIZON_ARG) $(_SEED_ARG) $(_CONFIG_ARG)

# ---------------------------------------------------------------------------
# Phony targets (never represent files)
# ---------------------------------------------------------------------------

.PHONY: help \
        setup install check \
        hpo final benchmark all \
        hpo-model final-model \
        hpo-horizon final-horizon \
        evaluate evaluate-model gen-best-params compute-params \
        plot-predictions plot-complexity post-process \
        clean clean-results clean-processed clean-models

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help:  ## Print this help message
	@echo ""
	@echo "  compare_forecasting_models — Makefile targets"
	@echo "  =============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*?## "}; \
		     {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Overridable variables:"
	@echo "    MODELS   = $(MODELS)"
	@echo "    HORIZON  = $(if $(HORIZON),$(HORIZON),(all))"
	@echo "    SEED     = $(SEED)"
	@echo "    CONFIG   = $(if $(CONFIG),$(CONFIG),(default))"
	@echo ""

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

setup:  ## Create the conda environment from environment.yml (name: forecasting)
	conda env create -f environment.yml
	@echo ""
	@echo "  Environment '$(CONDA_ENV)' created."
	@echo "  Activate it with:  conda activate $(CONDA_ENV)"
	@echo ""

install:  ## Install Python dependencies from requirements.txt into current env
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

check:  ## Verify the environment — Python version and critical imports
	$(PYTHON) -c "\
import sys; \
print(f'Python {sys.version}'); \
import torch; print(f'PyTorch  {torch.__version__}  |  CUDA: {torch.cuda.is_available()}'); \
import optuna; print(f'Optuna   {optuna.__version__}'); \
import numpy; print(f'NumPy    {numpy.__version__}'); \
import pandas; print(f'pandas   {pandas.__version__}'); \
import sklearn; print(f'sklearn  {sklearn.__version__}'); \
print('All critical imports OK.')"

# ---------------------------------------------------------------------------
# Core pipeline — full runs (all models)
# ---------------------------------------------------------------------------

hpo:  ## Run hyperparameter optimisation for ALL models and ALL horizons
	$(PYTHON) $(SRC) --mode hpo $(EXTRA_ARGS)

final:  ## Run multi-seed final training for ALL models and ALL horizons
	$(PYTHON) $(SRC) --mode final $(EXTRA_ARGS)

benchmark:  ## Generate benchmark leaderboards and comparison plots
	$(PYTHON) $(SRC) --mode benchmark $(EXTRA_ARGS)

all: hpo final benchmark  ## Full pipeline — HPO → Final training → Benchmark

# ---------------------------------------------------------------------------
# Selective runs — target specific models
# ---------------------------------------------------------------------------

hpo-model:  ## HPO for specific models.  Usage: make hpo-model MODELS="LSTM DLinear"
ifndef MODELS
	$(error MODELS is not set. Usage: make hpo-model MODELS="LSTM DLinear")
endif
	$(PYTHON) $(SRC) --mode hpo --models $(MODELS) $(EXTRA_ARGS)

final-model:  ## Final training for specific models.  Usage: make final-model MODELS="LSTM"
ifndef MODELS
	$(error MODELS is not set. Usage: make final-model MODELS="LSTM")
endif
	$(PYTHON) $(SRC) --mode final --models $(MODELS) $(EXTRA_ARGS)

# ---------------------------------------------------------------------------
# Horizon-scoped runs
# ---------------------------------------------------------------------------

hpo-horizon:  ## HPO constrained to one horizon.  Usage: make hpo-horizon HORIZON=4
ifndef HORIZON
	$(error HORIZON is not set. Usage: make hpo-horizon HORIZON=4)
endif
	$(PYTHON) $(SRC) --mode hpo --horizon $(HORIZON) $(_SEED_ARG) $(_CONFIG_ARG)

final-horizon:  ## Final training constrained to one horizon.  Usage: make final-horizon HORIZON=24
ifndef HORIZON
	$(error HORIZON is not set. Usage: make final-horizon HORIZON=24)
endif
	$(PYTHON) $(SRC) --mode final --horizon $(HORIZON) $(_SEED_ARG) $(_CONFIG_ARG)

# ---------------------------------------------------------------------------
# Post-training scripts
# ---------------------------------------------------------------------------

gen-best-params:  ## Extract best HPO params → configs/models/<model>/<cat>/<horizon>/best_params.yaml
	$(PYTHON) $(SCRIPTS_DIR)/generate_best_params_from_hpo.py

evaluate:  ## Evaluate best & last checkpoints for all trained seeds (scripts/final_evaluation.py)
	$(PYTHON) $(SCRIPTS_DIR)/final_evaluation.py

evaluate-model:  ## Evaluate a single model.  Usage: make evaluate-model MODELS="LSTM"
ifndef MODELS
	$(error MODELS is not set. Usage: make evaluate-model MODELS="LSTM")
endif
	$(PYTHON) $(SCRIPTS_DIR)/final_evaluation.py --model $(MODELS)

compute-params:  ## Compute trainable parameter counts → results/model_parameter_counts.csv
	$(PYTHON) $(SCRIPTS_DIR)/compute_param_counts.py

plot-predictions:  ## Plot actual vs predicted for all assets → results/plots/
	$(PYTHON) $(SCRIPTS_DIR)/plot_actual_vs_predicted.py \
		--results-dir results/final \
		--output-dir results/plots

plot-complexity:  ## Scatter plot of model complexity vs performance (requires compute-params first)
	$(PYTHON) $(SCRIPTS_DIR)/plot_complexity_vs_performance.py

# Convenience aggregate: run all post-processing scripts in the correct order
post-process: gen-best-params evaluate compute-params plot-predictions plot-complexity  ## Run all post-training scripts in order

# ---------------------------------------------------------------------------
# Clean targets
# ---------------------------------------------------------------------------

clean:  ## Remove ALL generated outputs: processed data, checkpoints, results, logs
	rm -rf data/processed/
	rm -rf models/final/ models/hpo/
	rm -rf results/
	rm -rf logs/
	@echo "  All generated outputs removed."

clean-results:  ## Remove results/ and logs/ only — keeps model checkpoints and processed data
	rm -rf results/
	rm -rf logs/
	@echo "  results/ and logs/ removed."

clean-processed:  ## Remove data/processed/ only — forces data re-preprocessing on next run
	rm -rf data/processed/
	@echo "  data/processed/ removed."

clean-models:  ## Remove saved model checkpoints only (final + HPO)
	rm -rf models/final/
	rm -rf models/hpo/
	@echo "  Model checkpoints removed."
