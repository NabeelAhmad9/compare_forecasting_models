"""
Compute trainable parameter counts for every model+category+horizon configuration
that has a frozen hyperparameter file (best_params.yaml).

The script mirrors the final training setup by reading the project configuration
and using the recorded best hyperparameters to instantiate each architecture.  It
then sums up the number of `requires_grad` parameters and writes a CSV table
with the results.

Usage:
    python scripts/compute_param_counts.py [--output PATH]

The default output path is `results/model_parameter_counts.csv`.

The resulting CSV contains the following columns:
    model,category,horizon,parameter_count

This information is useful for the complexity--performance analysis described in
the paper (see Section 3 and Figure 9).

The script is deliberately lightweight and has no external dependencies beyond
those already required by the project.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Ensure project root is on sys.path so that `import src` works regardless of
# current working directory or how the script is invoked.
root = Path(__file__).parent.parent.resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

torch.manual_seed(0)  # for deterministic parameter initialization if any

# now imports should succeed
from src.utils.config import ProjectConfig
from src.models import get_model_class


def count_parameters(model: torch.nn.Module) -> int:
    """Return number of trainable parameters in ``model``.

    Args:
        model: Any torch module.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(output: Optional[Path] = None) -> None:
    # --- load project configuration ---
    root = Path(__file__).parent.parent.resolve()
    cfg = ProjectConfig(root)

    # horizons defined in dataset config
    horizons: List[int] = cfg.dataset.get("horizons", [])

    results: List[Dict[str, Any]] = []

    for model_name in cfg.get_available_models():
        ModelClass = get_model_class(model_name)

        for category in cfg.get_categories():
            for horizon in horizons:
                try:
                    best = cfg.get_model_best_config(model_name, category, horizon)
                except FileNotFoundError:
                    # no frozen config for this combination, skip
                    continue

                # add required keys that may not be present in best_params.yaml
                params = best.copy()
                # horizon and window_size must be provided by most constructors
                params.setdefault("horizon", horizon)
                params.setdefault("window_size", cfg.get_window_size(horizon))
                # ensure input_size is always set; default to number of features
                default_input = len(cfg.dataset.get("features", [])) or 5
                params.setdefault("input_size", default_input)

                # instantiate the model
                try:
                    model = ModelClass(**params)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to instantiate {model_name} with params {params}: {exc}"
                    )

                param_count = count_parameters(model)
                results.append(
                    {
                        "model": model_name,
                        "category": category,
                        "horizon": horizon,
                        "parameter_count": param_count,
                    }
                )

    if not results:
        print("No model configurations were found. Did you run HPO and generate best_params.yaml files?")
        return

    output_path = output or Path(root) / "results" / "model_parameter_counts.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "category", "horizon", "parameter_count"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Wrote {len(results)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute parameter counts from best hyperparameters.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to CSV output file (default: results/model_parameter_counts.csv)",
    )
    args = parser.parse_args()
    main(output=args.output)
