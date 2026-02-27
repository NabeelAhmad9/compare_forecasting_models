"""
plot_actual_vs_predicted.py
----------------------------
Generates publication-ready PDF plots of actual vs predicted values from
multi-step forecasting outputs stored in:

    results/final/<model>/<category>/<asset>/<horizon>/123/best/predictions.csv

Output PDFs are saved to:

    results/plots/<category>/<asset>/<model>/<horizon>/

Horizon-specific steps plotted:
    horizon = 4  → steps 1, 2, 3, 4
    horizon = 24 → steps 1, 4, 12, 24

Usage:
    python scripts/plot_actual_vs_predicted.py [--results-dir results/final]
                                                [--output-dir results/plots]
                                                [--max-obs 500]
                                                [--dpi 600]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend; safe for headless execution

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED_DIR: str = "123"
PRED_SUBPATH: str = "best/predictions.csv"

# Steps to plot per horizon
HORIZON_STEPS: dict[int, list[int]] = {
    4: [1, 2, 3, 4],
    24: [1, 4, 12, 24],
}

# Font and style settings
FONT_FAMILY: str = "serif"
FONT_SIZE_TITLE: int = 11
FONT_SIZE_AXIS: int = 10
FONT_SIZE_TICK: int = 9
FONT_SIZE_LEGEND: int = 9

LINE_WIDTH_TRUE: float = 1.2
LINE_WIDTH_PRED: float = 1.2
FIGURE_SIZE: tuple[float, float] = (10.0, 3.8)

COLOR_TRUE: str = "black"
COLOR_PRED: str = "#1f77b4"   # Matplotlib default blue — professional and clear


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _apply_institutional_style() -> None:
    """Configure global matplotlib rc parameters for a clean institutional look."""
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.6,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "text.usetex": False,
        }
    )


def plot_step(
    y_true: pd.Series,
    y_pred: pd.Series,
    step: int,
    horizon: int,
    category: str,
    asset: str,
    model: str,
    output_path: Path,
    max_obs: int,
    dpi: int,
) -> None:
    """
    Generate and save a single actual-vs-predicted PDF plot for one forecast step.

    Args:
        y_true:      True values for this step (full series).
        y_pred:      Predicted values for this step (full series).
        step:        Forecast step index (1-based).
        horizon:     Total forecast horizon (4 or 24).
        category:    Asset category label (e.g. "crypto").
        asset:       Asset ticker label (e.g. "BTCUSDT").
        model:       Model name (e.g. "DLinear").
        output_path: Full path (including filename) to save the PDF.
        max_obs:     Maximum number of observations to visualise.
        dpi:         Output resolution in dots per inch.
    """
    # Slice to maximum observations
    y_true_plot = y_true.iloc[:max_obs].reset_index(drop=True)
    y_pred_plot = y_pred.iloc[:max_obs].reset_index(drop=True)

    assert len(y_true_plot) == len(y_pred_plot), (
        f"Length mismatch after slicing: y_true={len(y_true_plot)}, "
        f"y_pred={len(y_pred_plot)}"
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.plot(
        y_true_plot.index,
        y_true_plot.values,
        color=COLOR_TRUE,
        linestyle="-",
        linewidth=LINE_WIDTH_TRUE,
        label="Actual",
    )
    ax.plot(
        y_pred_plot.index,
        y_pred_plot.values,
        color=COLOR_PRED,
        linestyle="--",
        linewidth=LINE_WIDTH_PRED,
        label="Predicted",
    )

    title_line1 = f"Actual vs Predicted \u2014 Step {step}"
    title_line2 = f"{category} | {asset} | {model} | Horizon {horizon}"
    ax.set_title(
        f"{title_line1}\n{title_line2}",
        fontsize=FONT_SIZE_TITLE,
        pad=8,
    )
    ax.set_xlabel("Time Index", fontsize=FONT_SIZE_AXIS)
    ax.set_ylabel("Value", fontsize=FONT_SIZE_AXIS)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)

    ax.legend(
        fontsize=FONT_SIZE_LEGEND,
        frameon=False,
        loc="upper right",
    )

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"  [saved] {output_path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(csv_path: Path) -> pd.DataFrame:
    """
    Load a predictions CSV and validate its structure.

    Args:
        csv_path: Absolute path to predictions.csv.

    Returns:
        DataFrame with columns y_true_step_k and y_pred_step_k.

    Raises:
        ValueError: If expected columns are missing or horizon is unsupported.
    """
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"Empty predictions file: {csv_path}")

    # Detect horizon from column names
    true_cols = sorted(
        [c for c in df.columns if c.startswith("y_true_step_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    pred_cols = sorted(
        [c for c in df.columns if c.startswith("y_pred_step_")],
        key=lambda c: int(c.split("_")[-1]),
    )

    if not true_cols or not pred_cols:
        raise ValueError(
            f"No y_true_step_* or y_pred_step_* columns found in {csv_path}"
        )

    detected_horizon = len(true_cols)

    if detected_horizon not in HORIZON_STEPS:
        raise ValueError(
            f"Unsupported horizon {detected_horizon} detected in {csv_path}. "
            f"Supported: {list(HORIZON_STEPS.keys())}"
        )

    if len(true_cols) != len(pred_cols):
        raise ValueError(
            f"Column count mismatch: {len(true_cols)} true cols vs "
            f"{len(pred_cols)} pred cols in {csv_path}"
        )

    return df


# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------

def discover_prediction_files(results_dir: Path) -> list[dict]:
    """
    Recursively discover all predictions.csv files under results_dir.

    Expected directory layout:
        results_dir/<model>/<category>/<asset>/<horizon>/123/best/predictions.csv

    Args:
        results_dir: Root directory to search.

    Returns:
        List of dicts with keys: model, category, asset, horizon, csv_path.
    """
    records: list[dict] = []
    seed_dir = SEED_DIR
    pred_subpath = PRED_SUBPATH

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for category_dir in sorted(model_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category_name = category_dir.name

            for asset_dir in sorted(category_dir.iterdir()):
                if not asset_dir.is_dir():
                    continue
                asset_name = asset_dir.name

                for horizon_dir in sorted(asset_dir.iterdir()):
                    if not horizon_dir.is_dir():
                        continue

                    # Validate horizon directory name is numeric
                    try:
                        horizon_val = int(horizon_dir.name)
                    except ValueError:
                        continue  # Skip non-numeric directories

                    csv_path = horizon_dir / seed_dir / pred_subpath
                    if not csv_path.is_file():
                        print(
                            f"  [skip] Missing predictions: {csv_path}"
                        )
                        continue

                    records.append(
                        {
                            "model": model_name,
                            "category": category_name,
                            "asset": asset_name,
                            "horizon": horizon_val,
                            "csv_path": csv_path,
                        }
                    )

    return records


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def process_prediction_file(
    record: dict,
    output_dir: Path,
    max_obs: int,
    dpi: int,
) -> None:
    """
    Load one predictions.csv and generate all required step plots.

    Args:
        record:     Dict with keys model, category, asset, horizon, csv_path.
        output_dir: Root output directory for plots.
        max_obs:    Maximum observations per plot.
        dpi:        Output DPI.
    """
    model = record["model"]
    category = record["category"]
    asset = record["asset"]
    horizon = record["horizon"]
    csv_path: Path = record["csv_path"]

    print(
        f"\n[processing] model={model} | category={category} | "
        f"asset={asset} | horizon={horizon}"
    )
    print(f"  [source] {csv_path}")

    try:
        df = load_predictions(csv_path)
    except ValueError as exc:
        print(f"  [error] {exc}")
        return

    steps_to_plot = HORIZON_STEPS.get(horizon)
    if steps_to_plot is None:
        print(f"  [skip] No plot configuration for horizon={horizon}")
        return

    # Output directory: results/plots/<category>/<asset>/<model>/<horizon>/
    plot_dir = output_dir / category / asset / model / str(horizon)

    for step in steps_to_plot:
        true_col = f"y_true_step_{step}"
        pred_col = f"y_pred_step_{step}"

        if true_col not in df.columns or pred_col not in df.columns:
            print(
                f"  [skip] Columns {true_col} / {pred_col} not found "
                f"in {csv_path.name}"
            )
            continue

        output_filename = f"actual_vs_pred_step_{step}.pdf"
        output_path = plot_dir / output_filename

        plot_step(
            y_true=df[true_col],
            y_pred=df[pred_col],
            step=step,
            horizon=horizon,
            category=category,
            asset=asset,
            model=model,
            output_path=output_path,
            max_obs=max_obs,
            dpi=dpi,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-ready PDF plots of actual vs predicted "
            "values from multi-step forecasting outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/final"),
        help="Root directory containing model prediction files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plots"),
        help="Root directory to save output PDF plots.",
    )
    parser.add_argument(
        "--max-obs",
        type=int,
        default=500,
        help="Maximum number of observations to include in each plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Output resolution in dots per inch.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Main entry point for the plotting script."""
    args = parse_args(argv)

    results_dir: Path = args.results_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not results_dir.is_dir():
        print(
            f"[fatal] Results directory does not exist: {results_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    _apply_institutional_style()

    print(f"[info] Scanning results directory: {results_dir}")
    records = discover_prediction_files(results_dir)

    if not records:
        print("[warn] No prediction files found. Exiting.")
        return

    print(f"[info] Found {len(records)} prediction files. Generating plots...")

    total_plots = 0
    for record in records:
        horizon = record["horizon"]
        steps = HORIZON_STEPS.get(horizon, [])
        process_prediction_file(record, output_dir, args.max_obs, args.dpi)
        total_plots += len(steps)

    print(f"\n[done] Generated plots for {len(records)} prediction files.")
    print(f"[done] Output directory: {output_dir}")


if __name__ == "__main__":
    main()
