"""
Create publication-quality scatter plot of model complexity vs. performance.

Reads parameter counts from:
    results/test_counts.csv

Reads aggregated performance metrics from:
    results/benchmark/categories/<category>/category_summary/tables/category_metrics_aggregated.csv

Enhancements:
    • Log-scale complexity axis
    • Spearman correlation statistic
    • OLS regression line (log-parameter space)
    • Institutional styling
    • Smart label offsets
    • High-resolution 600 DPI export
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scatter of parameter count versus performance metric."
    )
    parser.add_argument(
        "--category",
        type=str,
        default="forex",
        help="Asset category to plot (crypto, forex, indices).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["rmse_mean", "mae_mean"],
        default="rmse_mean",
        help="Performance metric for y-axis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where output PDFs will be written.",
    )
    parser.add_argument(
        "--exclude-lstm",
        action="store_true",
        help="If set, omit the LSTM model from plots.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    counts_csv = Path("results") / "test_counts.csv"
    metrics_csv = (
        Path("results")
        / "benchmark"
        / "categories"
        / args.category
        / "category_summary"
        / "tables"
        / "category_metrics_aggregated.csv"
    )

    if not counts_csv.exists():
        raise FileNotFoundError(f"Missing file: {counts_csv}")
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing file: {metrics_csv}")

    counts = pd.read_csv(counts_csv)
    metrics = pd.read_csv(metrics_csv)

    counts = counts[counts["category"] == args.category]
    if counts.empty:
        raise ValueError(f"No parameter counts for category '{args.category}'")

    merged = pd.merge(counts, metrics, on="model", how="inner")
    if merged.empty:
        raise ValueError("Merge produced no rows — check model names.")

    # -----------------------------------------------------------------
    # STYLE CONFIGURATION (Institutional White)
    # -----------------------------------------------------------------

    sns.set_theme(style="whitegrid")

    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "grid.alpha": 0.4,
    })

    # common marker list large enough for models
    marker_list = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "H"]
    # fixed color palette mapping models to distinct hues
    MODEL_PALETTE = {
        "Autoformer":   "#8E44AD",  # Purple
        "DLinear":      "#27AE60",  # Green
        "iTransformer": "#F1C40F",  # Yellow
        "LSTM":         "#E67E22",  # Orange
        "ModernTCN":    "#1ABC9C",  # Turquoise
        "N-HiTS":       "#D35400",  # Dark Orange
        "PatchTST":     "#C0392B",  # Dark Red
        "TimesNet":     "#2980B9",  # Medium Blue
        "TimeXer":      "#7F8C8D",  # Gray
    }

    for horizon in [4, 24]:
        horizon_data = merged[merged["horizon"] == horizon].copy()
        if args.exclude_lstm:
            horizon_data = horizon_data[horizon_data["model"] != "LSTM"]
        if horizon_data.empty:
            print(f"Warning: no data for horizon={horizon}, skipping")
            continue

        # ensure unique model rows
        horizon_data = horizon_data.drop_duplicates(subset=["model"])
        models = sorted(horizon_data["model"].unique())
        if len(models) > len(marker_list):
            raise ValueError("Not enough distinct markers for the number of models")
        marker_map = {m: marker_list[i] for i, m in enumerate(models)}

        fig, ax = plt.subplots(figsize=(11, 7))

        # scatter each model individually to assign distinct symbols
        for _, row in horizon_data.iterrows():
            mdl = row.model
            color = MODEL_PALETTE.get(mdl, "#000000")
            ax.scatter(
                row.parameter_count,
                row[args.metric],
                marker=marker_map[mdl],
                color=color,
                s=140,
                edgecolor="#222222",
                linewidth=0.8,
                alpha=0.9,
                label=mdl,
            )

        # -----------------------------------------------------------------
        # LOG SCALE + FORMAT
        # -----------------------------------------------------------------

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")

        # -----------------------------------------------------------------
        # REGRESSION LINE (log complexity space)
        # -----------------------------------------------------------------

        x_vals = horizon_data["parameter_count"].values
        y_vals = horizon_data[args.metric].values
        log_x = np.log10(x_vals)
        coeffs = np.polyfit(log_x, y_vals, 1)
        poly = np.poly1d(coeffs)
        x_line = np.linspace(log_x.min(), log_x.max(), 200)
        y_line = poly(x_line)
        ax.plot(
            10 ** x_line,
            y_line,
            linestyle="--",
            linewidth=1.5,
            color="#444444",
            label="OLS Trend (log scale)",
        )

        # -----------------------------------------------------------------
        # SPEARMAN CORRELATION
        # -----------------------------------------------------------------

        rho, pval = spearmanr(x_vals, y_vals)
        ax.text(
            0.02,
            0.95,
            f"Spearman ρ = {rho:.3f}\n"
            f"p-value = {pval:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.85,
                edgecolor="#CCCCCC",
            ),
        )

        # -----------------------------------------------------------------
        # LABELS + TITLE
        # -----------------------------------------------------------------

        metric_label = args.metric.replace("_", " ").upper()
        ax.set_xlabel("Parameter Count (log scale)")
        ax.set_ylabel(metric_label)
        ax.set_title(
            f"Model Complexity vs {metric_label} "
            f"({args.category.capitalize()}, h={horizon})",
            pad=15,
        )

        legend = ax.legend(title="Model", frameon=False)
        legend.get_title().set_fontsize(11)

        plt.tight_layout()

        # -----------------------------------------------------------------
        # SAVE HIGH RESOLUTION
        # -----------------------------------------------------------------

        suffix = "_no_lstm" if args.exclude_lstm else ""
        out_path = args.output_dir / f"complexity_vs_performance_h{horizon}{suffix}.pdf"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close()

        print(f"Plot saved to {out_path}")
        print(f"Spearman correlation: rho={rho:.4f}, p-value={pval:.4f}")


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()