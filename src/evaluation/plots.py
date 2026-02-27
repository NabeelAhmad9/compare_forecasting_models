"""
Plotting utilities for model comparison and benchmarking.

Generates publication-quality plots for metric comparisons,
leaderboards, and degradation analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
# Use non-interactive backend for scripts to avoid GUI issues, 
# but allow interactive plotting if in a notebook/shell
try:
    if "IPython" not in sys.modules:
        matplotlib.use("Agg")
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Plot style settings for publication quality
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Totally distinct colors for each model
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


MODEL_ORDER = [
    "Autoformer", "DLinear", "iTransformer", "LSTM", "ModernTCN", 
    "N-HiTS", "PatchTST", "TimesNet", "TimeXer"
]

# Colors for forecast horizons when they are used as a hue variable.
# The palette is intentionally high-contrast and print-friendly.
HORIZON_PALETTE = {
    4: "#F1C40F",  # Yellow (short horizon)
    24: "#7F8C8D", # Black  (long horizon)
}

def get_model_color(model_name: str) -> str:
    """Get consistent color for a model."""
    return MODEL_PALETTE.get(model_name, "#333333")

def get_color_list(models: List[str]) -> List[str]:
    """Get list of colors corresponding to specific models."""
    return [get_model_color(m) for m in models]

def finalize_plot(ax: plt.Axes, title: Optional[str] = None) -> None:
    """Apply final styling to a plot."""
    if title:
        ax.set_title(title, pad=20)
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout()


def _save_plot(fig: plt.Figure, output_path: Union[str, Path]) -> None:
    """Helper to save plot with standardized settings.

    The output is written as a vector graphic PDF. The provided path is
    converted to have a ``.pdf`` suffix regardless of what the caller
    passed, ensuring consistent file formats across the project.
    """
    output_path = Path(output_path).with_suffix('.pdf')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved: {output_path}")


def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Union[str, Path],
    figsize: tuple = (10, 6),
) -> None:
    """Plot a bar chart comparing a metric across models and assets."""
    # 1. Plot full version
    _draw_metric_comparison(df, metric, title, output_path, figsize)
    
    # 2. Plot version without LSTM if it exists as an outlier
    if "LSTM" in df["model"].unique() and len(df["model"].unique()) > 1:
        no_lstm_df = df[df["model"] != "LSTM"]
        no_lstm_path = Path(output_path).with_name(Path(output_path).stem + "_no_lstm" + Path(output_path).suffix)
        no_lstm_title = f"{title} (Excluding LSTM for visualization clarity)"
        _draw_metric_comparison(no_lstm_df, metric, no_lstm_title, no_lstm_path, figsize)


def _draw_metric_comparison(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Union[str, Path],
    figsize: tuple = (10, 6),
) -> None:
    """Internal drawing logic for bar chart metric comparison."""
    available_models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    remaining_models = [m for m in df["model"].unique() if m not in MODEL_ORDER]
    models = available_models + sorted(remaining_models)
    
    assets = sorted(df["asset"].unique())
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(assets))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        model_data = df[df["model"] == model]
        means = [
            model_data[model_data["asset"] == a][f"{metric}_mean"].values[0]
            if a in model_data["asset"].values else 0
            for a in assets
        ]
        stds = [
            model_data[model_data["asset"] == a][f"{metric}_std"].values[0]
            if a in model_data["asset"].values else 0
            for a in assets
        ]
        ax.bar(
            x + i * width - (len(models) - 1) * width / 2,
            means, width, yerr=stds,
            label=model, color=get_model_color(model),
            capsize=3, alpha=0.9, edgecolor="black", linewidth=0.5
        )

    ax.set_xlabel("Asset")
    ax.set_ylabel(metric.upper().replace("_MEAN", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(assets, rotation=45, ha="right")
    ax.legend(frameon=True, loc="upper right")
    finalize_plot(ax, title)
    _save_plot(fig, output_path)


def plot_horizon_degradation(
    data: Dict[str, Dict[int, float]],
    metric: str,
    title: str,
    output_path: Union[str, Path],
    figsize: tuple = (10, 6),
) -> None:
    """Plot performance degradation across horizons with consistent colors."""
    # 1. Full version
    _draw_horizon_degradation(data, metric, title, output_path, figsize)
    
    # 2. No-LSTM version
    if "LSTM" in data and len(data) > 1:
        no_lstm_data = {k: v for k, v in data.items() if k != "LSTM"}
        no_lstm_path = Path(output_path).with_name(Path(output_path).stem + "_no_lstm" + Path(output_path).suffix)
        no_lstm_title = f"{title} (Excluding LSTM for visualization clarity)"
        _draw_horizon_degradation(no_lstm_data, metric, no_lstm_title, no_lstm_path, figsize)


def _draw_horizon_degradation(
    data: Dict[str, Dict[int, float]],
    metric: str,
    title: str,
    output_path: Union[str, Path],
    figsize: tuple = (10, 6),
) -> None:
    """Internal drawing logic for horizon degradation curves."""
    fig, ax = plt.subplots(figsize=figsize)
    sorted_models = [m for m in MODEL_ORDER if m in data]
    other_models = [m for m in data if m not in MODEL_ORDER]
    models = sorted_models + sorted(other_models)

    for model_name in models:
        horizon_data = data[model_name]
        horizons = sorted(horizon_data.keys())
        values = [horizon_data[h] for h in horizons]
        ax.plot(
            horizons, values, marker="o", label=model_name, 
            linewidth=2, markersize=8, color=get_model_color(model_name)
        )

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel(metric.upper())
    ax.set_xticks(horizons)
    ax.legend(frameon=True)
    finalize_plot(ax, title)
    _save_plot(fig, output_path)


def plot_rank_heatmap(
    rank_df: pd.DataFrame,
    title: str,
    output_path: Union[str, Path],
    figsize: tuple = (10, 8),
) -> None:
    """Plot a heatmap of model rankings across assets with consistent styling."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Use a clean diverging colormap for rankings (Green is better/1, Red is worse)
    im = ax.imshow(rank_df.values, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(np.arange(len(rank_df.columns)))
    ax.set_yticks(np.arange(len(rank_df.index)))
    ax.set_xticklabels(rank_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(rank_df.index)

    # Add text annotations with clean formatting
    for i in range(len(rank_df.index)):
        for j in range(len(rank_df.columns)):
            val = rank_df.iloc[i, j]
            color = "white" if val > 6 else "black" # Contrast based on heatmap color
            ax.text(j, i, f"{val:.0f}",
                    ha="center", va="center", fontsize=11, color=color, fontweight="bold")

    finalize_plot(ax, title)
    plt.colorbar(im, ax=ax, label="Rank", pad=0.02)
    output_path = Path(output_path).with_suffix('.pdf')
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Plot saved: {output_path}")


def plot_model_comparison_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Union[str, Path],
    hue_col: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """Generic bar plot for model comparison with consistent colors."""
    # 1. Full version
    _draw_model_comparison_bar(df, x_col, y_col, title, output_path, hue_col, figsize)
    
    # 2. No-LSTM version
    # Check if 'model' is present in either x_col or hue_col
    model_col = None
    if x_col == "model":
        model_col = "model"
    elif hue_col == "model":
        model_col = "model"

    if model_col and "LSTM" in df[model_col].unique() and len(df[model_col].unique()) > 1:
        no_lstm_df = df[df[model_col] != "LSTM"].copy()
        no_lstm_path = Path(output_path).with_name(Path(output_path).stem + "_no_lstm" + Path(output_path).suffix)
        no_lstm_title = f"{title} (Excluding LSTM for visualization clarity)"
        _draw_model_comparison_bar(no_lstm_df, x_col, y_col, no_lstm_title, no_lstm_path, hue_col, figsize)


def _draw_model_comparison_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Union[str, Path],
    hue_col: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """Internal drawing logic for bar plots."""
    # local import ensures sns is always defined even if the module-level
    # variable is accidentally removed or not reloaded properly.
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort data for consistency
    if x_col == "model":
        df = df.copy()
        df["order"] = df["model"].apply(lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
        df = df.sort_values("order")

    # Determine palette logic strictly
    palette = None
    if hue_col == "model":
        # color each bar according to model identity
        palette = MODEL_PALETTE
    elif hue_col is None and x_col == "model":
        # simple model-vs-metric plot
        palette = MODEL_PALETTE
    elif hue_col == "horizon":
        # specialized palette for horizons ensures yellow vs black contrast
        palette = HORIZON_PALETTE

    sns.barplot(
        data=df, x=x_col, y=y_col, hue=hue_col, ax=ax, 
        palette=palette, edgecolor="black", linewidth=0.5
    )
    
    if x_col == "model":
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    finalize_plot(ax, title)
    _save_plot(fig, output_path)


def plot_model_distribution(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Union[str, Path],
    plot_type: str = "box",
    figsize: tuple = (10, 6),
) -> None:
    """Plot distribution (box or violin) with consistent styling."""
    # 1. Full version
    _draw_model_distribution(df, x_col, y_col, title, output_path, plot_type, figsize)
    
    # 2. No-LSTM version
    if x_col == "model" and "LSTM" in df[x_col].unique() and len(df[x_col].unique()) > 1:
        no_lstm_df = df[df[x_col] != "LSTM"].copy()
        no_lstm_path = Path(output_path).with_name(Path(output_path).stem + "_no_lstm" + Path(output_path).suffix)
        no_lstm_title = f"{title} (Excluding LSTM for visualization clarity)"
        _draw_model_distribution(no_lstm_df, x_col, y_col, no_lstm_title, no_lstm_path, plot_type, figsize)


def _draw_model_distribution(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Union[str, Path],
    plot_type: str = "box",
    figsize: tuple = (10, 6),
) -> None:
    """Internal drawing logic for distributions."""
    # ensure seaborn is imported in this scope as well
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    order = [m for m in MODEL_ORDER if m in df[x_col].unique()]
    palette = MODEL_PALETTE if x_col == "model" else None

    if plot_type == "box":
        sns.boxplot(data=df, x=x_col, y=y_col, order=order, ax=ax, palette=palette)
    else:
        sns.violinplot(data=df, x=x_col, y=y_col, order=order, ax=ax, palette=palette)
        
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    finalize_plot(ax, title)
    _save_plot(fig, output_path)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    title: str,
    output_path: Union[str, Path],
    figsize: tuple = (10, 5),
) -> None:
    """Plot training and validation loss curves with consistent styling."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2, color="#4c72b0")
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=2, color="#dd8452")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend(frameon=True)
    
    finalize_plot(ax, title)
    output_path = Path(output_path).with_suffix('.pdf')
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Plot saved: {output_path}")


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Union[str, Path],
    n_samples: int = 200,
    figsize: tuple = (14, 5),
) -> None:
    """Plot predicted vs actual values with consistent styling."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten multi-step predictions for visualization
    y_true_flat = y_true.flatten()[:n_samples]
    y_pred_flat = y_pred.flatten()[:n_samples]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(y_true_flat, label="Actual", linewidth=1.5, alpha=0.8, color="black")
    ax.plot(y_pred_flat, label="Predicted", linewidth=1.5, alpha=0.8, color="#d62728")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend(frameon=True)
    
    finalize_plot(ax, title)
    output_path = Path(output_path).with_suffix('.pdf')
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Plot saved: {output_path}")


def plot_rank_barplot(
    leaderboard: pd.DataFrame,
    title: str,
    output_path: Union[str, Path],
    figsize: tuple = (10, 6),
) -> None:
    """Plot horizontal bar chart of model rankings with consistent colors."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Sort leaderboard by MODEL_ORDER if possible
    leaderboard = leaderboard.copy()
    leaderboard["_sort"] = leaderboard["model"].apply(lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
    # Actually, for rank barplots, sorting by rank is usually better, but colors should stay consistent
    leaderboard = leaderboard.sort_values("mean_rank")

    models = leaderboard["model"].values
    ranks = leaderboard["mean_rank"].values
    colors = [get_model_color(m) for m in models]
    
    y_pos = np.arange(len(models))
    ax.barh(y_pos, ranks, color=colors, alpha=0.9, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.invert_yaxis()  # Best rank at top
    ax.set_xlabel("Mean Rank (Lower is Better)")
    
    # Clean up grid
    ax.grid(True, alpha=0.3, axis="x", linestyle="--")
    
    # Add rank values as text with consistent precision
    for i, rank in enumerate(ranks):
        ax.text(rank + 0.05, i, f"{rank:.2f}", va="center", fontsize=11, fontweight="bold")

    finalize_plot(ax, title)
    output_path = Path(output_path).with_suffix('.pdf')
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Plot saved: {output_path}")


def plot_summary_heatmap(
    pivot_df: pd.DataFrame,
    title: str,
    output_path: Union[str, Path],
    cmap: str = "viridis",
    fmt: str = ".3f",
    figsize: tuple = (12, 8),
) -> None:
    """Plot summary heatmap for metrics with consistent styling."""
    # 1. Full version
    _draw_summary_heatmap(pivot_df, title, output_path, cmap, fmt, figsize)
    
    # 2. No-LSTM version
    if "LSTM" in pivot_df.index and len(pivot_df.index) > 1:
        no_lstm_df = pivot_df.drop("LSTM")
        no_lstm_path = Path(output_path).with_name(Path(output_path).stem + "_no_lstm" + Path(output_path).suffix)
        no_lstm_title = f"{title} (Excluding LSTM for visualization clarity)"
        _draw_summary_heatmap(no_lstm_df, no_lstm_title, no_lstm_path, cmap, fmt, figsize)


def _draw_summary_heatmap(
    pivot_df: pd.DataFrame,
    title: str,
    output_path: Union[str, Path],
    cmap: str = "viridis",
    fmt: str = ".3f",
    figsize: tuple = (12, 8),
) -> None:
    """Internal drawing logic for summary heatmaps."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort index (models) by MODEL_ORDER
    available_idx = [m for m in MODEL_ORDER if m in pivot_df.index]
    other_idx = [m for m in pivot_df.index if m not in MODEL_ORDER]
    pivot_df = pivot_df.reindex(available_idx + sorted(other_idx))

    sns.heatmap(
        pivot_df, annot=True, fmt=fmt, cmap=cmap, ax=ax, 
        cbar_kws={"label": "Value"}, annot_kws={"size": 10, "weight": "bold"}
    )
    
    finalize_plot(ax, title)
    _save_plot(fig, output_path)


def plot_variance_pie(
    df: pd.DataFrame,
    title: str,
    output_path: Union[str, Path],
    figsize: tuple = (8, 8),
) -> None:
    """Plot variance decomposition pie chart with consistent styling."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ["#4c72b0", "#dd8452", "#55a868"] # Consistent muted colors
    
    wedges, texts, autotexts = ax.pie(
        df["variance_explained"],
        labels=df["factor"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors[:len(df)],
        pctdistance=0.85,
        explode=[0.05] * len(df)
    )
    
    plt.setp(autotexts, size=12, weight="bold", color="white")
    plt.setp(texts, size=12)
    
    # Draw circle for donut effect (optional but cleaner)
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    finalize_plot(ax, title)
    output_path = Path(output_path).with_suffix('.pdf')
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_rank_correlation(
    rank_df: pd.DataFrame,
    title: str,
    output_path: Union[str, Path],
    figsize: tuple = (10, 8),
) -> None:
    """Plot correlation heatmap of model ranks across horizons with consistent styling."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute Spearman correlation between horizons
    corr_matrix = rank_df.corr(method="spearman")

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(corr_matrix.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns)
    ax.set_yticklabels(corr_matrix.index)

    # Add text annotations with consistent styling
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=12, weight="bold")

    finalize_plot(ax, title)
    plt.colorbar(im, ax=ax, label="Spearman Correlation", pad=0.02)
    output_path = Path(output_path).with_suffix('.pdf')
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Plot saved: {output_path}")


def plot_horizon_degradation_curve(
    degradation_df: pd.DataFrame,
    title: str,
    output_path: Union[str, Path],
    figsize: tuple = (10, 6),
) -> None:
    """Plot degradation percentage curves across horizons with consistent colors."""
    # 1. Full version
    _draw_horizon_degradation_curve(degradation_df, title, output_path, figsize)
    
    # 2. No-LSTM version
    if "LSTM" in degradation_df["model"].unique() and len(degradation_df["model"].unique()) > 1:
        no_lstm_df = degradation_df[degradation_df["model"] != "LSTM"].copy()
        no_lstm_path = Path(output_path).with_name(Path(output_path).stem + "_no_lstm" + Path(output_path).suffix)
        no_lstm_title = f"{title} (Excluding LSTM for visualization clarity)"
        _draw_horizon_degradation_curve(no_lstm_df, no_lstm_title, no_lstm_path, figsize)


def _draw_horizon_degradation_curve(
    degradation_df: pd.DataFrame,
    title: str,
    output_path: Union[str, Path],
    figsize: tuple = (10, 6),
) -> None:
    """Internal drawing logic for degradation curves."""
    fig, ax = plt.subplots(figsize=figsize)
    available_models = [m for m in MODEL_ORDER if m in degradation_df["model"].unique()]
    other_models = [m for m in degradation_df["model"].unique() if m not in MODEL_ORDER]
    models = available_models + sorted(other_models)
    
    for model in models:
        model_data = degradation_df[degradation_df["model"] == model]
        horizons = model_data["horizon"].values
        degradation = model_data["degradation_pct"].values
        ax.plot(
            horizons, degradation, marker="o", label=model, 
            linewidth=2, markersize=8, color=get_model_color(model)
        )

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Degradation (%)")
    ax.legend(frameon=True)
    ax.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.5)
    
    finalize_plot(ax, title)
    _save_plot(fig, output_path)


def plot_scatter_comparison(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Union[str, Path],
    hue_col: str = "model",
    figsize: tuple = (10, 8),
) -> None:
    """Plot scatter comparison (e.g., RMSE vs MAE) with consistent colors."""
    # 1. Full version
    _draw_scatter_comparison(df, x_col, y_col, title, output_path, hue_col, figsize)
    
    # 2. No-LSTM version
    if hue_col == "model" and "LSTM" in df[hue_col].unique() and len(df[hue_col].unique()) > 1:
        no_lstm_df = df[df[hue_col] != "LSTM"].copy()
        no_lstm_path = Path(output_path).with_name(Path(output_path).stem + "_no_lstm" + Path(output_path).suffix)
        no_lstm_title = f"{title} (Excluding LSTM for visualization clarity)"
        _draw_scatter_comparison(no_lstm_df, x_col, y_col, no_lstm_title, no_lstm_path, hue_col, figsize)


def _draw_scatter_comparison(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Union[str, Path],
    hue_col: str = "model",
    figsize: tuple = (10, 8),
) -> None:
    """Internal drawing logic for scatter plots."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort for consistent legend
    temp_df = df.copy()
    if hue_col == "model":
        temp_df["order"] = temp_df["model"].apply(lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
        temp_df = temp_df.sort_values("order")

    palette = MODEL_PALETTE if hue_col == "model" else None

    sns.scatterplot(
        data=temp_df, x=x_col, y=y_col, hue=hue_col, ax=ax, 
        palette=palette, s=100, alpha=0.8, edgecolor="black", linewidth=0.5
    )
    
    ax.set_xlabel(x_col.upper().replace("_MEAN", ""))
    ax.set_ylabel(y_col.upper().replace("_MEAN", ""))
    ax.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')

    finalize_plot(ax, title)
    _save_plot(fig, output_path)
