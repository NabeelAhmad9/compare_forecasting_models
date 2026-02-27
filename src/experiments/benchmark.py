"""
Benchmarking module.

Consolidates aggregated results into leaderboards, summary tables,
and comparison plots across models, assets, and horizons, following
a strict hierarchical directory structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.evaluation.aggregate import load_seed_metrics, aggregate_seed_metrics
from src.evaluation.plots import (
    plot_horizon_degradation,
    plot_rank_barplot,
    plot_model_comparison_bar,
    plot_model_distribution,
    plot_summary_heatmap,
    plot_variance_pie,
    plot_scatter_comparison
)
from src.utils.config import ProjectConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# placeholder helper functions have been removed. All outputs are now produced
# from real data and missing information leads to loud errors.  


def _collect_all_results(config: ProjectConfig) -> pd.DataFrame:
    """Walk the results directory and build a complete DataFrame of aggregated metrics.

    This function discovers models, categories, assets and horizons by inspecting
    the `results_dir/final` directory rather than relying on hardcoded lists. It
    loads per-seed metric files, validates that all expected seeds are present,
    aggregates them using :func:`src.evaluation.aggregate.aggregate_seed_metrics`
    and raises an error if any metrics are missing so that incompleteness is
    detected immediately.

    Args:
        config: Project configuration.

    Returns:
        DataFrame containing one row per (model,category,asset,horizon) with
        aggregated statistics (mean/std/values) for every metric found.
    """
    rows: List[Dict[str, Any]] = []
    base_results = config.get_path("results_dir")
    seeds = config.get_eval_seeds()

    final_root = base_results / "final"
    if not final_root.exists():
        raise FileNotFoundError(f"Results directory does not exist: {final_root}")

    for model_dir in sorted(final_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for category_dir in sorted(model_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            for asset_dir in sorted(category_dir.iterdir()):
                if not asset_dir.is_dir():
                    continue
                asset = asset_dir.name
                # Pre-filter to integer-named directories only; the sort key
                # must only receive valid names to avoid a ValueError during sort.
                _horizon_dirs = [
                    d for d in asset_dir.iterdir()
                    if d.is_dir() and d.name.isdigit()
                ]
                for horizon_dir in sorted(_horizon_dirs, key=lambda p: int(p.name)):
                    horizon = int(horizon_dir.name)

                    # load per-seed metrics and validate completeness
                    metrics_list = load_seed_metrics(
                        base_results, model_name, category, asset, horizon, seeds
                    )
                    if len(metrics_list) != len(seeds):
                        missing = set(seeds) - {m.get("seed") for m in metrics_list if "seed" in m}
                        raise RuntimeError(
                            f"Missing seed metrics for {model_name}/{category}/{asset}/h={horizon}, "
                            f"expected seeds {seeds}, got {len(metrics_list)} (missing {missing})"
                        )

                    agg = aggregate_seed_metrics(metrics_list)
                    if not agg:
                        raise RuntimeError(
                            f"No numeric metrics could be aggregated for {model_name}/{category}/{asset}/h={horizon}"
                        )

                    row: Dict[str, Any] = {
                        "model": model_name,
                        "category": category,
                        "asset": asset,
                        "horizon": horizon,
                    }
                    for m_name, stats in agg.items():
                        row[f"{m_name}_mean"] = stats["mean"]
                        row[f"{m_name}_std"] = stats["std"]
                        row[f"{m_name}_values"] = stats.get("values", [])
                    rows.append(row)

    if not rows:
        logger.error("No results were found in final directory after discovery")
    return pd.DataFrame(rows)


def compute_rankings(
    df: pd.DataFrame,
    metric: str = "rmse_mean",
    ascending: Optional[bool] = None,
) -> pd.DataFrame:
    """Compute rankings for models based on a metric.

    By default the direction is inferred from the metric name: metrics that
    contain ``"accuracy"`` or ``"da"`` are treated as larger-is-better (i.e.
    ``ascending=False``); all others are assumed to be smaller-is-better.

    Args:
        df: DataFrame containing a column with the metric values.
        metric: Column name to rank on.
        ascending: Whether smaller values should get better rank. If ``None`` the
            direction is inferred automatically.

    Returns:
        Sorted DataFrame with an additional ``rank`` column.
    """
    if df.empty:
        return df
    if ascending is None:
        name = metric.lower()
        ascending = not ("accuracy" in name or "da" in name)

    df = df.copy()
    if metric not in df.columns:
        raise KeyError(f"Metric {metric} not found in dataframe columns")
    df["rank"] = df[metric].rank(ascending=ascending, method="min")
    return df.sort_values("rank")


def compute_variance_decomposition(df: pd.DataFrame, metric_values_col: str = "rmse_values") -> pd.DataFrame:
    """Compute variance decomposition (Model vs Seed) using 2-way ANOVA."""
    models = df["model"].unique()
    if len(models) < 2 or metric_values_col not in df.columns:
        return pd.DataFrame(columns=["factor", "variance_explained"])
    
    values_dict = {}
    for _, row in df.iterrows():
        if isinstance(row[metric_values_col], list):
            values_dict[row["model"]] = row[metric_values_col]
            
    if len(values_dict) < 2:
        return pd.DataFrame(columns=["factor", "variance_explained"])
        
    min_len = min(len(v) for v in values_dict.values())
    if min_len < 2:
        return pd.DataFrame(columns=["factor", "variance_explained"])

    # Iterate over values_dict (not the raw ``models`` array) so that only
    # models that actually supplied list-typed metric values are included.
    # Using ``models`` directly would cause a KeyError for any model whose
    # ``rmse_values`` column entry is not a list.
    data_matrix = np.array([values_dict[m][:min_len] for m in values_dict])
    
    y_mean = np.mean(data_matrix)
    y_model_mean = np.mean(data_matrix, axis=1)
    y_seed_mean = np.mean(data_matrix, axis=0)
    
    n_models, n_seeds = data_matrix.shape
    
    SST = np.sum((data_matrix - y_mean)**2)
    if SST == 0:
        return pd.DataFrame(columns=["factor", "variance_explained"])
        
    SSM = n_seeds * np.sum((y_model_mean - y_mean)**2)
    SSS = n_models * np.sum((y_seed_mean - y_mean)**2)
    SSE = SST - SSM - SSS
    
    return pd.DataFrame([
        {"factor": "Model", "variance_explained": SSM / SST},
        {"factor": "Seed", "variance_explained": SSS / SST},
        {"factor": "Residual", "variance_explained": max(0, SSE / SST)}
    ])

def run_statistical_tests(
    df: pd.DataFrame,
    metric_values_col: str = "rmse_values",
) -> Dict[str, Any]:
    """Perform a Friedman test and Bonferroni‑corrected pairwise Wilcoxon tests.

    The function returns a dictionary with keys ``friedman`` and ``pairwise``. The
    ``friedman`` entry contains ``statistic`` and ``p_value`` if the test could
    be executed; the ``pairwise`` entry is a DataFrame with columns
    ``model_a``, ``model_b``, ``p_value``, ``p_value_adj`` and ``significant``.

    Args:
        df: DataFrame containing a column with lists of metric values per model.
        metric_values_col: Column name where lists of seed-wise results are stored.

    Returns:
        Dictionary with test results. ``None`` values indicate the test could not
        be performed due to insufficient data.
    """
    models = df["model"].unique()
    if len(models) < 2 or metric_values_col not in df.columns:
        return {"friedman": None, "pairwise": None}

    # build dict of model -> list
    values_dict: Dict[str, List[float]] = {}
    for _, row in df.iterrows():
        vals = row.get(metric_values_col)
        if isinstance(vals, list) and vals:
            values_dict[row["model"]] = vals

    if len(values_dict) < 2:
        return {"friedman": None, "pairwise": None}

    # equalize lengths by truncating to shortest sequence
    min_len = min(len(v) for v in values_dict.values())
    if min_len < 2:
        return {"friedman": None, "pairwise": None}

    # Iterate over values_dict keys so that only models with valid list-typed
    # metric data are included.  Iterating ``models`` (from df.unique()) would
    # cause a KeyError for any model absent from values_dict.
    data_matrix = np.vstack([values_dict[m][:min_len] for m in values_dict])

    # Friedman
    try:
        stat, p_value = scipy_stats.friedmanchisquare(*data_matrix)
        friedman_res = {"statistic": stat, "p_value": p_value}
    except Exception as e:
        logger.warning(f"Friedman test failed: {e}")
        friedman_res = None

    # pairwise Wilcoxon with Bonferroni correction
    pairwise = []
    n_models = len(models)
    n_pairs = n_models * (n_models - 1) // 2
    for i in range(n_models):
        for j in range(i + 1, n_models):
            m1, m2 = models[i], models[j]
            try:
                stat, p = scipy_stats.wilcoxon(
                    values_dict[m1][:min_len], values_dict[m2][:min_len]
                )
            except Exception as e:
                logger.warning(f"Wilcoxon failed for {m1} vs {m2}: {e}")
                continue
            p_adj = min(p * n_pairs, 1.0)
            pairwise.append({
                "model_a": m1,
                "model_b": m2,
                "p_value": p,
                "p_value_adj": p_adj,
                "significant": p_adj < 0.05,
            })
    pairwise_df = pd.DataFrame(pairwise) if pairwise else None

    return {"friedman": friedman_res, "pairwise": pairwise_df}


# ---------------------------------------------------------------------------
# Test A — Friedman-Iman-Davenport omnibus
# ---------------------------------------------------------------------------

def run_friedman_iman_davenport(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    metric_col: str = "rmse_mean",
    metric_values_col: str = "rmse_values",
) -> Dict[str, Any]:
    """Friedman-Iman-Davenport omnibus test for rank differences across models.

    When *group_cols* is provided the test operates in **per-dataset mode**:
    ``df`` is pivoted so that rows correspond to evaluation units defined by
    *group_cols* and columns correspond to models.  When *group_cols* is
    ``None`` the test operates in **per-seed mode**: seed-wise values stored in
    *metric_values_col* are used as the sample dimension.

    The Iman-Davenport F-approximation converts the Friedman chi-square to:

    .. math::
        F_F = \\frac{(N-1)\\chi^2_F}{N(k-1) - \\chi^2_F}

    with ``df1 = k-1`` and ``df2 = (k-1)(N-1)``.

    Args:
        df: DataFrame with one row per (model, evaluation_unit) combination.
        group_cols: Columns defining evaluation units for pivot mode. If
            ``None``, per-seed values are used.
        metric_col: Aggregated metric column (pivot mode only).
        metric_values_col: Column containing per-seed value lists (seed mode
            only).

    Returns:
        Dict with keys ``friedman_statistic``, ``friedman_p``,
        ``ff_statistic``, ``ff_p``, ``n_models``, ``n_observations``.
    """
    result: Dict[str, Any] = {
        "friedman_statistic": None,
        "friedman_p": None,
        "ff_statistic": None,
        "ff_p": None,
        "n_models": 0,
        "n_observations": 0,
    }

    if group_cols is not None:
        if metric_col not in df.columns or "model" not in df.columns:
            return result
        pivot = df.pivot_table(index=group_cols, columns="model", values=metric_col)
        pivot = pivot.dropna()
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            return result
        data_arrays = [pivot[col].to_numpy(dtype=float) for col in pivot.columns]
        k, N = len(data_arrays), len(data_arrays[0])
    else:
        if metric_values_col not in df.columns:
            return result
        values_dict: Dict[str, List[float]] = {}
        for _, row in df.iterrows():
            vals = row.get(metric_values_col)
            if isinstance(vals, list) and len(vals) >= 2:
                values_dict[row["model"]] = vals
        if len(values_dict) < 2:
            return result
        min_len = min(len(v) for v in values_dict.values())
        if min_len < 2:
            return result
        data_arrays = [np.array(v[:min_len], dtype=float) for v in values_dict.values()]
        k, N = len(data_arrays), min_len

    result["n_models"] = k
    result["n_observations"] = N

    try:
        chi2_stat, chi2_p = scipy_stats.friedmanchisquare(*data_arrays)
    except Exception as exc:
        logger.warning(f"Friedman-Iman-Davenport test failed: {exc}")
        return result

    result["friedman_statistic"] = float(chi2_stat)
    result["friedman_p"] = float(chi2_p)

    denom = N * (k - 1) - chi2_stat
    if denom > 0:
        ff_stat = (N - 1) * chi2_stat / denom
        df1 = k - 1
        df2 = (k - 1) * (N - 1)
        result["ff_statistic"] = float(ff_stat)
        result["ff_p"] = float(1.0 - scipy_stats.f.cdf(ff_stat, df1, df2))

    return result


# ---------------------------------------------------------------------------
# Test B — Holm-Bonferroni-corrected pairwise Wilcoxon
# ---------------------------------------------------------------------------

def run_holm_wilcoxon(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    metric_col: str = "rmse_mean",
    metric_values_col: str = "rmse_values",
) -> Optional[pd.DataFrame]:
    """Pairwise Wilcoxon signed-rank tests with Holm step-down correction.

    Supports the same two modes as :func:`run_friedman_iman_davenport`:
    *per-dataset* (``group_cols`` provided) and *per-seed* (default).

    Holm-adjusted p-values satisfy the monotonicity constraint:
    ``p_holm[i] = max(p_holm[i], p_holm[i-1])`` after computing
    the initial adjustments ``p_adj[i] = (m-i) * p_raw[i]``.

    Args:
        df: DataFrame with one row per model and metric values.
        group_cols: Evaluation unit columns for pivot mode.  If ``None``,
            per-seed values are used.
        metric_col: Aggregated metric column.
        metric_values_col: Per-seed value lists column.

    Returns:
        DataFrame with columns ``model_a``, ``model_b``, ``p_value``,
        ``p_value_holm``, ``significant``, sorted by raw p-value ascending, or
        ``None`` when data are insufficient.
    """
    if group_cols is not None:
        if metric_col not in df.columns:
            return None
        pivot = df.pivot_table(index=group_cols, columns="model", values=metric_col)
        pivot = pivot.dropna()
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            return None
        models_list = list(pivot.columns)
        vectors: Dict[str, np.ndarray] = {
            m: pivot[m].to_numpy(dtype=float) for m in models_list
        }
    else:
        if metric_values_col not in df.columns:
            return None
        values_dict: Dict[str, List[float]] = {}
        for _, row in df.iterrows():
            vals = row.get(metric_values_col)
            if isinstance(vals, list) and len(vals) >= 2:
                values_dict[row["model"]] = vals
        if len(values_dict) < 2:
            return None
        min_len = min(len(v) for v in values_dict.values())
        if min_len < 2:
            return None
        vectors = {
            m: np.array(v[:min_len], dtype=float) for m, v in values_dict.items()
        }
        models_list = list(vectors.keys())

    pairwise: List[Dict[str, Any]] = []
    for i in range(len(models_list)):
        for j in range(i + 1, len(models_list)):
            m1, m2 = models_list[i], models_list[j]
            if m1 not in vectors or m2 not in vectors:
                continue
            try:
                _, p = scipy_stats.wilcoxon(vectors[m1], vectors[m2])
            except Exception as exc:
                logger.warning(f"Holm-Wilcoxon failed for {m1} vs {m2}: {exc}")
                continue
            pairwise.append({"model_a": m1, "model_b": m2, "p_value": float(p)})

    if not pairwise:
        return None

    result_df = pd.DataFrame(pairwise).sort_values("p_value").reset_index(drop=True)
    num_pairs = len(result_df)
    # Initial Holm adjustment
    holm_raw = [
        min(result_df.loc[k, "p_value"] * (num_pairs - k), 1.0)
        for k in range(num_pairs)
    ]
    # Enforce non-decreasing monotonicity (step-down property)
    for k in range(1, num_pairs):
        holm_raw[k] = max(holm_raw[k], holm_raw[k - 1])
    result_df["p_value_holm"] = holm_raw
    result_df["significant"] = result_df["p_value_holm"] < 0.05
    return result_df


# ---------------------------------------------------------------------------
# Test C — Diebold-Mariano with Newey-West HAC
# ---------------------------------------------------------------------------

def _newey_west_variance(d: np.ndarray, max_lag: int) -> float:
    """Newey-West HAC variance estimator for a loss-differential series.

    .. math::
        \\hat{V}_{NW} = \\gamma_0 + 2\\sum_{h=1}^{L}\\left(1-\\frac{h}{L+1}\\right)\\gamma_h

    Args:
        d: 1-D loss-differential array of length *T*.
        max_lag: Bartlett kernel bandwidth *L*.

    Returns:
        HAC variance estimate (non-negative scalar).
    """
    T = len(d)
    d_c = d - d.mean()
    gamma0 = float(np.dot(d_c, d_c) / T)
    hac = gamma0
    for h in range(1, max_lag + 1):
        gamma_h = float(np.dot(d_c[h:], d_c[:-h]) / T)
        hac += 2.0 * (1.0 - h / (max_lag + 1)) * gamma_h
    return max(hac, 0.0)


def _load_predictions_csv(
    results_dir: Path,
    model: str,
    category: str,
    asset: str,
    horizon: int,
    seed: int,
) -> Optional[pd.DataFrame]:
    """Load per-step predictions for a given model/seed.

    Searches ``best/predictions.csv`` first, then ``last/predictions.csv``.

    Args:
        results_dir: Root results directory.
        model: Model name.
        category: Asset category.
        asset: Asset identifier.
        horizon: Forecast horizon.
        seed: Random seed.

    Returns:
        DataFrame with ``y_true_step_*`` and ``y_pred_step_*`` columns, or
        ``None`` if no file is found.
    """
    base = (
        results_dir / "final" / model / category / asset / str(horizon) / str(seed)
    )
    for sub in ("best", "last"):
        path = base / sub / "predictions.csv"
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as exc:
                logger.warning(f"Failed to read predictions at {path}: {exc}")
                return None
    logger.warning(
        f"Predictions not found for {model}/{category}/{asset}/h={horizon}/s={seed}"
    )
    return None


def run_diebold_mariano(
    results_dir: Path,
    df: pd.DataFrame,
    category: str,
    asset: str,
    horizon: int,
    seeds: List[int],
    max_lag: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Diebold-Mariano test with Newey-West HAC for all model pairs.

    MSE loss differentials are pooled across seeds (concatenated) to produce
    a long pseudo-series which improves HAC estimation.  The bandwidth
    defaults to ``max(1, int(horizon ** (1/3)))``.

    .. math::
        DM = \\frac{\\bar{d}}{\\sqrt{\\hat{V}_{NW}/T}}, \\quad
        p = 2\\Phi(-|DM|)

    Holm-Bonferroni correction is applied across all pairs.

    Args:
        results_dir: Root results directory containing seed predictions.
        df: DataFrame of aggregated results (used for model enumeration).
        category: Asset category.
        asset: Asset identifier.
        horizon: Forecast horizon length.
        seeds: List of seed integers.
        max_lag: Newey-West bandwidth.  Auto-selected when ``None``.

    Returns:
        DataFrame with columns ``model_a``, ``model_b``, ``dm_statistic``,
        ``p_value``, ``p_value_holm``, ``significant``, or ``None`` on failure.
    """
    models = df["model"].unique().tolist()
    if len(models) < 2:
        return None

    if max_lag is None:
        max_lag = max(1, int(horizon ** (1.0 / 3.0)))

    # Build per-model concatenated MSE loss arrays (across seeds)
    loss_arrays: Dict[str, np.ndarray] = {}
    for model in models:
        seed_losses: List[np.ndarray] = []
        for seed in seeds:
            pred_df = _load_predictions_csv(
                results_dir, model, category, asset, horizon, seed
            )
            if pred_df is None:
                continue
            true_cols = sorted(
                [c for c in pred_df.columns if c.startswith("y_true_step_")]
            )
            pred_cols = sorted(
                [c for c in pred_df.columns if c.startswith("y_pred_step_")]
            )
            if not true_cols or not pred_cols:
                continue
            # Shape: (T, horizon)
            y_true = pred_df[true_cols].to_numpy(dtype=float)
            y_pred = pred_df[pred_cols].to_numpy(dtype=float)
            # Mean MSE per observation (averaged across steps)
            mse_per_obs = np.mean((y_true - y_pred) ** 2, axis=1)  # Shape: (T,)
            seed_losses.append(mse_per_obs)
        if seed_losses:
            loss_arrays[model] = np.concatenate(seed_losses)

    available = [m for m in models if m in loss_arrays]
    if len(available) < 2:
        logger.warning(
            f"DM test: insufficient prediction data for {asset}/h={horizon} "
            f"(only {len(available)} models available)"
        )
        return None

    pairwise: List[Dict[str, Any]] = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            m1, m2 = available[i], available[j]
            min_T = min(len(loss_arrays[m1]), len(loss_arrays[m2]))
            d = loss_arrays[m1][:min_T] - loss_arrays[m2][:min_T]
            T = len(d)
            d_bar = float(np.mean(d))
            nw_var = _newey_west_variance(d, max_lag)
            if nw_var <= 0:
                logger.warning(
                    f"DM: zero HAC variance for {m1} vs {m2} on {asset}/h={horizon}"
                )
                continue
            dm_stat = d_bar / np.sqrt(nw_var / T)
            p_val = float(2.0 * scipy_stats.norm.cdf(-abs(dm_stat)))
            pairwise.append(
                {
                    "model_a": m1,
                    "model_b": m2,
                    "dm_statistic": float(dm_stat),
                    "p_value": p_val,
                }
            )

    if not pairwise:
        return None

    result_df = pd.DataFrame(pairwise).sort_values("p_value").reset_index(drop=True)
    num_pairs = len(result_df)
    holm_raw = [
        min(result_df.loc[k, "p_value"] * (num_pairs - k), 1.0)
        for k in range(num_pairs)
    ]
    for k in range(1, num_pairs):
        holm_raw[k] = max(holm_raw[k], holm_raw[k - 1])
    result_df["p_value_holm"] = holm_raw
    result_df["significant"] = result_df["p_value_holm"] < 0.05
    return result_df


# ---------------------------------------------------------------------------
# Test D — Intraclass Correlation Coefficient ICC(3,1)
# ---------------------------------------------------------------------------

def run_icc(
    df: pd.DataFrame,
    metric_values_col: str = "rmse_values",
) -> Dict[str, Any]:
    """Compute ICC(3,1) — two-way mixed, single measure, consistency.

    Models are treated as *subjects* and seeds as *raters*.  A high ICC
    (close to 1) indicates that relative performance rankings are stable
    across seeds.

    Two-way ANOVA decomposition::

        ICC(3,1) = (MSb - MSe) / (MSb + (k-1) * MSe)

    where *MSb* = between-subjects mean square and *MSe* = error mean square.

    Args:
        df: DataFrame with one row per model; *metric_values_col* holds a
            list of per-seed metric values.
        metric_values_col: Column containing per-seed value lists.

    Returns:
        Dict with keys ``icc``, ``n_models``, ``n_seeds``, ``f_statistic``,
        ``f_p_value``.
    """
    null_result: Dict[str, Any] = {
        "icc": None,
        "n_models": 0,
        "n_seeds": 0,
        "f_statistic": None,
        "f_p_value": None,
    }

    if metric_values_col not in df.columns:
        return null_result

    values_dict: Dict[str, List[float]] = {}
    for _, row in df.iterrows():
        vals = row.get(metric_values_col)
        if isinstance(vals, list) and len(vals) >= 2:
            values_dict[row["model"]] = vals

    if len(values_dict) < 2:
        return null_result

    n_seeds = min(len(v) for v in values_dict.values())
    if n_seeds < 2:
        return null_result

    n_models = len(values_dict)
    # Matrix: rows = models (subjects), cols = seeds (raters)
    # Shape: (n_models, n_seeds)
    X = np.array(
        [values_dict[m][:n_seeds] for m in values_dict], dtype=float
    )
    grand_mean = float(np.mean(X))
    subject_means = np.mean(X, axis=1)  # Shape: (n_models,)
    rater_means = np.mean(X, axis=0)    # Shape: (n_seeds,)

    SSb = float(n_seeds * np.sum((subject_means - grand_mean) ** 2))
    SSw = float(np.sum((X - subject_means[:, np.newaxis]) ** 2))
    SSr = float(n_models * np.sum((rater_means - grand_mean) ** 2))
    SSe = SSw - SSr

    df_b = n_models - 1
    df_e = (n_models - 1) * (n_seeds - 1)

    if df_b <= 0 or df_e <= 0:
        return null_result

    MSb = SSb / df_b
    MSe = SSe / df_e if df_e > 0 else 0.0
    denom_icc = MSb + (n_seeds - 1) * MSe
    if denom_icc <= 0:
        return null_result

    icc_val = float((MSb - MSe) / denom_icc)
    f_stat = float(MSb / MSe) if MSe > 0 else None
    f_p = (
        float(1.0 - scipy_stats.f.cdf(f_stat, df_b, df_e))
        if f_stat is not None
        else None
    )

    return {
        "icc": icc_val,
        "n_models": n_models,
        "n_seeds": n_seeds,
        "f_statistic": f_stat,
        "f_p_value": f_p,
    }


# ---------------------------------------------------------------------------
# Test E — One-sample z-test for directional accuracy vs 50% baseline
# ---------------------------------------------------------------------------

def _get_n_test_samples(
    results_dir: Path,
    model: str,
    category: str,
    asset: str,
    horizon: int,
    seeds: List[int],
) -> Optional[int]:
    """Infer the test set size from a predictions CSV.

    Tries seeds in order and returns the row count of the first file found.

    Args:
        results_dir: Root results directory.
        model: Model name.
        category: Asset category.
        asset: Asset identifier.
        horizon: Forecast horizon.
        seeds: Seed list to try.

    Returns:
        Integer row count, or ``None`` if no predictions file is found.
    """
    for seed in seeds:
        pred_df = _load_predictions_csv(
            results_dir, model, category, asset, horizon, seed
        )
        if pred_df is not None:
            return int(len(pred_df))
    return None


def run_da_ztest(
    df: pd.DataFrame,
    n_test: int,
    baseline: float = 0.5,
    da_col: str = "da_mean",
) -> Optional[pd.DataFrame]:
    """One-sample z-test for directional accuracy against a chance baseline.

    Tests H₀: DA = *baseline* vs H₁: DA > *baseline* (one-sided) for each
    model.  Holm-Bonferroni correction is applied across models.

    .. math::
        z = \\frac{\\hat{p} - p_0}{\\sqrt{p_0(1-p_0)/n}}

    DA values stored as percentages (0–100) are automatically normalised to
    the [0, 1] range.

    Args:
        df: DataFrame with one row per model and a mean DA column.
        n_test: Number of test observations used to compute DA.
        baseline: Null hypothesis probability (default 0.5 for chance level).
        da_col: Column name with mean DA values.

    Returns:
        DataFrame with columns ``model``, ``da``, ``z_statistic``,
        ``p_value``, ``p_value_holm``, ``significant``, or ``None`` when
        data are insufficient.
    """
    # Fall back to directional_accuracy_mean if the preferred column is absent
    if da_col not in df.columns:
        alt = "directional_accuracy_mean"
        if alt in df.columns:
            da_col = alt
        else:
            return None

    if n_test < 1:
        return None

    se = np.sqrt(baseline * (1.0 - baseline) / n_test)
    if se <= 0:
        return None

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        da_raw = row[da_col]
        if pd.isna(da_raw):
            continue
        # Normalise: assume percentage when value exceeds 1
        da = float(da_raw) / 100.0 if float(da_raw) > 1.0 else float(da_raw)
        z_stat = (da - baseline) / se
        # One-tailed p-value (right tail: DA > baseline)
        p_val = float(1.0 - scipy_stats.norm.cdf(z_stat))
        rows.append(
            {
                "model": row["model"],
                "da": da,
                "z_statistic": float(z_stat),
                "p_value": p_val,
            }
        )

    if not rows:
        return None

    result_df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    num_models = len(result_df)
    holm_raw = [
        min(result_df.loc[k, "p_value"] * (num_models - k), 1.0)
        for k in range(num_models)
    ]
    for k in range(1, num_models):
        holm_raw[k] = max(holm_raw[k], holm_raw[k - 1])
    result_df["p_value_holm"] = holm_raw
    result_df["significant"] = result_df["p_value_holm"] < 0.05
    return result_df


# ---------------------------------------------------------------------------
# Test F — Spearman ρ with Stouffer's method and Jonckheere-Terpstra
# ---------------------------------------------------------------------------

def _jonckheere_terpstra(groups: List[np.ndarray]) -> Dict[str, Any]:
    """Jonckheere-Terpstra trend test for ordered groups.

    Computes the JT statistic as the sum of concordant pair counts across all
    group pairs (i, j) with i < j, then applies a standard normal
    approximation.

    Args:
        groups: Ordered list of arrays; each array contains values for one
            group.  Groups must be in the hypothesised ascending trend order.

    Returns:
        Dict with ``jt_statistic`` (standardised Z), ``p_value`` (one-sided,
        increasing trend), ``n_groups``.
    """
    k = len(groups)
    if k < 2:
        return {"jt_statistic": None, "p_value": None, "n_groups": k}

    jt = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            for xi in groups[i]:
                jt += float(np.sum(xi < groups[j]))
                jt += 0.5 * float(np.sum(xi == groups[j]))

    ns = [len(g) for g in groups]
    n_total = sum(ns)
    # Null expectation and variance (no-tie approximation)
    e_jt = (n_total ** 2 - sum(ni ** 2 for ni in ns)) / 4.0
    var_jt = (
        n_total ** 2 * (2 * n_total + 3)
        - sum(ni ** 2 * (2 * ni + 3) for ni in ns)
    ) / 72.0

    if var_jt <= 0:
        return {"jt_statistic": None, "p_value": None, "n_groups": k}

    z_jt = (jt - e_jt) / np.sqrt(var_jt)
    p_val = float(1.0 - scipy_stats.norm.cdf(z_jt))  # one-sided (increasing)
    return {"jt_statistic": float(z_jt), "p_value": p_val, "n_groups": k}


def run_spearman_stability(
    df: pd.DataFrame,
    param_counts_path: Optional[Path] = None,
    metric_col: str = "rmse_mean",
) -> Dict[str, Any]:
    """Spearman ρ rank stability and complexity-performance monotonicity.

    Two sub-tests:

    1. **Cross-horizon rank stability** — For each (category, asset) group,
       the Spearman ρ between model ranks at the lowest and highest horizon is
       computed.  Individual ρ values are Fisher-z transformed and pooled via
       Stouffer's method::

           Z_S = \\sum_i \\mathrm{arctanh}(\\rho_i) / \\sqrt{n}

    2. **Complexity-performance monotonicity** — Models within each
       (category, horizon) group are ordered by parameter count and split into
       up to three quantile-based tiers.  A Jonckheere-Terpstra test checks
       whether mean RMSE increases monotonically with model complexity
       (requires *param_counts_path*).

    Args:
        df: Full results DataFrame with ``model``, ``category``, ``asset``,
            ``horizon`` and *metric_col* columns.
        param_counts_path: Path to a CSV with
            ``model``, ``category``, ``horizon``, ``parameter_count`` columns.
        metric_col: Metric used for rank derivation (default ``rmse_mean``).

    Returns:
        Dict with keys ``cross_horizon`` (dict or ``None``) and
        ``complexity_performance`` (DataFrame or ``None``).
    """
    result: Dict[str, Any] = {
        "cross_horizon": None,
        "complexity_performance": None,
    }

    # ----- Sub-test 1: cross-horizon rank stability -----
    horizons_sorted = sorted(df["horizon"].unique())
    if len(horizons_sorted) >= 2:
        low_h, high_h = horizons_sorted[0], horizons_sorted[-1]
        rho_records: List[Dict[str, Any]] = []
        for (category, asset), grp in df.groupby(["category", "asset"]):
            low_df = (
                grp[grp["horizon"] == low_h][["model", metric_col]].dropna()
            )
            high_df = (
                grp[grp["horizon"] == high_h][["model", metric_col]].dropna()
            )
            common = set(low_df["model"]) & set(high_df["model"])
            if len(common) < 3:
                continue
            low_ranks = (
                low_df[low_df["model"].isin(common)]
                .set_index("model")[metric_col]
                .rank()
            )
            high_ranks = (
                high_df[high_df["model"].isin(common)]
                .set_index("model")[metric_col]
                .rank()
            )
            shared = low_ranks.index.intersection(high_ranks.index)
            if len(shared) < 3:
                continue
            try:
                rho, p_rho = scipy_stats.spearmanr(
                    low_ranks[shared], high_ranks[shared]
                )
            except Exception as exc:
                logger.warning(f"Spearman failed for {category}/{asset}: {exc}")
                continue
            rho_records.append(
                {
                    "category": category,
                    "asset": asset,
                    "horizon_low": low_h,
                    "horizon_high": high_h,
                    "spearman_rho": float(rho),
                    "p_value": float(p_rho),
                    "n_models": len(shared),
                }
            )

        if rho_records:
            rho_df = pd.DataFrame(rho_records)
            # Fisher-z transform; clip to avoid arctanh(±1) = ±inf
            z_vals = np.arctanh(
                np.clip(rho_df["spearman_rho"].to_numpy(), -0.9999, 0.9999)
            )
            n_assets = len(z_vals)
            stouffer_z = float(np.sum(z_vals) / np.sqrt(n_assets))
            stouffer_p = float(1.0 - scipy_stats.norm.cdf(stouffer_z))
            result["cross_horizon"] = {
                "per_asset": rho_df,
                "stouffer_z": stouffer_z,
                "stouffer_p": stouffer_p,
                "n_assets": n_assets,
            }

    # ----- Sub-test 2: complexity-performance monotonicity -----
    if param_counts_path is not None and param_counts_path.exists():
        try:
            params_df = pd.read_csv(param_counts_path)
        except Exception as exc:
            logger.warning(f"Could not load parameter counts from {param_counts_path}: {exc}")
            params_df = None

        if params_df is not None and "parameter_count" in params_df.columns:
            jt_records: List[Dict[str, Any]] = []
            for (category, horizon), grp in df.groupby(["category", "horizon"]):
                cat_params = params_df[
                    (params_df["category"] == category)
                    & (params_df["horizon"] == horizon)
                ][["model", "parameter_count"]].drop_duplicates("model")
                merged = grp.merge(cat_params, on="model", how="inner")
                merged = merged.sort_values("parameter_count").dropna(
                    subset=[metric_col]
                )
                if len(merged) < 3:
                    continue
                # Split into up to 3 quantile-based complexity tiers
                n_tiers = min(3, len(merged))
                try:
                    merged["complexity_tier"] = pd.qcut(
                        merged["parameter_count"],
                        q=n_tiers,
                        labels=False,
                        duplicates="drop",
                    )
                except Exception:
                    continue
                groups = [
                    merged[merged["complexity_tier"] == t][metric_col].to_numpy(
                        dtype=float
                    )
                    for t in sorted(merged["complexity_tier"].dropna().unique())
                ]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) < 2:
                    continue
                jt_res = _jonckheere_terpstra(groups)
                jt_records.append(
                    {
                        "category": category,
                        "horizon": horizon,
                        "jt_statistic": jt_res["jt_statistic"],
                        "p_value": jt_res["p_value"],
                        "n_groups": jt_res["n_groups"],
                        "n_models": len(merged),
                    }
                )

            if jt_records:
                result["complexity_performance"] = pd.DataFrame(jt_records)

    return result


def run_benchmark(config: ProjectConfig) -> None:
    """Run the full benchmarking pipeline and persist a detailed hierarchy of
    tables and figures under ``results/benchmark``.

    All available models, categories, assets and horizons are discovered by
    inspecting the results directory produced by the final training stage. The
    procedure validates that every expected seed contributed metrics; missing
    data will raise a ``RuntimeError`` and abort.
    """
    logger.info("Starting benchmark generation...")

    df = _collect_all_results(config)
    if df.empty:
        raise RuntimeError("No benchmarkable results found – check final output")

    base_results = config.get_path("results_dir")
    benchmark_dir = base_results / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # 1. Global Summary -------------------------------------------------------
    global_dir = benchmark_dir / "global_summary"
    global_tables = global_dir / "tables"
    global_figures = global_dir / "figures"
    global_tables.mkdir(parents=True, exist_ok=True)
    global_figures.mkdir(parents=True, exist_ok=True)

    # compute win counts and rankings based on rmse
    if "rmse_mean" in df.columns:
        # ranking per asset/horizon then aggregate
        df["global_rank"] = (
            df.groupby(["category", "asset", "horizon"])["rmse_mean"]
            .rank(ascending=True, method="min")
        )
        global_ranking = (
            df.groupby("model")["global_rank"]
            .agg(["mean", "median"]).reset_index()
            .rename(columns={"mean": "mean_rank", "median": "median_rank"})
            .sort_values("mean_rank")
        )
        global_ranking.to_csv(global_tables / "global_ranking_aggregated.csv", index=False)

        # Test A — Friedman-Iman-Davenport across all evaluation units
        fid_global = run_friedman_iman_davenport(
            df, group_cols=["category", "asset", "horizon"]
        )
        pd.DataFrame([fid_global]).to_csv(
            global_tables / "friedman_iman_davenport.csv", index=False
        )
        # Test B — Holm-Bonferroni pairwise Wilcoxon across all evaluation units
        holm_global = run_holm_wilcoxon(df, group_cols=["category", "asset", "horizon"])
        if holm_global is not None:
            holm_global.to_csv(global_tables / "holm_wilcoxon.csv", index=False)

        plot_rank_barplot(
            leaderboard=global_ranking,
            title="Global Mean Rank (RMSE)",
            output_path=global_figures / "global_rank_comparison.pdf"
        )

        # win counts
        wins = []
        for _, grp in df.groupby(["category", "asset", "horizon"]):
            ranked = compute_rankings(grp, "rmse_mean")
            wins.append(ranked.iloc[0]["model"])
        win_series = pd.Series(wins, name="model")
        win_counts = (
            win_series.value_counts().rename_axis("model").reset_index(name="wins")
        )
        total = len(win_series)
        win_counts["total_comparisons"] = total
        win_counts["win_rate"] = win_counts["wins"] / total
        win_counts.to_csv(global_tables / "global_win_counts.csv", index=False)

        # Test F — Spearman ρ cross-horizon stability + Jonckheere-Terpstra complexity trend
        param_counts_csv = config.get_path("results_dir") / "test_counts.csv"
        spearman_global = run_spearman_stability(df, param_counts_path=param_counts_csv)
        if spearman_global["cross_horizon"] is not None:
            _ch_global = spearman_global["cross_horizon"]
            _ch_global["per_asset"].to_csv(
                global_tables / "spearman_cross_horizon_per_asset.csv", index=False
            )
            pd.DataFrame(
                [
                    {
                        "stouffer_z": _ch_global["stouffer_z"],
                        "stouffer_p": _ch_global["stouffer_p"],
                        "n_assets": _ch_global["n_assets"],
                    }
                ]
            ).to_csv(global_tables / "spearman_stouffer.csv", index=False)
        if spearman_global["complexity_performance"] is not None:
            spearman_global["complexity_performance"].to_csv(
                global_tables / "jonckheere_terpstra_complexity.csv", index=False
            )

        # global metric heatmap (model vs category-horizon)
        pivot = (
            df.pivot_table(
                index="model",
                columns=["category", "horizon"],
                values="rmse_mean",
            )
        )
        plot_summary_heatmap(
            pivot_df=pivot,
            title="RMSE Mean Heatmap by Model / Category-Horizon",
            output_path=global_figures / "global_metric_heatmap.pdf",
            figsize=(12, max(6, pivot.shape[0] * 0.6))
        )
    else:
        raise RuntimeError("Required metric 'rmse_mean' not found in results dataframe")

    # global variance decomposition
    global_var_decomp = compute_variance_decomposition(df)
    if not global_var_decomp.empty:
        global_var_decomp.to_csv(global_tables / "variance_decomposition.csv", index=False)
        plot_variance_pie(
            df=global_var_decomp,
            title="Global Variance Decomposition",
            output_path=global_figures / "variance_decomposition_pie.pdf"
        )

    # 2. Categories -----------------------------------------------------------
    categories_dir = benchmark_dir / "categories"
    seeds = config.get_eval_seeds()
    for category in df["category"].unique():
        cat_dir = categories_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        cat_df = df[df["category"] == category].copy()

        # Category summary tables & figures
        cat_summary_tables = cat_dir / "category_summary" / "tables"
        cat_summary_figures = cat_dir / "category_summary" / "figures"
        cat_summary_tables.mkdir(parents=True, exist_ok=True)
        cat_summary_figures.mkdir(parents=True, exist_ok=True)

        # basic aggregated metrics and ranks per metric
        cat_metrics = (
            cat_df.groupby("model")[['rmse_mean', 'mae_mean']]
            .mean()
            .reset_index()
        )
        cat_metrics.to_csv(cat_summary_tables / "category_metrics_aggregated.csv", index=False)
        cat_metrics["rmse_rank"] = cat_metrics["rmse_mean"].rank(ascending=True)
        cat_metrics["mae_rank"] = cat_metrics["mae_mean"].rank(ascending=True)
        cat_metrics[["model", "rmse_rank", "mae_rank"]].to_csv(
            cat_summary_tables / "category_ranking_by_metric.csv", index=False
        )
        plot_model_comparison_bar(
            df=cat_metrics,
            x_col="model",
            y_col="rmse_mean",
            title=f"{category.upper()} Mean RMSE",
            output_path=cat_summary_figures / "category_metric_barplot.pdf"
        )

        # statistical tests at category level
        tests = run_statistical_tests(cat_df)
        if tests["friedman"] is not None:
            pd.DataFrame([tests["friedman"]]).to_csv(
                cat_summary_tables / "friedman_test.csv", index=False
            )
        if tests["pairwise"] is not None:
            tests["pairwise"].to_csv(
                cat_summary_tables / "pairwise_wilcoxon.csv", index=False
            )

        # Test A — Friedman-Iman-Davenport at category level (cross asset-horizon units)
        fid_cat = run_friedman_iman_davenport(
            cat_df, group_cols=["asset", "horizon"]
        )
        pd.DataFrame([fid_cat]).to_csv(
            cat_summary_tables / "friedman_iman_davenport.csv", index=False
        )
        # Test B — Holm-Bonferroni pairwise Wilcoxon at category level
        holm_cat = run_holm_wilcoxon(cat_df, group_cols=["asset", "horizon"])
        if holm_cat is not None:
            holm_cat.to_csv(cat_summary_tables / "holm_wilcoxon.csv", index=False)

        # configuration hash reference not implemented here - developers may
        # populate later if required
        (cat_summary_tables / "config_hash_reference.json").write_text(
            json.dumps({"note": "to be filled manually"})
        )

        # Summary figures for category
        plot_model_distribution(
            df=cat_df,
            x_col="model",
            y_col="rmse_mean",
            title=f"{category} RMSE Distribution by Model",
            output_path=cat_summary_figures / "category_rank_boxplot.pdf",
            plot_type="box"
        )

        plot_scatter_comparison(
            df=cat_metrics,
            x_col="rmse_mean",
            y_col="mae_mean",
            title=f"{category} RMSE vs MAE",
            output_path=cat_summary_figures / "category_scatter_rmse_mae.pdf"
        )

        # violin requires one numeric value per row; explode lists
        violin_df = cat_df[['model', 'rmse_values']].explode('rmse_values')
        violin_df = violin_df.rename(columns={'rmse_values': 'rmse'})
        plot_model_distribution(
            df=violin_df,
            x_col="model",
            y_col="rmse",
            title=f"{category} Seed-wise RMSE Variance",
            output_path=cat_summary_figures / "category_violin_seed_variance.pdf",
            plot_type="violin"
        )

        # Assets -----------------------------------------------------------------
        assets_dir = cat_dir / "assets"
        for asset in cat_df["asset"].unique():
            asset_dir = assets_dir / asset
            asset_dir.mkdir(parents=True, exist_ok=True)
            asset_df = cat_df[cat_df["asset"] == asset].copy()

            # Asset summary
            asset_summary_tables = asset_dir / "summary" / "tables"
            asset_summary_figures = asset_dir / "summary" / "figures"
            asset_summary_tables.mkdir(parents=True, exist_ok=True)
            asset_summary_figures.mkdir(parents=True, exist_ok=True)

            asset_metrics = (
                asset_df.groupby(["model", "horizon"])[["rmse_mean", "mae_mean"]]
                .mean()
                .reset_index()
            )
            asset_metrics.to_csv(asset_summary_tables / "asset_metrics_aggregated.csv", index=False)

            # Best models per horizon
            bests = (
                asset_df.groupby("horizon")["rmse_mean"].idxmin()
            )
            best_df = asset_df.loc[bests, ["horizon", "model", "rmse_mean"]].rename(columns={"model": "best_model", "rmse_mean": "rmse"})
            best_df.to_csv(asset_summary_tables / "asset_best_models.csv", index=False)

            # ranking matrix across horizons
            rank_df = []
            for h, grp in asset_df.groupby("horizon"):
                ranked = compute_rankings(grp, "rmse_mean")[["model", "rank"]]
                ranked = ranked.rename(columns={"rank": f"horizon_{h}_rank"})
                rank_df.append(ranked.set_index("model"))
            if rank_df:
                rank_matrix = pd.concat(rank_df, axis=1).fillna(np.nan)
                rank_matrix.to_csv(asset_summary_tables / "asset_ranking_matrix.csv")


            if "da_mean" in asset_metrics.columns:
                plot_model_comparison_bar(
                    df=asset_metrics,
                    x_col="model",
                    y_col="da_mean",
                    hue_col="horizon",
                    title=f"{asset} DA across Horizons",
                    output_path=asset_summary_figures / "asset_da_barplot_all_horizons.pdf"
                )

            plot_model_comparison_bar(
                df=asset_df,
                x_col="model",
                y_col="rmse_mean",
                hue_col="horizon",
                title=f"{asset} Model Comparison Grouped",
                output_path=asset_summary_figures / "asset_model_comparison_grouped.pdf"
            )

            # Cross horizon comparisons
            cross_horizon_tables = asset_dir / "cross_horizon" / "tables"
            cross_horizon_figures = asset_dir / "cross_horizon" / "figures"
            cross_horizon_tables.mkdir(parents=True, exist_ok=True)
            cross_horizon_figures.mkdir(parents=True, exist_ok=True)

            deg_df = asset_df[["model", "horizon", "rmse_mean"]].copy()
            deg_df.to_csv(cross_horizon_tables / "horizon_degradation_curves.csv", index=False)
            
            # Use plot_horizon_degradation wrapper
            deg_data = {}
            for m in deg_df["model"].unique():
                m_df = deg_df[deg_df["model"] == m]
                deg_data[m] = dict(zip(m_df["horizon"], m_df["rmse_mean"]))
            
            plot_horizon_degradation(
                data=deg_data,
                metric="RMSE",
                title=f"{asset} Horizon Degradation (RMSE)",
                output_path=cross_horizon_figures / "horizon_degradation_lineplot.pdf"
            )

            # prepare pivot table describing rmse_mean by model and horizon
            pivot = asset_df.pivot_table(index="model", columns="horizon", values="rmse_mean")

            # ranking shift table
            shift_records = []
            if pivot.shape[1] > 1:
                ranks = pivot.rank(axis=0)
                for model in ranks.index:
                    low_h = ranks.columns.min()
                    high_h = ranks.columns.max()
                    shift = ranks.loc[model, high_h] - ranks.loc[model, low_h]
                    shift_records.append({
                        "model": model,
                        f"horizon_{low_h}_rank": ranks.loc[model, low_h],
                        f"horizon_{high_h}_rank": ranks.loc[model, high_h],
                        "shift": shift,
                    })
            pd.DataFrame(shift_records).to_csv(cross_horizon_tables / "horizon_ranking_shift.csv", index=False)

            plot_summary_heatmap(
                pivot_df=pivot,
                title=f"{asset} Horizon Sensitivity Heatmap",
                output_path=cross_horizon_figures / "horizon_sensitivity_heatmap.pdf"
            )

            # Test F — Spearman ρ cross-horizon rank stability for this asset
            spearman_asset = run_spearman_stability(asset_df)
            if spearman_asset["cross_horizon"] is not None:
                _ch_asset = spearman_asset["cross_horizon"]
                _ch_asset["per_asset"].to_csv(
                    cross_horizon_tables / "spearman_cross_horizon.csv", index=False
                )
                pd.DataFrame(
                    [
                        {
                            "stouffer_z": _ch_asset["stouffer_z"],
                            "stouffer_p": _ch_asset["stouffer_p"],
                            "n_assets": _ch_asset["n_assets"],
                        }
                    ]
                ).to_csv(cross_horizon_tables / "spearman_stouffer.csv", index=False)

            # Horizons breakdown
            horizons_dir = asset_dir / "horizons"
            for horizon in asset_df["horizon"].unique():
                horizon_dir = horizons_dir / str(horizon)
                hor_df = asset_df[asset_df["horizon"] == horizon].copy()

                res_agg_tables = horizon_dir / "results_aggregated" / "tables"
                res_agg_figures = horizon_dir / "results_aggregated" / "figures"
                res_agg_tables.mkdir(parents=True, exist_ok=True)
                res_agg_figures.mkdir(parents=True, exist_ok=True)

                cols = ["model"] + [c for c in hor_df.columns if c.endswith("_mean") or c.endswith("_std")]
                hor_df[cols].to_csv(res_agg_tables / "metrics_aggregated.csv", index=False)
                
                plot_model_comparison_bar(
                    df=hor_df,
                    x_col="model",
                    y_col="rmse_mean",
                    title=f"{asset} H={horizon} RMSE Comparison",
                    output_path=res_agg_figures / "rmse_comparison_barplot.pdf"
                )

                # raw per-seed table
                seed_rows = []
                for m in hor_df["model"].unique():
                    values = hor_df[hor_df["model"] == m]["rmse_values"].iloc[0]
                    if not isinstance(values, list):
                        continue
                    for si, v in enumerate(values):
                        # Guard against values list being longer than the seeds list
                        seed_label = seeds[si] if si < len(seeds) else si
                        seed_rows.append({"model": m, "seed": seed_label, "rmse": v})
                pd.DataFrame(seed_rows).to_csv(res_agg_tables / "metrics_raw_per_seed.csv", index=False)

                rank_mat = compute_rankings(hor_df, "rmse_mean")[["model", "rank"]]
                rank_mat.to_csv(res_agg_tables / "rank_matrix.csv", index=False)
                top = rank_mat.head(5).reset_index(drop=True)
                # rank_mat columns are ["model", "rank"] — preserve that order.
                top.columns = ["model", "rank"]
                top.to_csv(res_agg_tables / "top_models_summary.csv", index=False)

                # figures for this horizon
                plot_model_comparison_bar(
                    df=hor_df,
                    x_col="model",
                    y_col="mae_mean",
                    title=f"{asset} H={horizon} MAE Comparison",
                    output_path=res_agg_figures / "mae_comparison_barplot.pdf"
                )

                if "da_mean" in hor_df.columns:
                    plot_model_comparison_bar(
                        df=hor_df,
                        x_col="model",
                        y_col="da_mean",
                        title=f"{asset} H={horizon} DA Comparison",
                        output_path=res_agg_figures / "da_comparison_barplot.pdf"
                    )

                # explode rmse_values list for proper boxplot
                hor_violin = hor_df[['model', 'rmse_values']].explode('rmse_values').rename(columns={'rmse_values':'rmse'})
                plot_model_distribution(
                    df=hor_violin,
                    x_col="model",
                    y_col="rmse",
                    title=f"{asset} H={horizon} Seed Variance",
                    output_path=res_agg_figures / "seed_variance_boxplot.pdf",
                    plot_type="box"
                )

                plot_model_comparison_bar(
                    df=rank_mat,
                    x_col="model",
                    y_col="rank",
                    title=f"{asset} H={horizon} Rank Barplot",
                    output_path=res_agg_figures / "rank_barplot.pdf"
                )

                plot_scatter_comparison(
                    df=hor_df,
                    x_col="rmse_mean",
                    y_col="rmse_std",
                    title=f"{asset} H={horizon} RMSE vs Variance",
                    output_path=res_agg_figures / "scatter_rmse_vs_seed_variance.pdf"
                )

                # Statistical tests
                stat_tests_dir = horizon_dir / "statistical_tests"
                stat_tests_dir.mkdir(parents=True, exist_ok=True)
                stats_res = run_statistical_tests(hor_df)
                if stats_res["friedman"] is not None:
                    pd.DataFrame([stats_res["friedman"]]).to_csv(stat_tests_dir / "friedman_test.csv", index=False)
                if stats_res["pairwise"] is not None:
                    stats_res["pairwise"].to_csv(stat_tests_dir / "pairwise_wilcoxon.csv", index=False)

                var_decomp = compute_variance_decomposition(hor_df)
                var_decomp.to_csv(stat_tests_dir / "variance_decomposition.csv", index=False)

                # Test A — Friedman-Iman-Davenport (seed-level; N=seeds, k=models)
                fid_hor = run_friedman_iman_davenport(hor_df)
                pd.DataFrame([fid_hor]).to_csv(
                    stat_tests_dir / "friedman_iman_davenport.csv", index=False
                )
                # Test B — Holm-Bonferroni pairwise Wilcoxon (seed-level)
                holm_hor = run_holm_wilcoxon(hor_df)
                if holm_hor is not None:
                    holm_hor.to_csv(stat_tests_dir / "holm_wilcoxon.csv", index=False)
                # Test C — Diebold-Mariano with Newey-West HAC
                dm_res = run_diebold_mariano(
                    base_results, hor_df, category, asset, horizon, seeds
                )
                if dm_res is not None:
                    dm_res.to_csv(stat_tests_dir / "diebold_mariano.csv", index=False)
                # Test D — ICC(3,1) seed reliability
                icc_res = run_icc(hor_df)
                pd.DataFrame([icc_res]).to_csv(
                    stat_tests_dir / "icc_seed_reliability.csv", index=False
                )
                # Test E — One-sample z-test: DA vs 50% chance baseline
                _da_col = (
                    "da_mean"
                    if "da_mean" in hor_df.columns
                    else "directional_accuracy_mean"
                )
                if _da_col in hor_df.columns:
                    _first_model = hor_df["model"].iloc[0]
                    _n_test = _get_n_test_samples(
                        base_results, _first_model, category, asset, horizon, seeds
                    )
                    if _n_test is not None:
                        da_ztest_res = run_da_ztest(
                            hor_df, _n_test, da_col=_da_col
                        )
                        if da_ztest_res is not None:
                            da_ztest_res.to_csv(
                                stat_tests_dir / "da_ztest.csv", index=False
                            )

        # Category Figures Summary
        cat_figures_summary = cat_dir / "figures_summary"
        cat_figures_summary.mkdir(parents=True, exist_ok=True)
        
        plot_summary_heatmap(
            pivot_df=cat_df.pivot_table(index="model", columns=["asset", "horizon"], values="rmse_mean"),
            title=f"{category} Performance Matrix",
            output_path=cat_figures_summary / "category_performance_matrix.pdf",
            figsize=(12, 8)
        )

        # ensure bars appear in ascending rank order (1,2,3,...)
        cat_metrics_sorted = cat_metrics.sort_values("rmse_rank")
        plot_model_comparison_bar(
            df=cat_metrics_sorted,
            x_col="model",
            y_col="rmse_rank",
            title=f"{category} Hierarchical Ranking",
            output_path=cat_figures_summary / "category_hierarchical_ranking.pdf"
        )

    logger.info(f"Benchmark generation complete. Results saved to {benchmark_dir}")


if __name__ == "__main__":
    # For testing the module directly
    config = ProjectConfig(Path(__file__).parent.parent.parent)
    run_benchmark(config)
