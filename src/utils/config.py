"""
Configuration management utilities.

Handles loading, merging, and saving YAML configuration files for
the forecasting pipeline. Supports hierarchical config resolution.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save a dictionary to a YAML file.

    Args:
        data: Dictionary to save.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge multiple config dictionaries (later values override earlier).

    Args:
        *configs: Variable number of config dictionaries to merge.

    Returns:
        Merged configuration dictionary.
    """
    result: Dict[str, Any] = {}
    for cfg in configs:
        result = _deep_merge(result, cfg)
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class ProjectConfig:
    """Centralized project configuration manager.

    Loads and merges all configuration files, providing unified access
    to project settings.

    Attributes:
        project_root: Root directory of the project.
        base: Base configuration dictionary.
        dataset: Dataset configuration dictionary.
        training: Training configuration dictionary.
        hpo: HPO configuration dictionary.
        assets: Asset configuration dictionary.
    """

    def __init__(self, project_root: Union[str, Path]) -> None:
        """Initialize configuration from project root.

        Args:
            project_root: Path to the project root directory.
        """
        self.project_root = Path(project_root)
        self.configs_dir = self.project_root / "configs"

        # Load core configs
        self.base = load_yaml(self.configs_dir / "base.yaml")
        self.dataset = load_yaml(self.configs_dir / "dataset.yaml")
        self.training = load_yaml(self.configs_dir / "training.yaml")
        self.hpo = load_yaml(self.configs_dir / "hpo.yaml")
        self.assets = load_yaml(self.configs_dir / "asset.yaml")

    def get_path(self, key: str) -> Path:
        """Get an absolute path from the base config paths section.

        Args:
            key: Path key from base.yaml paths section.

        Returns:
            Absolute path.
        """
        return self.project_root / self.base["paths"][key]

    def get_model_search_space(self, model_name: str) -> Dict[str, Any]:
        """Load the search space config for a specific model.

        Args:
            model_name: Name of the model (e.g., 'TimesNet', 'LSTM').

        Returns:
            Search space configuration dictionary.
        """
        path = self.configs_dir / "models" / model_name / "search_space.yaml"
        return load_yaml(path)

    def get_model_best_config(
        self, model_name: str, category: Optional[str] = None, horizon: Optional[int] = None
    ) -> Dict[str, Any]:
        """Load the best hyperparameter config for a model.

        Args:
            model_name: Name of the model.
            category: Optional asset category for category-specific configs.
            horizon: Optional forecast horizon for horizon-specific configs.

        Returns:
            Best configuration dictionary.

        Raises:
            FileNotFoundError: If best_params.yaml does not exist yet.
        """
        if category and horizon:
            path = self.configs_dir / "models" / model_name / category / str(horizon) / "best_params.yaml"
        elif category:
            path = self.configs_dir / "models" / model_name / category / "best.yaml"
        else:
            path = self.configs_dir / "models" / model_name / "best.yaml"
        return load_yaml(path)

    def save_model_best_config(
        self,
        model_name: str,
        config: Dict[str, Any],
        category: Optional[str] = None,
        horizon: Optional[int] = None,
    ) -> Path:
        """Save best hyperparameters for a model after HPO.

        Args:
            model_name: Name of the model.
            config: Best hyperparameter dictionary.
            category: Optional asset category.
            horizon: Optional forecast horizon.

        Returns:
            Path where the config was saved.
        """
        if category and horizon:
            path = self.configs_dir / "models" / model_name / category / str(horizon) / "best_params.yaml"
        elif category:
            path = self.configs_dir / "models" / model_name / category / "best.yaml"
        else:
            path = self.configs_dir / "models" / model_name / "best.yaml"
        save_yaml(config, path)
        return path

    def get_categories(self) -> List[str]:
        """Get list of asset categories.

        Returns:
            List of category names.
        """
        return list(self.assets["categories"].keys())

    def get_assets_for_category(self, category: str) -> List[Dict[str, str]]:
        """Get the asset list for a given category.

        Args:
            category: Category name (e.g., 'crypto', 'forex', 'indices').

        Returns:
            List of asset dictionaries with 'name' and 'file' keys.
        """
        return self.assets["categories"][category]["assets"]

    def get_representative_asset(self, category: str) -> str:
        """Get the representative asset for HPO in a category.

        Args:
            category: Category name.

        Returns:
            Name of the representative asset.
        """
        return self.assets["categories"][category]["representative"]

    def get_horizons(self) -> List[int]:
        """Get list of forecasting horizons.

        Returns:
            List of horizon values.
        """
        return self.dataset["horizons"]

    def get_window_size(self, horizon: int) -> int:
        """Get the paired window size for a specific horizon.

        Pairing rule:
        - Use window_size[0] for horizons[0]
        - Use window_size[1] for horizons[1]

        Args:
            horizon: The forecasting horizon.

        Returns:
            The associated window size.

        Raises:
            ValueError: If the horizon is not in the config.
        """
        horizons = self.dataset["horizons"]
        window_sizes = self.dataset["window_size"]

        if horizon not in horizons:
            raise ValueError(
                f"Horizon {horizon} not found in dataset config (available: {horizons})"
            )

        idx = horizons.index(horizon)
        return window_sizes[idx]

    def get_eval_seeds(self) -> List[int]:
        """Get evaluation seeds for multi-seed training.

        Returns:
            List of seed values.
        """
        return self.training["eval_seeds"]

    def get_available_models(self) -> List[str]:
        """Discover available models that are both registered and have config.

        A model is available if:
        1. It has a search_space.yaml in configs/models/<name>/
        2. It is registered in the model registry (via @register_model)

        Returns:
            Sorted list of model names ready for training/HPO.
        """
        # Import here to avoid circular imports
        from src.models import list_models as registry_list_models

        # Get models from config directory
        models_dir = self.configs_dir / "models"
        config_models = set()
        if models_dir.exists():
            config_models = {
                d.name
                for d in models_dir.iterdir()
                if d.is_dir() and (d / "search_space.yaml").exists()
            }

        # Get models from registry
        registry_models = set(registry_list_models())

        # Return intersection - models that have both code and config
        available = config_models & registry_models

        # Warn about mismatches
        config_only = config_models - registry_models
        registry_only = registry_models - config_models

        if config_only:
            import warnings
            warnings.warn(
                f"Models with config but not registered: {config_only}. "
                f"Add @register_model decorator."
            )
        if registry_only:
            import warnings
            warnings.warn(
                f"Models registered but missing search_space.yaml: {registry_only}. "
                f"Add configs/models/<name>/search_space.yaml."
            )

        return sorted(available)
