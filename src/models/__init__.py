"""Models subpackage with automatic model discovery and registration."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List, Type

from src.models.base import BaseForecaster

# Model registry - maps model name to class
_MODEL_REGISTRY: Dict[str, Type[BaseForecaster]] = {}


def register_model(name: str):
    """Decorator to register a model class in the registry.

    Usage:
        @register_model("MyModel")
        class MyModelForecaster(BaseForecaster):
            ...

    Args:
        name: Model name used for lookup. Should match the config folder name.

    Returns:
        Decorator function.
    """
    def decorator(cls: Type[BaseForecaster]) -> Type[BaseForecaster]:
        if name in _MODEL_REGISTRY:
            # Allow re-registration (useful for module reloading)
            pass
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str) -> Type[BaseForecaster]:
    """Retrieve a model class by name from the registry.

    Args:
        name: Registered model name (e.g., 'LSTM', 'DLinear', 'TimesNet').

    Returns:
        Model class that can be instantiated.

    Raises:
        KeyError: If model name is not registered.
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Available models: {list_models()}. "
            f"Ensure the model file exists in src/models/ and uses @register_model decorator."
        )
    return _MODEL_REGISTRY[name]


def list_models() -> List[str]:
    """List all registered model names.

    Returns:
        Sorted list of model names available for training.
    """
    return sorted(_MODEL_REGISTRY.keys())


def _auto_discover_models() -> None:
    """Automatically discover and import all model modules in src/models/.

    This function scans the models directory for Python files (excluding
    base.py, __init__.py, and files starting with _), imports them, and
    triggers their @register_model decorators.

    New models are automatically discovered when placed in src/models/
    as long as they:
    1. Inherit from BaseForecaster
    2. Use the @register_model("ModelName") decorator
    3. Filename does not start with underscore (e.g., _template.py is skipped)
    """
    models_dir = Path(__file__).parent

    # Find all Python files in the models directory
    for module_info in pkgutil.iter_modules([str(models_dir)]):
        module_name = module_info.name

        # Skip base module, __init__, and private modules (starting with _)
        if module_name in ("base", "__init__") or module_name.startswith("_"):
            continue

        # Import the module to trigger registration
        try:
            importlib.import_module(f"src.models.{module_name}")
        except ImportError as e:
            # Log but don't fail - allows partial loading
            import warnings
            warnings.warn(
                f"Failed to import model module '{module_name}': {e}",
                ImportWarning,
            )


def get_model_info(name: str) -> Dict[str, any]:
    """Get information about a registered model.

    Args:
        name: Model name.

    Returns:
        Dictionary with model metadata.
    """
    model_cls = get_model_class(name)
    return {
        "name": name,
        "class": model_cls.__name__,
        "module": model_cls.__module__,
        "docstring": model_cls.__doc__,
    }


# Auto-discover all models on import
_auto_discover_models()


__all__ = [
    "BaseForecaster",
    "register_model",
    "get_model_class",
    "list_models",
    "get_model_info",
]
