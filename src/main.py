"""
CLI entry point for the forecasting comparison project.

Provides three main commands:
    - hpo:       Run hyperparameter optimization
    - final:     Run multi-seed final training
    - benchmark: Generate comparison leaderboards and plots

Usage:
    python src/main.py --mode hpo [--models MODEL1 MODEL2 ...]
    python src/main.py --mode final [--models MODEL1 MODEL2 ...]
    python src/main.py --mode benchmark [--models MODEL1 MODEL2 ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ProjectConfig
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Comparative Evaluation of Deep Learning Models "
                    "for Financial Time-Series Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["hpo", "final", "benchmark"],
        required=True,
        help="Execution mode: hpo, final, or benchmark.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to run (default: all available models). "
             "Example: --models LSTM DLinear TimesNet",
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=str(PROJECT_ROOT),
        help="Path to the project root directory.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (default: 42).",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()

    # Initialize configuration
    config = ProjectConfig(args.project_root)
    logger = get_logger(
        "main",
        log_file=config.get_path("logs_dir") / "system.log",
    )

    # Set global seed
    set_seed(args.seed)

    # List available models
    available_models = config.get_available_models()
    logger.info(f"Available models: {available_models}")

    # Validate model filter
    models_filter = args.models
    if models_filter:
        invalid = [m for m in models_filter if m not in available_models]
        if invalid:
            logger.error(f"Unknown models: {invalid}. Available: {available_models}")
            sys.exit(1)

    # Execute command
    if args.mode == "hpo":
        logger.info("=" * 60)
        logger.info("Starting Hyperparameter Optimization")
        logger.info("=" * 60)

        from src.experiments.hpo_runner import run_all_hpo
        run_all_hpo(config, models_filter=models_filter)

    elif args.mode == "final":
        logger.info("=" * 60)
        logger.info("Starting Multi-Seed Final Training")
        logger.info("=" * 60)

        from src.experiments.final_runner import run_all_final
        run_all_final(config, models_filter=models_filter)

    elif args.mode == "benchmark":
        logger.info("=" * 60)
        logger.info("Starting Benchmarking")
        logger.info("=" * 60)

        from src.experiments.benchmark import run_benchmark
        run_benchmark(config, models_filter=models_filter)

    logger.info("Done!")


if __name__ == "__main__":
    main()
