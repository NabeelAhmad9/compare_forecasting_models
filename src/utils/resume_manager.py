"""
Resume state management using JSON-only tracking.

Handles training state persistence for both HPO trials and multi-seed final training.
Resume state is stored exclusively in models/ directory using training_state.json files.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResumeManager:
    """Manages training resume state using JSON files.
    
    Resume tracking is stored in models/ directory only:
    - HPO: models/hpo/<model>/<category>/<asset>/<horizon>/trial_XXX/training_state.json
    - Final: models/<model>/<category>/<asset>/<horizon>/<seed>/training_state.json
    
    Schema:
    {
        "model_name": str,
        "category": str,
        "asset": str,
        "horizon": int,
        "seed": int,  // Optional for HPO
        "epoch": int,  // Last completed epoch (0-indexed)
        "best_val_loss": float,
        "current_val_loss": float,
        "training_completed": bool,
        "timestamp": str,
        "hyperparameters": {}
    }
    """
    
    STATE_FILE = "training_state.json"
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        model_name: str,
        category: str,
        asset: str,
        horizon: int,
        hyperparameters: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize ResumeManager.
        
        Args:
            checkpoint_dir: Directory where checkpoints and state are stored.
            model_name: Name of the model.
            category: Asset category (crypto, forex, indices).
            asset: Asset name.
            horizon: Forecast horizon.
            hyperparameters: Model and training hyperparameters.
            seed: Random seed (optional for HPO, required for final training).
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.state_path = self.checkpoint_dir / self.STATE_FILE
        
        # Metadata
        self.model_name = model_name
        self.category = category
        self.asset = asset
        self.horizon = horizon
        self.seed = seed
        self.hyperparameters = hyperparameters or {}
        
        # Ensure directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load training state from JSON file.
        
        Returns:
            State dictionary if exists and valid, None otherwise.
        """
        if not self.state_path.exists():
            logger.info(f"No resume state found at {self.state_path}")
            return None
        
        try:
            with open(self.state_path, "r") as f:
                state = json.load(f)
            
            # Validate schema
            required_fields = [
                "model_name", "category", "asset", "horizon",
                "epoch", "best_val_loss", "current_val_loss",
                "training_completed", "timestamp", "hyperparameters"
            ]
            
            for field in required_fields:
                if field not in state:
                    logger.warning(f"Invalid state file: missing field '{field}'")
                    return None
            
            logger.info(
                f"Loaded resume state: epoch={state['epoch']}, "
                f"best_val_loss={state['best_val_loss']:.6f}, "
                f"completed={state['training_completed']}"
            )
            return state
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load resume state from {self.state_path}: {e}")
            return None
    
    def is_training_completed(self) -> bool:
        """Check if training is already completed.
        
        Returns:
            True if training_state.json indicates completion, False otherwise.
        """
        state = self.load_state()
        if state is None:
            return False
        return state.get("training_completed", False)
    
    def initialize_state(self) -> Dict[str, Any]:
        """Initialize new training state.
        
        Returns:
            Initial state dictionary.
        """
        state = {
            "model_name": self.model_name,
            "category": self.category,
            "asset": self.asset,
            "horizon": self.horizon,
            "epoch": 0,
            "best_val_loss": float("inf"),
            "current_val_loss": float("inf"),
            "training_completed": False,
            "timestamp": datetime.now().isoformat(),
            "hyperparameters": self.hyperparameters,
        }
        
        # Add seed if provided (mandatory for final training, optional for HPO)
        if self.seed is not None:
            state["seed"] = self.seed
        
        self.save_state(state)
        logger.info(f"Initialized new training state at {self.state_path}")
        return state
    
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save training state to JSON file atomically.
        
        Args:
            state: State dictionary to save.
        """
        # Update timestamp
        state["timestamp"] = datetime.now().isoformat()
        
        # Atomic write: write to temp file, then rename
        temp_path = self.state_path.with_suffix(".json.tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(state, f, indent=2)
            temp_path.replace(self.state_path)
        except Exception as e:
            logger.error(f"Failed to save training state to {self.state_path}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def update_epoch(
        self,
        epoch: int,
        current_val_loss: float,
        best_val_loss: float,
    ) -> None:
        """Update training state after an epoch.
        
        Args:
            epoch: Completed epoch number (0-indexed).
            current_val_loss: Validation loss for this epoch.
            best_val_loss: Best validation loss so far.
        """
        state = self.load_state()
        if state is None:
            # Initialize if missing (shouldn't happen)
            logger.warning("State file missing during update, reinitializing")
            state = self.initialize_state()
        
        state["epoch"] = epoch
        state["current_val_loss"] = current_val_loss
        state["best_val_loss"] = best_val_loss
        
        self.save_state(state)
    
    def mark_completed(self) -> None:
        """Mark training as completed."""
        state = self.load_state()
        if state is None:
            logger.warning("Cannot mark completed: state file not found")
            return
        
        state["training_completed"] = True
        self.save_state(state)
        logger.info(f"Training marked as completed at {self.state_path}")
    
    def get_resume_info(self) -> Optional[Dict[str, Any]]:
        """Get resume information for restarting training.
        
        Returns:
            Dictionary with epoch, best_val_loss, or None if starting fresh.
        """
        state = self.load_state()
        if state is None:
            return None
        
        if state.get("training_completed", False):
            logger.info("Training already completed, no need to resume")
            return {
                "training_completed": True,
                "epoch": state["epoch"],
                "best_val_loss": state["best_val_loss"],
            }
        
        return {
            "training_completed": False,
            "epoch": state["epoch"],
            "best_val_loss": state["best_val_loss"],
        }
