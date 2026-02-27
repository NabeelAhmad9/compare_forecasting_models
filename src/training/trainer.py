"""
Model trainer with checkpoint resume support.

Handles the full training loop including forward/backward passes,
validation, checkpointing, early stopping, and learning rate scheduling.
Optimized for maximum throughput on CPU, CUDA, and MPS devices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import gc
import os
import platform

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.models.base import BaseForecaster
from src.training.callbacks import EarlyStopping
from src.utils.logger import get_logger
from src.utils.resume_manager import ResumeManager
from src.utils.seed import seed_worker, get_rng_state, set_rng_state

logger = get_logger(__name__)


class Trainer:
    """Unified, hardware-adaptive trainer optimized for throughput.

    Key features:
      - Automatic device adaptation (CPU / CUDA / MPS)
      - TF32 acceleration on Ampere+ GPUs
      - Mixed precision (AMP) with GradScaler on CUDA
      - Optional torch.compile() for PyTorch 2.0+
      - DataParallel / DistributedDataParallel support
      - Optimized DataLoader (pin_memory, persistent_workers, prefetch)
      - Gradient accumulation and optional gradient clipping
      - Robust checkpoint save/load across CPU/GPU and DP/DDP

    The Trainer adapts automatically to the provided device and configuration
    while preserving checkpoint semantics and backward compatibility.
    """

    def __init__(
        self,
        model: BaseForecaster,
        device: torch.device,
        config: Dict[str, Any],
        seed: int = 42,
    ) -> None:
        self.config = config
        self.seed = seed
        self.device = device

        # --- Hardware tuning (TF32, cuDNN benchmark, deterministic) ---
        self.deterministic = config.get("deterministic", False)
        self._configure_hardware()

        # Move model to device
        model = model.to(self.device)

        # --- Multi-GPU / Distributed setup ---
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        self.using_dp = False
        self.using_ddp = False

        if self.device.type == "cuda" and self.n_gpus > 1 and not self.is_distributed:
            model = nn.DataParallel(model)
            self.using_dp = True

        if self.is_distributed and self.world_size > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(
                    model, device_ids=[self.device.index or local_rank]
                )
            else:
                model = nn.parallel.DistributedDataParallel(model)
            self.using_ddp = True

        # --- Optional torch.compile (PyTorch 2.0+) ---
        self.use_compile = config.get("use_compile", False)
        if self.use_compile:
            if hasattr(torch, "compile"):
                logger.info("Compiling model with torch.compile()")
                model = torch.compile(model)
            else:
                logger.warning(
                    "use_compile=True but torch.compile not available (requires PyTorch >= 2.0)"
                )
                self.use_compile = False

        self.model = model

        # --- Training hyperparameters ---
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 64)
        self.lr = config.get("learning_rate", 0.001)
        self.weight_decay = config.get("weight_decay", 0.0001)

        # --- DataLoader config ---
        default_workers = min(8, os.cpu_count() or 1)
        self.num_workers = config.get("num_workers", default_workers)
        self.pin_memory = config.get("pin_memory", True) and self.device.type == "cuda"
        self.prefetch_factor = config.get("prefetch_factor", 2)
        # Disable persistent_workers on Windows due to multiprocessing cleanup issues
        # See: https://github.com/pytorch/pytorch/issues/60583
        is_windows = platform.system() == "Windows"
        self.persistent_workers = (
            config.get("persistent_workers", True) 
            and self.num_workers > 0 
            and not is_windows
        )

        # --- Gradient accumulation and clipping ---
        self.grad_accum_steps = max(1, config.get("grad_accum_steps", 1))
        self.gradient_clip_norm = config.get("grad_clip", config.get("gradient_clip_norm", None))

        # --- AMP (mixed precision) - only effective on CUDA ---
        self.use_amp = config.get("use_amp", False) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        # Autocast device_type ('cuda' and 'cpu' are universally supported)
        self._autocast_dtype = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"

        # Loss
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self._get_model_parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Scheduler
        sched_config = config.get("scheduler", {})
        sched_type = sched_config.get("type", "reduce_on_plateau")
        if sched_type == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=sched_config.get("patience", 5),
                factor=sched_config.get("factor", 0.5),
                min_lr=sched_config.get("min_lr", 1e-6),
            )
        elif sched_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        elif sched_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get("step_size", 10),
                gamma=sched_config.get("factor", 0.5),
            )
        else:
            self.scheduler = None

        # Early stopping
        es_config = config.get("early_stopping", {})
        self.early_stopping = None
        if es_config.get("enabled", True):
            self.early_stopping = EarlyStopping(
                patience=es_config.get("patience", 10),
                min_delta=es_config.get("min_delta", 0.0001),
            )

        # Tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float("inf")
        self.start_epoch = 0

        # Log environment summary
        logger.info(
            f"Trainer initialized: device={self.device} | amp={self.use_amp} | "
            f"tf32={self.device.type == 'cuda' and torch.backends.cuda.matmul.allow_tf32} | "
            f"compile={self.use_compile} | dp={self.using_dp} | ddp={self.using_ddp} | "
            f"n_gpus={self.n_gpus} | workers={self.num_workers} | "
            f"pin_memory={self.pin_memory} | persistent_workers={self.persistent_workers} | "
            f"grad_accum={self.grad_accum_steps} | batch_size={self.batch_size}"
        )

    # ------------------------------------------------------------------
    # Hardware & model helpers
    # ------------------------------------------------------------------

    def _configure_hardware(self) -> None:
        """Configure hardware backends for optimal throughput.

        Sets TF32, cuDNN benchmark, and deterministic mode based on config.
        Must be called before model.to(device).
        """
        if self.device.type == "cuda":
            if self.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            else:
                # Maximum throughput: cuDNN auto-tuner + TF32
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                if self.config.get("use_tf32", True):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

    @property
    def _base_model(self) -> BaseForecaster:
        """Unwrap DataParallel / DDP / torch.compile wrappers to get the base model."""
        m = self.model
        if hasattr(m, "_orig_mod"):  # torch.compile wrapper
            m = m._orig_mod
        if hasattr(m, "module"):  # DataParallel / DDP
            m = m.module
        return m

    def _get_model_parameters(self):
        """Get trainable parameters, unwrapping wrappers if needed."""
        return self._base_model.parameters()

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    def _create_dataloader(
        self,
        x: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> DataLoader:
        """Create a DataLoader optimized for the current device.

        - CPU tensors with pin_memory for async CUDA transfers
        - On-device tensors for CPU (avoids transfer overhead)
        - DistributedSampler for DDP mode
        - persistent_workers to avoid per-epoch fork overhead
        """
        if self.device.type == "cpu":
            x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        else:
            # Keep on CPU for pin_memory -> non_blocking CUDA transfer
            x_tensor = torch.as_tensor(x, dtype=torch.float32)
            y_tensor = torch.as_tensor(y, dtype=torch.float32)

        dataset = TensorDataset(x_tensor, y_tensor)

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        sampler = None
        if self.is_distributed and shuffle:
            sampler = DistributedSampler(dataset, shuffle=shuffle)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=drop_last,
        )

        return dataloader

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch with gradient accumulation and optional AMP.

        Optimized for throughput: no GPU sync or cache clearing inside the loop.
        Uses torch.autocast for device-agnostic mixed precision.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad(set_to_none=True)

        for step, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self._autocast_dtype, enabled=self.use_amp):
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            # Optimizer step at accumulation boundary
            if (step + 1) % self.grad_accum_steps == 0:
                if self.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._get_model_parameters(),
                        max_norm=self.gradient_clip_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.grad_accum_steps
            n_batches += 1

        # Flush remaining accumulated gradients from incomplete window
        if self.grad_accum_steps > 1 and n_batches % self.grad_accum_steps != 0:
            if self.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._get_model_parameters(),
                    max_norm=self.gradient_clip_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Run validation loop (no grad, optional AMP for consistent numerics)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self._autocast_dtype, enabled=self.use_amp):
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _cleanup_resources(self, train_loader=None, val_loader=None, epoch_pbar=None) -> None:
        """Cleanup all training resources to prevent hangs.
        
        Ensures DataLoader workers, tqdm threads, and other resources are properly
        terminated before function exit. Critical on Windows where worker processes
        may not terminate automatically.
        
        Args:
            train_loader: Training DataLoader to cleanup (optional).
            val_loader: Validation DataLoader to cleanup (optional).
            epoch_pbar: tqdm progress bar to cleanup (optional).
        """
        # Close tqdm progress bar first
        if epoch_pbar is not None:
            try:
                epoch_pbar.close()
                logger.debug("Closed tqdm progress bar")
            except Exception as e:
                logger.warning(f"Error closing progress bar: {e}")
        
        # Delete DataLoader references
        # This triggers __del__ which should shutdown workers
        loaders_deleted = False
        if train_loader is not None:
            try:
                del train_loader
                loaders_deleted = True
            except Exception as e:
                logger.warning(f"Error deleting train_loader: {e}")
        
        if val_loader is not None:
            try:
                del val_loader
                loaders_deleted = True
            except Exception as e:
                logger.warning(f"Error deleting val_loader: {e}")
        
        # Force garbage collection if we deleted any loaders
        # This is critical on Windows to ensure multiprocessing workers terminate
        if loaders_deleted:
            gc.collect()
            logger.debug("Forced garbage collection for DataLoader cleanup")

    def fit(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        resume: bool = True,
        model_name: Optional[str] = None,
        category: Optional[str] = None,
        asset: Optional[str] = None,
        horizon: Optional[int] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Full training loop with resume support and hardware optimizations.
        
        Implements robust resource cleanup to prevent hangs on all platforms,
        especially Windows where multiprocessing workers may not terminate properly.
        """
        # Initialize tracking variables for cleanup
        train_loader = None
        val_loader = None
        epoch_pbar = None
        
        try:
            # Initialize ResumeManager if checkpoint_dir provided
            resume_manager = None
            if checkpoint_dir and resume:
                if not all([model_name, category, asset, horizon is not None]):
                    logger.warning(
                        "Resume tracking requires model_name, category, asset, and horizon. "
                        "Falling back to checkpoint-only resume."
                    )
                else:
                    resume_manager = ResumeManager(
                        checkpoint_dir=checkpoint_dir,
                        model_name=model_name,
                        category=category,
                        asset=asset,
                        horizon=horizon,
                        hyperparameters=hyperparameters,
                        seed=self.seed,
                    )

                    # Check if training already completed
                    if resume_manager.is_training_completed():
                        logger.info("Training already completed according to ResumeManager, skipping...")
                        result = self._load_completed_results(Path(checkpoint_dir))
                        logger.info("Returning completed results (early exit path 1)")
                        return result

            # Resume from checkpoint if available (sets self.start_epoch)
            if checkpoint_dir and resume:
                logger.info("Checking for checkpoints to resume...")
                self._try_resume(checkpoint_dir, resume_manager)

            # Early exit if we already reached target epochs
            if self.start_epoch >= self.epochs:
                logger.info(
                    f"Training already completed (epoch {self.start_epoch}/{self.epochs}), "
                    "loading results..."
                )
                if resume_manager:
                    resume_manager.mark_completed()
                result = self._load_completed_results(Path(checkpoint_dir))
                logger.info("Returning completed results (early exit path 2)")
                return result

            # Create data loaders ONLY if we are actually going to train
            logger.info(f"Creating DataLoaders for epoch {self.start_epoch} to {self.epochs}...")
            train_loader = self._create_dataloader(train_x, train_y, shuffle=True, drop_last=True)
            val_loader = self._create_dataloader(val_x, val_y, shuffle=False, drop_last=False)

            # log summary
            n_params = self._base_model.count_parameters()
            logger.info(
                f"Training {self._base_model.model_name}: "
                f"epochs={self.epochs}, batch_size={self.batch_size}, lr={self.lr}, "
                f"params={n_params:,}"
            )

            # Epoch loop
            epoch_pbar = tqdm(
                range(self.start_epoch, self.epochs),
                initial=self.start_epoch,
                total=self.epochs,
                desc=f"Training {self._base_model.model_name}",
                unit="epoch",
                leave=True,
            )

            for epoch in epoch_pbar:
                # if using distributed sampler, set epoch for shuffling
                if self.is_distributed and hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)

                train_loss = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)

                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)

                # Scheduler step
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]["lr"]

                epoch_pbar.set_postfix({
                    "train": f"{train_loss:.4f}",
                    "val": f"{val_loss:.4f}",
                    "best": f"{self.best_val_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, lr: {current_lr:.2e}"
                )

                # Save best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    if checkpoint_dir:
                        self._save_checkpoint(
                            Path(checkpoint_dir) / "model_best.pt",
                            epoch=epoch + 1,
                            is_best=True,
                        )

                # Save last checkpoint
                if checkpoint_dir:
                    self._save_checkpoint(
                        Path(checkpoint_dir) / "model_last.pt",
                        epoch=epoch + 1,
                        is_best=False,
                    )

                if resume_manager:
                    resume_manager.update_epoch(
                        epoch=epoch + 1,
                        current_val_loss=val_loss,
                        best_val_loss=self.best_val_loss,
                    )

                if self.early_stopping is not None and self.early_stopping(val_loss):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    epoch_pbar.set_description(f"Early stopped at epoch {epoch + 1}")
                    break

            # Training loop completed - cleanup resources immediately
            logger.info("Training loop completed, cleaning up resources...")
            self._cleanup_resources(train_loader, val_loader, epoch_pbar)
            
            # Clear references to cleaned resources
            train_loader = None
            val_loader = None
            epoch_pbar = None
            
            # Mark training as completed
            if resume_manager:
                resume_manager.mark_completed()
                logger.info("Marked training as completed in ResumeManager")

            # Load best model (map to correct device)
            if checkpoint_dir and (Path(checkpoint_dir) / "model_best.pt").exists():
                logger.info("Loading best model checkpoint...")
                self._base_model.load_checkpoint(
                    Path(checkpoint_dir) / "model_best.pt",
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    device=self.device,
                )
                logger.info("Successfully loaded best model checkpoint")

            # Prepare results
            result = {
                "best_val_loss": self.best_val_loss,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "epochs_trained": len(self.train_losses),
                "n_parameters": self._base_model.count_parameters(),
            }
            
            logger.info(
                f"Training completed successfully: "
                f"best_val_loss={self.best_val_loss:.6f}, "
                f"epochs_trained={len(self.train_losses)}"
            )
            logger.info("Returning from fit() - normal completion path")
            return result
            
        except Exception as e:
            # Ensure cleanup happens even on error
            logger.error(f"Error during training: {e}", exc_info=True)
            self._cleanup_resources(train_loader, val_loader, epoch_pbar)
            raise
        finally:
            # Final safety cleanup in case anything was missed
            # Use try-except to prevent errors in cleanup from masking original errors
            try:
                if train_loader is not None or val_loader is not None or epoch_pbar is not None:
                    logger.warning("Final cleanup catching unreleased resources")
                    self._cleanup_resources(train_loader, val_loader, epoch_pbar)
            except Exception as cleanup_error:
                logger.error(f"Error in final cleanup: {cleanup_error}")

    def _save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        is_best: bool = False,
    ) -> None:
        """Save a complete training checkpoint with all state.

        Args:
            path: Path to save the checkpoint.
            epoch: Current epoch number (1-indexed, completed epoch).
            is_best: Whether this is the best model checkpoint.
        """
        # Capture all training state for full resume
        extra = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "rng_state": get_rng_state(),
            "scaler_state": self.scaler.state_dict() if self.use_amp else None,
            "early_stopping_state": {
                "counter": self.early_stopping.counter,
                "best_loss": self.early_stopping.best_loss,
                "should_stop": self.early_stopping.should_stop,
            } if self.early_stopping else None,
            "seed": self.seed,
            "is_best": is_best,
        }

        # save using the unwrapped base model
        self._base_model.save_checkpoint(
            path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            best_val_loss=self.best_val_loss,
            extra=extra,
        )

    def _try_resume(
        self,
        checkpoint_dir: Union[str, Path],
        resume_manager: Optional[ResumeManager] = None,
    ) -> bool:
        """Attempt to resume from the last checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoints.
            resume_manager: Optional ResumeManager for state tracking.

        Returns:
            True if resumed from checkpoint, False otherwise.
        """
        # Check resume state first
        if resume_manager:
            resume_info = resume_manager.get_resume_info()
            if resume_info and resume_info.get("training_completed"):
                logger.info("Training already completed, no need to resume")
                return False
            
            # Initialize state if not exists
            if resume_info is None:
                resume_manager.initialize_state()

        last_path = Path(checkpoint_dir) / "model_last.pt"
        if last_path.exists():
            logger.info(f"Resuming from checkpoint: {last_path}")
            info = self._base_model.load_checkpoint(
                last_path,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )
            self.start_epoch = info["epoch"]
            self.best_val_loss = info["best_val_loss"]

            # Restore extra state
            extra = info.get("extra", {})
            if extra:
                # Restore training history
                self.train_losses = extra.get("train_losses", [])
                self.val_losses = extra.get("val_losses", [])

                # Restore RNG state for reproducibility
                rng_state = extra.get("rng_state")
                if rng_state:
                    set_rng_state(rng_state)
                    logger.info("Restored RNG states")

                # Restore GradScaler state for AMP
                scaler_state = extra.get("scaler_state")
                if scaler_state and self.use_amp:
                    self.scaler.load_state_dict(scaler_state)
                    logger.info("Restored GradScaler state")

                # Restore early stopping state
                es_state = extra.get("early_stopping_state")
                if es_state and self.early_stopping:
                    self.early_stopping.counter = es_state.get("counter", 0)
                    self.early_stopping.best_loss = es_state.get("best_loss")
                    self.early_stopping.should_stop = es_state.get("should_stop", False)
                    logger.info(
                        f"Restored EarlyStopping state: counter={self.early_stopping.counter}, "
                        f"best_loss={self.early_stopping.best_loss}"
                    )

            logger.info(
                f"Resumed at epoch {self.start_epoch}/{self.epochs}, "
                f"best_val_loss={self.best_val_loss:.6f}, "
                f"history_len={len(self.train_losses)}"
            )
            return True
        else:
            logger.info("No checkpoint found, starting fresh")
            # Initialize state if resume_manager provided
            if resume_manager:
                resume_manager.initialize_state()
            return False
def _load_completed_results(self, checkpoint_dir: Path) -> Dict[str, Any]:
    """Load results from completed training.

    Args:
        checkpoint_dir: Directory containing checkpoints.

    Returns:
        Dictionary with training results.
    """
    logger.info(f"Loading completed results from {checkpoint_dir}")

    # Load best model checkpoint
    best_path = checkpoint_dir / "model_best.pt"
    if best_path.exists():
        logger.info(f"Loading best checkpoint from {best_path}")
        info = self._base_model.load_checkpoint(
            best_path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        logger.info("Best checkpoint loaded successfully")

        extra = info.get("extra", {})
        n_params = self._base_model.count_parameters()

        result = {
            "best_val_loss": info.get("best_val_loss", float("inf")),
            "train_losses": extra.get("train_losses", []),
            "val_losses": extra.get("val_losses", []),
            "epochs_trained": len(extra.get("train_losses", [])),
            "n_parameters": n_params,
        }

        logger.info(
            f"Loaded results: best_val_loss={result['best_val_loss']:.6f}, "
            f"epochs_trained={result['epochs_trained']}, "
            f"n_parameters={n_params:,}"
        )

        return result

    # Fallback: return minimal info
    logger.warning(
        f"Best checkpoint not found at {best_path}, returning minimal info"
    )

    n_params = self._base_model.count_parameters()

    return {
        "best_val_loss": float("inf"),
        "train_losses": [],
        "val_losses": [],
        "epochs_trained": 0,
        "n_parameters": n_params,
    }


@torch.no_grad()
def predict(
    self,
    x: np.ndarray,
) -> np.ndarray:
    """Generate predictions in batches.

    Keeps tensors on CPU and transfers per-batch for CUDA to limit peak memory.
    No GPU sync or cache clearing inside the loop.
    """
    self.model.eval()

    if self.device.type == "cpu":
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
    else:
        x_tensor = torch.as_tensor(x, dtype=torch.float32)

    predictions = []

    for i in range(0, len(x_tensor), self.batch_size):
        batch = x_tensor[i : i + self.batch_size]

        if self.device.type != "cpu":
            batch = batch.to(self.device, non_blocking=True)

        with torch.autocast(
            device_type=self._autocast_dtype,
            enabled=self.use_amp,
        ):
            pred = self.model(batch)

        predictions.append(
            pred.float().detach().cpu().numpy()
        )

    return np.concatenate(predictions, axis=0)
