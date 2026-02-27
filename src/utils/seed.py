"""
Reproducibility utilities.

Provides functions to set deterministic random seeds across
Python, NumPy, and PyTorch for experiment reproducibility.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
        deterministic: If True, enables PyTorch deterministic mode
            (may reduce performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Resolve and validate a torch device from a preference string.

    Supports: "cpu", "cuda", "cuda:<index>", "mps", and "auto".

    Behavior for "auto": prefer CUDA, then MPS, then CPU.

    Raises:
        ValueError: when a requested device is not available or the index is invalid.

    Returns:
        torch.device instance.
    """
    # allow passing a torch.device directly
    if isinstance(preference, torch.device):
        return preference

    pref = str(preference).lower()

    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if pref == "cpu":
        return torch.device("cpu")

    if pref == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        raise ValueError("MPS requested but not available on this machine")

    # handle cuda and cuda:<idx>
    if pref.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but CUDA is not available")
        # allow "cuda" or "cuda:<idx>"
        if pref == "cuda":
            return torch.device("cuda:0")
        try:
            _, idx = pref.split(":", 1)
            idx = int(idx)
        except Exception as exc:  # invalid format
            raise ValueError(f"Invalid CUDA device string: {preference}") from exc
        n_gpus = torch.cuda.device_count()
        if idx < 0 or idx >= n_gpus:
            raise ValueError(f"CUDA device index {idx} out of range (0..{n_gpus-1})")
        return torch.device(f"cuda:{idx}")

    raise ValueError(f"Unrecognized device preference: {preference}")


def seed_worker(worker_id: int) -> None:
    """Seed function for DataLoader workers to ensure reproducibility.

    Args:
        worker_id: Worker ID assigned by DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_rng_state() -> dict:
    """Capture current RNG states for checkpointing.

    Returns:
        Dictionary containing RNG states for Python, NumPy, and PyTorch.
    """
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng_state


def set_rng_state(rng_state: dict) -> None:
    """Restore RNG states from a checkpoint (robust + defensive).

    This function accepts RNG state entries stored as:
      - `torch.ByteTensor` (CPU or CUDA)
      - list / numpy array convertible to `torch.uint8`
    and normalizes them to the expected CPU `torch.uint8` ByteTensor(s)
    before calling `torch.set_rng_state` / `torch.cuda.set_rng_state_all`.

    Args:
        rng_state: Dictionary containing saved RNG states.

    Raises:
        TypeError: if `rng_state` has an unexpected structure.
    """
    if not isinstance(rng_state, dict):
        raise TypeError(f"rng_state must be a dict, got {type(rng_state)}")

    # Python and NumPy states - unchanged
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])

    # CPU torch RNG state: ensure a CPU ByteTensor (dtype=torch.uint8)
    if "torch" in rng_state:
        state = rng_state["torch"]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.uint8)
        else:
            if state.dtype != torch.uint8:
                state = state.to(dtype=torch.uint8)
            if state.device.type != "cpu":
                state = state.cpu()
        assert isinstance(state, torch.Tensor) and state.dtype == torch.uint8, (
            f"Expected CPU torch.uint8 tensor for CPU RNG state, got dtype={getattr(state, 'dtype', None)}"
        )
        torch.set_rng_state(state)

    # CUDA RNG states (list of ByteTensors) - normalize to CPU ByteTensors first
    if "torch_cuda" in rng_state and torch.cuda.is_available():
        cuda_states = rng_state["torch_cuda"]
        if not isinstance(cuda_states, (list, tuple)):
            raise TypeError("torch_cuda entry must be a list/tuple of RNG states")

        processed = []
        for s in cuda_states:
            if not isinstance(s, torch.Tensor):
                t = torch.as_tensor(s, dtype=torch.uint8)
            else:
                t = s
                if t.dtype != torch.uint8:
                    t = t.to(dtype=torch.uint8)
            # set_rng_state_all expects CPU ByteTensors; convert if needed
            if t.device.type != "cpu":
                t = t.cpu()
            processed.append(t)

        # If checkpoint was created on a machine with different GPU count, trim/pad safely
        expected = torch.cuda.device_count()
        if len(processed) != expected:
            import warnings

            warnings.warn(
                f"Loaded {len(processed)} CUDA RNG states but current device_count={expected}; "
                "will use available states up to the current GPU count."
            )
            processed = processed[:expected]

        # final validation
        for p in processed:
            assert isinstance(p, torch.Tensor) and p.dtype == torch.uint8, (
                "Each CUDA RNG state must be a CPU torch.uint8 tensor"
            )

        torch.cuda.set_rng_state_all(processed)
