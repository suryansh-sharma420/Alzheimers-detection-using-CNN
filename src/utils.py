"""Shared helpers: seeding, device selection, checkpoint I/O."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module, unwrapping ``DataParallel`` if present."""
    return model.module if isinstance(model, nn.DataParallel) else model


def save_checkpoint(model: nn.Module, path: str) -> None:
    torch.save(unwrap(model).state_dict(), path)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """Load a state dict saved by :func:`save_checkpoint` into ``model``."""
    state = torch.load(path)
    unwrap(model).load_state_dict(state, strict=False)
    return model
