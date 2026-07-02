"""Configuration for the Alzheimer's detection pipeline.

Values can be overridden on the command line (see the ``argparse`` helpers in
:mod:`src.train` and :mod:`src.evaluate`) or via environment variables so the
code is portable across Kaggle, Colab, and local machines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _default_data_dir() -> str:
    """Resolve the dataset root.

    Priority: ``ALZ_DATA_DIR`` env var, then the Kaggle default, then a local
    ``data/`` directory relative to the repo root.
    """
    env = os.environ.get("ALZ_DATA_DIR")
    if env:
        return env
    kaggle_default = "/kaggle/input/imagesoasis/Data"
    if os.path.isdir(kaggle_default):
        return kaggle_default
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Data")


def _default_output_dir() -> str:
    env = os.environ.get("ALZ_OUTPUT_DIR")
    if env:
        return env
    kaggle_default = "/kaggle/working"
    if os.path.isdir(kaggle_default):
        return kaggle_default
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


@dataclass
class Config:
    """Hyperparameters and paths for training and evaluation."""

    # Paths
    data_dir: str = field(default_factory=_default_data_dir)
    output_dir: str = field(default_factory=_default_output_dir)
    checkpoint_name: str = "alzheimers_cnn_best.pth"

    # Data
    image_size: int = 256
    num_classes: int = 4
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 2

    # Patient/subject grouping. OASIS filenames look like
    # ``OAS1_0031_MR1_mpr-1_100.jpg``; this regex captures the subject id so
    # that all slices of one subject stay within a single split (no leakage).
    subject_id_regex: str = r"(OAS\d+_\d+)"

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 10
    weight_decay: float = 0.0
    dropout: float = 0.5
    use_class_weights: bool = True
    early_stopping_patience: int = 5

    # Reproducibility
    seed: int = 42

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.output_dir, self.checkpoint_name)
