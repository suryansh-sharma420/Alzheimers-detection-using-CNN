"""Shared fixtures: a tiny OASIS-style ImageFolder tree on disk."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

CLASSES = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]


@pytest.fixture
def oasis_tree(tmp_path):
    """Create <tmp>/Data/<class>/OAS1_<subject>_MR1_..._<slice>.jpg.

    Six subjects per class, five slices per subject. Returns the Data dir.
    """
    rng = np.random.default_rng(0)
    data_dir = tmp_path / "Data"
    sid = 0
    for cls in CLASSES:
        (data_dir / cls).mkdir(parents=True, exist_ok=True)
        for _ in range(6):
            sid += 1
            for sl in range(5):
                arr = rng.integers(0, 256, (32, 32), dtype=np.uint8)
                Image.fromarray(arr, mode="L").save(
                    data_dir / cls / f"OAS1_{sid:04d}_MR1_mpr-1_{sl}.jpg"
                )
    return str(data_dir)
