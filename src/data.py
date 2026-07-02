"""Dataset loading, subject-level splitting, and data loaders.

Two correctness properties this module guarantees (both were broken in the
original notebook):

1. **No patient leakage.** OASIS provides many 2D slices per subject. Splitting
   at the image level lets slices of the same subject appear in both train and
   test, which massively inflates metrics. We split at the *subject* level.

2. **Per-split transforms.** ``random_split`` returns ``Subset`` objects that
   share a single underlying dataset, so mutating ``subset.dataset.transform``
   changes it for *every* split. We instead wrap each split in a dataset that
   owns its own transform, so training augmentation is applied only to the
   training split.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .config import Config


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Return ``(train_transform, eval_transform)``."""
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return train_transform, eval_transform


class TransformedSubset(Dataset):
    """A subset of ``base`` restricted to ``indices`` with its own transform."""

    def __init__(self, base: ImageFolder, indices: list[int], transform) -> None:
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        image, target = self.base[self.indices[i]]  # transform=None -> PIL image
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def extract_subject_ids(paths: list[str], regex: str) -> list[str]:
    """Map each file path to a subject id using ``regex``.

    Falls back to the full path (i.e. treats the image as its own group) when
    the pattern does not match, which degrades gracefully to an image-level
    split rather than crashing.
    """
    pattern = re.compile(regex)
    groups: list[str] = []
    unmatched = 0
    for path in paths:
        match = pattern.search(path)
        if match:
            groups.append(match.group(1) if match.groups() else match.group(0))
        else:
            groups.append(path)
            unmatched += 1
    if unmatched:
        print(
            f"⚠️  subject_id_regex matched none of {unmatched}/{len(paths)} files; "
            "those images fall back to an image-level split."
        )
    return groups


def _group_split(
    groups: list[str],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train/val/test with disjoint groups."""
    n = len(groups)
    indices = np.arange(n)
    groups_arr = np.asarray(groups)

    # First carve out the test set.
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(indices, groups=groups_arr))

    # Then split the remainder into train/val (rescale val fraction).
    val_fraction = val_split / (1.0 - test_split)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    rel_train_idx, rel_val_idx = next(
        gss_val.split(train_val_idx, groups=groups_arr[train_val_idx])
    )
    train_idx = train_val_idx[rel_train_idx]
    val_idx = train_val_idx[rel_val_idx]
    return train_idx, val_idx, test_idx


def compute_class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean 1."""
    counts = Counter(targets)
    freqs = np.array([counts.get(c, 0) for c in range(num_classes)], dtype=np.float64)
    freqs[freqs == 0] = 1.0  # avoid division by zero for absent classes
    weights = 1.0 / freqs
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def get_datasets(
    config: Config,
) -> tuple[TransformedSubset, TransformedSubset, TransformedSubset, list[str], list[int]]:
    """Build subject-disjoint train/val/test datasets.

    Returns ``(train_ds, val_ds, test_ds, class_names, train_targets)``.
    """
    # transform=None so the base returns PIL images; each split adds its own.
    base = ImageFolder(root=config.data_dir, transform=None)
    paths = [s for s, _ in base.samples]
    targets_all = [t for _, t in base.samples]

    groups = extract_subject_ids(paths, config.subject_id_regex)
    train_idx, val_idx, test_idx = _group_split(
        groups, config.val_split, config.test_split, config.seed
    )

    train_tf, eval_tf = build_transforms(config.image_size)
    train_ds = TransformedSubset(base, list(train_idx), train_tf)
    val_ds = TransformedSubset(base, list(val_idx), eval_tf)
    test_ds = TransformedSubset(base, list(test_idx), eval_tf)

    train_targets = [targets_all[i] for i in train_idx]

    _assert_disjoint_subjects(groups, train_idx, val_idx, test_idx)
    return train_ds, val_ds, test_ds, base.classes, train_targets


def _assert_disjoint_subjects(
    groups: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    groups_arr = np.asarray(groups)
    train_g = set(groups_arr[train_idx])
    val_g = set(groups_arr[val_idx])
    test_g = set(groups_arr[test_idx])
    assert not (train_g & test_g), "subject leakage between train and test"
    assert not (train_g & val_g), "subject leakage between train and val"
    assert not (val_g & test_g), "subject leakage between val and test"


def get_data_loaders(
    config: Config,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], torch.Tensor]:
    """Build train/val/test loaders plus class names and class weights."""
    train_ds, val_ds, test_ds, class_names, train_targets = get_datasets(config)
    class_weights = compute_class_weights(train_targets, config.num_classes)

    common = dict(num_workers=config.num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, **common
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, **common
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, **common
    )
    return train_loader, val_loader, test_loader, class_names, class_weights
