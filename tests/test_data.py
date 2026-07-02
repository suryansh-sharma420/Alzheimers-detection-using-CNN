"""Tests that lock in the three bug fixes in src.data."""

from __future__ import annotations

import numpy as np

from src.config import Config
from src.data import (
    build_transforms,
    compute_class_weights,
    extract_subject_ids,
    get_datasets,
)


def _cfg(data_dir: str) -> Config:
    cfg = Config()
    cfg.data_dir = data_dir
    cfg.image_size = 32
    cfg.num_workers = 0
    return cfg


def test_subjects_are_disjoint_across_splits(oasis_tree):
    """Bug 1: no subject may appear in more than one split."""
    cfg = _cfg(oasis_tree)
    train_ds, val_ds, test_ds, _, _ = get_datasets(cfg)

    def subjects(ds):
        paths = [ds.base.samples[i][0] for i in ds.indices]
        return set(extract_subject_ids(paths, cfg.subject_id_regex))

    tr, va, te = subjects(train_ds), subjects(val_ds), subjects(test_ds)
    assert tr and va and te
    assert tr.isdisjoint(te)
    assert tr.isdisjoint(va)
    assert va.isdisjoint(te)


def test_splits_cover_all_images(oasis_tree):
    cfg = _cfg(oasis_tree)
    train_ds, val_ds, test_ds, _, _ = get_datasets(cfg)
    total = len(train_ds) + len(val_ds) + len(test_ds)
    assert total == len(train_ds.base.samples)
    all_idx = set(train_ds.indices) | set(val_ds.indices) | set(test_ds.indices)
    assert len(all_idx) == total  # no index reused


def test_per_split_transforms_are_independent(oasis_tree):
    """Bug 3: eval splits must NOT carry train-time augmentation."""
    cfg = _cfg(oasis_tree)
    train_ds, val_ds, test_ds, _, _ = get_datasets(cfg)

    train_tf, eval_tf = build_transforms(cfg.image_size)
    train_names = [type(t).__name__ for t in train_ds.transform.transforms]
    val_names = [type(t).__name__ for t in val_ds.transform.transforms]

    assert "RandomHorizontalFlip" in train_names
    assert "RandomRotation" in train_names
    assert "RandomHorizontalFlip" not in val_names
    assert "RandomRotation" not in val_names
    assert val_ds.transform is test_ds.transform or val_names == [
        type(t).__name__ for t in test_ds.transform.transforms
    ]
    # base dataset itself holds no transform (returns PIL for the wrappers)
    assert train_ds.base.transform is None


def test_eval_transform_is_deterministic(oasis_tree):
    cfg = _cfg(oasis_tree)
    _, val_ds, _, _, _ = get_datasets(cfg)
    a, _ = val_ds[0]
    b, _ = val_ds[0]
    assert np.allclose(a.numpy(), b.numpy())


def test_class_weights_favor_rare_classes():
    targets = [0] * 90 + [1] * 10  # class 1 is rare
    weights = compute_class_weights(targets, num_classes=2)
    assert weights[1] > weights[0]


def test_unmatched_regex_falls_back_to_path():
    paths = ["/a/b/random_name.jpg", "/a/b/OAS1_0001_MR1_1.jpg"]
    groups = extract_subject_ids(paths, Config().subject_id_regex)
    assert groups[0] == paths[0]  # fallback to full path
    assert groups[1] == "OAS1_0001"
