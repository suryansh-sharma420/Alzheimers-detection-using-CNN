"""Empirical proof that the split fix removes patient leakage.

We build a synthetic dataset that mimics the key property of OASIS: each subject
contributes many near-identical slices, and a *label is assigned per subject*.
Crucially the label here is **arbitrary** (random per subject) with no feature a
model could generalize from — the only way to "predict" it is to recognize the
subject. So:

* The original notebook approach (image-level ``random_split`` + evaluating on
  the *entire* dataset) leaks subjects into the test set and reports high
  accuracy — pure memorization.
* The fixed approach (subject-level split + held-out test) reports ~chance,
  which is the honest result for an unlearnable label.

Run: ``python -m scripts.leakage_demo``
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.config import Config
from src.data import get_data_loaders
from src.model import build_model
from src.utils import set_seed

CLASSES = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
IMAGE_SIZE = 32
N_SUBJECTS = 48
SLICES_PER_SUBJECT = 12
EPOCHS = 4


def make_dataset(root: str, seed: int = 0) -> None:
    """Write an ImageFolder tree; each subject has a distinctive appearance."""
    rng = np.random.default_rng(seed)
    subject_labels = rng.integers(0, len(CLASSES), size=N_SUBJECTS)  # arbitrary
    for cls in CLASSES:
        os.makedirs(os.path.join(root, cls), exist_ok=True)

    for sid in range(N_SUBJECTS):
        # A low-frequency pattern unique to the subject (its "anatomy").
        base = rng.normal(0.5, 0.15, size=(4, 4))
        base = np.clip(base, 0, 1)
        cls = CLASSES[subject_labels[sid]]
        for sl in range(SLICES_PER_SUBJECT):
            up = np.kron(base, np.ones((IMAGE_SIZE // 4, IMAGE_SIZE // 4)))
            noise = rng.normal(0, 0.03, size=up.shape)
            img = np.clip(up + noise, 0, 1)
            arr = (img * 255).astype(np.uint8)
            Image.fromarray(arr, mode="L").save(
                os.path.join(root, cls, f"OAS1_{sid:04d}_MR1_mpr-1_{sl}.jpg")
            )


def _eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def _train(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(EPOCHS):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / max(total, 1)


def buggy_pipeline(data_dir: str, device: torch.device) -> float:
    """Replicates the original notebook: image-level split + eval on all data."""
    set_seed(42)
    full = ImageFolder(data_dir, transform=_eval_transform())
    n = len(full)
    val_n, test_n = int(n * 0.15), int(n * 0.15)
    train_ds, _, _ = random_split(
        full, [n - val_n - test_n, val_n, test_n],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = build_model(len(CLASSES), device=device)
    _train(model, train_loader, device)

    # The notebook's "test": a fresh loader over the ENTIRE dataset.
    whole_loader = DataLoader(full, batch_size=32, shuffle=False)
    return _accuracy(model, whole_loader, device)


def fixed_pipeline(data_dir: str, device: torch.device) -> float:
    """Uses src: subject-level split + held-out test only."""
    cfg = Config()
    cfg.data_dir = data_dir
    cfg.image_size = IMAGE_SIZE
    cfg.num_workers = 0
    cfg.num_epochs = EPOCHS
    cfg.use_class_weights = False
    set_seed(42)

    train_loader, _, test_loader, _, _ = get_data_loaders(cfg)
    model = build_model(len(CLASSES), device=device)
    _train(model, train_loader, device)
    return _accuracy(model, test_loader, device)


def main() -> None:
    device = torch.device("cpu")
    root = tempfile.mkdtemp()
    data_dir = os.path.join(root, "Data")
    make_dataset(data_dir)

    chance = 100.0 / len(CLASSES)
    buggy = buggy_pipeline(data_dir, device)
    fixed = fixed_pipeline(data_dir, device)

    print("\n" + "=" * 60)
    print("Leakage demonstration (label is arbitrary per subject)")
    print("=" * 60)
    print(f"Chance level                         : {chance:.1f}%")
    print(f"BUGGY (image split, eval on all data): {buggy:.1f}%  <- inflated")
    print(f"FIXED (subject split, held-out test) : {fixed:.1f}%  <- honest")
    print("=" * 60)


if __name__ == "__main__":
    main()
