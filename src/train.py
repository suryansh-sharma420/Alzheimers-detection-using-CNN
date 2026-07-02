"""Training entry point.

Improvements over the original notebook:
  * subject-level split with correct per-split transforms (see :mod:`src.data`);
  * class-weighted loss to counter the heavy class imbalance;
  * best-on-validation checkpointing plus early stopping (not just the last epoch);
  * full seeding for reproducibility.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .data import get_data_loaders
from .model import build_model
from .utils import get_device, save_checkpoint, set_seed


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one epoch. Trains when ``optimizer`` is given, else evaluates."""
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / max(total, 1), 100.0 * correct / max(total, 1)


def train(config: Config) -> dict:
    """Train the model and return the training history."""
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")
    os.makedirs(config.output_dir, exist_ok=True)

    train_loader, val_loader, _, class_names, class_weights = get_data_loaders(config)
    print(f"Classes: {class_names}")
    print(f"Class weights: {class_weights.tolist()}")

    model = build_model(config.num_classes, config.dropout, device)
    weight = class_weights.to(device) if config.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.5
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(config.num_epochs):
        start = time.time()
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{config.num_epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
            f"{time.time() - start:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, config.checkpoint_path)
            print(f"  ✅ New best (val_loss={val_loss:.4f}) -> {config.checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(
                    f"  ⏹️  Early stopping after {epoch + 1} epochs "
                    f"(no improvement for {config.early_stopping_patience})."
                )
                break

    return history


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train the Alzheimer's CNN.")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    config = Config()
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.seed is not None:
        config.seed = args.seed
    if args.no_class_weights:
        config.use_class_weights = False
    return config


def main() -> None:
    train(_parse_args())


if __name__ == "__main__":
    main()
