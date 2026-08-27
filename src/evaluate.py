"""Evaluation entry point.

Unlike the original notebook, evaluation runs **only on the held-out test
split** (subject-disjoint from training) and leads with macro-averaged metrics,
which are the honest signal under heavy class imbalance.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .config import Config
from .data import get_data_loaders
from .model import build_model
from .utils import get_device, load_checkpoint, set_seed


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module, loader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred, y_prob)`` over ``loader``."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for images, labels in loader:
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1)
        _, preds = torch.max(probs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def _plot_confusion_matrix(cm: np.ndarray, class_names, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Confusion Matrix (held-out test set)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved confusion matrix -> {path}")


def evaluate(config: Config) -> dict:
    """Load the best checkpoint and evaluate on the held-out test set."""
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
    )

    set_seed(config.seed)
    device = get_device()
    _, _, test_loader, class_names, _ = get_data_loaders(config)

    model = build_model(config.num_classes, config.dropout, device)
    load_checkpoint(model, config.checkpoint_path, device)

    y_true, y_pred, y_prob = collect_predictions(model, test_loader, device)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    _plot_confusion_matrix(
        cm, class_names, os.path.join(config.output_dir, "confusion_matrix.png")
    )

    print("\n📊 Classification report (held-out test set):\n")
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        digits=4, zero_division=0,
    )
    print(report)

    metrics: dict[str, float] = {"accuracy": float(np.mean(y_true == y_pred))}
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")

    try:
        y_true_oh = np.eye(len(class_names))[y_true]
        metrics["macro_auc_ovo"] = float(
            roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovo")
        )
        print(f"Macro AUC-ROC (OvO): {metrics['macro_auc_ovo']:.4f}")
    except ValueError as exc:
        print(f"⚠️  Could not compute AUC-ROC: {exc}")

    _binary_report(y_true, y_prob, class_names)
    return metrics


def _binary_report(y_true: np.ndarray, y_prob: np.ndarray, class_names) -> None:
    """Collapse to Alzheimer's (any dementia) vs. Non-Demented."""
    from sklearn.metrics import classification_report, roc_auc_score

    non_demented_idx = next(
        (i for i, n in enumerate(class_names) if "non" in n.lower()), None
    )
    if non_demented_idx is None:
        return

    alz_prob = 1.0 - y_prob[:, non_demented_idx]
    binary_true = (y_true != non_demented_idx).astype(int)
    binary_pred = (alz_prob >= 0.5).astype(int)

    print("\n📊 Binary report (Alzheimer's vs. Non-Demented):\n")
    print(
        classification_report(
            binary_true, binary_pred,
            target_names=["Non-Demented", "Alzheimer's"], zero_division=0,
        )
    )
    try:
        print(f"Binary ROC AUC: {roc_auc_score(binary_true, alz_prob):.4f}")
    except ValueError as exc:
        print(f"⚠️  Could not compute binary AUC: {exc}")


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Evaluate the Alzheimer's CNN.")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = Config()
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.checkpoint:
        config.checkpoint_name = os.path.basename(args.checkpoint)
        config.output_dir = os.path.dirname(args.checkpoint) or config.output_dir
    if args.seed is not None:
        config.seed = args.seed
    return config


def main() -> None:
    evaluate(_parse_args())


if __name__ == "__main__":
    main()
