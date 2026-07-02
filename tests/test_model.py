"""Smoke tests for the model."""

from __future__ import annotations

import torch

from src.model import SimpleCNN, build_model


def test_forward_shape():
    model = SimpleCNN(num_classes=4)
    out = model(torch.randn(2, 1, 64, 64))
    assert out.shape == (2, 4)


def test_adaptive_pool_handles_variable_input_size():
    model = SimpleCNN(num_classes=4)
    for size in (32, 64, 128):
        out = model(torch.randn(1, 1, size, size))
        assert out.shape == (1, 4)


def test_build_model_on_cpu():
    model = build_model(num_classes=4, device=torch.device("cpu"))
    out = model(torch.randn(1, 1, 32, 32))
    assert out.shape == (1, 4)
