"""Integration tests for MDFAN classifier alignment wiring."""

import torch

from src.models import create_model


def test_forward_train_returns_align_loss_when_enabled():
    model = create_model(
        "mdfan",
        backbone="mobilenet_v3_small",
        num_classes=5,
        num_sources=2,
        pretrained=False,
    )

    batch_size = 2
    source_images = [
        torch.randn(batch_size, 3, 224, 224),
        torch.randn(batch_size, 3, 224, 224),
    ]
    source_labels = [
        torch.zeros(batch_size, dtype=torch.long),
        torch.zeros(batch_size, dtype=torch.long),
    ]
    target_images = torch.randn(batch_size, 3, 224, 224)

    outputs = model.forward_train(
        source_images,
        source_labels,
        target_images,
        compute_alignment_loss=True,
    )

    assert "align_loss" in outputs
    loss = outputs["align_loss"]
    assert loss.shape == ()
    assert torch.isfinite(loss)

    loss.backward()
    grads = [
        p.grad for p in model.source_classifiers[0].parameters() if p.requires_grad
    ]
    assert any(g is not None for g in grads)


def test_forward_train_align_loss_is_zero_when_disabled():
    model = create_model(
        "mdfan",
        backbone="mobilenet_v3_small",
        num_classes=5,
        num_sources=2,
        pretrained=False,
    )

    batch_size = 2
    source_images = [
        torch.randn(batch_size, 3, 224, 224),
        torch.randn(batch_size, 3, 224, 224),
    ]
    source_labels = [
        torch.zeros(batch_size, dtype=torch.long),
        torch.zeros(batch_size, dtype=torch.long),
    ]
    target_images = torch.randn(batch_size, 3, 224, 224)

    outputs = model.forward_train(
        source_images,
        source_labels,
        target_images,
        compute_alignment_loss=False,
    )

    loss = outputs["align_loss"]
    assert torch.isclose(loss, loss.new_tensor(0.0), atol=0.0)
