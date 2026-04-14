"""Unit tests for classifier alignment (MDFAN Stage 2)."""

import torch

from src.models.mdfan import ClassifierAlignment


def test_zero_loss_for_identical_predictions():
    criterion = ClassifierAlignment(loss_type="l1")

    logits = torch.randn(8, 5)
    probs = torch.softmax(logits, dim=1)

    loss = criterion([probs, probs.clone()])
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)


def test_positive_loss_for_different_predictions():
    criterion = ClassifierAlignment(loss_type="l1")

    probs_a = torch.tensor([[1.0, 0.0, 0.0]])
    probs_b = torch.tensor([[0.0, 1.0, 0.0]])

    loss = criterion([probs_a, probs_b])
    assert loss.item() > 0.0


def test_single_source_returns_zero():
    criterion = ClassifierAlignment(num_sources=1, loss_type="l1")

    logits = torch.randn(4, 3)
    probs = torch.softmax(logits, dim=1)

    loss = criterion([probs])
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)


def test_alignment_loss_backward_propagates_gradients():
    for loss_type in ["l1", "kl"]:
        criterion = ClassifierAlignment(num_sources=2, loss_type=loss_type)

        logits_a = torch.randn(3, 4, requires_grad=True)
        logits_b = torch.randn(3, 4, requires_grad=True)

        probs_a = torch.softmax(logits_a, dim=1)
        probs_b = torch.softmax(logits_b, dim=1)

        loss = criterion([probs_a, probs_b])
        loss.backward()

        assert logits_a.grad is not None
        assert logits_b.grad is not None
        assert torch.isfinite(logits_a.grad).all()
        assert torch.isfinite(logits_b.grad).all()


def test_kl_loss_is_finite():
    criterion = ClassifierAlignment(loss_type="kl")

    logits_a = torch.randn(6, 4)
    logits_b = torch.randn(6, 4)

    probs_a = torch.softmax(logits_a, dim=1)
    probs_b = torch.softmax(logits_b, dim=1)

    loss = criterion([probs_a, probs_b])
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
