"""Tests for gradient reversal layer and lambda scheduling."""

import pytest
import torch

from src.models.components.gradient_reversal import (
    GradientReversalLayer,
    get_lambda_schedule,
    get_lambda_schedule_dann,
)


class TestGradientReversalLayer:
    """Tests for GRL forward/backward behavior."""

    def test_forward_is_identity(self):
        """Forward pass should not modify the input."""
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 8)
        y = grl(x)
        assert torch.allclose(x, y)

    def test_backward_reverses_gradient(self):
        """Backward pass should negate gradients."""
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 8, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()

        expected = -torch.ones_like(x)
        assert torch.allclose(x.grad, expected)

    def test_backward_scales_by_lambda(self):
        """Gradient should be scaled by -lambda."""
        lambda_val = 0.5
        grl = GradientReversalLayer(lambda_=lambda_val)
        x = torch.randn(4, 8, requires_grad=True)
        y = grl(x)
        y.sum().backward()

        expected = -lambda_val * torch.ones_like(x)
        assert torch.allclose(x.grad, expected)

    def test_lambda_zero_blocks_gradient(self):
        """Lambda=0 should zero out gradients."""
        grl = GradientReversalLayer(lambda_=0.0)
        x = torch.randn(4, 8, requires_grad=True)
        y = grl(x)
        y.sum().backward()
        assert torch.allclose(x.grad, torch.zeros_like(x))

    def test_set_lambda_affects_gradients(self):
        """set_lambda should change the scaling used in backward pass."""
        grl = GradientReversalLayer(lambda_=1.0)
        grl.set_lambda(0.3)
        assert grl.get_lambda() == pytest.approx(0.3)

        x = torch.randn(4, 8, requires_grad=True)
        y = grl(x)
        y.sum().backward()
        expected = -0.3 * torch.ones_like(x)
        assert torch.allclose(x.grad, expected)

    def test_works_in_computation_graph(self):
        """GRL should work with linear layers in a graph."""
        linear = torch.nn.Linear(8, 4)
        grl = GradientReversalLayer(lambda_=1.0)

        x = torch.randn(2, 8)
        features = linear(x)
        reversed_features = grl(features)
        loss = reversed_features.sum()
        loss.backward()

        assert linear.weight.grad is not None


class TestLambdaSchedule:
    """Tests for GRL lambda scheduling functions."""

    def test_warmup_starts_at_initial(self):
        val = get_lambda_schedule(0, 50, initial=0.0, final=1.0, warmup_epochs=10)
        assert val == pytest.approx(0.0)

    def test_warmup_reaches_final(self):
        val = get_lambda_schedule(10, 50, initial=0.0, final=1.0, warmup_epochs=10)
        assert val == pytest.approx(1.0)

    def test_midway_warmup(self):
        val = get_lambda_schedule(5, 50, initial=0.0, final=1.0, warmup_epochs=10)
        assert val == pytest.approx(0.5)

    def test_after_warmup_stays_at_final(self):
        val = get_lambda_schedule(30, 50, initial=0.0, final=1.0, warmup_epochs=10)
        assert val == pytest.approx(1.0)

    def test_epoch_beyond_max_stays_at_final(self):
        val = get_lambda_schedule(60, 50, initial=0.0, final=1.0, warmup_epochs=10)
        assert val == pytest.approx(1.0)

    def test_zero_warmup_immediately_at_final(self):
        val = get_lambda_schedule(0, 50, initial=0.0, final=1.0, warmup_epochs=0)
        assert val == pytest.approx(1.0)

    def test_dann_schedule_bounds(self):
        """DANN schedule should go from ~0 to ~1."""
        val_start = get_lambda_schedule_dann(0.0)
        val_end = get_lambda_schedule_dann(1.0)
        assert val_start < 0.01
        assert val_end > 0.99

    def test_dann_schedule_monotonic(self):
        """DANN schedule should be monotonically increasing."""
        values = [get_lambda_schedule_dann(p / 10) for p in range(11)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]
