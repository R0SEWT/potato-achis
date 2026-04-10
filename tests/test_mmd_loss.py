"""Tests for MMD loss functions."""

import pytest
import torch

from src.losses.mmd_loss import LocalMMDLoss, MMDLoss, MultiSourceMMDLoss


class TestMMDLoss:
    """Tests for MMDLoss with RBF and linear kernels."""

    def test_same_distribution_gives_low_loss(self):
        """MMD between identical distributions should be near zero."""
        torch.manual_seed(0)
        mmd = MMDLoss(kernel_type="rbf")
        x = torch.randn(32, 64)
        loss = mmd(x, x.clone())
        assert loss.item() < 0.1

    def test_different_distributions_gives_high_loss(self):
        """MMD between shifted distributions should be much higher than same-distribution."""
        torch.manual_seed(0)
        mmd = MMDLoss(kernel_type="rbf")
        source = torch.randn(32, 64)
        target = torch.randn(32, 64) + 5.0
        loss_same = mmd(source, source.clone())
        loss_diff = mmd(source, target)
        assert loss_diff.item() > loss_same.item()

    def test_linear_kernel(self):
        """Linear MMD should also detect distribution shift."""
        torch.manual_seed(0)
        mmd = MMDLoss(kernel_type="linear")
        source = torch.randn(32, 64)
        target = torch.randn(32, 64) + 3.0

        loss_same = mmd(source, source.clone())
        loss_diff = mmd(source, target)
        assert loss_diff.item() > loss_same.item()

    def test_loss_is_scalar(self):
        torch.manual_seed(0)
        mmd = MMDLoss(kernel_type="rbf")
        loss = mmd(torch.randn(16, 32), torch.randn(16, 32))
        assert loss.dim() == 0

    def test_loss_is_non_negative(self):
        """RBF MMD should be non-negative for well-behaved inputs."""
        torch.manual_seed(0)
        mmd = MMDLoss(kernel_type="rbf")
        loss = mmd(torch.randn(32, 64), torch.randn(32, 64))
        # MMD estimate can be slightly negative due to finite samples,
        # but should not be very negative
        assert loss.item() > -0.5

    def test_different_batch_sizes(self):
        """MMD should work with different source/target sizes."""
        torch.manual_seed(0)
        mmd = MMDLoss(kernel_type="rbf")
        loss = mmd(torch.randn(16, 64), torch.randn(48, 64))
        assert loss.dim() == 0

    def test_fix_sigma(self):
        torch.manual_seed(0)
        mmd = MMDLoss(kernel_type="rbf", fix_sigma=1.0)
        loss = mmd(torch.randn(16, 32), torch.randn(16, 32))
        assert loss.dim() == 0

    def test_invalid_kernel_raises(self):
        mmd = MMDLoss(kernel_type="invalid")
        with pytest.raises(ValueError, match="Unknown kernel type"):
            mmd(torch.randn(8, 16), torch.randn(8, 16))

    def test_gradient_flows(self):
        """Loss should support backpropagation."""
        torch.manual_seed(0)
        mmd = MMDLoss(kernel_type="rbf")
        source = torch.randn(16, 32, requires_grad=True)
        target = torch.randn(16, 32)
        loss = mmd(source, target)
        loss.backward()
        assert source.grad is not None


class TestLocalMMDLoss:
    """Tests for class-conditional LMMD."""

    def test_basic_forward(self):
        torch.manual_seed(0)
        lmmd = LocalMMDLoss(num_classes=3)
        source = torch.randn(30, 64)
        target = torch.randn(30, 64)
        source_labels = torch.randint(0, 3, (30,))
        target_labels = torch.randint(0, 3, (30,))
        loss = lmmd(source, target, source_labels, target_labels)
        assert loss.dim() == 0

    def test_no_shared_classes_gives_zero(self):
        """If no class overlaps, loss should be zero."""
        lmmd = LocalMMDLoss(num_classes=4)
        source = torch.randn(16, 32)
        target = torch.randn(16, 32)
        source_labels = torch.zeros(16, dtype=torch.long)  # all class 0
        target_labels = torch.ones(16, dtype=torch.long)  # all class 1
        loss = lmmd(source, target, source_labels, target_labels)
        assert loss.item() == pytest.approx(0.0)

    def test_gradient_flows(self):
        """LMMD should support backpropagation."""
        torch.manual_seed(0)
        lmmd = LocalMMDLoss(num_classes=3)
        source = torch.randn(30, 64, requires_grad=True)
        target = torch.randn(30, 64)
        source_labels = torch.randint(0, 3, (30,))
        target_labels = torch.randint(0, 3, (30,))
        loss = lmmd(source, target, source_labels, target_labels)
        loss.backward()
        assert source.grad is not None


class TestMultiSourceMMDLoss:
    """Tests for multi-source MMD."""

    def test_basic_forward(self):
        torch.manual_seed(0)
        ms_mmd = MultiSourceMMDLoss(num_sources=2)
        sources = [torch.randn(16, 64), torch.randn(16, 64)]
        target = torch.randn(16, 64)
        loss = ms_mmd(sources, target)
        assert loss.dim() == 0

    def test_single_source(self):
        torch.manual_seed(0)
        ms_mmd = MultiSourceMMDLoss(num_sources=1)
        sources = [torch.randn(16, 64)]
        target = torch.randn(16, 64)
        loss = ms_mmd(sources, target)
        assert loss.dim() == 0

    def test_gradient_flows(self):
        """MultiSourceMMD should support backpropagation."""
        torch.manual_seed(0)
        ms_mmd = MultiSourceMMDLoss(num_sources=2)
        sources = [
            torch.randn(16, 64, requires_grad=True),
            torch.randn(16, 64, requires_grad=True),
        ]
        target = torch.randn(16, 64)
        loss = ms_mmd(sources, target)
        loss.backward()
        assert sources[0].grad is not None
        assert sources[1].grad is not None
