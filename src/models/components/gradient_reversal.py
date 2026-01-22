"""
Gradient Reversal Layer
=======================
Essential component for adversarial domain adaptation.
Reverses gradients during backpropagation.

Reference:
    Ganin et al., "Domain-Adversarial Training of Neural Networks" (2016)
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Function.
    
    Forward: identity operation
    Backward: negates gradients and scales by lambda
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """
        Forward pass - identity operation.
        
        Args:
            ctx: Context for saving tensors
            x: Input tensor
            lambda_: Scaling factor for gradient reversal
            
        Returns:
            Input tensor unchanged
        """
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass - reverse and scale gradients.
        
        Args:
            grad_output: Gradient from subsequent layers
            
        Returns:
            Reversed gradient, None for lambda
        """
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL).
    
    Used between feature extractor and domain discriminator
    in adversarial domain adaptation.
    
    Args:
        lambda_: Initial scaling factor (default: 1.0)
        
    Example:
        >>> grl = GradientReversalLayer(lambda_=1.0)
        >>> features = backbone(images)
        >>> reversed_features = grl(features)
        >>> domain_pred = domain_discriminator(reversed_features)
    """
    
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal."""
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_: float):
        """Update lambda value during training."""
        self.lambda_ = lambda_
    
    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.lambda_


def get_lambda_schedule(
    epoch: int,
    max_epochs: int,
    initial: float = 0.0,
    final: float = 1.0,
    warmup_epochs: int = 10,
) -> float:
    """
    Compute lambda value for GRL scheduling.
    
    Uses a gradual ramp-up schedule to stabilize training:
    - First `warmup_epochs`: linear increase from `initial` to `final`
    - After warmup: constant at `final`
    
    Args:
        epoch: Current epoch (0-indexed)
        max_epochs: Total training epochs
        initial: Initial lambda value
        final: Final lambda value
        warmup_epochs: Number of warmup epochs
        
    Returns:
        Lambda value for current epoch
    """
    if epoch < warmup_epochs:
        # Linear warmup
        progress = epoch / warmup_epochs
        return initial + (final - initial) * progress
    else:
        return final


def get_lambda_schedule_dann(
    progress: float,
    gamma: float = 10.0,
) -> float:
    """
    DANN-style lambda schedule (from original paper).
    
    lambda = 2 / (1 + exp(-gamma * p)) - 1
    
    where p is training progress from 0 to 1.
    
    Args:
        progress: Training progress (0.0 to 1.0)
        gamma: Scaling factor (default: 10)
        
    Returns:
        Lambda value
    """
    import math
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0
