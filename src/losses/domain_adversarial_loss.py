"""
Domain Adversarial Loss
=======================
Adversarial loss functions for domain adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DomainAdversarialLoss(nn.Module):
    """
    Domain adversarial loss for DANN-style training.
    
    Binary cross-entropy loss for domain classification.
    Used with gradient reversal layer for adversarial training.
    
    Args:
        reduction: Loss reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.bce = nn.BCELoss(reduction=reduction)
        self.bce_logits = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(
        self,
        domain_pred: torch.Tensor,
        domain_label: torch.Tensor,
        use_logits: bool = False,
    ) -> torch.Tensor:
        """
        Compute domain adversarial loss.
        
        Args:
            domain_pred: Domain predictions (B, 1) or (B,)
            domain_label: Domain labels (B,) - 0 for source, 1 for target
            use_logits: Whether domain_pred is logits (not sigmoid)
            
        Returns:
            Domain adversarial loss
        """
        # Ensure proper shapes
        domain_label = domain_label.float()
        if domain_pred.dim() > 1:
            domain_pred = domain_pred.squeeze(-1)
        
        if use_logits:
            return self.bce_logits(domain_pred, domain_label)
        return self.bce(domain_pred, domain_label)


class MultiSourceDomainLoss(nn.Module):
    """
    Domain loss for multi-source domain adaptation.
    
    Computes domain adversarial loss for each source-target pair.
    
    Args:
        num_sources: Number of source domains
        reduction: Loss reduction method
    """
    
    def __init__(
        self,
        num_sources: int,
        reduction: str = "mean",
    ):
        super().__init__()
        
        self.num_sources = num_sources
        self.domain_loss = DomainAdversarialLoss(reduction=reduction)
    
    def forward(
        self,
        source_domain_preds: List[torch.Tensor],
        target_domain_preds: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute total domain loss across all source-target pairs.
        
        Args:
            source_domain_preds: List of source domain predictions
            target_domain_preds: List of target domain predictions
                                 (one per source discriminator)
            
        Returns:
            Average domain loss
        """
        total_loss = torch.tensor(0.0, device=source_domain_preds[0].device)
        
        for i in range(self.num_sources):
            source_pred = source_domain_preds[i]
            target_pred = target_domain_preds[i]
            
            batch_s = source_pred.size(0)
            batch_t = target_pred.size(0)
            
            # Source labels = 0, Target labels = 1
            source_labels = torch.zeros(batch_s, device=source_pred.device)
            target_labels = torch.ones(batch_t, device=target_pred.device)
            
            # Compute loss for this source-target pair
            source_loss = self.domain_loss(source_pred, source_labels)
            target_loss = self.domain_loss(target_pred, target_labels)
            
            total_loss += (source_loss + target_loss) / 2
        
        return total_loss / self.num_sources


class EntropyLoss(nn.Module):
    """
    Entropy minimization loss.
    
    Encourages confident predictions on target domain.
    Used for semi-supervised domain adaptation.
    
    Args:
        reduction: Loss reduction method
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of predictions.
        
        Args:
            logits: Model logits (B, C)
            
        Returns:
            Entropy loss (lower = more confident)
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        
        entropy = -(probs * log_probs).sum(dim=1)
        
        if self.reduction == "mean":
            return entropy.mean()
        elif self.reduction == "sum":
            return entropy.sum()
        return entropy


class ClassificationLoss(nn.Module):
    """
    Classification loss with optional label smoothing.
    
    Args:
        num_classes: Number of classes
        label_smoothing: Label smoothing factor (0 = no smoothing)
        reduction: Loss reduction method
    """
    
    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            logits: Model predictions (B, C)
            labels: Ground truth labels (B,)
            
        Returns:
            Classification loss
        """
        return self.ce(logits, labels)
