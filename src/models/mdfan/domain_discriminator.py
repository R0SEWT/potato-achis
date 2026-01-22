"""
Domain Discriminator
====================
Discriminator network for adversarial domain adaptation.
Classifies features as belonging to source or target domain.
"""

import torch
import torch.nn as nn

from ..components.gradient_reversal import GradientReversalLayer


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for adversarial domain adaptation.
    
    Architecture: GRL -> FC -> ReLU -> FC -> ReLU -> FC -> Sigmoid
    
    Args:
        in_features: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_domains: Number of domains (2 for single-source DA)
        grl_lambda: Initial GRL lambda value
        use_sigmoid: Output sigmoid (True) or raw logits (False)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 1024,
        num_domains: int = 2,
        grl_lambda: float = 1.0,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        
        self.grl = GradientReversalLayer(lambda_=grl_lambda)
        
        self.discriminator = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_domains),
        )
        
        self.use_sigmoid = use_sigmoid
        self.num_domains = num_domains
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor,
        apply_grl: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (B, in_features)
            apply_grl: Whether to apply gradient reversal
            
        Returns:
            Domain predictions of shape (B, num_domains) or (B, 1)
        """
        if apply_grl:
            x = self.grl(x)
        
        logits = self.discriminator(x)
        
        if self.use_sigmoid and self.num_domains == 2:
            # Binary classification
            return torch.sigmoid(logits[:, 0:1])
        
        return logits
    
    def set_lambda(self, lambda_: float):
        """Update GRL lambda value."""
        self.grl.set_lambda(lambda_)
    
    def get_lambda(self) -> float:
        """Get current GRL lambda."""
        return self.grl.get_lambda()


class MultiSourceDomainDiscriminator(nn.Module):
    """
    Domain discriminator for multi-source domain adaptation.
    
    Creates separate discriminators for each source-target pair.
    
    Args:
        in_features: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_sources: Number of source domains
        grl_lambda: Initial GRL lambda value
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 1024,
        num_sources: int = 2,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        
        self.num_sources = num_sources
        
        # One discriminator per source domain
        self.discriminators = nn.ModuleList([
            DomainDiscriminator(
                in_features=in_features,
                hidden_dim=hidden_dim,
                num_domains=2,  # source_i vs target
                grl_lambda=grl_lambda,
                use_sigmoid=True,
            )
            for _ in range(num_sources)
        ])
    
    def forward(
        self,
        features: torch.Tensor,
        source_idx: int,
        apply_grl: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for a specific source domain.
        
        Args:
            features: Input features
            source_idx: Index of source domain (0 to num_sources-1)
            apply_grl: Whether to apply gradient reversal
            
        Returns:
            Domain predictions
        """
        return self.discriminators[source_idx](features, apply_grl=apply_grl)
    
    def set_lambda(self, lambda_: float):
        """Update GRL lambda for all discriminators."""
        for disc in self.discriminators:
            disc.set_lambda(lambda_)
    
    def get_lambda(self) -> float:
        """Get current GRL lambda (from first discriminator)."""
        return self.discriminators[0].get_lambda()
