"""
Feature Extractor (Bottleneck)
==============================
Additional feature extraction layers for domain adaptation.
Maps backbone features to a shared feature space.
"""

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """
    Feature extractor / bottleneck layer.
    
    Maps high-dimensional backbone features to a lower-dimensional
    shared feature space for domain alignment.
    
    Args:
        in_features: Input feature dimension (from backbone)
        out_features: Output feature dimension
        use_bn: Use batch normalization
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int = 256,
        use_bn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        layers = [nn.Linear(in_features, out_features)]
        
        if use_bn:
            layers.append(nn.BatchNorm1d(out_features))
        
        layers.append(nn.ReLU(inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        self.extractor = nn.Sequential(*layers)
        self.out_features = out_features
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features."""
        return self.extractor(x)


class MultiRepresentationExtractor(nn.Module):
    """
    Multi-representation feature extractor.
    
    Extracts multiple feature representations for richer
    domain alignment (inspired by MRAN's Inception Module).
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension per branch
        num_branches: Number of representation branches
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int = 256,
        num_branches: int = 3,
    ):
        super().__init__()
        
        self.num_branches = num_branches
        self.out_features = out_features * num_branches
        
        # Branch 1: Direct mapping
        self.branch1 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )
        
        # Branch 2: Two-layer bottleneck
        self.branch2 = nn.Sequential(
            nn.Linear(in_features, out_features // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_features // 2, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )
        
        # Branch 3: Wide bottleneck
        self.branch3 = nn.Sequential(
            nn.Linear(in_features, out_features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_features * 2, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-representation features.
        
        Returns:
            Concatenated features from all branches
        """
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        return torch.cat([b1, b2, b3], dim=1)
