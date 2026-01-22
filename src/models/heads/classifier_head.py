"""
Classifier Head
===============
Task-specific classification head with optional bottleneck.
"""


import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """
    Classification head with optional bottleneck layer.
    
    Architecture:
        Input -> [Bottleneck] -> Dropout -> Linear -> Output
        
    Args:
        in_features: Input feature dimension
        num_classes: Number of output classes
        bottleneck_dim: Bottleneck dimension (None to skip)
        dropout: Dropout probability
        use_bn: Use batch normalization in bottleneck
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        bottleneck_dim: int | None = 256,
        dropout: float = 0.5,
        use_bn: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim

        layers = []

        # Optional bottleneck
        if bottleneck_dim is not None:
            layers.append(nn.Linear(in_features, bottleneck_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(bottleneck_dim))
            layers.append(nn.ReLU(inplace=True))
            classifier_in = bottleneck_dim
        else:
            classifier_in = in_features

        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.bottleneck = nn.Sequential(*layers) if layers else nn.Identity()

        # Final classifier
        self.classifier = nn.Linear(classifier_in, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
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
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (B, in_features)
            return_features: Also return bottleneck features
            
        Returns:
            Logits of shape (B, num_classes)
            Optionally: (logits, features) tuple
        """
        features = self.bottleneck(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get bottleneck features without classification."""
        return self.bottleneck(x)


