"""
Model Factory
=============
Central factory for creating models: Baseline, MDFAN.
Supports MobileNet and ResNet backbones.
"""

from typing import Any

import torch
import torch.nn as nn

from .backbones import BackboneFactory
from .heads import ClassifierHead
from .mdfan import MDFAN


class BaselineModel(nn.Module):
    """
    Baseline classification model without domain adaptation.
    Architecture: Backbone -> Classifier Head
    
    Args:
        backbone_name: Name of backbone (e.g., 'resnet50', 'mobilenet_v3_small')
        num_classes: Number of output classes
        pretrained: Use pretrained backbone weights
        bottleneck_dim: Bottleneck dimension (None to skip)
        dropout: Dropout probability
        frozen_stages: Number of backbone stages to freeze
    """

    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_small",
        num_classes: int = 15,
        pretrained: bool = True,
        bottleneck_dim: int | None = 256,
        dropout: float = 0.5,
        frozen_stages: int = 0,
    ):
        super().__init__()

        # Create backbone
        self.backbone, backbone_dim = BackboneFactory.create(
            backbone_name,
            pretrained=pretrained,
            frozen_stages=frozen_stages,
        )

        # Create classifier head
        self.head = ClassifierHead(
            in_features=backbone_dim,
            num_classes=num_classes,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )

        self.backbone_name = backbone_name
        self.num_classes = num_classes

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple:
        """
        Forward pass.
        
        Args:
            x: Input images of shape (B, C, H, W)
            return_features: Also return intermediate features
            
        Returns:
            Logits of shape (B, num_classes)
            Optionally: (logits, features) tuple
        """
        features = self.backbone(x)

        if return_features:
            logits, bottleneck_features = self.head(features, return_features=True)
            return logits, bottleneck_features

        return self.head(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features."""
        return self.backbone(x)

    def get_bottleneck_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract bottleneck features."""
        backbone_features = self.backbone(x)
        return self.head.get_features(backbone_features)


class ModelFactory:
    """Factory for creating models based on configuration."""

    SUPPORTED_TYPES = ["baseline", "mdfan"]

    @classmethod
    def create(cls, config: dict[str, Any]) -> nn.Module:
        """
        Create model from configuration dictionary.
        
        Args:
            config: Model configuration with 'type' key
            
        Returns:
            Instantiated model
        """
        model_type = config.get("type", "baseline")

        if model_type == "baseline":
            return cls._create_baseline(config)
        elif model_type == "mdfan":
            return cls._create_mdfan(config)
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported: {cls.SUPPORTED_TYPES}"
            )

    @classmethod
    def _create_baseline(cls, config: dict[str, Any]) -> BaselineModel:
        """Create baseline classification model."""
        return BaselineModel(
            backbone_name=config.get("backbone", "mobilenet_v3_small"),
            num_classes=config.get("num_classes", 15),
            pretrained=config.get("pretrained", True),
            bottleneck_dim=config.get("bottleneck_dim", 256),
            dropout=config.get("dropout", 0.5),
            frozen_stages=config.get("frozen_stages", 0),
        )

    @classmethod
    def _create_mdfan(cls, config: dict[str, Any]) -> MDFAN:
        """Create MDFAN model for domain adaptation."""
        return MDFAN(
            backbone_name=config.get("backbone", "resnet50"),
            num_classes=config.get("num_classes", 15),
            num_sources=config.get("num_sources", 2),
            pretrained=config.get("pretrained", True),
            bottleneck_dim=config.get("bottleneck_dim", 256),
            hidden_dim=config.get("hidden_dim", 1024),
            dropout=config.get("dropout", 0.5),
        )


def create_model(
    model_type: str = "baseline",
    **kwargs
) -> nn.Module:
    """
    Convenience function to create a model.
    
    Args:
        model_type: Type of model ('baseline' or 'mdfan')
        **kwargs: Model-specific arguments
        
    Returns:
        Instantiated model
    """
    config = {"type": model_type, **kwargs}
    return ModelFactory.create(config)
