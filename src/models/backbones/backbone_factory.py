"""
Backbone Factory
================
Factory pattern for creating backbone feature extractors.
Supports: MobileNetV3, ResNet variants via timm.
"""

import torch.nn as nn

from .mobilenet import MobileNetBackbone
from .resnet import ResNetBackbone


class BackboneFactory:
    """Factory for creating backbone networks."""

    SUPPORTED_BACKBONES = {
        # MobileNet variants
        "mobilenet_v3_small": ("mobilenet", "mobilenetv3_small_100"),
        "mobilenet_v3_large": ("mobilenet", "mobilenetv3_large_100"),
        # ResNet variants
        "resnet18": ("resnet", "resnet18"),
        "resnet34": ("resnet", "resnet34"),
        "resnet50": ("resnet", "resnet50"),
        "resnet101": ("resnet", "resnet101"),
    }

    @classmethod
    def create(
        cls,
        name: str,
        pretrained: bool = True,
        frozen_stages: int = 0,
    ) -> tuple[nn.Module, int]:
        """
        Create a backbone feature extractor.
        
        Args:
            name: Backbone name (e.g., 'resnet50', 'mobilenet_v3_small')
            pretrained: Whether to use pretrained weights
            frozen_stages: Number of stages to freeze (0 = train all)
            
        Returns:
            Tuple of (backbone module, output feature dimension)
        """
        if name not in cls.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {name}. "
                f"Supported: {list(cls.SUPPORTED_BACKBONES.keys())}"
            )

        backbone_type, model_name = cls.SUPPORTED_BACKBONES[name]

        if backbone_type == "mobilenet":
            backbone = MobileNetBackbone(
                model_name=model_name,
                pretrained=pretrained,
                frozen_stages=frozen_stages,
            )
        elif backbone_type == "resnet":
            backbone = ResNetBackbone(
                model_name=model_name,
                pretrained=pretrained,
                frozen_stages=frozen_stages,
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        return backbone, backbone.output_dim

    @classmethod
    def list_available(cls) -> list:
        """List all available backbone names."""
        return list(cls.SUPPORTED_BACKBONES.keys())
