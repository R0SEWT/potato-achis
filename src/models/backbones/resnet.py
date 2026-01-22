"""
ResNet Backbone
===============
ResNet feature extractor using timm.
Standard backbone for domain adaptation experiments.
"""

from typing import Optional
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")


class ResNetBackbone(nn.Module):
    """
    ResNet backbone for feature extraction.
    
    Args:
        model_name: timm model name (e.g., 'resnet50')
        pretrained: Load pretrained ImageNet weights
        frozen_stages: Number of stages to freeze (0-4)
    """
    
    # Output dimensions for ResNet variants
    OUTPUT_DIMS = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
    }
    
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        frozen_stages: int = 0,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = self.OUTPUT_DIMS.get(model_name, 2048)
        
        # Load model without classifier head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool="avg",
        )
        
        # Freeze stages if specified
        if frozen_stages > 0:
            self._freeze_stages(frozen_stages)
    
    def _freeze_stages(self, num_stages: int):
        """
        Freeze early layers for transfer learning.
        
        ResNet stages:
        0: conv1, bn1
        1: layer1
        2: layer2
        3: layer3
        4: layer4
        """
        # Always freeze conv1 and bn1 if any stage is frozen
        if num_stages >= 1:
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False
        
        # Freeze layer1-4 progressively
        layers = ['layer1', 'layer2', 'layer3', 'layer4']
        for i, layer_name in enumerate(layers):
            if num_stages >= i + 2:  # +2 because stage 1 is conv1/bn1
                layer = getattr(self.backbone, layer_name, None)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Features of shape (B, output_dim)
        """
        return self.backbone(x)
    
    def get_trainable_params(self):
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_intermediate_features(self, x: torch.Tensor) -> dict:
        """
        Get intermediate features from each stage.
        Useful for multi-scale feature adaptation.
        
        Returns:
            Dictionary with features from each layer
        """
        features = {}
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        features['stem'] = x
        
        x = self.backbone.layer1(x)
        features['layer1'] = x
        
        x = self.backbone.layer2(x)
        features['layer2'] = x
        
        x = self.backbone.layer3(x)
        features['layer3'] = x
        
        x = self.backbone.layer4(x)
        features['layer4'] = x
        
        x = self.backbone.global_pool(x)
        features['pooled'] = x
        
        return features
