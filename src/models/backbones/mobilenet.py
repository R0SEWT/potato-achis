"""
MobileNet Backbone
==================
MobileNetV3 feature extractor using timm.
Lightweight backbone suitable for mobile/edge deployment.
"""

from typing import Optional
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")


class MobileNetBackbone(nn.Module):
    """
    MobileNetV3 backbone for feature extraction.
    
    Args:
        model_name: timm model name (e.g., 'mobilenetv3_small_100')
        pretrained: Load pretrained ImageNet weights
        frozen_stages: Number of stages to freeze (0-5)
    """
    
    # Output dimensions for MobileNetV3 variants
    OUTPUT_DIMS = {
        "mobilenetv3_small_100": 576,
        "mobilenetv3_large_100": 960,
    }
    
    def __init__(
        self,
        model_name: str = "mobilenetv3_small_100",
        pretrained: bool = True,
        frozen_stages: int = 0,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = self.OUTPUT_DIMS.get(model_name, 576)
        
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
        """Freeze early layers for transfer learning."""
        # Freeze conv_stem
        if num_stages >= 1:
            for param in self.backbone.conv_stem.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False
        
        # Freeze blocks progressively
        if hasattr(self.backbone, 'blocks'):
            blocks_to_freeze = min(num_stages - 1, len(self.backbone.blocks))
            for i in range(blocks_to_freeze):
                for param in self.backbone.blocks[i].parameters():
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
