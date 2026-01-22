"""
MDFAN: Multi-source Domain Feature Adaptation Network
=====================================================
Main MDFAN architecture for multi-source unsupervised domain adaptation.

Two-stage alignment:
1. Feature-level alignment: Aligns source-target distributions using
   adversarial training and MMD loss.
2. Classifier-level alignment (optional): Aligns decision boundaries
   across source classifiers.

Reference:
    Adapted from multi-source domain adaptation literature.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import BackboneFactory
from ..heads import ClassifierHead
from .feature_extractor import FeatureExtractor
from .domain_discriminator import DomainDiscriminator, MultiSourceDomainDiscriminator


class MDFAN(nn.Module):
    """
    Multi-source Domain Feature Adaptation Network.
    
    Architecture:
        Shared Backbone -> Feature Extractor -> [Domain Discriminators]
                                             -> [Source Classifiers]
    
    Args:
        backbone_name: Backbone network name
        num_classes: Number of disease classes
        num_sources: Number of source domains
        pretrained: Use pretrained backbone
        bottleneck_dim: Feature bottleneck dimension
        hidden_dim: Domain discriminator hidden dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 5,
        num_sources: int = 2,
        pretrained: bool = True,
        bottleneck_dim: int = 256,
        hidden_dim: int = 1024,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_sources = num_sources
        self.bottleneck_dim = bottleneck_dim
        
        # Shared backbone (MobileNet or ResNet)
        self.backbone, backbone_dim = BackboneFactory.create(
            backbone_name,
            pretrained=pretrained,
        )
        
        # Shared feature extractor (bottleneck)
        self.feature_extractor = FeatureExtractor(
            in_features=backbone_dim,
            out_features=bottleneck_dim,
            use_bn=True,
            dropout=dropout,
        )
        
        # Domain discriminators (one per source)
        self.domain_discriminators = MultiSourceDomainDiscriminator(
            in_features=bottleneck_dim,
            hidden_dim=hidden_dim,
            num_sources=num_sources,
        )
        
        # Source-specific classifiers (for multi-source combination)
        self.source_classifiers = nn.ModuleList([
            ClassifierHead(
                in_features=bottleneck_dim,
                num_classes=num_classes,
                bottleneck_dim=None,  # No additional bottleneck
                dropout=dropout,
            )
            for _ in range(num_sources)
        ])
        
        # Combined classifier for inference
        self.combined_classifier = ClassifierHead(
            in_features=bottleneck_dim,
            num_classes=num_classes,
            bottleneck_dim=None,
            dropout=dropout,
        )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract bottleneck features from images."""
        backbone_features = self.backbone(x)
        return self.feature_extractor(backbone_features)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for inference.
        
        Uses combined classifier for predictions.
        
        Args:
            x: Input images of shape (B, C, H, W)
            return_features: Also return bottleneck features
            
        Returns:
            Logits of shape (B, num_classes)
        """
        features = self.extract_features(x)
        logits = self.combined_classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def forward_source(
        self,
        x: torch.Tensor,
        source_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a source domain sample.
        
        Args:
            x: Source domain images
            source_idx: Index of source domain
            
        Returns:
            Tuple of (class_logits, domain_pred, features)
        """
        features = self.extract_features(x)
        class_logits = self.source_classifiers[source_idx](features)
        domain_pred = self.domain_discriminators(features, source_idx)
        
        return class_logits, domain_pred, features
    
    def forward_target(
        self,
        x: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Forward pass for target domain samples.
        
        Args:
            x: Target domain images
            
        Returns:
            Tuple of (class_logits_list, domain_preds_list, features)
        """
        features = self.extract_features(x)
        
        # Get predictions from all source classifiers
        class_logits_list = [
            classifier(features) 
            for classifier in self.source_classifiers
        ]
        
        # Get domain predictions from all discriminators
        domain_preds_list = [
            self.domain_discriminators(features, i)
            for i in range(self.num_sources)
        ]
        
        return class_logits_list, domain_preds_list, features
    
    def forward_train(
        self,
        source_images: List[torch.Tensor],
        source_labels: List[torch.Tensor],
        target_images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass for training.
        
        Args:
            source_images: List of source domain image batches
            source_labels: List of source domain label batches
            target_images: Target domain image batch
            
        Returns:
            Dictionary with all intermediate outputs
        """
        outputs = {
            'source_features': [],
            'source_logits': [],
            'source_domain_preds': [],
            'target_features': None,
            'target_logits': [],
            'target_domain_preds': [],
        }
        
        # Process each source domain
        for i, (src_img, src_lbl) in enumerate(zip(source_images, source_labels)):
            logits, domain_pred, features = self.forward_source(src_img, i)
            outputs['source_features'].append(features)
            outputs['source_logits'].append(logits)
            outputs['source_domain_preds'].append(domain_pred)
        
        # Process target domain
        target_logits, target_domain_preds, target_features = self.forward_target(
            target_images
        )
        outputs['target_features'] = target_features
        outputs['target_logits'] = target_logits
        outputs['target_domain_preds'] = target_domain_preds
        
        return outputs
    
    def get_combined_prediction(
        self,
        x: torch.Tensor,
        method: str = "average",
    ) -> torch.Tensor:
        """
        Get combined prediction from all source classifiers.
        
        Args:
            x: Input images
            method: Combination method ('average', 'weighted', 'voting')
            
        Returns:
            Combined predictions
        """
        features = self.extract_features(x)
        
        predictions = []
        for classifier in self.source_classifiers:
            logits = classifier(features)
            probs = F.softmax(logits, dim=1)
            predictions.append(probs)
        
        if method == "average":
            # Simple averaging
            combined = torch.stack(predictions).mean(dim=0)
        elif method == "voting":
            # Hard voting
            votes = torch.stack([p.argmax(dim=1) for p in predictions])
            combined = torch.mode(votes, dim=0).values
            # Convert back to one-hot for consistency
            combined = F.one_hot(combined, self.num_classes).float()
        else:
            combined = torch.stack(predictions).mean(dim=0)
        
        return combined
    
    def set_grl_lambda(self, lambda_: float):
        """Set GRL lambda for all domain discriminators."""
        self.domain_discriminators.set_lambda(lambda_)
    
    def get_grl_lambda(self) -> float:
        """Get current GRL lambda."""
        return self.domain_discriminators.get_lambda()
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
