"""
Andean Field Augmentations
==========================
Specialized augmentations to simulate Andean field conditions.
Handles illumination variations, shadows, and environmental factors.
"""

from typing import Tuple, Optional
import random

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF


class AndeanFieldAugmentation:
    """
    Augmentations to simulate Andean highland field conditions.
    
    Simulates:
    - High-altitude lighting (intense UV, harsh shadows)
    - Variable cloud cover
    - Morning mist/fog
    - Soil reflections
    - Variable sun angles
    
    Args:
        p: Probability of applying each augmentation
        intensity: Overall intensity ('light', 'medium', 'strong')
    """
    
    def __init__(
        self,
        p: float = 0.5,
        intensity: str = "medium",
    ):
        self.p = p
        self.intensity = intensity
        
        # Intensity-dependent parameters
        self.params = self._get_intensity_params(intensity)
    
    def _get_intensity_params(self, intensity: str) -> dict:
        """Get augmentation parameters based on intensity."""
        if intensity == "light":
            return {
                'brightness': (0.8, 1.2),
                'contrast': (0.9, 1.1),
                'saturation': (0.9, 1.1),
                'shadow_intensity': (0.6, 0.9),
                'fog_density': (0.0, 0.1),
            }
        elif intensity == "medium":
            return {
                'brightness': (0.6, 1.4),
                'contrast': (0.7, 1.3),
                'saturation': (0.7, 1.3),
                'shadow_intensity': (0.4, 0.8),
                'fog_density': (0.0, 0.2),
            }
        elif intensity == "strong":
            return {
                'brightness': (0.4, 1.6),
                'contrast': (0.5, 1.5),
                'saturation': (0.5, 1.5),
                'shadow_intensity': (0.2, 0.7),
                'fog_density': (0.0, 0.3),
            }
        else:
            raise ValueError(f"Unknown intensity: {intensity}")
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply Andean field augmentations.
        
        Args:
            image: Input image tensor (C, H, W)
            
        Returns:
            Augmented image tensor
        """
        # High-altitude bright sunlight
        if random.random() < self.p:
            image = self._apply_harsh_lighting(image)
        
        # Overcast/cloudy conditions
        if random.random() < self.p:
            image = self._apply_overcast(image)
        
        # Morning mist simulation
        if random.random() < self.p * 0.5:  # Less frequent
            image = self._apply_mist(image)
        
        # Shadow simulation
        if random.random() < self.p:
            image = self._apply_shadow(image)
        
        # Soil reflection (warm tones)
        if random.random() < self.p * 0.3:
            image = self._apply_soil_reflection(image)
        
        return image
    
    def _apply_harsh_lighting(self, image: torch.Tensor) -> torch.Tensor:
        """Simulate intense high-altitude sunlight."""
        brightness = random.uniform(*self.params['brightness'])
        contrast = random.uniform(*self.params['contrast'])
        
        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
        
        return image
    
    def _apply_overcast(self, image: torch.Tensor) -> torch.Tensor:
        """Simulate cloudy/overcast conditions."""
        # Reduce saturation and slightly blue shift
        saturation = random.uniform(0.6, 0.9)
        image = TF.adjust_saturation(image, saturation)
        
        # Slight blue tint
        if image.shape[0] == 3:
            blue_shift = random.uniform(1.0, 1.1)
            image[2] = torch.clamp(image[2] * blue_shift, 0, 1)
        
        return image
    
    def _apply_mist(self, image: torch.Tensor) -> torch.Tensor:
        """Simulate morning mist/fog."""
        fog_density = random.uniform(*self.params['fog_density'])
        
        # Add white fog overlay
        fog = torch.ones_like(image) * 0.9  # Light gray fog
        image = image * (1 - fog_density) + fog * fog_density
        
        # Reduce contrast
        image = TF.adjust_contrast(image, 1 - fog_density * 0.5)
        
        return torch.clamp(image, 0, 1)
    
    def _apply_shadow(self, image: torch.Tensor) -> torch.Tensor:
        """Simulate partial shadows on leaves."""
        C, H, W = image.shape
        
        # Random shadow region
        shadow_intensity = random.uniform(*self.params['shadow_intensity'])
        
        # Create shadow mask (random rectangular region)
        x1 = random.randint(0, W // 2)
        y1 = random.randint(0, H // 2)
        x2 = random.randint(x1 + W // 4, W)
        y2 = random.randint(y1 + H // 4, H)
        
        mask = torch.ones_like(image)
        mask[:, y1:y2, x1:x2] = shadow_intensity
        
        return image * mask
    
    def _apply_soil_reflection(self, image: torch.Tensor) -> torch.Tensor:
        """Simulate warm soil reflections (reddish-brown tint)."""
        if image.shape[0] == 3:
            # Warm color shift (increase red, decrease blue)
            red_boost = random.uniform(1.0, 1.15)
            blue_reduce = random.uniform(0.9, 1.0)
            
            image[0] = torch.clamp(image[0] * red_boost, 0, 1)  # Red
            image[2] = torch.clamp(image[2] * blue_reduce, 0, 1)  # Blue
        
        return image


class AndeanFieldTransform:
    """
    Complete transform pipeline with Andean field augmentations.
    
    Combines standard augmentations with field-specific ones.
    
    Args:
        image_size: Target image size
        augment: Whether to apply augmentations (False for validation)
        andean_intensity: Andean augmentation intensity
    """
    
    def __init__(
        self,
        image_size: int = 224,
        augment: bool = True,
        andean_intensity: str = "medium",
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.image_size = image_size
        self.augment = augment
        self.mean = mean
        self.std = std
        
        # Standard transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)
        self.resize = transforms.Resize((image_size, image_size))
        
        # Augmentations
        if augment:
            self.standard_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
            ])
            self.andean_aug = AndeanFieldAugmentation(
                p=0.5,
                intensity=andean_intensity,
            )
        else:
            self.standard_aug = None
            self.andean_aug = None
    
    def __call__(self, image) -> torch.Tensor:
        """Apply transforms to PIL image."""
        # Resize
        image = self.resize(image)
        
        # Standard augmentations (on PIL)
        if self.augment and self.standard_aug:
            image = self.standard_aug(image)
        
        # Convert to tensor
        image = self.to_tensor(image)
        
        # Andean augmentations (on tensor)
        if self.augment and self.andean_aug:
            image = self.andean_aug(image)
        
        # Normalize
        image = self.normalize(image)
        
        return image
