"""
Standard Augmentations
======================
Training and validation transforms for potato disease images.
"""

from typing import Tuple, Optional

import torch
from torchvision import transforms


def get_train_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    strength: str = "medium",
) -> transforms.Compose:
    """
    Get training transforms with data augmentation.
    
    Args:
        image_size: Target image size
        mean: Normalization mean (ImageNet default)
        std: Normalization std (ImageNet default)
        strength: Augmentation strength ('light', 'medium', 'strong')
        
    Returns:
        Composed transforms
    """
    if strength == "light":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    elif strength == "medium":
        transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    elif strength == "strong":
        transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.2), int(image_size * 1.2))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.15,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ])
    
    else:
        raise ValueError(f"Unknown strength: {strength}")
    
    return transform


def get_val_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_inference_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Alias for validation transforms."""
    return get_val_transforms(image_size, mean, std)


class Denormalize:
    """
    Denormalize images for visualization.
    
    Args:
        mean: Normalization mean
        std: Normalization std
    """
    
    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize a tensor image."""
        device = tensor.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        return tensor * std + mean
