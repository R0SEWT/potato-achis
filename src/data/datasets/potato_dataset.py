"""
Potato Disease Dataset
======================
Dataset class for loading potato disease images.
Supports both labeled and unlabeled modes for domain adaptation.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from .base_dataset import BaseDataset


class PotatoDiseaseDataset(BaseDataset):
    """
    Potato disease image dataset.
    
    Expected directory structure:
        root/
        ├── early_blight/
        │   ├── img1.jpg
        │   └── ...
        ├── late_blight/
        ├── healthy/
        ├── bacterial_wilt/
        └── virus/
    
    Args:
        root: Root directory containing class folders
        transform: Image transforms
        target_transform: Label transforms
        labeled: Whether to return labels (False for unlabeled target domain)
        domain_label: Optional domain label for domain adaptation
        extensions: Valid image file extensions
    """
    
    DEFAULT_CLASSES = [
        "early_blight",
        "late_blight", 
        "healthy",
        "bacterial_wilt",
        "virus",
    ]
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        labeled: bool = True,
        domain_label: Optional[int] = None,
        classes: Optional[List[str]] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ):
        super().__init__(root, transform, target_transform)
        
        self.labeled = labeled
        self.domain_label = domain_label
        self.extensions = extensions
        
        # Use provided classes or defaults
        self.classes = classes if classes is not None else self.DEFAULT_CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self._load_samples()
    
    def _load_samples(self):
        """Load image paths and labels from directory structure."""
        self.samples = []
        
        for class_name in self.classes:
            class_dir = self.root / class_name
            
            if not class_dir.exists():
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((str(img_path), class_idx))
        
        if len(self.samples) == 0:
            # Try loading from flat directory (unlabeled mode)
            self._load_flat_samples()
    
    def _load_flat_samples(self):
        """Load samples from flat directory (no class subfolders)."""
        for img_path in self.root.iterdir():
            if img_path.suffix.lower() in self.extensions:
                self.samples.append((str(img_path), -1))  # -1 for unknown label
    
    def __getitem__(
        self, 
        index: int
    ) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, int]]:
        """
        Get sample by index.
        
        Returns:
            If labeled: (image, label)
            If labeled with domain: (image, label, domain_label)
            If unlabeled: (image, -1) or (image, -1, domain_label)
        """
        img_path, label = self.samples[index]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None and label >= 0:
            label = self.target_transform(label)
        
        # Return based on mode
        if self.domain_label is not None:
            return image, label, self.domain_label
        
        return image, label
    
    def get_image_path(self, index: int) -> str:
        """Get image path for a given index."""
        return self.samples[index][0]


class UnlabeledDataset(BaseDataset):
    """
    Simple unlabeled dataset for target domain.
    
    Args:
        root: Directory containing images
        transform: Image transforms
        domain_label: Domain label for this dataset
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        domain_label: int = 1,  # Target domain
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ):
        super().__init__(root, transform)
        
        self.domain_label = domain_label
        self.extensions = extensions
        self._load_samples()
    
    def _load_samples(self):
        """Load all images from directory."""
        self.samples = []
        
        # Recursively find all images
        for ext in self.extensions:
            self.samples.extend([
                (str(p), -1) for p in self.root.rglob(f"*{ext}")
            ])
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """Get sample: (image, pseudo_label=-1, domain_label)."""
        img_path, _ = self.samples[index]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, -1, self.domain_label
