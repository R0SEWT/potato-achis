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

    Supports two directory structures:
    
    1. Standard (class subfolders):
        root/
        ├── early_blight/
        ├── late_blight/
        └── healthy/
    
    2. PlantVillage style:
        root/
        ├── Potato___Early_blight/
        ├── Potato___Late_blight/
        └── Potato___healthy/

    Args:
        root: Root directory containing class folders
        transform: Image transforms
        target_transform: Label transforms
        labeled: Whether to return labels (False for unlabeled target domain)
        domain_label: Optional domain label for domain adaptation
        classes: List of class names to use (auto-detected if None)
        class_filter: Optional prefix filter (e.g., "Potato" to only load potato classes)
        extensions: Valid image file extensions
    """

    DEFAULT_CLASSES = [
        # Pepper (2)
        "pepper_bell_bacterial_spot",
        "pepper_bell_healthy",
        # Potato (3)
        "potato_early_blight",
        "potato_healthy",
        "potato_late_blight",
        # Tomato (10)
        "tomato_bacterial_spot",
        "tomato_early_blight",
        "tomato_healthy",
        "tomato_late_blight",
        "tomato_leaf_mold",
        "tomato_mosaic_virus",
        "tomato_septoria_leaf_spot",
        "tomato_spider_mites_two_spotted_spider_mite",
        "tomato_target_spot",
        "tomato_yellowleaf_curl_virus",
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        labeled: bool = True,
        domain_label: Optional[int] = None,
        classes: Optional[List[str]] = None,
        class_filter: Optional[str] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".JPG"),
    ):
        super().__init__(root, transform, target_transform)

        self.labeled = labeled
        self.domain_label = domain_label
        self.extensions = extensions
        self.class_filter = class_filter

        # Auto-detect classes from directory structure if not provided
        if classes is None:
            self.classes, self.folder_to_class = self._detect_classes()
        else:
            self.classes = classes
            self.folder_to_class = {cls: cls for cls in classes}

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self._load_samples()

    def _detect_classes(self) -> Tuple[List[str], Dict[str, str]]:
        """
        Auto-detect classes from directory structure.

        Handles PlantVillage naming (Potato___Early_blight -> early_blight)
        and standard naming (early_blight -> early_blight).

        Returns:
            Tuple of (class_names, folder_to_class_mapping)
        """
        classes = []
        folder_to_class = {}

        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue

            folder_name = folder.name

            # Apply filter if specified (e.g., only "Potato" classes)
            if self.class_filter:
                if not folder_name.lower().startswith(self.class_filter.lower()):
                    continue

            # Normalize class name
            # Handle PlantVillage: "Potato___Early_blight" -> "potato_early_blight"
            # Preserve plant prefix to avoid class name collisions (e.g., multiple "healthy")
            # Clean up multiple underscores and redundant prefixes
            class_name = folder_name.lower().replace("___", "_").replace("__", "_").replace(" ", "_")
            
            # Remove redundant plant name prefix (e.g., "tomato_tomato_mosaic" -> "tomato_mosaic")
            parts = class_name.split("_")
            if len(parts) > 1 and parts[0] == parts[1]:
                class_name = "_".join(parts[1:])

            if class_name not in classes:
                classes.append(class_name)

            folder_to_class[folder_name] = class_name

        if not classes:
            raise ValueError(
                f"No class folders found in {self.root}. "
                f"Filter: {self.class_filter}"
            )

        return classes, folder_to_class

    def _load_samples(self):
        """Load image paths and labels from directory structure."""
        self.samples = []

        for folder_name, class_name in self.folder_to_class.items():
            class_dir = self.root / folder_name

            if not class_dir.exists():
                continue

            class_idx = self.class_to_idx[class_name]

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in [ext.lower() for ext in self.extensions]:
                    self.samples.append((str(img_path), class_idx))

        if len(self.samples) == 0:
            raise ValueError(
                f"No images found in {self.root}. "
                f"Classes: {self.classes}, Extensions: {self.extensions}"
            )
    
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
