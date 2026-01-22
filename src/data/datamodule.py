"""
Data Module
===========
Central data management for potato disease classification.
Handles dataset creation, transforms, and dataloaders.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .datasets import MultiSourceDataset, PotatoDiseaseDataset, UnlabeledDataset
from .transforms import (
    AndeanFieldAugmentation,
    get_train_transforms,
    get_val_transforms,
)


class TransformSubset(Dataset):
    """
    A subset of a dataset with a specific transform.
    
    Solves the problem of random_split sharing the same transform
    for both train and val subsets.
    
    Args:
        dataset: The full dataset (should have transform=None)
        indices: Indices of samples to include
        transform: Transform to apply to samples
    """
    
    def __init__(
        self,
        dataset: Dataset,
        indices: List[int],
        transform: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __getitem__(self, idx: int):
        sample = self.dataset[self.indices[idx]]
        
        # Handle both (image, label) and (image, label, domain) returns
        if len(sample) == 2:
            image, label = sample
        else:
            image, label, domain = sample
        
        if self.transform is not None:
            image = self.transform(image)
        
        if len(sample) == 2:
            return image, label
        return image, label, domain
    
    def __len__(self) -> int:
        return len(self.indices)
    
    @property
    def classes(self) -> List[str]:
        """Forward classes property from underlying dataset."""
        return self.dataset.classes
    
    @property
    def class_to_idx(self) -> Dict[str, int]:
        """Forward class_to_idx property from underlying dataset."""
        return self.dataset.class_to_idx


class PotatoDataModule:
    """
    Data module for potato disease classification.
    
    Handles:
    - Loading source and target datasets
    - Creating train/val/test splits
    - Setting up dataloaders
    - Multi-source domain adaptation setup
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        image_size: Target image size
        val_split: Validation split ratio
        use_andean_aug: Use Andean field augmentations
        aug_strength: Augmentation strength
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        val_split: float = 0.2,
        use_andean_aug: bool = True,
        aug_strength: str = "medium",
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_split = val_split
        self.use_andean_aug = use_andean_aug
        self.aug_strength = aug_strength
        
        # Datasets (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Multi-source datasets
        self.source_datasets = []
        self.target_dataset = None

        # Transforms
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup data transforms."""
        if self.use_andean_aug:
            # Combine standard transforms with Andean augmentation
            andean_aug = AndeanFieldAugmentation(
                p=0.5,
                intensity=self.aug_strength,
            )
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize(
                        (int(self.image_size * 1.1), int(self.image_size * 1.1))
                    ),
                    transforms.RandomCrop(self.image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor(),
                    andean_aug,
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            self.train_transform = get_train_transforms(
                image_size=self.image_size,
                strength=self.aug_strength,
            )
        
        self.val_transform = get_val_transforms(image_size=self.image_size)

    def setup_single_source(
        self,
        source_dir: str,
        classes: Optional[List[str]] = None,
        class_filter: Optional[str] = "Potato",
    ):
        """
        Setup for single-source training (baseline).

        Args:
            source_dir: Path to source dataset
            classes: List of class names (auto-detected if None)
            class_filter: Filter to select only certain classes (e.g., "Potato")
        """
        # Create dataset WITHOUT transform - transforms applied via TransformSubset
        full_dataset = PotatoDiseaseDataset(
            root=source_dir,
            transform=None,  # Important: no transform here
            classes=classes,
            class_filter=class_filter,
        )

        # Store detected classes
        self._detected_classes = full_dataset.classes
        self._full_dataset = full_dataset

        # Calculate split sizes
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size

        # Generate reproducible random indices
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_size, generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create subsets with DIFFERENT transforms
        self.train_dataset = TransformSubset(
            full_dataset, train_indices, self.train_transform
        )
        self.val_dataset = TransformSubset(
            full_dataset, val_indices, self.val_transform
        )
    
    def setup_multi_source(
        self,
        source_dirs: List[str],
        target_dir: str,
        source_names: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
    ):
        """
        Setup for multi-source domain adaptation.
        
        Args:
            source_dirs: List of source dataset directories
            target_dir: Target dataset directory
            source_names: Names for source domains
            classes: List of class names
        """
        # Create source datasets
        self.source_datasets = []
        for i, src_dir in enumerate(source_dirs):
            dataset = PotatoDiseaseDataset(
                root=src_dir,
                transform=self.train_transform,
                classes=classes,
                domain_label=i,  # Source domain labels: 0, 1, ...
            )
            self.source_datasets.append(dataset)
        
        # Create target dataset (unlabeled)
        self.target_dataset = UnlabeledDataset(
            root=target_dir,
            transform=self.train_transform,
            domain_label=len(source_dirs),  # Target domain label
        )
        
        # Create multi-source wrapper
        self.multi_source = MultiSourceDataset(
            source_datasets=self.source_datasets,
            target_dataset=self.target_dataset,
            source_names=source_names,
        )
    
    def get_train_loader(self) -> DataLoader:
        """Get training dataloader (single-source)."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup_single_source() first")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def get_val_loader(self) -> DataLoader:
        """Get validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("Call setup_single_source() first")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def get_multi_source_loaders(self) -> Tuple[List[DataLoader], DataLoader]:
        """
        Get dataloaders for multi-source domain adaptation.
        
        Returns:
            Tuple of (source_loaders, target_loader)
        """
        if self.multi_source is None:
            raise RuntimeError("Call setup_multi_source() first")
        
        return self.multi_source.get_dataloaders(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def get_test_loader(
        self,
        test_dir: str,
        classes: Optional[List[str]] = None,
    ) -> DataLoader:
        """
        Get test dataloader for evaluation.
        
        Args:
            test_dir: Test dataset directory
            classes: Class names (use source classes if None)
        """
        if classes is None and self.source_datasets:
            classes = self.source_datasets[0].classes
        
        test_dataset = PotatoDiseaseDataset(
            root=test_dir,
            transform=self.val_transform,
            classes=classes,
        )
        
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        if hasattr(self, "_detected_classes"):
            return len(self._detected_classes)
        if self.source_datasets:
            return len(self.source_datasets[0].classes)
        if self.train_dataset and hasattr(self.train_dataset.dataset, "classes"):
            return len(self.train_dataset.dataset.classes)
        return 5  # Default

    @property
    def classes(self) -> List[str]:
        """Get class names."""
        if hasattr(self, "_detected_classes"):
            return self._detected_classes
        if self.source_datasets:
            return self.source_datasets[0].classes
        if self.train_dataset and hasattr(self.train_dataset.dataset, "classes"):
            return self.train_dataset.dataset.classes
        return PotatoDiseaseDataset.DEFAULT_CLASSES
