"""
Multi-Source Dataset
====================
Dataset wrapper for multi-source domain adaptation.
Combines multiple source domains with a target domain.
"""

from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from .potato_dataset import PotatoDiseaseDataset, UnlabeledDataset


class MultiSourceDataset:
    """
    Wrapper for multi-source domain adaptation datasets.
    
    Manages multiple source datasets and one target dataset,
    providing synchronized iteration for MDFAN training.
    
    Args:
        source_datasets: List of source domain datasets
        target_dataset: Target domain dataset
        source_names: Names for each source domain
    """
    
    def __init__(
        self,
        source_datasets: List[PotatoDiseaseDataset],
        target_dataset: Dataset,
        source_names: Optional[List[str]] = None,
    ):
        self.source_datasets = source_datasets
        self.target_dataset = target_dataset
        self.num_sources = len(source_datasets)
        
        if source_names is None:
            source_names = [f"source_{i}" for i in range(self.num_sources)]
        self.source_names = source_names
        
        # Verify all sources have same classes
        self._verify_classes()
    
    def _verify_classes(self):
        """Verify all source datasets have the same classes."""
        if self.num_sources == 0:
            return
        
        base_classes = set(self.source_datasets[0].classes)
        for i, ds in enumerate(self.source_datasets[1:], 1):
            if set(ds.classes) != base_classes:
                raise ValueError(
                    f"Source {i} has different classes than source 0. "
                    f"Expected {base_classes}, got {set(ds.classes)}"
                )
    
    @property
    def classes(self) -> List[str]:
        """Get class names (from first source)."""
        if self.num_sources > 0:
            return self.source_datasets[0].classes
        return []
    
    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.classes)
    
    def get_source_sizes(self) -> List[int]:
        """Get size of each source dataset."""
        return [len(ds) for ds in self.source_datasets]
    
    def get_target_size(self) -> int:
        """Get size of target dataset."""
        return len(self.target_dataset)
    
    def get_dataloaders(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
    ) -> Tuple[List[DataLoader], DataLoader]:
        """
        Create dataloaders for all datasets.
        
        Returns:
            Tuple of (source_loaders, target_loader)
        """
        source_loaders = [
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=True,
            )
            for ds in self.source_datasets
        ]
        
        target_loader = DataLoader(
            self.target_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )
        
        return source_loaders, target_loader
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'num_sources': self.num_sources,
            'num_classes': self.num_classes,
            'classes': self.classes,
            'source_sizes': {},
            'target_size': self.get_target_size(),
        }
        
        for name, ds in zip(self.source_names, self.source_datasets):
            stats['source_sizes'][name] = {
                'total': len(ds),
                'per_class': ds.get_class_counts() if hasattr(ds, 'get_class_counts') else {}
            }
        
        return stats


class InfiniteDataLoader:
    """
    DataLoader wrapper that cycles infinitely.
    Useful for aligning iterations across datasets of different sizes.
    """
    
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
    
    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch
    
    def __iter__(self):
        return self


class MultiSourceIterator:
    """
    Iterator that synchronizes batches from multiple sources and target.
    
    At each iteration, returns:
        - List of source batches
        - Target batch
    """
    
    def __init__(
        self,
        source_loaders: List[DataLoader],
        target_loader: DataLoader,
        num_iterations: Optional[int] = None,
    ):
        self.source_loaders = [InfiniteDataLoader(dl) for dl in source_loaders]
        self.target_loader = InfiniteDataLoader(target_loader)
        
        # Default: iterate based on largest source
        if num_iterations is None:
            max_source_size = max(len(dl.dataloader.dataset) for dl in self.source_loaders)
            batch_size = source_loaders[0].batch_size
            num_iterations = max_source_size // batch_size
        
        self.num_iterations = num_iterations
        self.current = 0
    
    def __iter__(self):
        self.current = 0
        return self
    
    def __next__(self) -> Tuple[List, any]:
        if self.current >= self.num_iterations:
            raise StopIteration
        
        self.current += 1
        
        source_batches = [next(loader) for loader in self.source_loaders]
        target_batch = next(self.target_loader)
        
        return source_batches, target_batch
    
    def __len__(self):
        return self.num_iterations
