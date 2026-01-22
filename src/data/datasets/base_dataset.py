"""
Base Dataset
============
Abstract base class for all datasets.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for potato disease datasets.
    
    Args:
        root: Root directory of dataset
        transform: Image transforms
        target_transform: Label transforms
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
    
    @abstractmethod
    def _load_samples(self):
        """Load dataset samples. Must be implemented by subclasses."""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get sample by index. Must be implemented by subclasses."""
        pass
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get number of samples per class."""
        counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for balanced sampling."""
        counts = self.get_class_counts()
        total = sum(counts.values())
        
        weights = []
        for _, label in self.samples:
            class_name = self.classes[label]
            weight = total / (len(counts) * counts[class_name])
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
