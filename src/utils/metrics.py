"""
Evaluation Metrics
==================
Metrics for classification and domain adaptation evaluation.
"""


import numpy as np
import torch


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth labels
        
    Returns:
        Accuracy as a float (0-1)
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)

    correct = (predictions == targets).sum().item()
    total = targets.size(0)

    return correct / total if total > 0 else 0.0


def compute_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    average: str = "macro",
) -> float:
    """
    Compute F1 score.
    
    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth labels
        num_classes: Number of classes
        average: Averaging method ('macro', 'micro', 'weighted')
        
    Returns:
        F1 score
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)

    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # Compute per-class metrics
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)

    for c in range(num_classes):
        tp[c] = ((predictions == c) & (targets == c)).sum()
        fp[c] = ((predictions == c) & (targets != c)).sum()
        fn[c] = ((predictions != c) & (targets == c)).sum()

    # Compute precision and recall per class
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    # Compute F1 per class
    f1_per_class = 2 * precision * recall / (precision + recall + 1e-8)

    if average == "macro":
        return float(f1_per_class.mean())
    elif average == "micro":
        total_tp = tp.sum()
        total_fp = fp.sum()
        total_fn = fn.sum()
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        return float(2 * precision * recall / (precision + recall + 1e-8))
    elif average == "weighted":
        support = np.array([(targets == c).sum() for c in range(num_classes)])
        return float((f1_per_class * support).sum() / (support.sum() + 1e-8))
    else:
        raise ValueError(f"Unknown average method: {average}")


def compute_per_class_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute per-class accuracy.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        num_classes: Number of classes
        class_names: Optional class names
        
    Returns:
        Dictionary of class accuracies
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    accuracies = {}
    for c in range(num_classes):
        mask = (targets == c)
        if mask.sum() > 0:
            correct = ((predictions == c) & mask).sum().item()
            total = mask.sum().item()
            accuracies[class_names[c]] = correct / total
        else:
            accuracies[class_names[c]] = 0.0

    return accuracies


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        num_classes: Number of classes
        
    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)

    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, target in zip(predictions, targets):
        cm[target, pred] += 1

    return cm


class AverageMeter:
    """
    Computes and stores running average and current value.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class MetricTracker:
    """
    Track multiple metrics during training.
    """

    def __init__(self, metrics: list[str]):
        self.metrics = {name: AverageMeter(name) for name in metrics}

    def update(self, name: str, val: float, n: int = 1):
        if name in self.metrics:
            self.metrics[name].update(val, n)

    def reset(self):
        for meter in self.metrics.values():
            meter.reset()

    def get_averages(self) -> dict[str, float]:
        return {name: meter.avg for name, meter in self.metrics.items()}

    def __getitem__(self, name: str) -> AverageMeter:
        return self.metrics[name]


def compute_a_distance(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    device: str = "cuda",
) -> float:
    """
    Compute A-distance (proxy measure of domain discrepancy).
    
    Uses a simple linear classifier to distinguish domains.
    A-distance = 2 * (1 - 2 * error)
    
    Args:
        source_features: Source domain features (N_s, D)
        target_features: Target domain features (N_t, D)
        device: Computation device
        
    Returns:
        A-distance value (0 = identical, 2 = maximally different)
    """
    import torch.nn as nn
    import torch.optim as optim

    source_features = source_features.to(device)
    target_features = target_features.to(device)

    # Create labels
    source_labels = torch.zeros(source_features.size(0), device=device)
    target_labels = torch.ones(target_features.size(0), device=device)

    # Concatenate data
    features = torch.cat([source_features, target_features], dim=0)
    labels = torch.cat([source_labels, target_labels], dim=0)

    # Shuffle
    perm = torch.randperm(features.size(0))
    features = features[perm]
    labels = labels[perm]

    # Train simple classifier
    classifier = nn.Linear(features.size(1), 1).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    # Quick training
    for _ in range(100):
        optimizer.zero_grad()
        preds = classifier(features).squeeze()
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

    # Compute error
    with torch.no_grad():
        preds = torch.sigmoid(classifier(features).squeeze())
        preds = (preds > 0.5).float()
        error = (preds != labels).float().mean().item()

    # A-distance
    a_distance = 2 * (1 - 2 * error)
    return max(0, a_distance)  # Clamp to non-negative
