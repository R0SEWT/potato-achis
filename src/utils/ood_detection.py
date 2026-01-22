"""
OOD Detection
=============
Out-of-Distribution detection for open-set recognition.
Essential for Andean field deployment with unknown disease classes.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OODDetector:
    """
    Out-of-Distribution detector for open-set recognition.
    
    Supports multiple detection methods:
    - Maximum Softmax Probability (MSP)
    - Entropy-based detection
    - Energy-based detection
    - Mahalanobis distance (requires training statistics)
    
    Args:
        model: Trained classification model
        method: Detection method ('msp', 'entropy', 'energy', 'mahalanobis')
        threshold: Rejection threshold (samples below are OOD)
    """
    
    def __init__(
        self,
        model: nn.Module,
        method: str = "msp",
        threshold: Optional[float] = None,
    ):
        self.model = model
        self.method = method
        self.threshold = threshold
        
        # For Mahalanobis distance
        self.class_means = None
        self.precision_matrix = None
    
    def compute_scores(
        self,
        x: torch.Tensor,
        return_preds: bool = False,
    ) -> torch.Tensor:
        """
        Compute OOD scores for input samples.
        
        Higher scores = more likely to be in-distribution.
        
        Args:
            x: Input images (B, C, H, W)
            return_preds: Also return class predictions
            
        Returns:
            OOD scores (B,), optionally (scores, predictions)
        """
        self.model.eval()
        
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                logits = self.model(x)
            else:
                logits = self.model(x)
        
        if self.method == "msp":
            scores = self._msp_score(logits)
        elif self.method == "entropy":
            scores = self._entropy_score(logits)
        elif self.method == "energy":
            scores = self._energy_score(logits)
        elif self.method == "mahalanobis":
            # Requires features, not logits
            features = self._extract_features(x)
            scores = self._mahalanobis_score(features)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if return_preds:
            preds = logits.argmax(dim=1)
            return scores, preds
        
        return scores
    
    def _msp_score(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Maximum Softmax Probability score.
        
        Score = max(softmax(logits))
        """
        probs = F.softmax(logits, dim=1)
        scores, _ = probs.max(dim=1)
        return scores
    
    def _entropy_score(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Entropy-based score (negative entropy).
        
        High entropy = uncertain = likely OOD
        Score = -entropy (higher = more confident)
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)
        
        # Convert to confidence (negative entropy)
        # Normalize to [0, 1] range
        max_entropy = np.log(logits.size(1))  # log(num_classes)
        scores = 1 - (entropy / max_entropy)
        
        return scores
    
    def _energy_score(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Energy-based score.
        
        Score = -temperature * logsumexp(logits / temperature)
        Higher energy (less negative) = more likely in-distribution
        """
        energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
        
        # Convert to positive score (negate energy)
        scores = -energy
        
        return scores
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from model for Mahalanobis distance."""
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'extract_features'):
                features = self.model.extract_features(x)
            elif hasattr(self.model, 'get_bottleneck_features'):
                features = self.model.get_bottleneck_features(x)
            else:
                # Fall back to forward with features
                logits, features = self.model(x, return_features=True)
        return features
    
    def _mahalanobis_score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Mahalanobis distance-based score.
        
        Requires fit_mahalanobis() to be called first.
        """
        if self.class_means is None or self.precision_matrix is None:
            raise RuntimeError("Call fit_mahalanobis() first")
        
        device = features.device
        class_means = self.class_means.to(device)
        precision = self.precision_matrix.to(device)
        
        # Compute Mahalanobis distance to each class
        distances = []
        for mean in class_means:
            diff = features - mean.unsqueeze(0)
            dist = torch.sum(diff @ precision * diff, dim=1)
            distances.append(dist)
        
        # Take minimum distance (closest class)
        distances = torch.stack(distances, dim=1)
        min_distances, _ = distances.min(dim=1)
        
        # Convert to score (negative distance)
        scores = -min_distances
        
        return scores
    
    def fit_mahalanobis(
        self,
        dataloader,
        num_classes: int,
    ):
        """
        Fit Mahalanobis distance parameters from training data.
        
        Args:
            dataloader: Training data loader
            num_classes: Number of classes
        """
        self.model.eval()
        
        # Collect features per class
        class_features = [[] for _ in range(num_classes)]
        
        with torch.no_grad():
            for images, labels in dataloader:
                device = next(self.model.parameters()).device
                images = images.to(device)
                labels = labels.to(device)
                
                features = self._extract_features(images)
                
                for feat, label in zip(features, labels):
                    if label >= 0:  # Ignore unlabeled
                        class_features[label.item()].append(feat.cpu())
        
        # Compute class means
        class_means = []
        for feats in class_features:
            if feats:
                class_means.append(torch.stack(feats).mean(dim=0))
            else:
                class_means.append(torch.zeros_like(class_features[0][0]))
        
        self.class_means = torch.stack(class_means)
        
        # Compute shared covariance and precision matrix
        all_features = []
        all_means = []
        for i, feats in enumerate(class_features):
            if feats:
                all_features.extend(feats)
                all_means.extend([class_means[i]] * len(feats))
        
        all_features = torch.stack(all_features)
        all_means = torch.stack(all_means)
        
        centered = all_features - all_means
        cov = centered.T @ centered / (centered.size(0) - 1)
        
        # Regularize and invert
        cov += 1e-5 * torch.eye(cov.size(0))
        self.precision_matrix = torch.linalg.inv(cov)
    
    def predict_with_rejection(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with OOD rejection.
        
        Args:
            x: Input images
            threshold: Rejection threshold (uses self.threshold if None)
            
        Returns:
            Tuple of (predictions, is_ood, scores)
            - predictions: Class predictions (B,)
            - is_ood: Boolean mask for OOD samples (B,)
            - scores: OOD scores (B,)
        """
        if threshold is None:
            threshold = self.threshold
        if threshold is None:
            raise ValueError("Threshold must be provided")
        
        scores, predictions = self.compute_scores(x, return_preds=True)
        is_ood = scores < threshold
        
        return predictions, is_ood, scores
    
    def find_threshold(
        self,
        in_dist_loader,
        ood_loader,
        target_fpr: float = 0.05,
    ) -> float:
        """
        Find optimal threshold given in-distribution and OOD data.
        
        Args:
            in_dist_loader: In-distribution data loader
            ood_loader: OOD data loader
            target_fpr: Target false positive rate on in-distribution
            
        Returns:
            Optimal threshold
        """
        # Collect in-distribution scores
        in_scores = []
        for images, _ in in_dist_loader:
            device = next(self.model.parameters()).device
            images = images.to(device)
            scores = self.compute_scores(images)
            in_scores.append(scores.cpu())
        
        in_scores = torch.cat(in_scores).numpy()
        
        # Find threshold at target FPR
        threshold = np.percentile(in_scores, target_fpr * 100)
        self.threshold = float(threshold)
        
        return self.threshold


def compute_ood_metrics(
    in_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Compute OOD detection metrics.
    
    Args:
        in_scores: Scores for in-distribution samples
        ood_scores: Scores for OOD samples
        
    Returns:
        Dictionary with AUROC, AUPR, FPR@95TPR
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    
    # Labels: 1 = in-distribution, 0 = OOD
    labels = np.concatenate([
        np.ones(len(in_scores)),
        np.zeros(len(ood_scores)),
    ])
    scores = np.concatenate([in_scores, ood_scores])
    
    # AUROC
    auroc = roc_auc_score(labels, scores)
    
    # AUPR
    aupr = average_precision_score(labels, scores)
    
    # FPR at 95% TPR
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95tpr = fpr[idx]
    
    return {
        'auroc': auroc,
        'aupr': aupr,
        'fpr@95tpr': fpr_at_95tpr,
    }
