"""
MMD Loss
========
Maximum Mean Discrepancy loss for domain alignment.
Measures distance between source and target feature distributions.

Reference:
    Gretton et al., "A Kernel Two-Sample Test" (2012)
"""


import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) loss with multiple kernels.
    
    Computes the empirical estimate of MMD between two distributions
    using Gaussian RBF kernels with multiple bandwidths.
    
    Args:
        kernel_type: Type of kernel ('rbf', 'linear')
        kernel_mul: Multiplier for kernel bandwidth
        kernel_num: Number of kernels (for multi-kernel MMD)
        fix_sigma: Fixed sigma value (None for automatic)
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: float | None = None,
    ):
        super().__init__()

        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MMD loss between source and target features.
        
        Args:
            source: Source domain features (N_s, D)
            target: Target domain features (N_t, D)
            
        Returns:
            MMD loss scalar
        """
        if self.kernel_type == "linear":
            return self._linear_mmd(source, target)
        elif self.kernel_type == "rbf":
            return self._rbf_mmd(source, target)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def _linear_mmd(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Linear kernel MMD (faster but less expressive)."""
        delta = source.mean(dim=0) - target.mean(dim=0)
        return torch.sum(delta * delta)

    def _rbf_mmd(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian RBF kernel MMD with multiple bandwidths."""
        batch_size_s = source.size(0)
        batch_size_t = target.size(0)

        # Concatenate source and target
        total = torch.cat([source, target], dim=0)

        # Compute pairwise L2 distances
        total_size = total.size(0)
        total0 = total.unsqueeze(0).expand(total_size, total_size, -1)
        total1 = total.unsqueeze(1).expand(total_size, total_size, -1)
        L2_distance = ((total0 - total1) ** 2).sum(dim=2)

        # Compute bandwidth
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance) / (total_size ** 2 - total_size)

        # Compute multi-kernel
        bandwidth_list = [
            bandwidth * (self.kernel_mul ** (i - self.kernel_num // 2))
            for i in range(self.kernel_num)
        ]

        kernel_val = sum([
            torch.exp(-L2_distance / (bw + 1e-8))
            for bw in bandwidth_list
        ])

        # Extract kernel matrices
        XX = kernel_val[:batch_size_s, :batch_size_s]
        YY = kernel_val[batch_size_s:, batch_size_s:]
        XY = kernel_val[:batch_size_s, batch_size_s:]

        # Compute MMD
        loss = XX.mean() + YY.mean() - 2 * XY.mean()

        return loss


class LocalMMDLoss(nn.Module):
    """
    Local MMD (LMMD) loss for subdomain/class-conditional alignment.
    
    Aligns class-conditional distributions (subdomains) between 
    source and target domains.
    
    Args:
        num_classes: Number of classes
        kernel_type: Type of kernel ('rbf', 'linear')
        kernel_mul: Multiplier for kernel bandwidth
        kernel_num: Number of kernels
    """

    def __init__(
        self,
        num_classes: int,
        kernel_type: str = "rbf",
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.mmd = MMDLoss(
            kernel_type=kernel_type,
            kernel_mul=kernel_mul,
            kernel_num=kernel_num,
        )

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute LMMD loss with class-conditional alignment.
        
        Args:
            source_features: Source features (N_s, D)
            target_features: Target features (N_t, D)
            source_labels: Source class labels (N_s,)
            target_pseudo_labels: Target pseudo-labels (N_t,)
            
        Returns:
            LMMD loss scalar
        """
        loss = torch.tensor(0.0, device=source_features.device)
        num_valid_classes = 0

        for c in range(self.num_classes):
            source_mask = (source_labels == c)
            target_mask = (target_pseudo_labels == c)

            if source_mask.sum() > 0 and target_mask.sum() > 0:
                source_c = source_features[source_mask]
                target_c = target_features[target_mask]

                loss += self.mmd(source_c, target_c)
                num_valid_classes += 1

        if num_valid_classes > 0:
            loss /= num_valid_classes

        return loss


class MultiSourceMMDLoss(nn.Module):
    """
    Multi-source MMD loss for MDFAN.
    
    Computes MMD loss between each source domain and target domain.
    
    Args:
        num_sources: Number of source domains
        kernel_type: Type of kernel
        kernel_mul: Kernel bandwidth multiplier
        kernel_num: Number of kernels
    """

    def __init__(
        self,
        num_sources: int,
        kernel_type: str = "rbf",
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
    ):
        super().__init__()

        self.num_sources = num_sources
        self.mmd = MMDLoss(
            kernel_type=kernel_type,
            kernel_mul=kernel_mul,
            kernel_num=kernel_num,
        )

    def forward(
        self,
        source_features_list: list[torch.Tensor],
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute average MMD loss across all source-target pairs.
        
        Args:
            source_features_list: List of source features [(N_s1, D), ...]
            target_features: Target features (N_t, D)
            
        Returns:
            Average MMD loss
        """
        total_loss = torch.tensor(0.0, device=target_features.device)

        for source_features in source_features_list:
            total_loss += self.mmd(source_features, target_features)

        return total_loss / len(source_features_list)
