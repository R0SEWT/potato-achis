"""Classifier Alignment Module.

Implements an optional MDFAN Stage 2 loss that aligns the *predicted class
probabilities* across source-specific classifiers on target images.

This is a simple pairwise discrepancy loss (L1/L2 or symmetric KL) that can be
enabled during training via `--lambda_align`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierAlignment(nn.Module):
    """
    Classifier alignment for multi-source domain adaptation.

    Minimizes discrepancy between classifier predictions on target domain.
    This helps align decision boundaries across source-specific classifiers.

    Args:
        num_sources: Number of source classifiers
        loss_type: Type of alignment loss ('l1', 'l2', 'kl')
    """

    def __init__(
        self,
        num_sources: int = 2,
        loss_type: str = "l1",
    ):
        super().__init__()
        if num_sources < 1:
            raise ValueError("num_sources must be >= 1")
        self.num_sources = num_sources
        self.loss_type = loss_type

    @staticmethod
    def _safe_probs(probs: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(probs.dtype).eps
        safe = probs.clamp_min(eps)
        return safe / safe.sum(dim=-1, keepdim=True)

    def forward(
        self,
        predictions_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute classifier alignment loss.

        Args:
            predictions_list: List of softmax outputs from each source classifier
                              Shape: [(B, C), (B, C), ...]

        Returns:
            Alignment loss scalar
        """
        if len(predictions_list) != self.num_sources:
            raise ValueError(
                f"Expected {self.num_sources} prediction tensors, got {len(predictions_list)}"
            )

        if len(predictions_list) < 2:
            return predictions_list[0].new_tensor(0.0)

        total_loss = predictions_list[0].new_tensor(0.0)
        num_pairs = 0

        # Compute pairwise loss between all classifier pairs
        for i in range(len(predictions_list)):
            for j in range(i + 1, len(predictions_list)):
                pred_i = predictions_list[i]
                pred_j = predictions_list[j]

                if self.loss_type == "l1":
                    loss = F.l1_loss(pred_i, pred_j)
                elif self.loss_type == "l2":
                    loss = F.mse_loss(pred_i, pred_j)
                elif self.loss_type == "kl":
                    # Symmetric KL divergence
                    safe_pred_i = self._safe_probs(pred_i)
                    safe_pred_j = self._safe_probs(pred_j)
                    kl_ij = F.kl_div(
                        safe_pred_i.log(),
                        safe_pred_j,
                        reduction="batchmean",
                    )
                    kl_ji = F.kl_div(
                        safe_pred_j.log(),
                        safe_pred_i,
                        reduction="batchmean",
                    )
                    loss = (kl_ij + kl_ji) / 2
                else:
                    loss = F.l1_loss(pred_i, pred_j)

                total_loss += loss
                num_pairs += 1

        return total_loss / num_pairs if num_pairs > 0 else total_loss


# Future improvements:
# - Maximum Classifier Discrepancy (MCD) approach
# - Entropy minimization on target
# - Pseudo-label refinement
