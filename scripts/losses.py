"""Loss functions for handling class imbalance in binary classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Simple focal loss for binary classification with logits input.

    This is designed to be a drop-in replacement for BCEWithLogitsLoss
    in existing training loops.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Make sure targets have the same dtype as logits for BCE computation.
        targets = targets.float()

        # Compute standard BCE loss per sample (no reduction yet).
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Convert logits to probabilities.
        probs = torch.sigmoid(logits)

        # pt is the probability assigned to the true class.
        # If target=1, pt=probs. If target=0, pt=1-probs.
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)

        # Alpha weighting for class imbalance.
        # If target=1, use alpha. If target=0, use (1-alpha).
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        # Focal term reduces loss from easy examples and focuses on hard examples.
        focal_weight = (1.0 - pt) ** self.gamma

        # Final focal loss per sample.
        loss = alpha_factor * focal_weight * bce_loss

        # Return mean loss so it matches common training-loop expectations.
        return loss.mean()


def get_weighted_bce_loss(pos_weight: float) -> nn.Module:
    """Create a weighted BCEWithLogitsLoss for comparison with focal loss.

    Args:
        pos_weight: Extra weight for positive class examples.

    Returns:
        nn.BCEWithLogitsLoss instance.
    """
    weight_tensor = torch.tensor(float(pos_weight), dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)


# Usage example:
# from losses import FocalLoss
# criterion = FocalLoss(alpha=0.75, gamma=2.0)
#
# # Replace BCEWithLogitsLoss in your training code:
# # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=...)
# # becomes:
# # criterion = FocalLoss(alpha=0.75, gamma=2.0)
#
# # Optional weighted BCE helper for comparison:
# # from losses import get_weighted_bce_loss
# # criterion_bce = get_weighted_bce_loss(pos_weight=3.5)
