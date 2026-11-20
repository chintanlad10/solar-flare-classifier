"""Reusable plotting helpers for model evaluation.

These functions are intentionally simple and notebook-friendly.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_roc_curve(y_true, y_scores):
    """Plot ROC curve (FPR vs TPR) for binary classification."""
    # Compute false-positive and true-positive rates from labels and scores.
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # Draw the ROC line and a random baseline for reference.
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pr_curve(y_true, y_scores):
    """Plot Precision-Recall curve (Recall vs Precision)."""
    # Compute precision and recall from labels and predicted scores.
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    # Draw the PR curve.
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label="PR Curve")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_training_loss(train_losses, val_losses=None):
    """Plot training loss over epochs and optional validation loss."""
    # Epoch numbers start at 1 for easier reading.
    train_epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(train_epochs, train_losses, label="Train Loss")

    # If validation losses are provided, plot them as well.
    if val_losses is not None:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Usage examples:
# from plotting import plot_roc_curve, plot_pr_curve, plot_training_loss
# plot_roc_curve(y_true, y_scores)
# plot_pr_curve(y_true, y_scores)
# plot_training_loss(train_losses, val_losses)
