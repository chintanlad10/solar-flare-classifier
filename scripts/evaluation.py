"""Evaluation helpers for threshold selection under FPR constraints."""

from __future__ import annotations

import numpy as np

from scripts.metrics import compute_metrics, sweep_thresholds


def best_tss_under_fpr(y_true, y_prob, thresholds=None, fpr_limit=0.10):
    """Pick threshold with highest TSS while keeping FPR <= fpr_limit.

    Falls back to threshold=0.5 if no threshold satisfies the constraint.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 33)

    best, _ = sweep_thresholds(y_true, y_prob, thresholds=thresholds, max_fpr=fpr_limit)

    if best is None:
        return compute_metrics(y_true, y_prob, threshold=0.5)

    return best


def select_threshold_on_validation(y_val, y_val_prob, thresholds=None, fpr_limit=0.10):
    """Validation threshold selection logic used across sequence notebooks."""
    return best_tss_under_fpr(
        y_true=y_val,
        y_prob=y_val_prob,
        thresholds=thresholds,
        fpr_limit=fpr_limit,
    )


def apply_validation_threshold_to_split(y_true, y_prob, validation_pick):
    """Evaluate any split using threshold selected from validation data."""
    threshold = float(validation_pick["threshold"])
    return compute_metrics(y_true, y_prob, threshold=threshold)


def best_test_sweep_under_fpr(y_test, y_test_prob, thresholds=None, fpr_limit=0.10):
    """Oracle-style best test score under FPR limit (analysis only, not deployment)."""
    return best_tss_under_fpr(
        y_true=y_test,
        y_prob=y_test_prob,
        thresholds=thresholds,
        fpr_limit=fpr_limit,
    )
