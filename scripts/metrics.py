"""Reusable metric utilities for tabular and sequence model notebooks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def heidke_skill_score(tn, fp, fn, tp):
    """Heidke Skill Score (HSS) with safe zero-division handling."""
    denom = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    if denom == 0:
        return 0.0
    return (2.0 * (tp * tn - fp * fn)) / denom


def _safe_div(num, den):
    return num / den if den > 0 else 0.0


def compute_metrics(y_true, y_score, threshold=0.5):
    """Compute common binary classification metrics from scores + threshold."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).ravel()
    y_pred = (y_score >= threshold).astype(int)

    has_two_classes = len(np.unique(y_true)) > 1
    roc_auc = roc_auc_score(y_true, y_score) if has_two_classes else 0.0
    pr_auc = average_precision_score(y_true, y_score) if has_two_classes else 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    tpr = recall
    fpr = _safe_div(fp, fp + tn)
    tss = tpr - fpr
    hss = heidke_skill_score(tn, fp, fn, tp)

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "tss": float(tss),
        "hss": float(hss),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_pred": y_pred,
    }


def sweep_thresholds(y_true, y_score, thresholds=None, max_fpr=None):
    """Sweep thresholds and return the best metric dict plus all threshold stats."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    all_stats = [compute_metrics(y_true, y_score, thr) for thr in thresholds]

    if max_fpr is None:
        candidates = all_stats
    else:
        candidates = [m for m in all_stats if m["fpr"] <= max_fpr]
        if not candidates:
            candidates = all_stats

    best = max(candidates, key=lambda m: m["tss"])
    return best, all_stats


def format_metrics(metrics: dict, decimals: int = 3) -> dict:
    """Round numeric values for clean printing while preserving arrays/other values."""
    formatted = {}
    for key, value in metrics.items():
        if key == "y_pred":
            continue
        try:
            formatted[key] = round(float(value), decimals)
        except (TypeError, ValueError):
            formatted[key] = value
    return formatted
