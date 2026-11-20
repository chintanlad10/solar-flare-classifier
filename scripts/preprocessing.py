"""Common preprocessing helpers for tabular and sequence SWAN-SF models."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def scale_tabular_data(X_train, X_val, X_test):
    """Fit StandardScaler on train and transform train/val/test splits."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def compute_feature_stats(X: np.ndarray, min_std: float = 1e-6):
    """Compute per-feature mean/std from sequence tensor (samples, time, features)."""
    flattened = X.reshape(-1, X.shape[-1])
    mean = flattened.mean(axis=0)
    std = flattened.std(axis=0)
    std = np.where(std < min_std, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_sequence_data(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Normalize sequence data with precomputed train statistics."""
    return ((X - mean) / std).astype(np.float32)


def normalize_sequence_splits(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    min_std: float = 1e-6,
):
    """Compute train stats once and normalize train/val/test without leakage."""
    mean, std = compute_feature_stats(X_train, min_std=min_std)
    X_train_n = normalize_sequence_data(X_train, mean, std)
    X_val_n = normalize_sequence_data(X_val, mean, std)
    X_test_n = normalize_sequence_data(X_test, mean, std)
    return X_train_n, X_val_n, X_test_n, mean, std
