from __future__ import annotations

import os
import pickle
from typing import List, Tuple, Literal, Optional

import numpy as np

# Attribute order as provided by the dataset documentation
ATTRIBUTES: Tuple[str, ...] = (
    'R_VALUE', 'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP',
    'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSX', 'EPSY', 'EPSZ', 'MEANSHR', 'SHRGT45',
    'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD',
    'MEANALP', 'TOTFX',
)

Split = Literal['train', 'test']

TRAIN_DATA_FMT = 'Partition{idx}_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl'
TRAIN_LABELS_FMT = 'Partition{idx}_Labels_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl'

TEST_DATA_FMT = 'Partition{idx}_LSBZM-Norm_FPCKNN-impute.pkl'
TEST_LABELS_FMT = 'Partition{idx}_Labels_LSBZM-Norm_FPCKNN-impute.pkl'


def _load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_swansf_partitions(
    base_dir: str,
    split: Split,
    num_partitions: int = 5,
    strict: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load SWAN-SF cleaned dataset partitions.

    Returns lists of arrays: X_parts, y_parts, each length=num_partitions.
    X shape per partition: (n_samples, n_timestamps, n_attributes)
    y shape per partition: (n_samples,)
    """
    split_dir = os.path.join(base_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for i in range(1, num_partitions + 1):
        if split == 'train':
            data_fname = TRAIN_DATA_FMT.format(idx=i)
            label_fname = TRAIN_LABELS_FMT.format(idx=i)
        else:
            data_fname = TEST_DATA_FMT.format(idx=i)
            label_fname = TEST_LABELS_FMT.format(idx=i)

        data_path = os.path.join(split_dir, data_fname)
        label_path = os.path.join(split_dir, label_fname)

        if strict and (not os.path.exists(data_path) or not os.path.exists(label_path)):
            raise FileNotFoundError(f"Missing partition {i} files under {split_dir}")

        X = _load_pickle(data_path)
        y = _load_pickle(label_path)

        # Basic validation
        if X.ndim != 3:
            raise ValueError(f"Partition {i} X should be 3D, got {X.ndim}D shape {X.shape}")
        if X.shape[2] != len(ATTRIBUTES):
            raise ValueError(
                f"Partition {i} attributes mismatch: X.shape[2]={X.shape[2]}, expected {len(ATTRIBUTES)}"
            )
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(
                f"Partition {i} labels mismatch: y.shape={y.shape}, X.shape={X.shape}"
            )

        X_list.append(X)
        y_list.append(y)

    return X_list, y_list


def concat_partitions(
    X_parts: List[np.ndarray], y_parts: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate partitions along the sample axis."""
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y


def summarize_partitions(X_parts: List[np.ndarray], y_parts: List[np.ndarray]) -> str:
    lines = []
    total = 0
    for i, (X, y) in enumerate(zip(X_parts, y_parts), start=1):
        n, t, a = X.shape
        total += n
        pos = int((y == 1).sum())
        neg = n - pos
        lines.append(
            f"Partition {i}: X={X.shape}, y={y.shape}, pos={pos}, neg={neg}"
        )
    lines.append(f"Total samples across partitions: {total}")
    return "\n".join(lines)


def get_attribute_index(name: str) -> Optional[int]:
    try:
        return ATTRIBUTES.index(name)
    except ValueError:
        return None
