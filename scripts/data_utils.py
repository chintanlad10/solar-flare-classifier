"""Shared data-loading and feature-selection helpers for SWAN-SF notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


SELECTED_FEATURES = ("SAVNCPP", "ABSNJZH", "R_VALUE", "EPSZ", "TOTFZ")


def find_project_root(start: Path | None = None) -> Path:
    """Find project root by locating src/sdp from the current path upward."""
    start_path = (start or Path.cwd()).resolve()
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "src" / "sdp").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate project root containing 'src/sdp' starting from {start_path}"
    )


def ensure_src_on_path(project_root: Path | None = None) -> Path:
    """Add src directory to sys.path so sdp package imports work in notebooks/scripts."""
    root = project_root or find_project_root()
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return src_path


def get_prepared_dir(project_root: Path | None = None) -> Path:
    root = project_root or find_project_root()
    return root / "data" / "prepared"


def get_cleaned_dataset_dir(project_root: Path | None = None) -> Path:
    root = project_root or find_project_root()
    return root / "data" / "Cleaned SWANSF Dataset"


def load_tabular_splits(prepared_dir: Path | None = None):
    """Load train/validation/test tabular CSVs used by tabular baselines."""
    base_dir = prepared_dir or get_prepared_dir()
    train_df = pd.read_csv(base_dir / "swansf_tabular_train.csv")
    val_df = pd.read_csv(base_dir / "swansf_tabular_val.csv")
    test_df = pd.read_csv(base_dir / "swansf_tabular_test.csv")
    return train_df, val_df, test_df


def split_tabular_features_labels(
    df: pd.DataFrame,
    selected_features: Sequence[str] = SELECTED_FEATURES,
    label_col: str = "label",
):
    """Split tabular dataframe into X/y using the selected base feature names."""
    base_set = set(selected_features)
    feature_cols = [
        c for c in df.columns if c != label_col and c.split("_")[0] in base_set
    ]
    X = df[feature_cols].values
    y = df[label_col].values
    return X, y, feature_cols


def split_features_labels(
    df: pd.DataFrame,
    selected_features: Sequence[str] = SELECTED_FEATURES,
    label_col: str = "label",
):
    """Notebook-friendly wrapper that returns only X and y."""
    X, y, _ = split_tabular_features_labels(
        df=df,
        selected_features=selected_features,
        label_col=label_col,
    )
    return X, y


def get_selected_feature_indices(
    attributes: Sequence[str],
    selected_features: Sequence[str] = SELECTED_FEATURES,
) -> list[int]:
    """Map selected feature names to their index positions in ATTRIBUTES."""
    return [attributes.index(name) for name in selected_features]


def slice_sequence_features(parts: Iterable, selected_indices: Sequence[int]):
    """Apply feature-column slicing to each sequence partition array."""
    return [part[..., selected_indices] for part in parts]


def select_features(parts: Iterable, selected_indices: Sequence[int]):
    """Helper to select only required feature columns from sequence partitions."""
    return [part[..., selected_indices] for part in parts]


def load_sequence_data(data_dir, selected_features: Sequence[str] = SELECTED_FEATURES):
    """Load SWAN-SF sequence data and return train/val/test arrays.

    Steps:
    1. Load all train/test partitions.
    2. Select only requested features.
    3. Split validation from the last train partition.
    4. Concatenate remaining train partitions and all test partitions.
    """
    ensure_src_on_path()
    from sdp.data_loader import ATTRIBUTES, concat_partitions, load_swansf_partitions

    X_train_parts, y_train_parts = load_swansf_partitions(
        str(data_dir), split="train", num_partitions=5
    )
    X_test_parts, y_test_parts = load_swansf_partitions(
        str(data_dir), split="test", num_partitions=5
    )

    selected_indices = get_selected_feature_indices(ATTRIBUTES, selected_features)
    X_train_parts = select_features(X_train_parts, selected_indices)
    X_test_parts = select_features(X_test_parts, selected_indices)

    # Keep validation split identical to existing notebooks: last train partition.
    X_val_all = X_train_parts.pop(-1)
    y_val_all = y_train_parts.pop(-1)

    X_train_all, y_train_all = concat_partitions(X_train_parts, y_train_parts)
    X_test_all, y_test_all = concat_partitions(X_test_parts, y_test_parts)

    return (
        X_train_all,
        y_train_all,
        X_val_all,
        y_val_all,
        X_test_all,
        y_test_all,
    )


def load_sequence_splits(
    base_dir: Path | None = None,
    num_train_partitions: int = 5,
    num_test_partitions: int = 5,
    use_last_train_partition_for_validation: bool = True,
    selected_features: Sequence[str] = SELECTED_FEATURES,
):
    """Load SWAN-SF sequence partitions and return train/val/test arrays.

    This wraps load_swansf_partitions + concat_partitions used in sequence notebooks.
    """
    ensure_src_on_path()
    from sdp.data_loader import ATTRIBUTES, concat_partitions, load_swansf_partitions

    dataset_dir = base_dir or get_cleaned_dataset_dir()

    X_train_parts, y_train_parts = load_swansf_partitions(
        str(dataset_dir), split="train", num_partitions=num_train_partitions
    )
    X_test_parts, y_test_parts = load_swansf_partitions(
        str(dataset_dir), split="test", num_partitions=num_test_partitions
    )

    selected_idx = get_selected_feature_indices(ATTRIBUTES, selected_features)
    X_train_parts = slice_sequence_features(X_train_parts, selected_idx)
    X_test_parts = slice_sequence_features(X_test_parts, selected_idx)

    if use_last_train_partition_for_validation:
        X_val = X_train_parts.pop(-1)
        y_val = y_train_parts.pop(-1)
    else:
        X_val = None
        y_val = None

    X_train, y_train = concat_partitions(X_train_parts, y_train_parts)

    if num_test_partitions == 1:
        X_test = X_test_parts[0]
        y_test = y_test_parts[0]
    else:
        X_test, y_test = concat_partitions(X_test_parts, y_test_parts)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "selected_features": tuple(selected_features),
        "selected_indices": selected_idx,
        "attributes": ATTRIBUTES,
    }
