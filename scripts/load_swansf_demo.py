import os
import sys
from pathlib import Path
import numpy as np

# Ensure project src/ is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sdp.data_loader import load_swansf_partitions, concat_partitions, summarize_partitions, ATTRIBUTES

BASE_DIR = os.path.join('data', 'Cleaned SWANSF Dataset')

if __name__ == '__main__':
    # Load train partitions
    X_train_parts, y_train_parts = load_swansf_partitions(BASE_DIR, split='train', num_partitions=5)
    print('Train summary:')
    print(summarize_partitions(X_train_parts, y_train_parts))

    # Load test partitions
    X_test_parts, y_test_parts = load_swansf_partitions(BASE_DIR, split='test', num_partitions=5)
    print('\nTest summary:')
    print(summarize_partitions(X_test_parts, y_test_parts))

    # Concatenate (optional)
    X_train, y_train = concat_partitions(X_train_parts, y_train_parts)
    X_test, y_test = concat_partitions(X_test_parts, y_test_parts)

    print('\nConcatenated shapes:')
    print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
    print(f'X_test:  {X_test.shape}, y_test:  {y_test.shape}')

    # Example access: R_VALUE timeseries for first sample
    r_idx = ATTRIBUTES.index('R_VALUE')
    r_ts = X_train[0, :, r_idx]
    print(f"\nFirst sample R_VALUE series length: {len(r_ts)}; first 5 values: {r_ts[:5]}")
