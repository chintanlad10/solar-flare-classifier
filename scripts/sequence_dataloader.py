"""Reusable dataloaders for full-sequence LSTM/BiLSTM notebooks."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class SequenceDataset(Dataset):
    """Simple dataset that returns full sequences and binary labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.X[idx])
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return seq, label


def pad_collate(batch):
    """Pad variable-length sequences in a batch and return lengths."""

    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    return padded, labels, lengths


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
):
    """Create train/validation/test dataloaders with the standard collate fn."""

    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    test_dataset = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    return train_loader, val_loader, test_loader
