from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from blockcipher_nd.data.differential import DifferentialDataset, DiskDifferentialDataset


def make_loader(
    dataset: DifferentialDataset,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader:
    if isinstance(dataset, DiskDifferentialDataset):
        torch_dataset: Dataset = DiskDifferentialTorchDataset(dataset)
    else:
        features = torch.tensor(dataset.features, dtype=torch.float32)
        labels = torch.tensor(dataset.labels, dtype=torch.float32)
        torch_dataset = TensorDataset(features, labels)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


class DiskDifferentialTorchDataset(Dataset):
    def __init__(self, dataset: DiskDifferentialDataset) -> None:
        self.features = dataset.features
        self.labels = dataset.labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature = torch.as_tensor(np.asarray(self.features[index]).copy(), dtype=torch.float32)
        label = torch.tensor(float(self.labels[index]), dtype=torch.float32)
        return feature, label


def select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
