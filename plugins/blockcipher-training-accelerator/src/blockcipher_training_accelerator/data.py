from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from blockcipher_nd.data.differential import DifferentialDataset, DiskDifferentialDataset

from blockcipher_training_accelerator.profiles import SpeedProfile


def make_accelerated_loader(
    dataset: DifferentialDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    profile: SpeedProfile,
    device: torch.device,
) -> DataLoader:
    if isinstance(dataset, DiskDifferentialDataset):
        torch_dataset: Dataset = DiskDifferentialTorchDataset(dataset)
    else:
        features = torch.tensor(dataset.features, dtype=torch.float32)
        labels = torch.tensor(dataset.labels, dtype=torch.float32)
        torch_dataset = TensorDataset(features, labels)

    workers = max(0, int(profile.dataloader_workers))
    pin_memory = bool(profile.pin_memory and device.type == "cuda")
    persistent_workers = bool(profile.persistent_workers and workers > 0)
    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": torch.Generator().manual_seed(seed) if shuffle else None,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if workers > 0 and profile.prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(profile.prefetch_factor)
    return DataLoader(torch_dataset, **kwargs)


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
