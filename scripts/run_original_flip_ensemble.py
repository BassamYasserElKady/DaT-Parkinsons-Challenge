from __future__ import annotations

import torch
from torch.utils.data import Dataset

import evaluate_holdout_3d_ensemble as base


class FlipDataset(Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index):
        x, y = self.dataset[index]
        if torch.rand(()) < 0.5: x = x.flip(-1)
        return x, y


def main():
    original_loader = base.DataLoader
    def loader(dataset, *args, **kwargs):
        if kwargs.get("shuffle", False): dataset = FlipDataset(dataset)
        return original_loader(dataset, *args, **kwargs)
    base.DataLoader = loader
    base.OUTPUT = base.ROOT / "results" / "holdout_900_120_342_3d_ensemble_original_flip"
    try:
        base.main()
    finally:
        base.DataLoader = original_loader


if __name__ == "__main__": main()
