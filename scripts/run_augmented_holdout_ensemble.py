from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import evaluate_holdout_3d_ensemble as base


class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if torch.rand(()) < 0.5:
            x = x.flip(-1)
        angle = (torch.rand(()) * 20.0 - 10.0) * math.pi / 180.0
        cosine, sine = torch.cos(angle), torch.sin(angle)
        theta = torch.tensor([[cosine, -sine, 0.0, 0.0], [sine, cosine, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=x.dtype)
        grid = F.affine_grid(theta.unsqueeze(0), x.unsqueeze(0).shape, align_corners=False)
        x = F.grid_sample(x.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(0)
        x = x * (0.1 + 1.4 * torch.rand(()))
        return x, y


def main():
    original_loader = base.DataLoader
    original_train = base.train_model
    original_adamw = torch.optim.AdamW
    original_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    def loader(dataset, *args, **kwargs):
        if kwargs.get("shuffle", False):
            dataset = AugmentedDataset(dataset)
        return original_loader(dataset, *args, **kwargs)

    def train(*args, **kwargs):
        args = list(args)
        args[5] = 50
        return original_train(*args, **kwargs)

    base.DataLoader = loader
    base.train_model = train
    torch.optim.AdamW = lambda params, lr, weight_decay: torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    torch.optim.lr_scheduler.CosineAnnealingLR = lambda optimizer, T_max: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)
    base.OUTPUT = base.ROOT / "results" / "holdout_900_120_342_3d_ensemble_augmented"
    try:
        base.main()
    finally:
        base.DataLoader = original_loader
        base.train_model = original_train
        torch.optim.AdamW = original_adamw
        torch.optim.lr_scheduler.CosineAnnealingLR = original_scheduler


if __name__ == "__main__":
    main()
