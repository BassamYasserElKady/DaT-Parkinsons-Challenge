from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import evaluate_holdout_3d_ensemble as base


CONFIGS = {
    "baseline": (False, False, False),
    "flip": (True, False, False),
    "rotation": (False, True, False),
    "brightness": (False, False, True),
    "flip_rotation": (True, True, False),
    "flip_brightness": (True, False, True),
    "rotation_brightness": (False, True, True),
    "all_augmentation": (True, True, True),
}
ACTIVE = (True, True, True)


class AugmentedDataset(Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index):
        x, y = self.dataset[index]
        flip, rotate, brightness = ACTIVE
        if flip and torch.rand(()) < 0.5: x = x.flip(-1)
        if rotate:
            angle = (torch.rand(()) * 20 - 10) * math.pi / 180
            c, s = torch.cos(angle), torch.sin(angle)
            theta = torch.stack((torch.stack((c, -s, c * 0, c * 0)), torch.stack((s, c, c * 0, c * 0)), torch.tensor([0., 0., 1., 0.]))).to(x)
            grid = F.affine_grid(theta.unsqueeze(0), x.unsqueeze(0).shape, align_corners=False)
            x = F.grid_sample(x.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(0)
        if brightness: x = x * (0.1 + 1.4 * torch.rand(()))
        return x, y


def main():
    global ACTIVE
    original_loader, original_train = base.DataLoader, base.train_model
    def loader(dataset, *args, **kwargs):
        if kwargs.get("shuffle", False) and any(ACTIVE): dataset = AugmentedDataset(dataset)
        return original_loader(dataset, *args, **kwargs)
    def train(*args, **kwargs):
        args = list(args); args[5] = 20
        return original_train(*args, **kwargs)
    base.DataLoader, base.train_model = loader, train
    results = []
    try:
        for name, flags in CONFIGS.items():
            ACTIVE = flags
            base.OUTPUT = base.ROOT / "results" / "ablation_screen" / name
            print(f"START {name}", flush=True)
            base.main()
            metrics = __import__("json").loads((base.OUTPUT / "metrics.json").read_text())
            results.append({"configuration": name, "validation_log_loss": metrics["validation_raw"]["log_loss"], "validation_auroc": metrics["validation_raw"]["auroc"]})
            __import__("pandas").DataFrame(results).to_csv(base.ROOT / "results" / "ablation_screen" / "summary.csv", index=False)
    finally:
        base.DataLoader, base.train_model = original_loader, original_train


if __name__ == "__main__": main()
