#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=("ants", "itk"), default="ants")
parser.add_argument("--count", type=int, default=8)
parser.add_argument("--groups", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
scan_dir = root / ("results/ants_registration_mni" if args.method == "ants" else "results/rigid_registration_mni")
paths = sorted(scan_dir.glob("*.nii.gz"))
rng = np.random.default_rng(args.seed)
selected = rng.choice(paths, size=args.count * args.groups, replace=False)
template = nib.load(str(root / "data/external/drive_inputs/template.nii")).get_fdata()
template = np.rot90(template[:, :, template.shape[2] // 2])
template /= max(template.max(), 1e-12)

args.output_dir.mkdir(parents=True, exist_ok=True)
for group in range(args.groups):
    fig, axes = plt.subplots(args.count, 3, figsize=(12, 3.5 * args.count))
    axes = np.atleast_2d(axes)
    for row, path in enumerate(selected[group * args.count : (group + 1) * args.count]):
        scan = nib.load(str(path)).get_fdata()
        scan = np.rot90(scan[:, :, scan.shape[2] // 2])
        scan /= max(scan.max(), 1e-12)
        overlay = np.stack((scan, template, np.zeros_like(scan)), axis=-1)
        for col, image, title in ((0, scan, f"{path.stem} scan"), (1, template, "Template"), (2, overlay, "Overlay")):
            axes[row, col].imshow(image, cmap=None if col == 2 else "gray")
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{args.method}_scan_group_{group + 1:02d}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
