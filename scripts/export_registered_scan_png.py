#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--uid", required=True)
parser.add_argument("--method", choices=("ants", "itk"), default="ants")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
scan_dir = root / ("results/ants_registration_mni" if args.method == "ants" else "results/rigid_registration_mni")
scan = nib.load(str(scan_dir / f"{args.uid}.nii.gz")).get_fdata()
template = nib.load(str(root / "data/external/drive_inputs/template.nii")).get_fdata()
z = scan.shape[2] // 2
scan = np.rot90(scan[:, :, z])
template = np.rot90(template[:, :, z])
scan = scan / max(scan.max(), 1e-12)
template = template / max(template.max(), 1e-12)
overlay = np.stack((scan, template, np.zeros_like(scan)), axis=-1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, image, title in zip(axes, (scan, template, overlay), ("Registered scan", "Template", "Overlay")):
    ax.imshow(image, cmap=None if title == "Overlay" else "gray")
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=160, bbox_inches="tight")
