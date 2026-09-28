#!/usr/bin/env python3
"""Render five representative colored 3D quarter-shape previews."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "adaptive8_3d_quarter_shapes"
OUT = BASE / "preview_5_colored_3d_shapes"
COLORS = ("#e41a1c", "#377eb8", "#4daf4a", "#984ea3")


def draw_side(axis, labels: np.ndarray, lower: int, title: str) -> None:
    for quarter, color in enumerate(COLORS):
        xyz = np.argwhere(labels == lower + quarter)
        if len(xyz):
            axis.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=7, c=color, alpha=0.80, depthshade=False,
                         label=f"quarter {quarter + 1}")
    axis.set_title(title)
    axis.set_xlabel("MNI x voxel"); axis.set_ylabel("MNI y voxel"); axis.set_zlabel("axial z")
    axis.view_init(elev=22, azim=-62)
    axis.legend(loc="upper right", fontsize=8)
    axis.set_box_aspect((1.0, 1.0, 0.55))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(BASE / "shape_audit.csv", dtype={"uid": str}).sort_values("left_shape_voxels")
    nonempty = audit.loc[(audit.left_shape_voxels > 0) & (audit.right_shape_voxels > 0)]
    selected = nonempty.iloc[np.linspace(0, len(nonempty) - 1, 5, dtype=int)]
    for row in selected.itertuples(index=False):
        labels = np.asarray(nib.load(str(BASE / "labels" / f"{row.uid}_adaptive8_3d_quarters.nii.gz")).dataobj)
        fig = plt.figure(figsize=(12, 5.7), facecolor="white")
        draw_side(fig.add_subplot(1, 2, 1, projection="3d"), labels, 1, f"{row.uid} — left 3D shape")
        draw_side(fig.add_subplot(1, 2, 2, projection="3d"), labels, 5, f"{row.uid} — right 3D shape")
        fig.suptitle("Eight-slice stacked contours, divided at each side's 3D center", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(OUT / f"{row.uid}_adaptive8_3d_quarters.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(row.uid, flush=True)


if __name__ == "__main__":
    main()
