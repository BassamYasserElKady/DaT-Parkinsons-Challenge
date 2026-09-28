#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
UID = "bmae38sr"
img = nib.load(str(ROOT / "results/ants_registration_mni" / f"{UID}.nii.gz"))
volume = img.get_fdata().astype(np.float32)
positive = volume[np.isfinite(volume) & (volume > 0)]
cutoff = float(np.percentile(positive, 99.75))
hot = np.isfinite(volume) & (volume >= cutoff)
coords = np.indices(volume.shape)
mid = volume.shape[0] / 2.0
labels = np.zeros(volume.shape, dtype=np.uint8)

for selector in (coords[0] < mid, coords[0] >= mid):
    points = np.argwhere(hot & selector)
    if not len(points):
        continue
    y = points[:, 1]
    edges = np.linspace(y.min(), y.max() + 1, 11)
    for band in range(10):
        chosen = points[(y >= edges[band]) & (y < edges[band + 1])]
        labels[tuple(chosen.T)] = band + 1

z = int(np.argmax(hot.sum(axis=(0, 1))))
base = np.rot90(volume[:, :, z].T)
label_slice = np.rot90(labels[:, :, z].T)
vmax = float(np.percentile(positive, 99.5))
colors = ["#ff3333", "#ff7f0e", "#ffd700", "#7fff00", "#00cc66",
          "#00cccc", "#3388ff", "#6633cc", "#aa33cc", "#ff33aa"]

fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
ax.imshow(base, cmap="gray", vmin=0, vmax=vmax)
for band in range(1, 11):
    region = label_slice == band
    if region.any():
        ax.contour(region, levels=[0.5], colors=colors[band - 1], linewidths=1.7)
ax.axvline(volume.shape[0] / 2, color="white", linestyle="--", linewidth=1)
ax.set_title(
    f"{UID}: top 0.25% voxels divided into 10 horizontal bands per side\n"
    f"cutoff={cutoff:.6g}, hotspot voxels={int(hot.sum())}, axial z={z}"
)
ax.axis("off")
out = ROOT / "results" / f"{UID}_top025_10_horizontal_bands.png"
fig.savefig(out, dpi=180)
plt.close(fig)
print(out)
