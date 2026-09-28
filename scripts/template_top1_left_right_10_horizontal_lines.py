#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
img = nib.load(str(ROOT/'submission_src_ensemble/assets/template.nii'))
a = img.get_fdata().astype(np.float32)
pos = a[np.isfinite(a) & (a > 0)]
cutoff = float(np.percentile(pos, 99.0))
hot = np.isfinite(a) & (a >= cutoff)
mid = a.shape[0] / 2.0
labels = np.zeros(a.shape, dtype=np.int16)
for side, selector in ((1, lambda x: x < mid), (2, lambda x: x >= mid)):
    c = np.argwhere(hot & selector(np.indices(a.shape)[0]))
    if len(c):
        ys = c[:,1]; edges = np.linspace(ys.min(), ys.max()+1, 11)
        for band in range(10):
            pick = c[(ys >= edges[band]) & (ys < edges[band+1])]
            labels[pick[:,0], pick[:,1], pick[:,2]] = side * (band + 1)

z = int(np.argmax(hot.sum(axis=(0,1))))
base = np.rot90(a[:,:,z].T); lab = np.rot90(labels[:,:,z].T)
vmax = np.percentile(pos, 99.5)
fig, ax = plt.subplots(figsize=(10,8), constrained_layout=True)
ax.imshow(base, cmap='gray', vmin=0, vmax=vmax)
colors = ['#ff3333','#ff7f0e','#ffd700','#7fff00','#00cc66','#00cccc','#3388ff','#6633cc','#aa33cc','#ff33aa']
for code in range(1,21):
    m = lab == code
    if m.any(): ax.contour(m, levels=[0.5], colors=colors[(code-1)%10], linewidths=1.4)
for side, selector in ((1, lambda x:x<mid),(2,lambda x:x>=mid)):
    c=np.argwhere(hot & selector(np.indices(a.shape)[0]))
    if len(c):
        edges=np.linspace(c[:,1].min(),c[:,1].max()+1,11)
        for band,y in enumerate(edges):
            yy = a.shape[1]-1-y
            xx = [0, a.shape[0]-1] if side==1 else [a.shape[0]-1, a.shape[0]-1]
            # Boundaries are represented by mask contours; no full-image lines are added.
ax.axvline(a.shape[0]/2, color='white', linestyle='--', linewidth=1)
ax.set_title(f'Top 1% template voxels: 10 horizontal bands per side | cutoff={cutoff:.6g} | axial z={z}')
ax.axis('off')
fig.savefig(ROOT/'results/template_top1_left_right_10_horizontal_lines.png', dpi=180)
plt.close(fig)
print(f'cutoff={cutoff} top1_voxels={int(hot.sum())} display_z={z}')
print(ROOT/'results/template_top1_left_right_10_horizontal_lines.png')
