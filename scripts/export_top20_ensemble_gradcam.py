#!/usr/bin/env python3
from pathlib import Path
import sys, zipfile
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D
from train_registered_consecutive_densenet121_3d import DenseNet121_3D

REG = ROOT / 'results/ants_registration_mni'
INPUTS = ROOT / 'results/cv5_real8_post_registration_gaussian/real8_native_post_registration_gaussian.npy'
SELECTIONS = ROOT / 'results/cv5_real8_post_registration_gaussian/slice_selection.csv'
CKPT = ROOT / 'results/full_data_real8_sigma075_ensemble/ensemble_weights.pt'
MEANS = ROOT / 'results/full_data_real8_sigma075_ensemble/means.npy'
STDS = ROOT / 'results/full_data_real8_sigma075_ensemble/stds.npy'
OUT = ROOT / 'results/top20_ensemble_gradcam'
UIDS = ['bmae38sr','7875aud9','xid4n1u3','y08p0fod','uh17j8qo','bglmy62x','c1m3o55i','rqq5qob8','07gbasf5','vfua9bq1','0imjyjcm','344mw04i','0uc87nxp','l5u7nted','nal3gywi','nkb4ehou','6tx0tbkg','ccprhvlz','f5gzpl1q','dhvs41lq']

def cam(model, layer, x):
    acts, grads = [], []
    def fh(_, __, out): acts.append(out)
    def bh(_, gin, gout): grads.append(gout[0])
    h1, h2 = layer.register_forward_hook(fh), layer.register_full_backward_hook(bh)
    model.zero_grad(set_to_none=True); logits = model(x); logits[:, 1].sum().backward()
    h1.remove(); h2.remove()
    a, g = acts[0], grads[0]
    w = g.mean(dim=(2, 3, 4), keepdim=True)
    c = F.relu((w * a).sum(dim=1, keepdim=True))
    c = F.interpolate(c, size=x.shape[2:], mode='trilinear', align_corners=False)[0, 0]
    c = c.detach().cpu().numpy()
    return c / (c.max() + 1e-12), float(torch.softmax(logits.detach(), 1)[0, 1])

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    native = np.load(INPUTS).astype(np.float32, copy=False)
    means, stds = np.load(MEANS), np.load(STDS)
    xall = np.empty((len(native), 16, 64, 64), np.float32)
    from scipy import ndimage
    for i, v in enumerate(native):
        z = (v - means[:, None, None]) / stds[:, None, None]
        xall[i] = ndimage.zoom(z, (2, 64/109, 64/91), order=1)
    rows = pd.read_csv(SELECTIONS).set_index('uid')
    order = pd.read_csv(ROOT / 'results/cv5_real8_post_registration_gaussian/oof_predictions.csv').set_index('uid')
    index = {u:i for i,u in enumerate(order.index)}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obj = torch.load(CKPT, map_location=device, weights_only=False)
    r, d = ResNet18CBAM3D().to(device).eval(), DenseNet121_3D().to(device).eval()
    r.load_state_dict(obj['resnet_state_dict']); d.load_state_dict(obj['densenet_state_dict'])
    for uid in UIDS:
        i = index[uid]; x = torch.from_numpy(xall[i:i+1]).unsqueeze(1).to(device).requires_grad_(True)
        cr, pr = cam(r, r.layer4[-1], x); cd, pd_ = cam(d, d.features.norm5, x)
        ce = (cr + cd) / 2; ce = ce / (ce.max() + 1e-12)
        start, end = int(rows.loc[uid, 'start']), int(rows.loc[uid, 'end'])
        raw = nib.load(str(REG / f'{uid}.nii.gz')).get_fdata()
        fig, ax = plt.subplots(4, 8, figsize=(20, 10), constrained_layout=True)
        vmax = np.percentile(raw[raw > 0], 99.5) if np.any(raw > 0) else 1.0
        for j, z in enumerate(range(start, end + 1)):
            base = np.rot90(raw[:, :, z].T)
            inp = np.rot90(xall[i, j].T)
            ax[0,j].imshow(base, cmap='gray', vmin=0, vmax=vmax); ax[0,j].set_title(f'Input z={z}')
            for row, heat, title in ((1, cr[j], 'ResNet CAM'), (2, cd[j], 'DenseNet CAM'), (3, ce[j], 'Ensemble CAM')):
                ax[row,j].imshow(inp, cmap='gray'); ax[row,j].imshow(np.rot90(heat), cmap='jet', alpha=0.45, vmin=0, vmax=1); ax[row,j].set_title(title)
            for k in range(4): ax[k,j].axis('off')
        for j in range(8):
            ax[0,j].set_title(f'Input z={start+j}')
        fig.suptitle(f'{uid} — disease-logit Grad-CAM | ResNet p={pr:.3f}, DenseNet p={pd_:.3f}, ensemble p={(pr+pd_)/2:.3f}', fontsize=14)
        fig.savefig(OUT / f'{uid}_gradcam.png', dpi=150); plt.close(fig)
    zp = ROOT / 'results/top20_ensemble_gradcam.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob('*.png')): z.write(p, p.name)
    print(zp)

if __name__ == '__main__': main()
