#!/usr/bin/env python3
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / 'results/ants_registration_mni'
PRED = ROOT / 'results/cv5_real8_post_registration_gaussian/oof_predictions.csv'
TEMPLATE_PATH = ROOT / 'submission_src_ensemble/assets/template.nii'
OUT = ROOT / 'results/top20_striatum_theft_composites'
UIDS = ['bmae38sr','7875aud9','xid4n1u3','y08p0fod','uh17j8qo','bglmy62x','c1m3o55i','rqq5qob8','07gbasf5','vfua9bq1','0imjyjcm','344mw04i','0uc87nxp','l5u7nted','nal3gywi','nkb4ehou','6tx0tbkg','ccprhvlz','f5gzpl1q','dhvs41lq']

def hotspot(a, percentile):
    pos = a[np.isfinite(a) & (a > 0)]
    cut = float(np.percentile(pos, 100-percentile))
    return np.isfinite(a) & (a >= cut), cut

def composite(template, scan, dest, source):
    out = template.copy(); mid = template.shape[0] / 2.0
    grid = np.indices(template.shape)
    for selector in (grid[0] < mid, grid[0] >= mid):
        d = np.argwhere(dest & selector); s = np.argwhere(source & selector)
        if not len(d) or not len(s): continue
        dmin, dmax = d.min(axis=0), d.max(axis=0); smin, smax = s.min(axis=0), s.max(axis=0)
        dspan = np.maximum(dmax-dmin, 1); sspan = np.maximum(smax-smin, 1)
        # Coordinate normalization maps all ten horizontal bands, including
        # the final two guide bands, onto their template counterparts.
        sn = (s-smin)/sspan; dn = (d-dmin)/dspan
        tree = cKDTree(sn)
        _, near = tree.query(dn, k=1)
        vals = scan[tuple(s[near].T)]
        out[tuple(d.T)] = vals
    return out

def guide_mask(mask, shape):
    g = np.zeros(shape, dtype=bool); mid=shape[0]/2.0; grid=np.indices(shape)
    for selector in (grid[0] < mid, grid[0] >= mid):
        p=np.argwhere(mask & selector)
        if not len(p): continue
        edges=np.linspace(p[:,1].min(), p[:,1].max()+1, 11)
        # Last two of ten horizontal bands.
        g[p[:,0],p[:,1],p[:,2]] = p[:,1] >= edges[8]
    return g

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    timg=nib.load(str(TEMPLATE_PATH)); t=timg.get_fdata().astype(np.float32)
    dest, tcut=hotspot(t, 1.0); tbase=t.copy(); tbase[dest]=0
    tg=guide_mask(dest,t.shape)
    pred=pd.read_csv(PRED).set_index('uid')
    for uid in UIDS:
        simg=nib.load(str(REG/f'{uid}.nii.gz')); s=simg.get_fdata().astype(np.float32)
        src, scut=hotspot(s, .25); comp=composite(tbase,s,dest,src)
        nimg=nib.Nifti1Image(comp,timg.affine,timg.header.copy()); nimg.header.set_data_dtype(np.float32)
        nib.save(nimg,str(OUT/f'{uid}_composite.nii.gz'))
        z=int(np.argmax(dest.sum(axis=(0,1)))); zs=range(max(0,z-3),min(t.shape[2],z+5))
        pos=t[t>0]; vmax=np.percentile(pos,99.5)
        fig,ax=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
        for axy,k in zip(ax.flat,zs):
            axy.imshow(np.rot90(tbase[:,:,k].T),cmap='gray',vmin=0,vmax=vmax)
            axy.contour(np.rot90(tg[:,:,k].T),levels=[.5],colors='lime',linewidths=1.3)
            axy.set_title(f'Template top 1% removed z={k}'); axy.axis('off')
        fig.suptitle(f'{uid} — template with top 0.25% hotspot injected | scan cutoff={scut:.4g} | log loss={-(int(pred.loc[uid,"actual"])*np.log(np.clip(pred.loc[uid,"equal_ensemble_clipped"],1e-15,1-1e-15))+(1-int(pred.loc[uid,"actual"]))*np.log(np.clip(1-pred.loc[uid,"equal_ensemble_clipped"],1e-15,1-1e-15))):.3f}',fontsize=13)
        fig.savefig(OUT/f'{uid}_template_injected.png',dpi=150);plt.close(fig)
    zp=ROOT/'results/top20_striatum_theft_composites.zip'
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()): z.write(p,p.name)
    print(f'template_cutoff={tcut} composites={len(list(OUT.glob("*_composite.nii.gz")))}')
    print(zp)

if __name__=='__main__': main()
