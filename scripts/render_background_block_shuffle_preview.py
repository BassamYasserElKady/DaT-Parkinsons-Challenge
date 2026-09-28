#!/usr/bin/env python3
"""Visualize striatum-protected 2x2 background-block shuffle augmentation."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
UID='01nouhtc'
REGISTERED=ROOT/'results/end_to_end_raw_registration_cv5_seed20260931/registered'
CONTOURS=ROOT/'results/adaptive8_mip6_guided_contours/contours'
AUDIT=ROOT/'results/adaptive8_mip6_guided_contours/contour_audit.csv'
OUT=ROOT/'results/augmentation_preview'

def shuffled(image: np.ndarray, striatum: np.ndarray, multiplier: float, seed: int) -> np.ndarray:
    """Permute non-overlapping 2x2 blocks wholly outside the striatal contour."""
    result=image.copy(); candidates=[]
    # Fixed 2-pixel grid guarantees source blocks do not overlap one another.
    for x in range(0,image.shape[0]-1,2):
        for y in range(0,image.shape[1]-1,2):
            if not striatum[x:x+2,y:y+2].any(): candidates.append((x,y))
    rng=np.random.default_rng(seed); order=rng.permutation(len(candidates)); source=image.copy()
    for destination_index,source_index in enumerate(order):
        dx,dy=candidates[destination_index]; sx,sy=candidates[source_index]
        result[dx:dx+2,dy:dy+2]=source[sx:sx+2,sy:sy+2]*multiplier
    return result

def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    row=pd.read_csv(AUDIT,dtype={'uid':str}).set_index('uid').loc[UID]
    volume=np.asarray(nib.load(str(REGISTERED/f'{UID}.nii.gz')).dataobj,dtype=np.float32)
    contour=np.asarray(nib.load(str(CONTOURS/f'{UID}_adaptive8_contours.nii.gz')).dataobj)>0
    # The actual six-slice MIP used as the guide; striatum is the projection of
    # the protected eight-slice contour, so no source/destination block touches it.
    mip=volume[:,:,int(row.mip6_start):int(row.mip6_end)+1].max(axis=2)
    striatum=contour[:,:,int(row.adaptive8_start):int(row.adaptive8_end)+1].any(axis=2)
    variants=[('Original',mip),('Shuffle +25%',shuffled(mip,striatum,1.25,25)),('Shuffle +50%',shuffled(mip,striatum,1.50,50)),('Shuffle +75%',shuffled(mip,striatum,1.75,75))]
    positive=mip[mip>0]; low,high=np.percentile(positive,[1,99.5])
    fig,axes=plt.subplots(1,4,figsize=(15,4.2),facecolor='white')
    for ax,(title,image) in zip(axes,variants):
        ax.imshow(image.T,cmap='gray',origin='lower',vmin=low,vmax=high)
        ax.contour(striatum.T,levels=[.5],colors='red',linewidths=1.3)
        ax.set_title(title);ax.axis('off')
    fig.suptitle(f'{UID}: six-slice MIP — 2×2 shuffle outside protected striatum (red)',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(OUT/f'{UID}_mip_background_shuffle_25_50_75.png',dpi=200,bbox_inches='tight');plt.close(fig)
    print(OUT/f'{UID}_mip_background_shuffle_25_50_75.png')
if __name__=='__main__':main()
