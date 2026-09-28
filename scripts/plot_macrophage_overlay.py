#!/usr/bin/env python3
"""Plot macrophage-enrichment scores from DeepSpot-M, adapter, and Xenium."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import tifffile
import torch
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from spatial_pipeline.adapter import AdapterConfig, DeepSpotCalibrationAdapter, default_device

MARKERS=("CD68","CD163","LYZ","APOC1")

def zscore_matrix(a):
    a=np.asarray(a,dtype=np.float32)
    mean=a.mean(axis=0); std=a.std(axis=0)
    std[std<1e-6]=1.0
    return (a-mean)/std

def score(a, genes):
    return zscore_matrix(a[:, [genes.index(g) for g in MARKERS]]).mean(axis=1)

def corr(x,y):
    return (float(np.corrcoef(x,y)[0,1]), float(spearmanr(x,y).statistic))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--sample-id',default='S2_Bot')
    ap.add_argument('--pyramid-level',type=int,default=4)
    ap.add_argument('--marker-size',type=float,default=2.5)
    ap.add_argument('--overlay-alpha',type=float,default=0.57)
    ap.add_argument('--output',type=Path,default=Path('results/s2_bot_macrophage_overlay.png'))
    args=ap.parse_args(); root=args.root.resolve()
    sample_dir=root/'results/deepspot_tables'/args.sample_id
    pred_df=pd.read_parquet(next(sample_dir.glob('*_deepspot_scgpt_predictions.parquet')))
    target_df=pd.read_parquet(next(sample_dir.glob('*_xenium_log1p_cp10k_targets.parquet')))
    ckpt=torch.load(root/'results/adapter_baseline/adapter_final.pt',map_location='cpu',weights_only=False)
    with np.load(root/'results/deepspot_tables/development_normalization.npz',allow_pickle=False) as st:
        genes=[str(g) for g in st['genes'].tolist()]
        pmean=st['prediction_mean'].astype(np.float32); pstd=st['prediction_std'].astype(np.float32)
    if any(g not in genes for g in MARKERS): raise ValueError('marker absent from fitted gene list')
    if not pred_df[['spot','x','y']].equals(target_df[['spot','x','y']]): raise ValueError('coordinates differ')
    pred=pred_df[genes].to_numpy(np.float32)
    target=target_df[genes].to_numpy(np.float32)
    device=default_device()
    model=DeepSpotCalibrationAdapter(AdapterConfig(**ckpt['adapter_config']))
    model.load_state_dict(ckpt['model_state_dict']); model.to(device).eval()
    with torch.inference_mode():
        out=model(torch.from_numpy((pred-pmean)/pstd).to(device)).cpu().numpy()
    calibrated=out*np.asarray(ckpt['target_std'],dtype=np.float32)+np.asarray(ckpt['target_mean'],dtype=np.float32)
    scores=[score(pred,genes),score(calibrated,genes),score(target,genes)]
    raw_corr=corr(scores[0],scores[2]); cal_corr=corr(scores[1],scores[2])

    row=pd.read_csv(root/'data/sample_manifest.csv')
    row=row.loc[row.sample_id.astype(str)==args.sample_id]
    if len(row)!=1: raise ValueError('sample not found in manifest')
    image_path=root/str(row.iloc[0].he_image_path)
    with tifffile.TiffFile(image_path) as tf:
        levels=tf.series[0].levels
        image=levels[args.pyramid_level].asarray()
        full_h,full_w=levels[0].shape[:2]
    scale=((full_w/image.shape[1])+(full_h/image.shape[0]))/2.0
    x=(pred_df.x.to_numpy(float)+112.0)/scale
    y=(pred_df.y.to_numpy(float)+112.0)/scale
    vlim=2.5
    titles=[f'Raw DeepSpot-M\nPearson {raw_corr[0]:.3f} | Spearman {raw_corr[1]:.3f}',
            f'Calibrated DeepSpot-M\nPearson {cal_corr[0]:.3f} | Spearman {cal_corr[1]:.3f}',
            'Actual Xenium\nreference']
    fig,axes=plt.subplots(1,3,figsize=(19,7),constrained_layout=True)
    for ax,title,val in zip(axes,titles,scores):
        ax.imshow(image,origin='upper')
        sc=ax.scatter(x,y,c=val,cmap='coolwarm',norm=Normalize(-vlim,vlim),s=args.marker_size,linewidths=0,alpha=args.overlay_alpha)
        ax.set_xlim(0,image.shape[1]); ax.set_ylim(image.shape[0],0); ax.set_aspect('equal')
        ax.set_title(title,fontsize=12); ax.set_xlabel('H&E x (display pixels)'); ax.set_ylabel('H&E y (display pixels)')
        fig.colorbar(sc,ax=ax,fraction=.046,pad=.04,label='macrophage marker z-score')
    fig.suptitle(f'{args.sample_id} macrophage-enrichment overlay on H&E\n'
                 f'Mean within-panel z-score of {", ".join(MARKERS)}; {len(scores[0]):,} tiles',fontsize=15)
    out=root/args.output if not args.output.is_absolute() else args.output
    out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=180,bbox_inches='tight'); plt.close(fig)
    print(f'output={out}')
    print(f'markers={",".join(MARKERS)}; tiles={len(scores[0]):,}; device={device}')
    print(f'raw_pearson={raw_corr[0]:.6f}; raw_spearman={raw_corr[1]:.6f}')
    print(f'calibrated_pearson={cal_corr[0]:.6f}; calibrated_spearman={cal_corr[1]:.6f}')

if __name__=='__main__': main()
