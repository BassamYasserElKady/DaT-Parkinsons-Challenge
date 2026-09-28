#!/usr/bin/env python3
"""Create one 20-scan refined-segmentation montage per histogram cluster."""
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib, matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; REG=ROOT/'results/ants_registration_mni'; WIN=ROOT/'results/adaptive_slice_best_mask_sbr_cv5/adaptive_windows.csv'; MASK=ROOT/'results/paper_style_adaptive5_segmentation_refined/mip_masks'; ASSIGN=ROOT/'results/combined_histogram_clusters_10/best_10_cluster_assignments.csv'; OUT=ROOT/'results/histogram_clusters_refined_segmentation_pngs'

def main():
 OUT.mkdir(parents=True,exist_ok=True); d=pd.read_csv(ASSIGN,dtype={'uid':str}).sort_values(['gmm_cluster','uid']); w=pd.read_csv(WIN,dtype={'uid':str}).set_index('uid'); rows=[]
 for cluster in sorted(d.gmm_cluster.unique()):
  members=d[d.gmm_cluster==cluster].head(20); fig,ax=plt.subplots(4,5,figsize=(15,11),facecolor='white')
  for k,(_,r) in enumerate(members.iterrows()):
   uid=str(r.uid); img=nib.load(str(REG/f'{uid}.nii.gz')); a=np.asarray(img.dataobj,dtype=np.float32); start=int(w.loc[uid,'start']); end=int(w.loc[uid,'end']); base=a[:,:,start:end+1].max(2); positive=base[base>0]; lo,hi=np.percentile(positive,[1,99]); ax.flat[k].imshow(np.clip((base-lo)/max(hi-lo,1e-8),0,1).T,cmap='gray',origin='lower'); m=np.asarray(nib.load(str(MASK/f'{uid}_mip_striatum_mask.nii.gz')).dataobj)>0; contour=m[:,:,start:end+1].max(2); ax.flat[k].contour(contour.T,levels=[.5],colors='red',linewidths=1); ax.flat[k].set_title(f'{uid}\nlabel={int(r.actual)} z={start}-{end}',fontsize=8); ax.flat[k].set_xticks([]);ax.flat[k].set_yticks([]); rows.append({'cluster':int(cluster),'uid':uid,'actual':int(r.actual),'window_start':start,'window_end':end})
  fig.suptitle(f'Histogram cluster {int(cluster)} — 20 scans with refined segmentation overlay\nred contour = paper-style refined bilateral striatal candidate mask',fontsize=14); fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(OUT/f'cluster_{int(cluster)}_20_scans.png',dpi=160,bbox_inches='tight'); plt.close(fig)
 pd.DataFrame(rows).to_csv(OUT/'manifest.csv',index=False); print(OUT)
if __name__=='__main__':main()
