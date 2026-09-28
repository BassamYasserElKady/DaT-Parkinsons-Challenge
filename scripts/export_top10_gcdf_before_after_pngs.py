#!/usr/bin/env python3
"""Export top-10 GCDF log-loss scans before/after intensity remapping."""
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import gamma

ROOT=Path(__file__).resolve().parents[1]; REG=ROOT/'results/ants_registration_mni'; MD=ROOT/'assets/atlases/fsl/masks'; GC=ROOT/'results/nested_gcdf_shape40_mask_cv5'; WIN=ROOT/'results/adaptive_slice_best_mask_sbr_cv5/adaptive_windows.csv'; OUT=ROOT/'results/top10_gcdf_before_after_pngs'

def scaled(uid,brain):
 p=ROOT/'results/healthy_mode500_tail_gamma_clusters_10/normalized_healthy_volumes'/f'{uid}.nii.gz'
 if p.exists(): return np.asarray(nib.load(str(p)).dataobj,dtype=np.float32)
 a=np.asarray(nib.load(str(REG/f'{uid}.nii.gz')).dataobj,dtype=np.float32); v=a[brain&np.isfinite(a)&(a>0)]; c,e=np.histogram(v,bins=500); i=c.argmax(); m=(e[i]+e[i+1])/2; z=np.zeros_like(a); q=brain&np.isfinite(a); z[q]=a[q]/m; return z

def main():
 OUT.mkdir(parents=True,exist_ok=True); brain=np.asarray(nib.load(str(MD/'MNI152_T1_2mm_brain_mask.nii.gz')).dataobj)>0; windows=pd.read_csv(WIN,dtype={'uid':str}).set_index('uid'); assign=pd.read_csv(GC/'fold_gcdf_features.csv',dtype={'uid':str}); pred=pd.read_csv(GC/'oof_predictions.csv',dtype={'uid':str}); pred=pred[pred.model=='xgboost'].set_index('uid'); pars=pd.read_csv(GC/'fold_gamma_parameters.csv'); pars=pars.set_index(['fold','cluster']); rows=[]
 for _,r in assign.iterrows():
  uid=str(r.uid); actual=int(r.actual); p=float(pred.loc[uid,'oof_probability']); loss=-(actual*np.log(np.clip(p,1e-12,1))+(1-actual)*np.log(np.clip(1-p,1e-12,1))); rows.append({'uid':uid,'actual':actual,'probability':p,'log_loss':loss,'fold':int(r.fold),'cluster':int(r.cluster)})
 top=pd.DataFrame(rows).sort_values('log_loss',ascending=False).head(10); top.to_csv(OUT/'top10_manifest.csv',index=False)
 fig,ax=plt.subplots(2,10,figsize=(25,6),facecolor='white')
 for j,(_,r) in enumerate(top.iterrows()):
  uid=r.uid; a=scaled(uid,brain); start=int(windows.loc[uid,'start']); end=int(windows.loc[uid,'end']); par=pars.loc[(int(r.fold),int(r.cluster))]; b=gamma.cdf(np.maximum(a,0),a=float(par['shape']),scale=float(par['scale'])); raw=a[:,:,start:end+1].max(2); after=b[:,:,start:end+1].max(2)
  lo,hi=np.percentile(raw[raw>0],[1,99.5]); ax[0,j].imshow(np.clip((raw-lo)/max(hi-lo,1e-8),0,1).T,cmap='gray',origin='lower'); ax[1,j].imshow(after.T,cmap='viridis',origin='lower',vmin=0,vmax=1); ax[0,j].set_title(f'{uid}\np={r.probability:.3f}, LL={r.log_loss:.2f}',fontsize=8); ax[1,j].set_title(f'GCDF\ncluster {int(r.cluster)}',fontsize=8)
  for rr in range(2): ax[rr,j].set_xticks([]); ax[rr,j].set_yticks([])
 fig.suptitle('Top 10 GCDF log-loss scans — same adaptive five-slice MIP before vs after GCDF',fontsize=14); fig.tight_layout(rect=[0,0,1,.91]); fig.savefig(OUT/'top10_gcdf_before_after.png',dpi=180,bbox_inches='tight'); plt.close(fig); print(OUT/'top10_gcdf_before_after.png'); print(top.to_string(index=False))
if __name__=='__main__': main()
