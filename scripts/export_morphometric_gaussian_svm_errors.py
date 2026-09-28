#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd,nibabel as nib,matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/paper_morphometric_only_cv5/misclassified_pngs'; MASK=ROOT/'results/paper_style_adaptive5_segmentation_refined/mip_masks'; REG=ROOT/'results/ants_registration_mni'; WIN=ROOT/'results/adaptive_slice_best_mask_sbr_cv5/adaptive_windows.csv'; PRED=ROOT/'results/paper_morphometric_only_cv5/oof_predictions.csv'
def main():
 OUT.mkdir(parents=True,exist_ok=True); p=pd.read_csv(PRED,dtype={'uid':str}); p=p[p.model=='gaussian_svm'].copy(); p=p[p.oof_prediction!=p.actual].sort_values('uid'); w=pd.read_csv(WIN,dtype={'uid':str}).set_index('uid'); rows=[]
 for _,r in p.iterrows():
  u=str(r.uid); a=np.asarray(nib.load(str(REG/f'{u}.nii.gz')).dataobj,dtype=float); s,e=int(w.loc[u,'start']),int(w.loc[u,'end']); x=a[:,:,s:e+1].max(2); pos=x[x>0]; lo,hi=np.percentile(pos,[1,99]); m=np.asarray(nib.load(str(MASK/f'{u}_mip_striatum_mask.nii.gz')).dataobj)>0; mask=m[:,:,s:e+1].max(2); fig,ax=plt.subplots(figsize=(5,5),facecolor='white'); ax.imshow(np.clip((x-lo)/max(hi-lo,1e-8),0,1).T,cmap='gray',origin='lower'); ax.contour(mask.T,levels=[.5],colors='red',linewidths=1.2); status='FP' if int(r.actual)==0 else 'FN'; ax.set_title(f'{u} — {status}\nactual={int(r.actual)} | predicted={int(r.oof_prediction)} | p={float(r.oof_probability):.3f}\nz={s}-{e}',fontsize=11);ax.set_xticks([]);ax.set_yticks([]); fig.tight_layout(); fig.savefig(OUT/f'{u}_{status}.png',dpi=150,bbox_inches='tight');plt.close(fig);rows.append({'uid':u,'actual':int(r.actual),'prediction':int(r.oof_prediction),'probability':float(r.oof_probability),'error_type':status,'window_start':s,'window_end':e})
 pd.DataFrame(rows).to_csv(OUT/'manifest.csv',index=False);print(f'created={len(rows)}');print(OUT)
if __name__=='__main__':main()
