#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd,nibabel as nib,matplotlib.pyplot as plt
import morphometric_gaussian_smoothing_cv5 as smooth
ROOT=Path(__file__).resolve().parents[1];REG=ROOT/'results/ants_registration_mni';WIN=ROOT/'results/adaptive_slice_best_mask_sbr_cv5/adaptive_windows.csv';ERRORS=ROOT/'results/paper_morphometric_only_cv5/oof_predictions.csv';OUT=ROOT/'results/morphometric_gaussian_smoothing_cv5/visualize_20_errors';BRAIN=ROOT/'assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz';SIGMA=.75
def main():
 OUT.mkdir(parents=True,exist_ok=True);d=pd.read_csv(ERRORS,dtype={'uid':str});d=d[(d.model=='gaussian_svm')&(d.oof_prediction!=d.actual)].copy();d['loss']=-(d.actual*np.log(np.clip(d.oof_probability,1e-12,1))+(1-d.actual)*np.log(np.clip(1-d.oof_probability,1e-12,1)));d=d.sort_values('loss',ascending=False).head(20);w=pd.read_csv(WIN,dtype={'uid':str}).set_index('uid');brain=np.asarray(nib.load(str(BRAIN)).dataobj)>0;manifest=[]
 for _,r in d.iterrows():
  u=str(r.uid);a=np.asarray(nib.load(str(REG/f'{u}.nii.gz')).dataobj,dtype=float);s,e=int(w.loc[u,'start']),int(w.loc[u,'end']);slab=a[:,:,s:e+1];bm=brain[:,:,s:e+1].any(2);raw=slab.max(2);smoothed=smooth.gaussian_filter(raw,SIGMA);m0=smooth.segment(raw,bm,0);m1=smooth.segment(smoothed,bm,0);fig,ax=plt.subplots(1,2,figsize=(9,4),facecolor='white')
  for j,(im,m,title) in enumerate([(raw,m0,'Before smoothing'),(smoothed,m1,f'After smoothing σ={SIGMA}')]):
   pos=im[im>0];lo,hi=np.percentile(pos,[1,99]);ax[j].imshow(np.clip((im-lo)/max(hi-lo,1e-8),0,1).T,cmap='gray',origin='lower');ax[j].contour(m.T,levels=[.5],colors='red',linewidths=1.2);ax[j].set_title(title);ax[j].set_xticks([]);ax[j].set_yticks([])
  fig.suptitle(f'{u} | actual={int(r.actual)} predicted={int(r.oof_prediction)} p={float(r.oof_probability):.3f} | z={s}-{e}',fontsize=11);fig.tight_layout();fig.savefig(OUT/f'{u}_before_after_sigma075.png',dpi=160,bbox_inches='tight');plt.close(fig);manifest.append({'uid':u,'actual':int(r.actual),'prediction':int(r.oof_prediction),'probability':float(r.oof_probability),'log_loss':float(r.loss),'window_start':s,'window_end':e})
 pd.DataFrame(manifest).to_csv(OUT/'manifest.csv',index=False);print(OUT)
if __name__=='__main__':main()
