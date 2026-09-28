from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib, matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; U=['fknwbhkv','n3b8pfy4','v1fkhnnk']; W=pd.read_csv(R/'results/adaptive_slice_best_mask_sbr_cv5/adaptive_windows.csv',dtype={'uid':str}).set_index('uid'); OUT=R/'results/paper_style_adaptive5_segmentation_refined/empty_mask_scans.png'
fig,ax=plt.subplots(1,3,figsize=(12,4),facecolor='white')
for j,u in enumerate(U):
 a=np.asarray(nib.load(str(R/'results/ants_registration_mni'/f'{u}.nii.gz')).dataobj,dtype=float); s,e=int(W.loc[u,'start']),int(W.loc[u,'end']); x=a[:,:,s:e+1].max(2); p=x[x>0]; lo,hi=np.percentile(p,[1,99]); ax[j].imshow(np.clip((x-lo)/max(hi-lo,1e-8),0,1).T,cmap='gray',origin='lower'); ax[j].set_title(f'{u}\nz={s}-{e}\nrefined mask: empty'); ax[j].set_xticks([]);ax[j].set_yticks([])
fig.suptitle('Scans rejected by refined paper-style segmentation',fontsize=13);fig.tight_layout();fig.savefig(OUT,dpi=180,bbox_inches='tight');print(OUT)
