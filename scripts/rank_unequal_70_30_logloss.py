from pathlib import Path
import inspect
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import log_loss
import morphometric_histogram_top_bins_cv5 as m

R = Path(__file__).resolve().parents[1]
lab = pd.read_csv(R/'results/cv5_submission_flip_clip/oof_predictions.csv', dtype={'uid': str}).sort_values('uid').reset_index(drop=True)
w = pd.read_csv(R/'results/adaptive_localizer_parameter_cv5/stage1_windows.csv', dtype={'uid': str})
w = w[(w.sigma == .5) & (w.localizer_percentile == 95)].set_index('uid')
brain = np.asarray(nib.load(str(R/'assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz')).dataobj) > 0
base = R/'results/paper_style_adaptive5_segmentation_refined/mip_masks'; reg = R/'results/ants_registration_mni'
valid = [u for u in lab.uid.astype(str) if u in w.index and (reg/f'{u}.nii.gz').exists() and (base/f'{u}_mip_striatum_mask.nii.gz').exists()]
lab = lab[lab.uid.isin(valid)].sort_values('uid').reset_index(drop=True)
y = lab.actual.to_numpy(int)
src = inspect.getsource(m.segment).replace('float(np.percentile(v, 70))', 'float(np.percentile(v, threshold_percentile))')
ns = {'np': np, 'gaussian_filter': m.gaussian_filter, 'generate_binary_structure': m.generate_binary_structure, 'binary_opening': m.binary_opening, 'binary_closing': m.binary_closing, 'label': m.label, 'otsu': m.otsu, 'threshold_percentile': 70}
exec('def seg(im,bm):\n' + '\n'.join('    '+x for x in src.splitlines()[1:]), ns); seg = ns['seg']
shape_rows=[]; raw_rows=[]
for u in lab.uid.astype(str):
    a = np.asarray(nib.load(str(reg/f'{u}.nii.gz')).dataobj, dtype=np.float32); s,e = int(w.loc[u,'start']), int(w.loc[u,'end'])
    raw = a[:,:,s:e+1].max(2); bm = brain[:,:,s:e+1].any(2); q = seg(m.retain_top_bins(raw,bm,96), bm); g = np.indices(q.shape)
    shape_rows.append(m.shape(q&(g[0]>=45)) + m.shape(q&(g[0]<45)))
    pos = raw[bm&(raw>0)]; raw_rows.append((q, raw/max(float(np.median(pos)), 1e-8), g))
X_shape = np.asarray(shape_rows, float); Xh=[]
for q,norm,g in raw_rows:
    vals=[]
    for side in (q&(g[0]>=45), q&(g[0]<45)):
        yy = np.where(side)[0]
        if len(yy) < 3: vals += [0.,0.]; continue
        cut = yy.min() + .70*(yy.max()-yy.min()); up = side&(g[1] <= cut); down = side&(g[1] > cut)
        vals += [float(norm[up].mean()) if up.any() else 0., float(norm[down].mean()) if down.any() else 0.]
    Xh.append(np.r_[vals, [vals[i]-vals[j] for i in range(4) for j in range(i+1,4)]])
X = np.c_[X_shape, np.asarray(Xh,float)]; p = np.full(len(y), np.nan); cv = StratifiedKFold(5, shuffle=True, random_state=20260931)
for fold,(tr,va) in enumerate(cv.split(X,y),1):
    sc=StandardScaler(); xt=sc.fit_transform(X[tr]); xv=sc.transform(X[va]); model=SVC(C=1,kernel='rbf',gamma='scale',probability=True,random_state=20260931+fold); model.fit(xt,y[tr]); p[va]=model.predict_proba(xv)[:,1]
q=np.clip(p,.005,.995); losses=-(y*np.log(q)+(1-y)*np.log(1-q)); out=lab[['uid','actual']].copy(); out['oof_probability']=p; out['clipped_probability']=q; out['scan_log_loss']=losses; out=out.sort_values('scan_log_loss',ascending=False).reset_index(drop=True); out.insert(0,'rank',np.arange(1,len(out)+1)); out.to_csv(R/'results/morphometric_unequal_two_band_cv5/70_30_oof_logloss_ranking.csv',index=False); print(out.head(20).to_string(index=False)); print('total_log_loss',log_loss(y,q,labels=[0,1]))
