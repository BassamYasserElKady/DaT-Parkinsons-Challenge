#!/usr/bin/env python3
"""Paper-style morphometric-only evaluation on valid refined masks."""
from pathlib import Path
import numpy as np,pandas as pd,nibabel as nib
from scipy.ndimage import binary_erosion
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score,log_loss,accuracy_score

ROOT=Path(__file__).resolve().parents[1]; MASKS=ROOT/'results/paper_style_adaptive5_segmentation_refined/mip_masks'; LABELS=ROOT/'results/cv5_submission_flip_clip/oof_predictions.csv'; OUT=ROOT/'results/paper_morphometric_only_cv5'; SEED=20260931

def shape(mask,spacing=2.):
    yy,xx=np.where(mask)
    if len(xx)<3:return [0.,0.,0.,0.,0.,0.]
    area=float(len(xx)*spacing*spacing); boundary=mask&~binary_erosion(mask); perimeter=float(boundary.sum()*spacing); circularity=float(4*np.pi*area/max(perimeter*perimeter,1e-12)); coords=np.column_stack([xx,yy])*spacing; cov=np.cov(coords,rowvar=False); ev=np.sort(np.linalg.eigvalsh(cov))[::-1]; major=float(4*np.sqrt(max(ev[0],0))); minor=float(4*np.sqrt(max(ev[-1],0))); ecc=float(np.sqrt(max(0.,1-(minor/max(major,1e-12))**2))); return [area,perimeter,circularity,major,minor,ecc]

def main():
    lab=pd.read_csv(LABELS,dtype={'uid':str}).sort_values('uid').reset_index(drop=True); rows=[]; valid=[]
    for _,r in lab.iterrows():
        p=MASKS/f'{r.uid}_mip_striatum_mask.nii.gz'
        if not p.exists():continue
        m=np.asarray(nib.load(str(p)).dataobj)>0; m=m.any(2)
        if not m.any():continue
        # Existing pipeline convention: x>=45 is anatomical left in the template orientation.
        x=np.indices(m.shape)[0]; left=m&(x>=45); right=m&(x<45); vals=shape(left)+shape(right); rows.append([str(r.uid),int(r.actual),*vals]); valid.append(r.uid)
    cols=['uid','actual']+[f'{side}_{feat}' for side in ('left','right') for feat in ('area','perimeter','circularity','major_axis','minor_axis','eccentricity')]; df=pd.DataFrame(rows,columns=cols); X=df.drop(columns=['uid','actual']).to_numpy(float); y=df.actual.to_numpy(int); cv=StratifiedKFold(5,shuffle=True,random_state=SEED); models={'linear_svm':SVC(C=1,kernel='linear',probability=True,random_state=SEED),'linear_discriminant':LinearDiscriminantAnalysis(),'weighted_knn':KNeighborsClassifier(n_neighbors=10,weights='distance'),'gaussian_svm':SVC(C=1,kernel='rbf',gamma='scale',probability=True,random_state=SEED),'gaussian_nb':GaussianNB(),'knn':KNeighborsClassifier(n_neighbors=10)}; preds={k:np.full(len(y),np.nan) for k in models}; folds=[]
    for fold,(tr,va) in enumerate(cv.split(X,y),1):
        for name,est in models.items():
            model=make_pipeline(StandardScaler(),est); model.fit(X[tr],y[tr]); p=model.predict_proba(X[va])[:,1]; preds[name][va]=p; folds.append({'fold':fold,'model':name,'auroc':roc_auc_score(y[va],p),'log_loss':log_loss(y[va],p,labels=[0,1]),'accuracy':accuracy_score(y[va],p>=.5),'incorrect':int(np.sum((p>=.5).astype(int)!=y[va]))})
        print(f'fold={fold} complete',flush=True)
    results=[]; frames=[]
    for name,p in preds.items():
        q=(p>=.5).astype(int); results.append({'model':name,'n':len(y),'auroc':roc_auc_score(y,p),'log_loss':log_loss(y,p,labels=[0,1]),'accuracy':accuracy_score(y,q),'incorrect':int(np.sum(q!=y))}); frames.append(pd.DataFrame({'uid':df.uid,'actual':y,'model':name,'oof_probability':p,'oof_prediction':q}))
    OUT.mkdir(parents=True,exist_ok=True); df.to_csv(OUT/'morphometric_features.csv',index=False); pd.DataFrame(results).to_csv(OUT/'metrics.csv',index=False); pd.DataFrame(folds).to_csv(OUT/'fold_metrics.csv',index=False); pd.concat(frames,ignore_index=True).to_csv(OUT/'oof_predictions.csv',index=False); print(f'valid_scans={len(df)}'); print(pd.DataFrame(results).sort_values('auroc',ascending=False).to_string(index=False))
if __name__=='__main__':main()
