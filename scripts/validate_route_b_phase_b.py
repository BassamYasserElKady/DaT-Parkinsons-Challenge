#!/usr/bin/env python3
from pathlib import Path
import csv, math, tempfile
import ants
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/route_b_ants_random25_affine_corrected'
RAW=ROOT/'data/external/niftis_original'
CLEAN=ROOT/'results/ants_registration_percentile10_masked/cleaned_masks'
TEMPLATE_PATH=ROOT/'submission_src_ensemble/assets/template.nii'
TEMPLATE=ants.image_read(str(TEMPLATE_PATH))
TMASK=ants.image_read(str(ROOT/'assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz'))
VOI=ants.image_read(str(ROOT/'assets/atlases/fsl/masks/striatal_voi_mask_LAS.nii.gz'))

def center_world(img, weights=None):
    a=img.numpy().astype(float)
    if weights is None: weights=np.maximum(a,0)
    else: weights=weights.astype(float)
    w=weights.ravel(); total=w.sum()
    if total<=0: return np.array([np.nan]*3)
    inds=np.indices(a.shape,dtype=float).reshape(3,-1).T
    c=inds.T@w/total
    return np.asarray(img.origin[:3])+np.asarray(img.direction)[:3,:3] @ (c*np.asarray(img.spacing[:3]))

def metric_row(uid):
    img=ants.image_read(str(OUT/f'{uid}.nii.gz'))
    tx=str(OUT/'transforms'/f'{uid}.mat')
    moving_mask=ants.image_read(str(CLEAN/f'{uid}.nii.gz'))
    warped_mask=ants.apply_transforms(fixed=TMASK,moving=moving_mask,transformlist=[tx],interpolator='nearestNeighbor')
    wm=warped_mask.numpy()>0; fm=TMASK.numpy()>0
    dice=2.0*np.sum(wm&fm)/(np.sum(wm)+np.sum(fm)) if (wm.sum()+fm.sum()) else 0
    va=VOI.numpy()>0; arr=np.maximum(img.numpy(),0); com=center_world(img,arr*va); tc=center_world(VOI,va)
    dist=float(np.linalg.norm(com-tc)) if np.all(np.isfinite(com)) else float('nan')
    p=np.asarray(ants.read_transform(tx).parameters[:9],float).reshape(3,3)
    det=float(np.linalg.det(p)); sv=np.linalg.svd(p,compute_uv=False)
    q,r=np.linalg.qr(p); d=np.sign(np.diag(r)); d[d==0]=1; q=q@np.diag(d); r=np.diag(d)@r
    scales=np.diag(r); shear=np.array([r[0,1]/scales[1],r[0,2]/scales[2],r[1,2]/scales[2]])
    return {'uid':uid,'dice':float(dice),'com_distance_mm':dist,'determinant':det,
            'scale_min':float(sv.min()),'scale_max':float(sv.max()),
            'shear_max_abs':float(np.max(np.abs(shear))),
            'dice_fail':int(dice<.85),'com_fail':int(dist>6),'det_fail':int(det<=0)}

def main():
    uids=sorted(p.name[:-7] for p in RAW.glob('*.nii.gz'))
    rows=[]; sumv=None; sumsq=None; n=0
    for i,uid in enumerate(uids,1):
        r=metric_row(uid); rows.append(r)
        a=ants.image_read(str(OUT/f'{uid}.nii.gz')).numpy().astype(np.float64)
        sumv=a if sumv is None else sumv+a; sumsq=a*a if sumsq is None else sumsq+a*a; n+=1
        if i%100==0 or i==len(uids): print(f'validated={i}/{len(uids)}',flush=True)
    rows.sort(key=lambda x:x['uid'])
    with (OUT/'phase_b_qc_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    mean=sumv/n; var=np.maximum(sumsq/n-mean*mean,0); cv=np.sqrt(var)/(np.abs(mean)+1e-6)
    template_affine=nib.load(str(TEMPLATE_PATH)).affine
    nib.save(nib.Nifti1Image(mean.astype(np.float32),template_affine),str(OUT/'cohort_mean.nii.gz'))
    nib.save(nib.Nifti1Image(cv.astype(np.float32),template_affine),str(OUT/'cohort_cv.nii.gz'))
    brain=TMASK.numpy()>0; cvb=cv[brain]
    print('scans',n)
    for k in ['dice','com_distance_mm','determinant','shear_max_abs']:
        a=np.array([r[k] for r in rows]); print(k,'mean',float(np.nanmean(a)),'median',float(np.nanmedian(a)),'min',float(np.nanmin(a)),'max',float(np.nanmax(a)))
    print('dice_below_085',sum(r['dice_fail'] for r in rows),'com_above_6mm',sum(r['com_fail'] for r in rows),'det_nonpositive',sum(r['det_fail'] for r in rows))
    print('brain_cv_median',float(np.median(cvb)),'brain_cv_p95',float(np.percentile(cvb,95)))
    # Select unique edge cases and generate orthogonal overlays.
    audit={r['uid']:r for r in rows}; ids=[]
    ids += [r['uid'] for r in sorted(rows,key=lambda x:x['dice'])[:5]]
    ids += [r['uid'] for r in sorted(rows,key=lambda x:-x['shear_max_abs'])[:5]]
    # Use the existing physical geometric translation report for the final group.
    trans=[]
    with (ROOT/'results/phase_a_initialization.csv').open() as f:
        trans=list(csv.DictReader(f))
    ids += [r['uid'] for r in sorted(trans,key=lambda x:-float(x['translation_norm_mm']))[:5]]
    ids=list(dict.fromkeys(ids)); vdir=OUT/'qc_extreme_overlays';vdir.mkdir(exist_ok=True)
    for uid in ids:
        a=ants.image_read(str(OUT/f'{uid}.nii.gz')).numpy(); t=TEMPLATE.numpy()
        z=tuple(int(s//2) for s in a.shape)
        fig,ax=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        views=[a[:,:,z[2]],a[:,z[1],:],a[z[0],:,:]]
        tv=[t[:,:,z[2]],t[:,z[1],:],t[z[0],:,:]]
        for j,(x,y) in enumerate(zip(views,tv)):
            ax[j].imshow(np.rot90(y),cmap='gray'); ax[j].contour(np.rot90(x),levels=[np.percentile(x[x>0],70)] if np.any(x>0) else [0],colors='r',linewidths=.5); ax[j].axis('off')
        fig.suptitle(f'{uid} | Dice {audit[uid]["dice"]:.3f} | shear {audit[uid]["shear_max_abs"]:.4f}')
        fig.savefig(vdir/f'{uid}.png',dpi=160);plt.close(fig)
    print('extreme_overlay_count',len(ids))
    print('output',OUT)

if __name__=='__main__':main()
