#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, os, tempfile, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import ants

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/external/niftis_original'
MASKS=ROOT/'results/ants_registration_percentile10_masked/cleaned_masks'
TEMPLATE=ROOT/'submission_src_ensemble/assets/template.nii'
FIXED_MASK=ROOT/'assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz'
OUT=ROOT/'results/phase_b_ants_affine_masked'

def geometric_center(img):
    import numpy as np
    size=np.asarray(img.shape[:3],dtype=float)
    idx=((size-1.0)/2.0)*np.asarray(img.spacing[:3],dtype=float)
    return np.asarray(img.origin[:3],dtype=float)+np.asarray(img.direction)[:3,:3].dot(idx)

def one(uid):
    try:
        fixed=ants.image_read(str(TEMPLATE)); moving=ants.image_read(str(RAW/f'{uid}.nii.gz'))
        fixed_mask=ants.image_read(str(FIXED_MASK)); moving_mask=ants.image_read(str(MASKS/f'{uid}.nii.gz'))
        fc=geometric_center(fixed); mc=geometric_center(moving)
        tx=ants.create_ants_transform(transform_type='AffineTransform',dimension=3,
                                      matrix=[[1,0,0],[0,1,0],[0,0,1]],
                                      center=fc.tolist(),translation=(fc-mc).tolist())
        with tempfile.TemporaryDirectory(prefix='ants_phase_b_') as td:
            init_path=str(Path(td)/f'{uid}_init.mat'); ants.write_transform(tx,init_path)
            result=ants.registration(fixed=fixed,moving=moving,type_of_transform='Affine',
                initial_transform=[init_path],mask=fixed_mask,moving_mask=moving_mask,mask_all_stages=True,
                aff_metric='mattes',aff_sampling=32,aff_random_sampling_rate=1.0,
                aff_iterations=(250,250,250,100),aff_shrink_factors=(8,4,2,1),
                aff_smoothing_sigmas=(3,2,1,0),smoothing_in_mm=False,
                outprefix=str(Path(td)/f'{uid}_'),verbose=False,singleprecision=True)
            ants.image_write(result['warpedmovout'],str(OUT/f'{uid}.nii.gz'))
            ants.write_transform(ants.read_transform(result['fwdtransforms'][-1]),str(OUT/'transforms'/f'{uid}.mat'))
        return {'uid':uid,'status':'ok','metric':'','iterations':'','error':''}
    except Exception as e:
        return {'uid':uid,'status':'failed','metric':'','iterations':'','error':f'{type(e).__name__}: {e}'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--jobs',type=int,default=max(1,min(16,os.cpu_count() or 1))); a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'transforms').mkdir(exist_ok=True)
    uids=sorted(p.name[:-7] for p in RAW.glob('*.nii.gz')); pending=[u for u in uids if not (OUT/f'{u}.nii.gz').exists() or not (OUT/'transforms'/f'{u}.mat').exists()]
    print(f'pending={len(pending)} total={len(uids)} jobs={a.jobs}',flush=True)
    summary=OUT/'registration_summary.csv'; existing={}
    if summary.exists():
        for r in csv.DictReader(summary.open(newline='')): existing[r['uid']]=r
    fields=['uid','status','metric','iterations','elapsed_sec','error']
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        fs={pool.submit(one,u):u for u in pending}
        for i,f in enumerate(as_completed(fs),1):
            start=time.monotonic(); r=f.result(); r['elapsed_sec']=round(time.monotonic()-start,3); existing[r['uid']]=r
            if i%10==0 or i==len(pending): print(f"completed={i}/{len(pending)} uid={r['uid']} status={r['status']}",flush=True)
            with summary.open('w',newline='') as h:
                w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(existing[k] for k in sorted(existing))
    print(f'output={OUT}',flush=True)

if __name__=='__main__':main()
