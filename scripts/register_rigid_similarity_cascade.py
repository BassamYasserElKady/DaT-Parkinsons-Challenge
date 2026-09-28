#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,os,tempfile,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import ants,numpy as np

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/external/niftis_original'; MASKS=ROOT/'results/ants_registration_percentile10_masked/cleaned_masks'
TEMPLATE=ROOT/'submission_src_ensemble/assets/template.nii'; FMASK=ROOT/'assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz'
OUT=ROOT/'results/rigid_similarity_cascade'

def one(uid):
    try:
        fixed=ants.image_read(str(TEMPLATE)); moving=ants.image_read(str(RAW/f'{uid}.nii.gz'))
        fm=ants.image_read(str(FMASK)); mm=ants.image_read(str(MASKS/f'{uid}.nii.gz'))
        common=dict(mask=fm,moving_mask=mm,aff_metric='mattes',aff_sampling=32,aff_random_sampling_rate=.25,
                    aff_iterations=(1000,500,250,100),aff_shrink_factors=(8,4,2,1),aff_smoothing_sigmas=(4,2,1,0),singleprecision=True)
        with tempfile.TemporaryDirectory(prefix='ants_cascade_') as td:
            r=ants.registration(fixed=fixed,moving=moving,type_of_transform='Rigid',outprefix=td+'/rigid_',verbose=False,**common)
            s=ants.registration(fixed=fixed,moving=moving,type_of_transform='Similarity',initial_transform=r['fwdtransforms'],outprefix=td+'/similarity_',verbose=False,**common)
            ants.image_write(s['warpedmovout'],str(OUT/f'{uid}.nii.gz'))
            ants.write_transform(ants.read_transform(r['fwdtransforms'][-1]),str(OUT/'transforms'/f'{uid}_rigid.mat'))
            ants.write_transform(ants.read_transform(s['fwdtransforms'][-1]),str(OUT/'transforms'/f'{uid}_similarity.mat'))
        return {'uid':uid,'status':'ok','error':''}
    except Exception as e:return {'uid':uid,'status':'failed','error':f'{type(e).__name__}: {e}'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=max(1,min(16,os.cpu_count() or 1)));a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'transforms').mkdir(exist_ok=True)
    uids=sorted(p.name[:-7] for p in RAW.glob('*.nii.gz')); pending=[u for u in uids if not (OUT/f'{u}.nii.gz').exists() or not (OUT/'transforms'/f'{u}_similarity.mat').exists()]
    print(f'pending={len(pending)} total={len(uids)} jobs={a.jobs}',flush=True); summary=OUT/'registration_summary.csv';existing={}
    if summary.exists():
        for r in csv.DictReader(summary.open(newline='')):existing[r['uid']]=r
    fields=['uid','status','elapsed_sec','error']
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        fs={pool.submit(one,u):u for u in pending}
        for i,f in enumerate(as_completed(fs),1):
            st=time.monotonic();r=f.result();r['elapsed_sec']=round(time.monotonic()-st,3);existing[r['uid']]=r
            if i%10==0 or i==len(pending):print(f"completed={i}/{len(pending)} uid={r['uid']} status={r['status']}",flush=True)
            with summary.open('w',newline='') as h:
                w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(existing[k] for k in sorted(existing))
    print(f'output={OUT}',flush=True)
if __name__=='__main__':main()
