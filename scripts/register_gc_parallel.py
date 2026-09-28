#!/usr/bin/env python3
import argparse, csv, os, shutil, tempfile, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/ants_registration_gc_full/registered/rigid_gc_standard'
ARCHIVE = ROOT / 'data/external/niftis.zip'
TEMPLATE = ROOT / 'data/external/drive_inputs/template.nii'
SPLIT = ROOT / 'results/ants_registered_consecutive_resnet18_cbam/split.csv'

def register_one(uid):
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
    import ants
    dest = OUT / f'{uid}.nii.gz'
    if dest.exists(): return uid, 'existing', 0.0
    started = time.monotonic()
    with ZipFile(ARCHIVE) as archive, tempfile.TemporaryDirectory(prefix='gc_registration_') as td:
        moving_path = Path(td) / f'{uid}.nii.gz'
        with archive.open(f'{uid}.nii.gz') as source, moving_path.open('wb') as target:
            shutil.copyfileobj(source, target)
        result = ants.registration(
            fixed=ants.image_read(str(TEMPLATE)), moving=ants.image_read(str(moving_path)),
            type_of_transform='Rigid', grad_step=0.1, aff_metric='gc', aff_sampling=32,
            aff_random_sampling_rate=0.20, aff_iterations=(100,70,50),
            aff_shrink_factors=(4,2,1), aff_smoothing_sigmas=(2,1,0),
            verbose=False, singleprecision=True)
        tmp = dest.with_suffix('.tmp.nii.gz')
        ants.image_write(result['warpedmovout'], str(tmp))
        tmp.replace(dest)
    return uid, 'ok', time.monotonic()-started

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=8); args=ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows=list(csv.DictReader(SPLIT.open(newline='', encoding='utf-8')))
    uids=[r['uid'] for r in rows]
    done=sum((OUT/f'{u}.nii.gz').exists() for u in uids)
    print(f'start total={len(uids)} existing={done} workers={args.workers} threads_per_worker=1', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(register_one,u):u for u in uids}
        for n,f in enumerate(as_completed(futures),1):
            uid,status,sec=f.result()
            print(f'completed={n}/{len(uids)} uid={uid} status={status} sec={sec:.1f}', flush=True)
    print('registration_complete=1', flush=True)
if __name__=='__main__': main()
