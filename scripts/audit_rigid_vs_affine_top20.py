#!/usr/bin/env python3
from __future__ import annotations
import csv, math, os, tempfile, time
from pathlib import Path
import ants
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/external/niftis_original'
MOVING_MASKS = ROOT / 'results/ants_registration_percentile10_masked/cleaned_masks'
TEMPLATE = ROOT / 'submission_src_ensemble/assets/template.nii'
FIXED_MASK = ROOT / 'assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz'
OUT = ROOT / 'results/audit_rigid_vs_affine_top20'

UIDS = [
    'bmae38sr','7875aud9','xid4n1u3','y08p0fod','uh17j8qo',
    'bglmy62x','c1m3o55i','rqq5qob8','07gbasf5','vfua9bq1',
    '0imjyjcm','344mw04i','0uc87nxp','l5u7nted','nal3gywi',
    'nkb4ehou','6tx0tbkg','ccprhvlz','f5gzpl1q','dhvs41lq',
]

def geometric_center(img):
    size = np.asarray(img.shape[:3], dtype=float)
    idx = ((size - 1.0) / 2.0) * np.asarray(img.spacing[:3], dtype=float)
    return np.asarray(img.origin[:3], dtype=float) + np.asarray(img.direction)[:3, :3].dot(idx)

def dice(a, b):
    a = np.asarray(a, dtype=bool); b = np.asarray(b, dtype=bool)
    den = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / den) if den else float('nan')

def transform_stats(path):
    tx = ants.read_transform(str(path))
    p = np.asarray(tx.parameters, dtype=float)
    M = p[:9].reshape(3, 3)
    t = p[9:12] if len(p) >= 12 else np.full(3, np.nan)
    sv = np.linalg.svd(M, compute_uv=False)
    # Polar decomposition separates the closest proper rotation from scale/shear.
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    angle = math.degrees(math.acos(float(np.clip((np.trace(R)-1.0)/2.0, -1.0, 1.0))))
    return {
        'transform_det': float(np.linalg.det(M)),
        'scale_s1': float(sv[0]), 'scale_s2': float(sv[1]), 'scale_s3': float(sv[2]),
        'scale_mean': float(sv.mean()), 'scale_min': float(sv.min()), 'scale_max': float(sv.max()),
        'rotation_angle_deg': angle,
        'transform_tx': float(t[0]), 'transform_ty': float(t[1]), 'transform_tz': float(t[2]),
        'transform_translation_norm': float(np.linalg.norm(t)),
    }

def register_one(uid, kind, fixed, fixed_mask, moving, moving_mask, init_path, outdir):
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='ants_audit_') as td:
        result = ants.registration(
            fixed=fixed, moving=moving, type_of_transform=kind,
            initial_transform=[str(init_path)], mask=fixed_mask,
            moving_mask=moving_mask, mask_all_stages=True,
            aff_metric='mattes', aff_sampling=32, aff_random_sampling_rate=0.25,
            aff_iterations=(1000, 500, 250, 100),
            aff_shrink_factors=(8, 4, 2, 1),
            aff_smoothing_sigmas=(4, 2, 1, 0), smoothing_in_mm=False,
            outprefix=str(Path(td) / f'{uid}_{kind}_'), verbose=False,
            singleprecision=True)
        warped = result['warpedmovout']
        txs = result['fwdtransforms']
        final_tx = txs[-1]
        out_img = outdir / 'warped' / f'{uid}_{kind}.nii.gz'
        out_tx = outdir / 'transforms' / f'{uid}_{kind}.mat'
        ants.image_write(warped, str(out_img))
        ants.write_transform(ants.read_transform(final_tx), str(out_tx))
        warped_mask = ants.apply_transforms(
            fixed=fixed, moving=moving_mask, transformlist=txs,
            interpolator='nearestNeighbor')
        mask_dice = dice(warped_mask.numpy() > 0, fixed_mask.numpy() > 0)
        row = {'uid': uid, 'method': kind, 'status': 'ok',
               'elapsed_sec': round(time.monotonic() - started, 3),
               'mask_dice': mask_dice,
               'n_forward_transforms': len(txs),
               'warped_path': str(out_img), 'transform_path': str(out_tx)}
        row.update(transform_stats(out_tx))
        return row

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'warped').mkdir(exist_ok=True); (OUT/'transforms').mkdir(exist_ok=True)
    fixed = ants.image_read(str(TEMPLATE)); fixed_mask = ants.image_read(str(FIXED_MASK))
    fixed_center = geometric_center(fixed)
    rows = []
    for i, uid in enumerate(UIDS, 1):
        moving = ants.image_read(str(RAW / f'{uid}.nii.gz'))
        moving_mask = ants.image_read(str(MOVING_MASKS / f'{uid}.nii.gz'))
        moving_center = geometric_center(moving)
        # Same physical geometric-center initialization for both arms.
        init = ants.create_ants_transform(
            transform_type='AffineTransform', dimension=3,
            matrix=np.eye(3).tolist(), center=fixed_center.tolist(),
            translation=(moving_center - fixed_center).tolist())
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as h:
            init_path = h.name
        try:
            ants.write_transform(init, init_path)
            for kind in ('Rigid', 'Affine'):
                try:
                    row = register_one(uid, kind, fixed, fixed_mask, moving,
                                       moving_mask, init_path, OUT)
                    row.update({'initial_tx': float(fixed_center[0]-moving_center[0]),
                                'initial_ty': float(fixed_center[1]-moving_center[1]),
                                'initial_tz': float(fixed_center[2]-moving_center[2]),
                                'initial_translation_norm': float(np.linalg.norm(fixed_center-moving_center))})
                except Exception as exc:
                    row = {'uid': uid, 'method': kind, 'status': 'failed',
                           'error': f'{type(exc).__name__}: {exc}'}
                rows.append(row)
                print(f"[{i}/{len(UIDS)}] {uid} {kind}: {row['status']}"
                      + (f" Dice={row.get('mask_dice', float('nan')):.4f}" if row['status']=='ok' else ''), flush=True)
        finally:
            try: os.unlink(init_path)
            except OSError: pass
    fields = sorted({k for r in rows for k in r})
    with (OUT/'registration_audit.csv').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f'Wrote {OUT/"registration_audit.csv"}', flush=True)

if __name__ == '__main__':
    main()
