#!/usr/bin/env python3
"""Small healthy/PD diagnostic for cluster-specific striatal Gamma CDF features."""
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import gamma
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / 'results/cv5_submission_flip_clip/oof_predictions.csv'
VOLUMES = ROOT / 'results/healthy_mode500_tail_gamma_clusters_10/normalized_healthy_volumes'
RAW = ROOT / 'results/ants_registration_mni'
HEALTHY_ASSIGN = ROOT / 'results/healthy_mode500_shape2_mode115_gamma_clusters_10/healthy_cluster_assignments.csv'
PARAMS = ROOT / 'results/healthy_mode500_shape2_mode115_gamma_clusters_10/gamma_mixture_components.csv'
MASKDIR = ROOT / 'assets/atlases/fsl/masks'
OUT = ROOT / 'results/validate_gcdf_regions'


def load_mask(names, ref):
    out = np.zeros(ref.shape, dtype=bool)
    for name in names:
        a = np.asarray(nib.load(str(MASKDIR / name)).dataobj) > 0
        if a.shape != ref.shape: raise ValueError(f'mask geometry mismatch: {name}')
        out |= a
    return out


def bg_score(x, row):
    z = x[x > 0]
    z = z[z <= 1.1]
    if len(z) == 0: return -np.inf
    return float(np.mean((row['shape']-1)*np.log(z) - z/row['scale'] - gammaln(row['shape']) - row['shape']*np.log(row['scale'])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS)
    healthy_assign = pd.read_csv(HEALTHY_ASSIGN)
    params = pd.read_csv(PARAMS)
    bg_params = params[params.label == 'background_uptake'].set_index('cluster')
    st_params = params[params.label == 'specific_striatal_uptake'].set_index('cluster')
    template = nib.load(str(MASKDIR / 'MNI152_T1_2mm_brain_mask.nii.gz'))
    caudate = load_mask(['left_caudate.nii.gz','right_caudate.nii.gz'], template)
    putamen = load_mask(['left_putamen.nii.gz','right_putamen.nii.gz'], template)
    selected = []
    for actual, name in [(0, 'healthy'), (1, 'PD')]:
        selected.extend(labels.loc[labels.actual == actual, ['uid','actual']].sort_values('uid').head(5).assign(group=name).to_dict('records'))
    rows=[]
    for rec in selected:
        uid=str(rec['uid'])
        if rec['group']=='healthy':
            cluster=int(healthy_assign.loc[healthy_assign.uid==uid,'cluster'].iloc[0])
            path=VOLUMES/f'{uid}.nii.gz'
        else:
            # Use the raw registered image and the same 500-bin mode scaling.
            img=nib.load(str(RAW/f'{uid}.nii.gz')); a=np.asarray(img.dataobj,dtype=np.float32)
            pos=a[(np.asarray(template.dataobj)>0) & np.isfinite(a) & (a>0)]
            counts,edges=np.histogram(pos,bins=500); mode=(edges[counts.argmax()]+edges[counts.argmax()+1])/2
            scaled=np.zeros_like(a); inside=(np.asarray(template.dataobj)>0)&np.isfinite(a); scaled[inside]=a[inside]/mode
            path=OUT/f'{uid}_scaled.nii.gz'; nib.save(nib.Nifti1Image(scaled,img.affine,img.header),str(path))
            x=scaled[np.asarray(template.dataobj)>0]
            scores={k:bg_score(x,row) for k,row in bg_params.iterrows()}; cluster=int(max(scores,key=scores.get))
        a=np.asarray(nib.load(str(path)).dataobj,dtype=np.float32)
        model=st_params.loc[cluster]
        cdf=gamma.cdf(np.maximum(a,0), a=model['shape'], scale=model['scale'])
        rows.append({'uid':uid,'group':rec['group'],'cluster':cluster,'striatal_shape':model['shape'],
                     'striatal_scale':model['scale'],'striatal_mode':model['mode'],
                     'mean_caudate_gcdf':float(np.mean(cdf[caudate])),
                     'mean_putamen_gcdf':float(np.mean(cdf[putamen])),
                     'mean_striatum_gcdf':float(np.mean(cdf[caudate|putamen]))})
    out=pd.DataFrame(rows); out.to_csv(OUT/'gcdf_region_values_5_healthy_5_pd.csv',index=False)
    print(out.to_string(index=False))
    print('\nGroup means:')
    print(out.groupby('group')[['mean_caudate_gcdf','mean_putamen_gcdf','mean_striatum_gcdf']].mean().to_string())


if __name__=='__main__': main()
