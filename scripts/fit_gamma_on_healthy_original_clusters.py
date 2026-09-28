#!/usr/bin/env python3

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from fit_healthy_cluster_gamma_mixtures import fit_cluster, MASK_PATH, VOLUMES


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "results" / "scan_clusters_10" / "feature_table.csv"
ASSIGNMENTS = ROOT / "results" / "scan_clusters_10" / "gmm_cluster_assignments.csv"
OUT = ROOT / "results" / "healthy_from_original_scan_clusters_gamma_mixtures"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(FEATURES)[["uid", "actual"]]
    assignments = pd.read_csv(ASSIGNMENTS)
    data = table.merge(assignments, on="uid", validate="one_to_one")
    healthy = data[data.actual == 0].copy()
    healthy.to_csv(OUT / "healthy_scan_memberships.csv", index=False)
    mask = np.asarray(nib.load(str(MASK_PATH)).dataobj) > 0

    components = []
    summaries = []
    for cluster in sorted(data.gmm_cluster.unique()):
        uids = healthy.loc[healthy.gmm_cluster == cluster, "uid"].tolist()
        arrays = []
        for uid in uids:
            image = np.asarray(nib.load(str(VOLUMES / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
            arrays.append(image[mask & np.isfinite(image) & (image > 0)])
        if not arrays:
            summaries.append({"cluster": int(cluster), "healthy_scans": 0, "positive_voxels": 0, "status": "no healthy scans"})
            continue
        values = np.concatenate(arrays)
        fitted, objective, method, fit_voxels = fit_cluster(values, int(cluster))
        for row in fitted:
            row["healthy_scans_in_original_cluster"] = len(uids)
            components.append(row)
        summaries.append({
            "cluster": int(cluster), "healthy_scans": len(uids),
            "positive_voxels": int(len(values)), "fit_voxels": int(fit_voxels),
            "fit_method": method, "objective": objective, "status": "completed",
        })
        print(f"cluster={cluster} healthy_scans={len(uids)} voxels={len(values)} method={method}", flush=True)

    pd.DataFrame(components).to_csv(OUT / "gamma_mixture_components.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "cluster_summary.csv", index=False)
    summary = {
        "source_clusters": "original all-scan metadata/image-derived GMM clusters",
        "healthy_scans": int(len(healthy)), "clusters": int(data.gmm_cluster.nunique()),
        "fit": "two-component Gamma mixture per original cluster after filtering actual==0",
        "component_order": "lower mean/mode = background_uptake; higher mean/mode = specific_striatal_uptake",
        "primary_fit": "VB-style Gamma responsibilities with Dirichlet mixing prior and moment-updated shape/scale",
        "fallback": "Gamma-mixture EM with moment updates",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
