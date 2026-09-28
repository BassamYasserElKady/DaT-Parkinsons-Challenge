#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import math
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
REGISTERED = ROOT / "results" / "ants_registration_mni"
TRANSFORMS = ROOT / "results" / "rigid_registration_mni" / "transforms"
TEMPLATE = ROOT / "submission_ensemble_flip_clip_005_995" / "assets" / "template.nii"
OOF = ROOT / "results" / "cv5_submission_flip_clip" / "oof_predictions.csv"
QC = ROOT / "results" / "downloaded_rigid_adaptive_3d_ensemble" / "adaptive_slice_selection.csv"
SLICE_QC = ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "slice_selection.csv"
OUT = ROOT / "results" / "registration_metrics_1362.xlsx"


def load_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def scale_image(a):
    a = np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    positive = a[a > 0]
    if positive.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(positive, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def image_metrics(moving, fixed):
    m = scale_image(moving); f = scale_image(fixed)
    moving_mask = moving > 0
    fixed_mask = fixed > 0
    intersection = int(np.logical_and(moving_mask, fixed_mask).sum())
    union = int(np.logical_or(moving_mask, fixed_mask).sum())
    dice = 2.0 * intersection / (int(moving_mask.sum()) + int(fixed_mask.sum())) if (moving_mask.sum() + fixed_mask.sum()) else float("nan")
    jaccard = intersection / union if union else float("nan")
    surface_m = moving_mask ^ ndimage.binary_erosion(moving_mask, structure=np.ones((3, 3, 3)), border_value=0)
    surface_f = fixed_mask ^ ndimage.binary_erosion(fixed_mask, structure=np.ones((3, 3, 3)), border_value=0)
    spacing = (2.0, 2.0, 2.0)
    if surface_m.any() and surface_f.any():
        dist_f = ndimage.distance_transform_edt(~fixed_mask, sampling=spacing)
        dist_m = ndimage.distance_transform_edt(~moving_mask, sampling=spacing)
        dm = dist_f[surface_m]; df = dist_m[surface_f]
        hausdorff = float(max(dm.max(), df.max()))
        assd = float((dm.sum() + df.sum()) / (len(dm) + len(df)))
    else:
        hausdorff = assd = float("nan")
    valid = np.logical_or(moving_mask, fixed_mask)
    mv, fv = m[valid], f[valid]
    mse = float(np.mean((mv - fv) ** 2)) if mv.size else float("nan")
    ssd = float(np.sum((mv - fv) ** 2)) if mv.size else float("nan")
    ncc = float(np.corrcoef(mv, fv)[0, 1]) if mv.size > 1 and np.std(mv) > 1e-8 and np.std(fv) > 1e-8 else float("nan")
    bins = 64
    joint, _, _ = np.histogram2d(mv, fv, bins=bins, range=[[0, 1], [0, 1]])
    pxy = joint / max(joint.sum(), 1.0); px = pxy.sum(1); py = pxy.sum(0)
    expected = px[:, None] * py[None, :]
    nz = pxy > 0
    mi = float(np.sum(pxy[nz] * np.log((pxy[nz] + 1e-12) / (expected[nz] + 1e-12))))
    data_range = 1.0; c1 = (0.01 * data_range) ** 2; c2 = (0.03 * data_range) ** 2
    mu_m, mu_f = float(m.mean()), float(f.mean())
    var_m, var_f = float(m.var()), float(f.var())
    cov = float(np.mean((m - mu_m) * (f - mu_f)))
    ssim = ((2 * mu_m * mu_f + c1) * (2 * cov + c2)) / ((mu_m**2 + mu_f**2 + c1) * (var_m + var_f + c2))
    return {
        "dice_foreground_proxy": dice, "jaccard_foreground_proxy": jaccard,
        "hausdorff_distance_mm_proxy": hausdorff, "assd_mm_proxy": assd,
        "mutual_information": mi, "normalized_cross_correlation": ncc,
        "ssd_normalized": ssd, "mse_normalized": mse, "ssim_global": float(ssim),
        "moving_foreground_voxels": int(moving_mask.sum()), "template_foreground_voxels": int(fixed_mask.sum()),
    }


def transform_metrics(path):
    text = path.read_text(errors="replace")
    match = re.search(r"^Parameters:\s+(.+)$", text, flags=re.MULTILINE)
    if not match:
        return {"rotation_angle_degrees": float("nan"), "translation_norm_mm": float("nan"), "jacobian_determinant_rigid": 1.0, "harmonic_energy_rigid_proxy": float("nan"), "bending_energy_rigid": 0.0}
    values = np.asarray([float(v) for v in match.group(1).split()], dtype=float)
    angles = values[:3]; translation = values[3:6]
    try:
        matrix = Rotation.from_euler("xyz", angles).as_matrix()
        angle = math.degrees(math.acos(float(np.clip((np.trace(matrix) - 1) / 2, -1, 1))))
        harmonic = float(np.sum((matrix - np.eye(3)) ** 2))
    except Exception:
        angle = float("nan"); harmonic = float("nan")
    return {"rotation_angle_degrees": angle, "translation_norm_mm": float(np.linalg.norm(translation)), "jacobian_determinant_rigid": 1.0, "harmonic_energy_rigid_proxy": harmonic, "bending_energy_rigid": 0.0}


def xml_cell(ref, value, style=None):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        value = "N/A"
    attrs = f' r="{ref}"'
    if style is not None: attrs += f' s="{style}"'
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return f'<c{attrs}><v>{float(value) if isinstance(value, np.floating) else value}</v></c>'
    return f'<c{attrs} t="inlineStr"><is><t>{html.escape(str(value))}</t></is></c>'


def col_name(n):
    out = ""
    while n:
        n, rem = divmod(n - 1, 26); out = chr(65 + rem) + out
    return out


def write_xlsx(sheets):
    workbook = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
    rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    content = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    for i, (name, data) in enumerate(sheets, 1):
        workbook += f'<sheet name="{html.escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
        rels += f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        content += f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
    workbook += '</sheets></workbook>'; rels += '</Relationships>'
    content += '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>'
    with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content); z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr("xl/workbook.xml", workbook); z.writestr("xl/_rels/workbook.xml.rels", rels)
        z.writestr("xl/styles.xml", '<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders><cellXfs count="1"><xf/></cellXfs></styleSheet>')
        for i, (_, data) in enumerate(sheets, 1):
            rows = []
            for ri, row in enumerate(data, 1):
                cells = [xml_cell(f"{col_name(ci)}{ri}", value) for ci, value in enumerate(row, 1)]
                rows.append(f'<row r="{ri}">' + "".join(cells) + "</row>")
            z.writestr(f"xl/worksheets/sheet{i}.xml", '<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + "".join(rows) + '</sheetData></worksheet>')


def main():
    template = np.asarray(nib.load(str(TEMPLATE)).get_fdata(), dtype=np.float32)
    oof = {r["uid"]: r for r in load_csv(OOF)}
    qc = {r["uid"]: r for r in load_csv(QC)}
    for r in load_csv(SLICE_QC): qc.setdefault(r["uid"], {}).update({f"registration_{k}": v for k, v in r.items() if k != "uid"})
    uids = sorted(p.name.removesuffix(".nii.gz") for p in REGISTERED.glob("*.nii.gz"))
    headers = ["uid", "dice_similarity_coefficient", "jaccard_index", "hausdorff_distance_mm", "average_symmetric_surface_distance_mm", "target_registration_error", "mutual_information", "normalized_cross_correlation", "ssd", "mse", "ssim", "jacobian_determinant", "harmonic_energy", "inverse_consistency_error", "difference_image", "checkerboard_display", "flicker_display", "counter_overlay", "bending_energy", "downstream_accuracy", "downstream_log_loss", "robustness_variance", "registration_rotation_degrees", "registration_translation_norm_mm", "moving_foreground_voxels", "template_foreground_voxels", "localizer_score", "detected_slices", "nonempty_slices", "selected_slices", "hot_voxels_total", "notes"]
    data = [headers]; rows = []
    for n, uid in enumerate(uids, 1):
        image = np.asarray(nib.load(str(REGISTERED / f"{uid}.nii.gz")).get_fdata(), dtype=np.float32)
        im = image_metrics(image, template); tm = transform_metrics(TRANSFORMS / f"{uid}.tfm")
        pred = oof.get(uid, {}); actual = int(pred["actual"]) if pred else None; prob = float(pred["probability"]) if pred else float("nan")
        acc = float(int(pred["prediction"]) == actual) if pred else float("nan")
        ll = float(-math.log(max(prob if actual else 1 - prob, 1e-15))) if pred else float("nan")
        q = qc.get(uid, {}); sq = q
        rows.append([uid, im["dice_foreground_proxy"], im["jaccard_foreground_proxy"], im["hausdorff_distance_mm_proxy"], im["assd_mm_proxy"], "N/A", im["mutual_information"], im["normalized_cross_correlation"], im["ssd_normalized"], im["mse_normalized"], im["ssim_global"], tm["jacobian_determinant_rigid"], tm["harmonic_energy_rigid_proxy"], "N/A", "N/A", "N/A", "N/A", "N/A", tm["bending_energy_rigid"], acc, ll, "N/A", tm["rotation_angle_degrees"], tm["translation_norm_mm"], im["moving_foreground_voxels"], im["template_foreground_voxels"], q.get("localizer_score", "N/A"), q.get("detected", "N/A"), q.get("registration_nonempty_slices", "N/A"), q.get("registration_selected_slices", "N/A"), q.get("registration_hot_voxels_total", "N/A"), "Dice/Jaccard/Hausdorff/ASSD use positive-voxel foreground proxies; TRE/inverse consistency/robustness require landmarks or repeated registrations."])
        if n % 100 == 0: print(f"processed {n}/{len(uids)}", flush=True)
    data.extend(rows)
    numeric_cols = list(zip(*rows))
    summary = [["metric", "mean", "std", "count"]]
    for i, h in enumerate(headers):
        vals = np.asarray([v for v in numeric_cols[i] if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v)], dtype=float)
        if vals.size: summary.append([h, float(vals.mean()), float(vals.std()), int(vals.size)])
    definitions = [["field", "interpretation"]]
    definitions += [["dice_similarity_coefficient / jaccard_index", "Computed from positive-voxel foreground masks; proxy because no anatomical segmentations were provided."], ["hausdorff_distance_mm / average_symmetric_surface_distance_mm", "Computed between those proxy foreground surfaces using 2 mm voxel spacing."], ["target_registration_error", "N/A: no corresponding landmarks were provided."], ["mutual_information / NCC / SSD / MSE / SSIM", "Computed between registered scan and template after robust per-image intensity scaling; SSIM is a global SSIM formula, not a local-window implementation."], ["jacobian_determinant", "1.0 for an ideal rigid transform."], ["harmonic_energy", "Rigid-transform proxy based on squared Frobenius norm of R-I; convention-dependent."], ["inverse_consistency_error", "N/A: paired forward/inverse transforms were not available."], ["difference/checkerboard/flicker/counter overlay", "Visual assessments are not scalar metrics; these columns are N/A."], ["bending_energy", "0 for a rigid transform because its displacement field has no second spatial derivatives."], ["downstream_accuracy/log_loss", "Per-scan values from the completed 5-fold out-of-fold ensemble predictions."], ["robustness_variance", "N/A: repeated perturbation/re-registration runs were not available."], ["QC fields", "Registration/localization diagnostics from the saved preprocessing selection records, not independent registration ground truth."]]
    write_xlsx([("PerScan", data), ("Summary", summary), ("Definitions", definitions)])
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__": main()
