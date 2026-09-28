#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "submission_ensemble_flip_clip_005_995"
OUT = ROOT / "results" / "cv5_submission_flip_clip" / "gradcam_misclassified"
X_PATH = ROOT / "results" / "downloaded_rigid_adaptive_3d_ensemble" / "input_windows.npy"
OOF_PATH = ROOT / "results" / "cv5_submission_flip_clip" / "oof_predictions.csv"
QC_PATH = ROOT / "results" / "downloaded_rigid_adaptive_3d_ensemble" / "adaptive_slice_selection.csv"
SLICE_QC_PATH = ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "slice_selection.csv"


def load_qc():
    qc = {}
    with QC_PATH.open() as f:
        for row in csv.DictReader(f):
            qc[row["uid"]] = row
    with SLICE_QC_PATH.open() as f:
        for row in csv.DictReader(f):
            qc.setdefault(row["uid"], {}).update({f"slice_{k}": v for k, v in row.items() if k != "uid"})
    return qc


def cam_for_layer(model, layer, x, target):
    activation = {}
    gradient = {}

    def forward_hook(_, __, output):
        activation["value"] = output
        output.retain_grad()

    handle = layer.register_forward_hook(forward_hook)
    model.zero_grad(set_to_none=True)
    output = model(x)
    output[:, 1].sum().backward()
    handle.remove()
    act = activation["value"]
    grad = act.grad if act.grad is not None else gradient.get("value")
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=target, mode="trilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    lo, hi = float(cam.min()), float(cam.max())
    return (cam - lo) / (hi - lo + 1e-8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(OOF_PATH.open()))
    wrong = [r for r in rows if int(r["prediction"]) != int(r["actual"])]
    x = np.load(X_PATH, mmap_mode="r")
    uid_to_index = {r["uid"]: i for i, r in enumerate(rows)}
    qc = load_qc()
    sys.path.insert(0, str(PACKAGE))
    from src.model import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(PACKAGE / "assets" / "model" / "ensemble_weights.pt", device)
    model.eval()
    # Eight evenly spaced views represent the 16 depth planes supplied to the network.
    view_indices = np.linspace(0, 15, 8).round().astype(int)
    manifest = []

    for number, row in enumerate(wrong, 1):
        uid = row["uid"]
        actual = int(row["actual"]); prediction = int(row["prediction"]); p_oof = float(row["probability"])
        per_scan_ll = -np.log(np.clip(p_oof if actual else 1 - p_oof, 1e-15, 1.0))
        tensor = torch.from_numpy(np.asarray(x[uid_to_index[uid]], dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        tensor.requires_grad_(True)
        rcam = cam_for_layer(model.resnet, model.resnet.layer4[-1], tensor, (16, 64, 64))
        dcam = cam_for_layer(model.densenet, model.densenet.features.denseblock4, tensor, (16, 64, 64))
        rw = float(model.resnet_weight.detach().cpu()); dw = float(model.densenet_weight.detach().cpu())
        cam = (rw * rcam + dw * dcam) / (rw + dw)
        volume = tensor.detach().cpu().numpy()[0, 0]
        vmin, vmax = np.percentile(volume, [1, 99])
        info = qc.get(uid, {})
        detected = info.get("detected", "NA")
        localizer = info.get("localizer_score", "NA")
        nonempty = info.get("slice_nonempty_slices", "NA")
        selected = info.get("slice_selected_slices", "NA")
        hot = info.get("slice_hot_voxels_total", "NA")
        fig, axes = plt.subplots(2, 8, figsize=(22, 6.8), constrained_layout=False)
        for j, z in enumerate(view_indices):
            image = volume[z]
            axes[0, j].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
            axes[0, j].set_title(f"Plane {z + 1}", fontsize=9)
            axes[1, j].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
            axes[1, j].imshow(cam[z], cmap="jet", vmin=0, vmax=1, alpha=0.48)
            axes[1, j].set_title("Grad-CAM", fontsize=9)
            axes[0, j].axis("off"); axes[1, j].axis("off")
        fig.suptitle(
            f"{uid}_{number}   |   actual={actual}  prediction={prediction}  "
            f"OOF probability={p_oof:.4f}  per-scan log loss={per_scan_ll:.4f}\n"
            f"Registration/localization QC: detected={detected}  localizer_score={localizer}  "
            f"nonempty_slices={nonempty}  selected_slices={selected}  hot_voxels={hot}\n"
            "Top: 8 representative planes from the 16-plane model input  |  Bottom: full-data submission-model Grad-CAM",
            fontsize=11,
        )
        fig.subplots_adjust(top=0.79, wspace=0.03, hspace=0.10)
        output = OUT / f"{uid}_{number}.png"
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"number": number, "uid": uid, "filename": output.name, "actual": actual, "prediction": prediction, "oof_probability": p_oof, "per_scan_log_loss": float(per_scan_ll)})
        if number % 10 == 0 or number == len(wrong):
            print(f"generated {number}/{len(wrong)}", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"generated": len(manifest), "output": str(OUT), "device": str(device), "view_indices_zero_based": view_indices.tolist()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
