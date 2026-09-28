from pathlib import Path
import json
import time

import ants
import pandas as pd
import torch

from submission_src_ensemble.src.model import load_model, probability
from submission_src_ensemble.src.preprocessing import prepare_tensor


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIRS = [ROOT / "google_drive_folder_1", ROOT / "google_drive_folder_2"]
ASSETS_DIR = ROOT / "submission_src_ensemble" / "assets"
OUTPUT_PATH = ROOT / "results" / "drive_200_ensemble_predictions.csv"


def main():
    paths = sorted(
        path
        for input_dir in INPUT_DIRS
        for path in input_dir.glob("*.nii.gz")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with (ASSETS_DIR / "standardization.json").open() as file:
        standardization = json.load(file)
    template = ants.image_read(str(ASSETS_DIR / "template.nii"))
    model = load_model(ASSETS_DIR / "model" / "ensemble_weights.pt", device)

    rows = []
    started = time.monotonic()
    with torch.inference_mode():
        for index, path in enumerate(paths, start=1):
            tensor = prepare_tensor(
                path,
                template,
                standardization["channel_means"],
                standardization["channel_stds"],
            )
            rows.append(
                {
                    "uid": path.name.removesuffix(".nii.gz"),
                    "is_pathologic": probability(model, tensor, device),
                }
            )
            if index % 10 == 0 or index == len(paths):
                elapsed = time.monotonic() - started
                print(f"Processed {index}/{len(paths)} scans in {elapsed:.1f} seconds", flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
