#!/usr/bin/env python3
"""Independent-seed rerun of the 8-slice sigma=0.75 ResNet/DenseNet CV."""

from pathlib import Path

import cv5_real8_post_registration_gaussian as experiment


ROOT = Path(__file__).resolve().parents[1]
experiment.OUT = ROOT / "results/repro_seed20260932/cnn"
experiment.INPUT_PATH = experiment.OUT / "real8_native_post_registration_gaussian.npy"
experiment.SEED = 20260932


if __name__ == "__main__":
    experiment.main()
