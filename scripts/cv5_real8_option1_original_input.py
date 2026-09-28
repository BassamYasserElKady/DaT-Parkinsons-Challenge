#!/usr/bin/env python3
"""Evaluate the 8-slice ensemble on option-1 registrations."""

from pathlib import Path

import cv5_real8_post_registration_gaussian as base


ROOT = Path(__file__).resolve().parents[1]
base.OUT = ROOT / "results/cv5_real8_option1_original_input"
base.REGISTERED = ROOT / "results/ants_registration_smoothed_optimizer_original_input"
base.INPUT_PATH = base.OUT / "real8_native_option1_original_input.npy"
base.SIGMA = 0.0
base.ndimage.gaussian_filter = lambda volume, sigma, mode="nearest": volume


if __name__ == "__main__":
    base.main()
