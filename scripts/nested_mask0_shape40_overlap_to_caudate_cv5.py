#!/usr/bin/env python3
"""Nested CV for the 40% mask-0 shape variant with overlap owned by caudate."""
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
import nested_repositioned_mask0_sbr_cv5 as base

ALPHA = 0.40
base.OUT = base.R / 'results/nested_mask0_shape40_overlap_to_caudate_cv5'
original_reposition = base.reposition


def circularize(mask, alpha):
    count = int(mask.sum())
    points = np.argwhere(mask)
    center = points.mean(axis=0)
    grid = np.indices(mask.shape).astype(float)
    distance = np.sqrt(sum((grid[d] - center[d]) ** 2 for d in range(3)))
    sphere_score = -distance
    shape_score = distance_transform_edt(mask) - distance_transform_edt(~mask)
    sphere_score = (sphere_score - sphere_score.mean()) / (sphere_score.std() + 1e-8)
    shape_score = (shape_score - shape_score.mean()) / (shape_score.std() + 1e-8)
    score = (1.0 - alpha) * shape_score + alpha * sphere_score
    selected = np.argpartition(score.ravel(), -count)[-count:]
    result = np.zeros(mask.size, dtype=bool)
    result[selected] = True
    return result.reshape(mask.shape)


def shorten_posterior_tail(mask, fraction):
    points = np.argwhere(mask)
    minimum_y = points[:, 1].min()
    center_y = points[:, 1].mean()
    cutoff = minimum_y + fraction * (center_y - minimum_y)
    y_grid = np.indices(mask.shape)[1]
    return mask & (y_grid >= cutoff)


def reposition_shape40(template, original_masks):
    caudate, putamen, shifts = original_reposition(template, original_masks)
    x_grid = np.indices(template.shape)[0]
    caudate = circularize(caudate & (x_grid >= 45), ALPHA) | circularize(caudate & (x_grid < 45), ALPHA)
    putamen = shorten_posterior_tail(putamen & (x_grid >= 45), ALPHA) | shorten_posterior_tail(putamen & (x_grid < 45), ALPHA)
    putamen &= ~caudate
    shifts['shape_adjustment'] = '40pct circularization/tail shortening; overlap to caudate'
    shifts['remaining_overlap_voxels'] = int((caudate & putamen).sum())
    return caudate, putamen, shifts


base.reposition = reposition_shape40

if __name__ == '__main__':
    base.main()
