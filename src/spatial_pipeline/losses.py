"""Loss functions for a future trainer; this module performs no training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def spatial_pearson_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """One minus mean gene-wise Pearson correlation across tiles."""

    pred_centered = prediction - prediction.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=0)
    denominator = torch.sqrt(
        (pred_centered.square().sum(dim=0) * target_centered.square().sum(dim=0)) + eps
    )
    correlation = numerator / denominator.clamp_min(eps)
    return 1.0 - torch.nan_to_num(correlation, nan=0.0).mean()


def pairwise_gene_rank_loss(prediction: torch.Tensor, target: torch.Tensor, pairs: int = 3000) -> torch.Tensor:
    """Logistic ordering loss over sampled within-tile gene pairs."""

    if pairs <= 0:
        raise ValueError("pairs must be positive")
    genes = prediction.shape[1]
    left = torch.randint(0, genes, (pairs,), device=prediction.device)
    right = torch.randint(0, genes, (pairs,), device=prediction.device)
    keep = left != right
    left, right = left[keep], right[keep]
    target_sign = torch.sign(target[:, left] - target[:, right])
    prediction_difference = prediction[:, left] - prediction[:, right]
    informative = target_sign != 0
    if not informative.any():
        return prediction.sum() * 0.0
    return F.softplus(-target_sign[informative] * prediction_difference[informative]).mean()


def calibration_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    pearson_weight: float = 0.5,
    rank_weight: float = 0.15,
    rank_pairs: int = 3000,
) -> dict[str, torch.Tensor]:
    """Return unweighted terms and total for a future, explicitly configured trainer."""

    mse = F.mse_loss(prediction, target)
    pearson = spatial_pearson_loss(prediction, target)
    rank = pairwise_gene_rank_loss(prediction, target, pairs=rank_pairs)
    total = mse + pearson_weight * pearson + rank_weight * rank
    return {"mse": mse, "spatial_pearson": pearson, "gene_rank": rank, "total": total}
