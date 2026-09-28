"""Spatial EGFR predictor built from DeepSpot tile features.

The model is intentionally small enough for the project's sample-level split:
an MLP encodes the 276 DeepSpot features, two GraphSAGE-style layers pass
information between neighboring tiles, and separate heads predict EGFR
abundance and EGFR detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EGFRSpatialConfig:
    input_genes: int
    hidden_width: int = 96
    dropout: float = 0.15


class _GraphSAGELayer(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.update = nn.Linear(width * 2, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def aggregate(hidden: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, destination = edge_index
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(0, destination, hidden[source])
        degree = torch.zeros(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)
        degree.index_add_(0, destination, torch.ones_like(destination, dtype=hidden.dtype))
        return aggregate / degree.clamp_min(1.0).unsqueeze(1)

    def forward(self, hidden: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        neighbors = self.aggregate(hidden, edge_index)
        updated = F.gelu(self.norm(self.update(torch.cat([hidden, neighbors], dim=1))))
        return hidden + self.dropout(updated)


class EGFRSpatialGraphNet(nn.Module):
    """Predict measured Xenium EGFR from DeepSpot features and tile neighbors.

    ``abundance`` is on the standardized Xenium log1p(CP10K) scale. ``presence``
    is a logit for whether measured EGFR is nonzero.
    """

    def __init__(self, config: EGFRSpatialConfig):
        super().__init__()
        if config.input_genes <= 0:
            raise ValueError("input_genes must be positive")
        self.config = config
        width = config.hidden_width
        self.input_encoder = nn.Sequential(
            nn.Linear(config.input_genes + 2, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Dropout(config.dropout),
        )
        self.graph_one = _GraphSAGELayer(width, config.dropout)
        self.graph_two = _GraphSAGELayer(width, config.dropout)
        self.abundance_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(width // 2, 1),
        )
        self.presence_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(width // 2, 1),
        )

    def forward(self, features: torch.Tensor, coordinates: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or coordinates.shape != (features.shape[0], 2):
            raise ValueError("features must be [nodes, genes] and coordinates must be [nodes, 2]")
        hidden = self.input_encoder(torch.cat([features, coordinates], dim=1))
        hidden = self.graph_one(hidden, edge_index)
        hidden = self.graph_two(hidden, edge_index)
        return {
            "abundance": self.abundance_head(hidden).squeeze(1),
            "presence": self.presence_head(hidden).squeeze(1),
        }


def egfr_spatial_loss(
    output: dict[str, torch.Tensor],
    target_standardized: torch.Tensor,
    target_present: torch.Tensor,
    abundance_weight: float = 1.0,
    presence_weight: float = 0.35,
    correlation_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Combined abundance, detection, and spatial-correlation objective."""

    abundance = F.smooth_l1_loss(output["abundance"], target_standardized, beta=0.5)
    presence = F.binary_cross_entropy_with_logits(output["presence"], target_present)
    pred_centered = output["abundance"] - output["abundance"].mean()
    target_centered = target_standardized - target_standardized.mean()
    denominator = torch.sqrt(pred_centered.square().sum() * target_centered.square().sum() + 1e-8)
    correlation = 1.0 - (pred_centered * target_centered).sum() / denominator.clamp_min(1e-8)
    total = abundance_weight * abundance + presence_weight * presence + correlation_weight * correlation
    return {"abundance": abundance, "presence": presence, "correlation": correlation, "total": total}
