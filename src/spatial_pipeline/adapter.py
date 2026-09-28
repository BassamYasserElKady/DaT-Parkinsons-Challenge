"""DeepSpot-M-only calibration adapter definition.

This module defines a model but intentionally does not train or load data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AdapterConfig:
    genes: int
    hidden_width: int = 128
    latent_width: int = 96
    dropout: float = 0.20


class DeepSpotCalibrationAdapter(nn.Module):
    """Residual calibration from normalized DeepSpot-M tiles to Xenium targets.

    Only ``tile_predictions`` is accepted by ``forward``. No external patient-level
    conditioning input is present by design.
    """

    def __init__(self, config: AdapterConfig):
        super().__init__()
        if config.genes <= 0:
            raise ValueError("genes must be positive")
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.genes, config.hidden_width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_width, config.latent_width),
            nn.GELU(),
        )
        self.correction_head = nn.Sequential(
            nn.Linear(config.latent_width, config.hidden_width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_width, config.genes),
        )
        self.gene_scale = nn.Parameter(torch.ones(config.genes))
        self.gene_bias = nn.Parameter(torch.zeros(config.genes))

    def forward(self, tile_predictions: torch.Tensor) -> torch.Tensor:
        if tile_predictions.ndim != 2 or tile_predictions.shape[1] != self.config.genes:
            raise ValueError("tile_predictions must have shape [tiles, configured_genes]")
        context = self.encoder(tile_predictions)
        base = tile_predictions * self.gene_scale.unsqueeze(0) + self.gene_bias.unsqueeze(0)
        return base + self.correction_head(context)


def default_device() -> torch.device:
    """Use CUDA automatically when available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
