"""Compact PyTorch task-progress prior for Colab-scale training."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from promptmorph.learning.features import PROGRESS_FEATURE_NAMES

PROGRESS_CLASSES = ("approach", "grasp", "transport", "place")


class SubgoalProgressNet(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        input_dim = len(PROGRESS_FEATURE_NAMES)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(PROGRESS_CLASSES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != len(PROGRESS_FEATURE_NAMES):
            raise ValueError(
                f"expected {len(PROGRESS_FEATURE_NAMES)} features, got {features.shape[-1]}"
            )
        return cast(torch.Tensor, self.network(features))
