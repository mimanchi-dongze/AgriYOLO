from __future__ import annotations

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """Parameter-free attention module described in SimAM."""

    def __init__(self, channels: int | None = None, lambda_val: float = 1e-4) -> None:
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        energy = (x - mean) ** 2
        energy = energy.sum(dim=(2, 3), keepdim=True)
        attention = energy / (energy + self.lambda_val)
        return x * attention


__all__ = ("SimAM",)


