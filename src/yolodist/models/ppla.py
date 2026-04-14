from __future__ import annotations

import torch
from torch import nn


class PPLALite(nn.Module):
    """Prompt-Prior Lite Attention.

    Lightweight channel gating module for detection features.
    Designed for P3/P4/P5 neck outputs with minimal overhead.
    """

    def __init__(self, channels: int, prior_dim: int = 16, eca_kernel: int = 3, prior_scale: float = 0.5) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if eca_kernel % 2 == 0:
            raise ValueError("eca_kernel must be odd to preserve length")

        self.channels = int(channels)
        self.prior_dim = int(prior_dim)
        self.prior_scale = float(prior_scale)

        self.pool = nn.AdaptiveAvgPool2d(1)
        # ECA-style local channel interaction on pooled descriptor.
        self.eca = nn.Conv1d(1, 1, kernel_size=int(eca_kernel), padding=int(eca_kernel // 2), bias=False)
        self.prior_proj = nn.Linear(self.channels, self.prior_dim, bias=False)
        # Learnable prompt prior anchor. This is intentionally tiny.
        self.prior_token = nn.Parameter(torch.zeros(self.prior_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        if c != self.channels:
            raise ValueError(f"PPLALite expected {self.channels} channels, got {c}")

        pooled = self.pool(x).view(b, c)  # [B, C]
        eca_feat = self.eca(pooled.unsqueeze(1)).squeeze(1)  # [B, C]

        prior_feat = self.prior_proj(pooled)  # [B, D]
        prior_token = self.prior_token.unsqueeze(0).expand(b, -1)  # [B, D]
        cosine = nn.functional.cosine_similarity(prior_feat, prior_token, dim=1).unsqueeze(1)  # [B, 1]

        gate = self.sigmoid(eca_feat + self.prior_scale * cosine)  # [B, C]
        gate = gate.view(b, c, 1, 1)
        return x * gate

