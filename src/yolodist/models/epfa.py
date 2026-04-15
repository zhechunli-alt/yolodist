from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class EPFALite(nn.Module):
    """Edge-Prior Feature Attention with tiny channel and spatial gates."""

    current_input: torch.Tensor | None = None

    def __init__(self, channels: int = 0, eca_kernel: int = 3, edge_kernel: int = 3, alpha_init: float = 0.5) -> None:
        super().__init__()
        if eca_kernel % 2 == 0 or edge_kernel % 2 == 0:
            raise ValueError("eca_kernel and edge_kernel must be odd")
        self.channels = int(channels) if channels > 0 else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Conv1d(1, 1, kernel_size=eca_kernel, padding=eca_kernel // 2, bias=False)
        self.edge_conv = nn.Conv2d(1, 1, kernel_size=edge_kernel, padding=edge_kernel // 2, bias=True)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.sigmoid = nn.Sigmoid()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    @classmethod
    def set_current_input(cls, image: torch.Tensor | None) -> None:
        cls.current_input = image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        channel_gate = self.pool(x).view(b, 1, c)
        channel_gate = self.sigmoid(self.channel_conv(channel_gate)).view(b, c, 1, 1)

        image = self.current_input
        if image is None or image.ndim != 4:
            edge_prior = x.mean(dim=1, keepdim=True)
        else:
            gray = image.mean(dim=1, keepdim=True)
            gray = gray.to(device=x.device, dtype=x.dtype)
            edge_x = F.conv2d(gray, self.sobel_x.to(x.dtype), padding=1)
            edge_y = F.conv2d(gray, self.sobel_y.to(x.dtype), padding=1)
            edge_prior = torch.sqrt(edge_x.square() + edge_y.square() + 1e-6)
            edge_prior = edge_prior / edge_prior.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            edge_prior = F.interpolate(edge_prior, size=(h, w), mode="bilinear", align_corners=False)

        spatial_gate = self.sigmoid(self.edge_conv(edge_prior))
        return x * channel_gate * (1.0 + self.alpha.view(1, 1, 1, 1) * spatial_gate)
