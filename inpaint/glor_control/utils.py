"""Utility functions for GLoR-Control."""

import torch
import torch.nn.functional as F


def unfold_local_patches(
    feat: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
) -> torch.Tensor:
    """Return local patches as [B, H*W, K, C]."""

    batch, channels, height, width = feat.shape
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    if dilation < 1:
        raise ValueError("dilation must be positive.")
    padding = dilation * (kernel_size // 2)
    patches = F.unfold(
        feat,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
    )
    patch_count = kernel_size * kernel_size
    patches = patches.transpose(1, 2)
    return patches.view(batch, height * width, patch_count, channels)
