"""Utility functions for GLoR-Control."""

import torch
import torch.nn.functional as F


def unfold_local_patches(feat: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Return local patches as [B, H*W, K, C]."""

    batch, channels, height, width = feat.shape
    padding = kernel_size // 2
    patches = F.unfold(feat, kernel_size=kernel_size, padding=padding)
    patch_count = kernel_size * kernel_size
    patches = patches.transpose(1, 2)
    return patches.view(batch, height * width, patch_count, channels)
