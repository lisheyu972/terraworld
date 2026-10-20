"""Task-aware soft correspondence bias builders."""

from __future__ import annotations

import torch
from torch import nn


def make_coord_grid(height: int, width: int, device, dtype) -> torch.Tensor:
    ys = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    xs = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy, xx], dim=-1).view(height * width, 2)


def gaussian_spatial_bias(
    target_hw: tuple[int, int],
    cond_hw: tuple[int, int],
    sigma: float,
    scale_factor: float | None = None,
    device=None,
    dtype=None,
) -> torch.Tensor:
    target_height, target_width = target_hw
    cond_height, cond_width = cond_hw
    target_coords = make_coord_grid(target_height, target_width, device, dtype)
    cond_coords = make_coord_grid(cond_height, cond_width, device, dtype)

    if scale_factor is not None:
        target_coords = target_coords / float(scale_factor)
    else:
        target_coords = target_coords.clone()
        target_coords[:, 0] *= (cond_height - 1) / max(target_height - 1, 1)
        target_coords[:, 1] *= (cond_width - 1) / max(target_width - 1, 1)

    dist2 = ((target_coords[:, None, :] - cond_coords[None, :, :]) ** 2).sum(dim=-1)
    bias = -dist2 / (2 * float(sigma) ** 2)
    return bias - bias.amax(dim=-1, keepdim=True)


class SoftCorrespondenceBiasBuilder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config or {}

    def forward(
        self,
        task: str,
        target_hw: tuple[int, int],
        condition_hw: tuple[int, int],
        batch: dict,
        device,
        dtype,
    ) -> torch.Tensor:
        if task == "super_resolution":
            return self.build_sr_bias(target_hw, condition_hw, batch, device, dtype)
        if task == "frame_interpolation":
            return self.build_interp_bias(target_hw, condition_hw, batch, device, dtype)
        if task == "optical_to_sar":
            return self.build_o2sar_bias(target_hw, condition_hw, batch, device, dtype)
        raise ValueError(f"Unknown SCB task: {task}")

    def build_sr_bias(self, target_hw, condition_hw, batch, device, dtype):
        config = self.config.get("super_resolution", {})
        sigma = float(config.get("sigma", 1.5))
        scale = batch.get("scale", batch.get("position_scale", 4.0))
        if torch.is_tensor(scale):
            scale_value = float(scale.detach().flatten()[0].item())
            batch_size = int(scale.shape[0]) if scale.ndim > 0 else 1
        else:
            scale_value = float(scale)
            batch_size = int(batch.get("batch_size", 1))
        bias = gaussian_spatial_bias(
            target_hw,
            condition_hw,
            sigma=sigma,
            scale_factor=scale_value,
            device=device,
            dtype=dtype,
        )
        return bias[None, None].expand(batch_size, 1, -1, -1).contiguous()

    def build_interp_bias(self, target_hw, condition_hw, batch, device, dtype):
        config = self.config.get("frame_interpolation", {})
        sigma = float(config.get("sigma", 1.5))
        eps = float(config.get("eps", 1.0e-6))
        tau = batch.get("tau", 0.5)
        if not torch.is_tensor(tau):
            tau = torch.tensor([tau], device=device, dtype=dtype)
        tau = tau.to(device=device, dtype=dtype).flatten()
        spatial = gaussian_spatial_bias(
            target_hw, condition_hw, sigma=sigma, device=device, dtype=dtype
        )
        bias0 = spatial[None] + torch.log(1.0 - tau[:, None, None] + eps)
        bias1 = spatial[None] + torch.log(tau[:, None, None] + eps)
        return torch.cat([bias0, bias1], dim=-1)[:, None]

    def build_o2sar_bias(self, target_hw, condition_hw, batch, device, dtype):
        config = self.config.get("optical_to_sar", {})
        sigma = float(config.get("sigma", 2.0))
        batch_size = int(batch.get("batch_size", 1))
        bias = gaussian_spatial_bias(
            target_hw, condition_hw, sigma=sigma, device=device, dtype=dtype
        )
        return bias[None, None].expand(batch_size, 1, -1, -1).contiguous()
