"""Target-aligned control field builders.

The first implementation focuses on SEN2NAIP super-resolution. It maps the LR
observation to the packed FLUX target-token grid so every generated image token
receives an aligned control feature.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SimpleConditionEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScalarEmbedding(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 0:
            value = value[None]
        dtype = next(self.net.parameters()).dtype
        return self.net(value.to(dtype=dtype).reshape(-1, 1))


class TargetObservationControlField(nn.Module):
    """Build task-specific target-aligned control fields."""

    def __init__(
        self,
        ctrl_dim: int = 512,
        sr_in_channels: int = 3,
        sr_align_mode: str = "nearest",
        use_scale_embedding: bool = True,
    ):
        super().__init__()
        self.ctrl_dim = int(ctrl_dim)
        self.sr_align_mode = sr_align_mode
        self.use_scale_embedding = bool(use_scale_embedding)
        self.sr_encoder = SimpleConditionEncoder(sr_in_channels, self.ctrl_dim)
        self.scale_embedding = (
            ScalarEmbedding(self.ctrl_dim) if self.use_scale_embedding else None
        )

    @classmethod
    def from_config(cls, config: dict) -> "TargetObservationControlField":
        toa_config = config.get("toa_control", config)
        task_config = toa_config.get("tasks", {}).get("super_resolution", {})
        return cls(
            ctrl_dim=toa_config.get("ctrl_dim", 512),
            sr_in_channels=task_config.get("in_channels", 3),
            sr_align_mode=task_config.get("align_mode", "nearest"),
            use_scale_embedding=task_config.get("use_scale_embedding", True),
        )

    def forward(
        self,
        batch: dict,
        task: str,
        target_latent_shape: tuple[int, int],
    ) -> torch.Tensor:
        if task == "super_resolution":
            return self.build_sr_field(batch, target_latent_shape)
        raise ValueError(f"Unknown task: {task}")

    def build_sr_field(
        self,
        batch: dict,
        target_latent_shape: tuple[int, int],
    ) -> torch.Tensor:
        lr_image = batch.get("lr_image", batch.get("condition_0"))
        if lr_image is None:
            raise KeyError("Super-resolution TOA-Control requires 'lr_image'.")
        if lr_image.ndim == 3:
            lr_image = lr_image.unsqueeze(0)
        if lr_image.ndim != 4:
            raise ValueError(
                f"Expected BCHW LR image tensor, got {tuple(lr_image.shape)}."
            )

        ctrl = self.sr_encoder(lr_image)
        interpolate_kwargs = {"size": target_latent_shape, "mode": self.sr_align_mode}
        if self.sr_align_mode in {"bilinear", "bicubic"}:
            interpolate_kwargs["align_corners"] = False
        ctrl = F.interpolate(ctrl, **interpolate_kwargs)

        if self.scale_embedding is not None:
            scale = batch.get("scale", batch.get("position_scale", 1.0))
            if not torch.is_tensor(scale):
                scale = torch.tensor(scale, device=ctrl.device)
            scale = scale.to(ctrl.device)
            if scale.numel() == 1 and ctrl.shape[0] > 1:
                scale = scale.expand(ctrl.shape[0])
            ctrl = ctrl + self.scale_embedding(scale).to(ctrl.dtype)[:, :, None, None]

        return ctrl.flatten(2).transpose(1, 2).contiguous()
