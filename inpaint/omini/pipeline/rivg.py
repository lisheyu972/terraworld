"""Training-free reference-information velocity guidance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


@dataclass(frozen=True)
class RIVGConfig:
    method: str = "stable_rivg"
    lambda_max: float = 0.25
    time_power: float = 2.0
    guidance_start_fraction: float = 0.70
    absorption_start_fraction: float = 0.70
    absorption_temperature: float = 0.25
    information_power: float = 1.0
    deficit_power: float = 1.5
    active_quantile: float = 0.70
    max_deficit_weight: float = 2.0
    ema_decay: float = 0.80
    residual_norm_ratio: float = 0.50
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.method not in {"stable_rivg", "deficit_rivg"}:
            raise ValueError("RIVG method must be stable_rivg or deficit_rivg.")
        if self.lambda_max < 0 or self.time_power < 0:
            raise ValueError("RIVG lambda_max and time_power must be non-negative.")
        for name in (
            "guidance_start_fraction",
            "absorption_start_fraction",
            "active_quantile",
            "ema_decay",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"RIVG {name} must be in [0, 1].")
        if self.method == "deficit_rivg" and (
            self.absorption_start_fraction < self.guidance_start_fraction
        ):
            raise ValueError(
                "RIVG absorption_start_fraction cannot precede guidance_start_fraction."
            )
        if self.absorption_temperature <= 0 or self.max_deficit_weight <= 0:
            raise ValueError(
                "RIVG absorption_temperature and max_deficit_weight must be positive."
            )
        if self.information_power <= 0 or self.deficit_power <= 0:
            raise ValueError(
                "RIVG information_power and deficit_power must be positive."
            )
        if self.residual_norm_ratio < 0 or self.eps <= 0:
            raise ValueError(
                "RIVG residual_norm_ratio must be non-negative and eps positive."
            )

    @classmethod
    def from_value(cls, value) -> "RIVGConfig":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("RIVG configuration must be a dict or RIVGConfig.")
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RIVGState:
    deficit_ema: Optional[Tensor] = None


def fuse_rivg_information_maps(
    information_maps: list[Tensor],
    method: str = "max",
) -> Tensor:
    """Fuse equally aligned information maps from multiple references."""

    if not information_maps:
        raise ValueError("At least one RIVG information map is required.")
    reference_shape = information_maps[0].shape
    if any(value.shape != reference_shape for value in information_maps):
        raise ValueError("RIVG information maps must have equal shapes.")
    stacked = torch.stack(information_maps, dim=0)
    if method == "max":
        return stacked.amax(dim=0)
    if method == "mean":
        return stacked.mean(dim=0)
    raise ValueError("RIVG information-map fusion must be max or mean.")


def merge_rivg_condition_latents(condition_latents: list[Tensor]):
    """Build one absorption reference without privileging temporal frame 0."""

    if not condition_latents:
        return None
    if all(
        latent.shape == condition_latents[0].shape
        for latent in condition_latents
    ):
        return torch.stack(condition_latents, dim=0).mean(dim=0)
    return condition_latents[0]


def normalize_per_sample(value: Tensor, eps: float = 1e-6) -> Tensor:
    flat = value.float().flatten(1)
    minimum = flat.amin(dim=1, keepdim=True)
    maximum = flat.amax(dim=1, keepdim=True)
    normalized = (flat - minimum) / (maximum - minimum + eps)
    return normalized.view_as(value).to(value.dtype)


def gaussian_kernel_2d(
    kernel_size: int,
    sigma: float,
    device,
    dtype,
) -> Tensor:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Gaussian kernel_size must be a positive odd integer.")
    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive.")
    radius = kernel_size // 2
    coordinate = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coordinate.square()) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d[:, None] * kernel_1d[None, :]


def gaussian_blur(
    value: Tensor,
    kernel_size: int = 9,
    sigma: float = 2.0,
) -> Tensor:
    kernel = gaussian_kernel_2d(
        kernel_size,
        sigma,
        value.device,
        value.dtype,
    )
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(
        value.shape[1], 1, 1, 1
    )
    return F.conv2d(
        value,
        kernel,
        padding=kernel_size // 2,
        groups=value.shape[1],
    )


def sobel_xy(value: Tensor) -> tuple[Tensor, Tensor]:
    if value.shape[1] != 1:
        value = value.mean(dim=1, keepdim=True)
    kernel_x = value.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    return (
        F.conv2d(value, kernel_x, padding=1),
        F.conv2d(value, kernel_y, padding=1),
    )


def build_sar_information_map(
    sar: Tensor,
    lower_quantile: float = 0.40,
    upper_quantile: float = 0.95,
    eps: float = 1e-6,
) -> Tensor:
    if sar.ndim != 4:
        raise ValueError("SAR tensor must have shape [B, C, H, W].")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Information-map quantiles must satisfy 0 <= low < high <= 1.")
    sar = sar.float()
    if sar.shape[1] != 1:
        sar = sar.mean(dim=1, keepdim=True)
    log_sar = torch.log1p(sar.clamp_min(0.0))
    gx, gy = sobel_xy(log_sar)
    gradient = torch.sqrt(gx.square() + gy.square() + eps)

    j_xx = gaussian_blur(gx.square(), kernel_size=7, sigma=1.5)
    j_yy = gaussian_blur(gy.square(), kernel_size=7, sigma=1.5)
    j_xy = gaussian_blur(gx * gy, kernel_size=7, sigma=1.5)
    trace = j_xx + j_yy
    discriminant = torch.sqrt(
        (j_xx - j_yy).square() + 4.0 * j_xy.square() + eps
    )
    lambda_1 = 0.5 * (trace + discriminant)
    lambda_2 = 0.5 * (trace - discriminant)
    coherence = (lambda_1 - lambda_2) / (lambda_1 + lambda_2 + eps)

    smooth = gaussian_blur(log_sar, kernel_size=7, sigma=1.5)
    local_mean = gaussian_blur(smooth, kernel_size=7, sigma=1.5)
    local_square_mean = gaussian_blur(
        smooth.square(), kernel_size=7, sigma=1.5
    )
    variance = (local_square_mean - local_mean.square()).clamp_min(0.0)
    raw = (
        0.50 * normalize_per_sample(gradient, eps) * coherence
        + 0.30 * normalize_per_sample(variance, eps)
        + 0.20 * coherence
    )
    flat = raw.flatten(1)
    lower = torch.quantile(
        flat.float(), lower_quantile, dim=1, keepdim=True
    ).to(raw.dtype)
    upper = torch.quantile(
        flat.float(), upper_quantile, dim=1, keepdim=True
    ).to(raw.dtype)
    normalized = (flat - lower) / (upper - lower + eps)
    result = normalized.clamp(0.0, 1.0).view_as(raw)
    ensure_finite("RIVG SAR information map", result)
    return result


def pil_to_unit_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).div_(255.0)


def tensor_to_pil(value: Tensor) -> Image.Image:
    value = value.detach().float().squeeze(0).clamp(0.0, 1.0)
    array = value.permute(1, 2, 0).mul(255.0).round().byte().cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def prepare_rivg_reference(
    sar_image: Image.Image,
    *,
    weak_kernel_size: int = 9,
    weak_sigma: float = 2.0,
    lower_quantile: float = 0.40,
    upper_quantile: float = 0.95,
) -> tuple[Image.Image, Tensor]:
    sar = pil_to_unit_tensor(sar_image)
    weak = gaussian_blur(sar, weak_kernel_size, weak_sigma).clamp(0.0, 1.0)
    information_map = build_sar_information_map(
        sar,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    return tensor_to_pil(weak), information_map


def resize_map_to_tokens(
    value: Tensor,
    latent_image_ids: Tensor,
    *,
    device,
    dtype,
) -> tuple[Tensor, tuple[int, int]]:
    height = int(latent_image_ids[:, 1].max().item()) + 1
    width = int(latent_image_ids[:, 2].max().item()) + 1
    resized = F.interpolate(
        value.float().to(device),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    tokens = resized.flatten(2).transpose(1, 2).to(dtype=dtype)
    if tokens.shape[1] != latent_image_ids.shape[0]:
        raise ValueError(
            "RIVG information-map token count does not match generated latents: "
            f"{tokens.shape[1]} != {latent_image_ids.shape[0]}."
        )
    return tokens, (height, width)


def clamp_residual_norm(
    residual: Tensor,
    reference: Tensor,
    ratio: float,
    eps: float = 1e-6,
) -> Tensor:
    dimensions = tuple(range(1, residual.ndim))
    residual_norm = torch.sqrt(
        residual.float().square().sum(dim=dimensions, keepdim=True) + eps
    )
    reference_norm = torch.sqrt(
        reference.float().square().sum(dim=dimensions, keepdim=True) + eps
    )
    scale = (ratio * reference_norm / (residual_norm + eps)).clamp(max=1.0)
    return (residual.float() * scale).to(residual.dtype)


def predict_clean_latent_flow(
    latent: Tensor,
    velocity: Tensor,
    normalized_timestep: Tensor,
) -> Tensor:
    timestep = normalized_timestep
    while timestep.ndim < latent.ndim:
        timestep = timestep.unsqueeze(-1)
    return latent - timestep * velocity


def _spatial_structure(
    tokens: Tensor,
    spatial_shape: tuple[int, int],
    eps: float,
) -> Tensor:
    height, width = spatial_shape
    if tokens.shape[1] != height * width:
        raise ValueError("RIVG latent tokens do not match their spatial shape.")
    feature_map = tokens.float().transpose(1, 2).reshape(
        tokens.shape[0], tokens.shape[2], height, width
    )
    scalar = feature_map.square().mean(dim=1, keepdim=True).add(eps).sqrt()
    gx, gy = sobel_xy(scalar)
    return normalize_per_sample(torch.sqrt(gx.square() + gy.square() + eps), eps)


def latent_structural_absorption(
    predicted_clean_latent: Tensor,
    condition_latent: Tensor,
    spatial_shape: tuple[int, int],
    temperature: float = 0.25,
    eps: float = 1e-6,
) -> Tensor:
    predicted = _spatial_structure(predicted_clean_latent, spatial_shape, eps)
    condition = _spatial_structure(condition_latent, spatial_shape, eps)
    absorption = torch.exp(-(predicted - condition).abs() / (temperature + eps))
    return absorption.flatten(2).transpose(1, 2).clamp(0.0, 1.0)


def normalize_deficit(
    deficit: Tensor,
    active_quantile: float = 0.70,
    max_weight: float = 2.0,
    eps: float = 1e-6,
) -> Tensor:
    flat = deficit.float().flatten(1)
    threshold = torch.quantile(
        flat, active_quantile, dim=1, keepdim=True
    ).to(deficit.dtype)
    active = (flat >= threshold).to(flat.dtype)
    selected = flat * active
    active_mean = selected.sum(dim=1, keepdim=True) / active.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)
    normalized = (selected / (active_mean + eps)).clamp(0.0, max_weight)
    return normalized.view_as(deficit).to(deficit.dtype)


def build_deficit_map(
    information_map: Tensor,
    absorption: Tensor,
    config: RIVGConfig,
) -> Tensor:
    deficit = information_map.clamp(0.0, 1.0).pow(config.information_power) * (
        1.0 - absorption
    ).clamp(0.0, 1.0).pow(config.deficit_power)
    return normalize_deficit(
        deficit,
        active_quantile=config.active_quantile,
        max_weight=config.max_deficit_weight,
        eps=config.eps,
    )


def apply_rivg_velocity(
    *,
    latent: Tensor,
    velocity_full: Tensor,
    velocity_weak: Tensor,
    condition_latent: Optional[Tensor],
    information_map: Tensor,
    spatial_shape: tuple[int, int],
    normalized_timestep: Tensor,
    progress_fraction: float,
    config: RIVGConfig,
    state: RIVGState,
) -> tuple[Tensor, RIVGState, Tensor]:
    if progress_fraction < config.guidance_start_fraction:
        return velocity_full, state, torch.zeros_like(information_map)

    residual = clamp_residual_norm(
        velocity_full - velocity_weak,
        velocity_full,
        ratio=config.residual_norm_ratio,
        eps=config.eps,
    )
    gating_map = information_map
    if (
        config.method == "deficit_rivg"
        and progress_fraction >= config.absorption_start_fraction
    ):
        if condition_latent is None:
            raise ValueError("Deficit RIVG requires the full condition latent.")
        clean_latent = predict_clean_latent_flow(
            latent,
            velocity_full,
            normalized_timestep,
        )
        absorption = latent_structural_absorption(
            clean_latent,
            condition_latent,
            spatial_shape,
            temperature=config.absorption_temperature,
            eps=config.eps,
        )
        deficit = build_deficit_map(information_map, absorption, config)
        if state.deficit_ema is None:
            deficit_ema = deficit
        else:
            deficit_ema = (
                config.ema_decay * state.deficit_ema
                + (1.0 - config.ema_decay) * deficit
            )
        state.deficit_ema = deficit_ema.detach()
        gating_map = deficit_ema

    timestep = normalized_timestep
    while timestep.ndim < velocity_full.ndim:
        timestep = timestep.unsqueeze(-1)
    strength = config.lambda_max * (1.0 - timestep).clamp(0.0, 1.0).pow(
        config.time_power
    )
    guided = (
        velocity_full.float()
        + strength.float() * gating_map.float() * residual.float()
    ).to(velocity_full.dtype)
    for name, value in (
        ("full velocity", velocity_full),
        ("weak velocity", velocity_weak),
        ("guided velocity", guided),
        ("gating map", gating_map),
    ):
        ensure_finite(f"RIVG {name}", value)
    return guided, state, gating_map


def ensure_finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf values.")
