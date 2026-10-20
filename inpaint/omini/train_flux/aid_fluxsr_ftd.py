"""AID paired-data adaptation of FluxSR Flow Trajectory Distillation.

The implementation follows Equations (16)--(23) and Algorithm 1 of:
"One Diffusion Step to Real-World Super-Resolution via Flow Trajectory
Distillation" (Li et al., ICML 2025).

The paper trains from offline FLUX-generated noise/image pairs. For the paired
AID setting requested here, the provided HR image supplies z_0 and its paired
bicubic-upsampled LR image supplies z_L. The FTD trajectory, one-step
reconstruction, TV-LPIPS, and ADL objectives remain unchanged.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from ..pipeline.flux_omini import transformer_forward
from .trainer import OminiModel


FLUXSR_LORA_TARGET_MODULES = (
    r"(.*x_embedder"
    r"|.*time_text_embed\.(timestep_embedder|guidance_embedder|text_embedder)"
    r"\.linear_[12]"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.norm1\.linear"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_k"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_q"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_v"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_out\.0"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.ff\.net\.0\.proj"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.ff\.net\.2"
    r"|.*single_transformer_blocks\.[0-9]+\.norm\.linear"
    r"|.*single_transformer_blocks\.[0-9]+\.proj_mlp"
    r"|.*single_transformer_blocks\.[0-9]+\.proj_out"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.to_k"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.to_q"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.to_v"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.to_out"
    r"|norm_out\.linear|proj_out"
    r"|context_embedder"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.norm1_context\.linear"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.add_q_proj"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.add_k_proj"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.add_v_proj"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_add_out"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.ff_context\.net\.0\.proj"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.ff_context\.net\.2)"
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _resolution_role(metadata: dict[str, Any]) -> str | None:
    role = metadata.get("metadata", {}).get("resolution_role")
    if not isinstance(role, str):
        return None
    normalized = role.strip().lower().replace("_", "-")
    if normalized in {"hr", "high", "high-resolution"}:
        return "hr"
    if normalized in {"lr", "low", "low-resolution"}:
        return "lr"
    return None


def _pair_id(metadata: dict[str, Any], json_path: Path) -> str:
    candidates = (
        str(metadata.get("image_id", "")),
        Path(str(metadata.get("image_path", ""))).stem,
        json_path.stem,
    )
    patterns = (
        re.compile(r"^SR_(?:HR|LR)_(.+)$", re.IGNORECASE),
        re.compile(r"^(?:HR|LR)_AID_(.+?)(?:_ps|_L2A_data)$", re.IGNORECASE),
    )
    for candidate in candidates:
        for pattern in patterns:
            match = pattern.match(candidate)
            if match:
                return match.group(1)
    raise ValueError(f"Cannot extract paired sample id from {json_path}")


def _resolve_image_path(
    metadata: dict[str, Any],
    json_path: Path,
    dataset_root: Path,
    role: str,
) -> Path:
    raw_path = metadata.get("image_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"Missing image_path in {json_path}")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = (
        dataset_root / path,
        dataset_root / "images" / role / path.name,
        json_path.parent / path,
    )
    return next((item for item in candidates if item.is_file()), candidates[0])


def _prompt_from_metadata(hr_metadata: dict[str, Any]) -> str:
    description = hr_metadata.get("description", {})
    sentence = description.get("content_sentence")
    if isinstance(sentence, str) and sentence.strip():
        return sentence.strip()
    scene = description.get("scene") or hr_metadata.get("metadata", {}).get(
        "class_name"
    )
    if isinstance(scene, str) and scene.strip():
        return f"An optical remote sensing image showing a {scene.strip()} scene."
    return "A high-resolution optical remote sensing image."


def discover_aid_pairs(
    root: str | Path,
    splits: tuple[str, ...] | list[str] | set[str],
) -> list[dict[str, Any]]:
    root = Path(root).expanduser().resolve()
    json_dir = root / "json"
    if not json_dir.is_dir():
        raise FileNotFoundError(f"AID JSON directory does not exist: {json_dir}")

    records: dict[str, dict[str, dict[str, Any]]] = {}
    for json_path in sorted(json_dir.rglob("*.json")):
        metadata = _load_json(json_path)
        role = _resolution_role(metadata)
        if role is None:
            continue
        sample_id = _pair_id(metadata, json_path)
        pair = records.setdefault(sample_id, {})
        if role in pair:
            raise ValueError(f"Duplicate {role.upper()} JSON for sample {sample_id}")
        pair[role] = {"metadata": metadata, "json_path": json_path}

    requested_splits = {str(split).strip().lower() for split in splits}
    samples = []
    for sample_id, pair in sorted(records.items()):
        if set(pair) != {"hr", "lr"}:
            raise ValueError(
                f"Incomplete AID pair {sample_id}: found {sorted(pair)}, "
                "expected HR and LR"
            )
        hr_split = str(pair["hr"]["metadata"].get("split", "")).strip().lower()
        lr_split = str(pair["lr"]["metadata"].get("split", "")).strip().lower()
        if not hr_split or hr_split != lr_split:
            raise ValueError(
                f"Split mismatch for {sample_id}: HR={hr_split!r}, LR={lr_split!r}"
            )
        if hr_split not in requested_splits:
            continue

        hr_path = _resolve_image_path(
            pair["hr"]["metadata"], pair["hr"]["json_path"], root, "hr"
        )
        lr_path = _resolve_image_path(
            pair["lr"]["metadata"], pair["lr"]["json_path"], root, "lr"
        )
        if not hr_path.is_file() or not lr_path.is_file():
            raise FileNotFoundError(
                f"Missing image for {sample_id}: HR={hr_path}, LR={lr_path}"
            )
        samples.append(
            {
                "sample_id": sample_id,
                "split": hr_split,
                "hr_path": hr_path,
                "lr_path": lr_path,
                "hr_json": pair["hr"]["json_path"],
                "lr_json": pair["lr"]["json_path"],
                "prompt": _prompt_from_metadata(pair["hr"]["metadata"]),
            }
        )
    if not samples:
        raise ValueError(
            f"No complete AID pairs found for splits {sorted(requested_splits)}"
        )
    return samples


def _read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


class AIDFluxSRDataset(Dataset):
    """Read paired AID TIFF/JSON files directly, without requiring index.jsonl."""

    def __init__(
        self,
        root: str | Path,
        splits: tuple[str, ...] | list[str] = ("train",),
        lr_size: int = 128,
        hr_size: int = 512,
        random_flip: bool = True,
    ):
        self.samples = discover_aid_pairs(root, splits)
        self.lr_size = int(lr_size)
        self.hr_size = int(hr_size)
        self.random_flip = bool(random_flip)
        if self.lr_size <= 0 or self.hr_size <= self.lr_size:
            raise ValueError("Expected 0 < lr_size < hr_size")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        lr = _read_rgb(sample["lr_path"])
        hr = _read_rgb(sample["hr_path"])
        if tuple(lr.shape[-2:]) != (self.lr_size, self.lr_size):
            raise ValueError(
                f"Expected {self.lr_size}x{self.lr_size} LR image, got "
                f"{tuple(lr.shape[-2:])}: {sample['lr_path']}"
            )
        if tuple(hr.shape[-2:]) != (self.hr_size, self.hr_size):
            raise ValueError(
                f"Expected {self.hr_size}x{self.hr_size} HR image, got "
                f"{tuple(hr.shape[-2:])}: {sample['hr_path']}"
            )
        if self.random_flip:
            if random.random() < 0.5:
                lr, hr = lr.flip(-1), hr.flip(-1)
            if random.random() < 0.5:
                lr, hr = lr.flip(-2), hr.flip(-2)
        lr_up = F.interpolate(
            lr.unsqueeze(0),
            size=(self.hr_size, self.hr_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0).clamp_(0.0, 1.0)
        return {
            "sample_id": sample["sample_id"],
            "split": sample["split"],
            "lr": lr,
            "lr_up": lr_up,
            "hr": hr,
            "prompt": sample["prompt"],
            "lr_path": str(sample["lr_path"]),
            "hr_path": str(sample["hr_path"]),
        }


def encode_rgb_to_packed(pipe, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically encode RGB [0,1] images into packed FLUX latents."""
    dtype = next(pipe.transformer.parameters()).dtype
    images = pipe.image_processor.preprocess(images)
    images = images.to(device=pipe.device, dtype=dtype)
    latent = pipe.vae.encode(images).latent_dist.mode()
    latent = (
        latent - pipe.vae.config.shift_factor
    ) * pipe.vae.config.scaling_factor
    tokens = pipe._pack_latents(latent, *latent.shape)
    ids = pipe._prepare_latent_image_ids(
        latent.shape[0],
        latent.shape[2] // 2,
        latent.shape[3] // 2,
        pipe.device,
        dtype,
    )
    return tokens, ids


def decode_packed_to_rgb(
    pipe,
    tokens: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    latent = pipe._unpack_latents(tokens, height, width, pipe.vae_scale_factor)
    latent = (
        latent / pipe.vae.config.scaling_factor
    ) + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(latent, return_dict=False)[0]
    return decoded.float().div(2.0).add(0.5).clamp(0.0, 1.0)


def tv_map(image: torch.Tensor) -> torch.Tensor:
    center = image[:, :, :-1, :-1]
    vertical = (image[:, :, 1:, :-1] - center).abs()
    horizontal = (image[:, :, :-1, 1:] - center).abs()
    return vertical + horizontal


class AIDFluxSRFTDModel(OminiModel):
    """One-step FLUX.1-dev super-resolution LoRA trained with FluxSR losses."""

    def __init__(
        self,
        *args,
        fluxsr_config: dict[str, Any],
        image_size: int = 512,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fluxsr_config = dict(fluxsr_config)
        self.image_size = int(image_size)
        self.t_l = float(self.fluxsr_config.get("t_l", 0.4))
        if not 0.0 < self.t_l < 1.0:
            raise ValueError("fluxsr.t_l must be strictly between 0 and 1")
        self.guidance_scale = float(
            self.fluxsr_config.get("guidance_scale", 3.5)
        )
        self.ftd_weight = float(self.fluxsr_config.get("ftd_weight", 1.0))
        self.mse_weight = float(self.fluxsr_config.get("mse_weight", 1.0))
        self.lpips_weight = float(self.fluxsr_config.get("lpips_weight", 1.0))
        self.tv_lpips_gamma = float(
            self.fluxsr_config.get("tv_lpips_gamma", 1.0)
        )
        self.adl_weight = float(self.fluxsr_config.get("adl_weight", 0.01))
        self.lpips_model = lpips.LPIPS(net="vgg").eval().to(self.device)
        self.lpips_model.requires_grad_(False)
        self.last_loss_components: dict[str, float] = {}

    def train(self, mode: bool = True):
        result = super().train(mode)
        # These frozen modules must stay deterministic while LoRA is trained.
        self.flux_pipe.vae.eval()
        self.flux_pipe.text_encoder.eval()
        self.flux_pipe.text_encoder_2.eval()
        if hasattr(self, "lpips_model"):
            self.lpips_model.eval()
        return result

    def _encode_prompt(self, prompts: list[str]):
        with torch.no_grad():
            return self.flux_pipe.encode_prompt(
                prompt=prompts,
                prompt_2=prompts,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=self.device,
                num_images_per_prompt=1,
                max_sequence_length=self.model_config.get(
                    "max_sequence_length", 512
                ),
                lora_scale=None,
            )

    def _predict_velocity(
        self,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
        timestep: torch.Tensor,
        prompt_features,
        return_adl_loss: bool = False,
    ):
        prompt_embeds, pooled_prompt_embeds, text_ids = prompt_features
        guidance = (
            torch.full_like(timestep, self.guidance_scale)
            if self.transformer.config.guidance_embeds
            else None
        )
        outputs = transformer_forward(
            self.transformer,
            image_features=[latents],
            text_features=[prompt_embeds],
            img_ids=[latent_ids],
            txt_ids=[text_ids],
            timesteps=[timestep, timestep],
            pooled_projections=[pooled_prompt_embeds, pooled_prompt_embeds],
            guidances=[guidance, guidance],
            adapters=self.adapter_names,
            return_dict=False,
            return_adl_loss=return_adl_loss,
        )
        return outputs

    def generate_one_step(self, batch: dict[str, Any]) -> torch.Tensor:
        lr_up = batch["lr_up"].to(self.device)
        prompts = list(batch["prompt"])
        with torch.no_grad():
            z_l, latent_ids = encode_rgb_to_packed(self.flux_pipe, lr_up)
            prompt_features = self._encode_prompt(prompts)
            timestep = torch.full(
                (z_l.shape[0],),
                self.t_l,
                device=self.device,
                dtype=z_l.dtype,
            )
            velocity = self._predict_velocity(
                z_l, latent_ids, timestep, prompt_features
            )[0]
            z_hat = z_l - self.t_l * velocity
            return decode_packed_to_rgb(
                self.flux_pipe, z_hat, self.image_size, self.image_size
            )

    def training_step(self, batch, batch_idx):
        hr = batch["hr"].to(self.device)
        lr_up = batch["lr_up"].to(self.device)
        prompts = list(batch["prompt"])
        batch_size = hr.shape[0]

        with torch.no_grad():
            z_0, latent_ids = encode_rgb_to_packed(self.flux_pipe, hr)
            z_l, lr_ids = encode_rgb_to_packed(self.flux_pipe, lr_up)
            if z_0.shape != z_l.shape or not torch.equal(latent_ids, lr_ids):
                raise ValueError("HR and bicubic-LR latent layouts do not match")
            prompt_features = self._encode_prompt(prompts)
            epsilon = torch.randn_like(z_0)
            timestep = torch.rand(
                batch_size, device=self.device, dtype=z_0.dtype
            )
            timestep = self.t_l + (1.0 - self.t_l) * timestep
            t_view = timestep[:, None, None]
            x_t = (
                ((1.0 - t_view) / (1.0 - self.t_l)) * z_l
                + ((t_view - self.t_l) / (1.0 - self.t_l)) * epsilon
            )
            teacher_velocity = epsilon - z_0

        velocity_t = self._predict_velocity(
            x_t, latent_ids, timestep, prompt_features
        )[0]
        ftd_residual = (
            teacher_velocity
            - self.t_l * velocity_t
            - (epsilon - z_l)
        )
        loss_ftd = ftd_residual.float().square().mean()

        t_l_batch = torch.full(
            (batch_size,), self.t_l, device=self.device, dtype=z_l.dtype
        )
        velocity_l, loss_adl = self._predict_velocity(
            z_l,
            latent_ids,
            t_l_batch,
            prompt_features,
            return_adl_loss=True,
        )
        z_hat = z_l - self.t_l * velocity_l
        prediction = decode_packed_to_rgb(
            self.flux_pipe, z_hat, self.image_size, self.image_size
        )
        target = hr.float()

        loss_mse = F.mse_loss(prediction, target)
        loss_lpips = self.lpips_model(
            prediction, target, normalize=True
        ).mean()
        prediction_tv = tv_map(prediction).div(2.0).clamp(0.0, 1.0)
        target_tv = tv_map(target).div(2.0).clamp(0.0, 1.0)
        loss_tv_lpips = self.lpips_model(
            prediction_tv, target_tv, normalize=True
        ).mean()
        loss_perceptual = loss_lpips + self.tv_lpips_gamma * loss_tv_lpips
        loss_reconstruction = (
            self.mse_weight * loss_mse
            + self.lpips_weight * loss_perceptual
        )
        loss = (
            self.ftd_weight * loss_ftd
            + loss_reconstruction
            + self.adl_weight * loss_adl
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite FluxSR loss at batch {batch_idx}")

        self.last_t = float(timestep.float().mean())
        self.log_loss = float(loss.detach())
        self.last_loss_components = {
            "loss/total": float(loss.detach()),
            "loss/ftd": float(loss_ftd.detach()),
            "loss/reconstruction": float(loss_reconstruction.detach()),
            "loss/mse": float(loss_mse.detach()),
            "loss/lpips": float(loss_lpips.detach()),
            "loss/tv_lpips": float(loss_tv_lpips.detach()),
            "loss/adl": float(loss_adl.detach()),
            "train/timestep": self.last_t,
        }
        self.log_dict(
            {key: value for key, value in self.last_loss_components.items()},
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=False,
            sync_dist=False,
        )
        return loss


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def make_comparison_panel(
    lr: torch.Tensor,
    prediction: torch.Tensor,
    hr: torch.Tensor,
) -> Image.Image:
    lr_image = tensor_to_pil(lr).resize(
        (hr.shape[-1], hr.shape[-2]), Image.Resampling.NEAREST
    )
    prediction_image = tensor_to_pil(prediction)
    hr_image = tensor_to_pil(hr)
    width, height = hr_image.size
    header = 28
    panel = Image.new("RGB", (width * 3, height + header), "white")
    panel.paste(lr_image, (0, header))
    panel.paste(prediction_image, (width, header))
    panel.paste(hr_image, (width * 2, header))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(panel)
    for index, label in enumerate(("LR input", "FluxSR prediction", "HR truth")):
        draw.text((index * width + 8, 7), label, fill="black")
    return panel
