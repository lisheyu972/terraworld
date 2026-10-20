"""Global-Local Observation Retrieval Control."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .utils import unfold_local_patches


def _selected_indices(total: int, mode: str) -> set[int]:
    if total <= 0:
        return set()
    if mode == "all":
        return set(range(total))
    if mode == "late":
        return set(range(total // 2, total))
    if mode == "middle":
        return set(range(total // 3, (2 * total) // 3))
    if mode == "middle_late":
        return set(range(total // 3, total))
    raise ValueError(f"Unsupported GLoR inject_blocks mode: {mode}")


def _masked_softmax(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Softmax over valid keys, returning zeros when a row has no valid key."""

    valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    masked_logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
    probabilities = masked_logits.softmax(dim=-1)
    probabilities = probabilities * valid_mask.to(probabilities.dtype)
    normalizer = probabilities.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return probabilities / normalizer


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


class GLoRControl(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ctrl_dim: int = 512,
        num_dual_blocks: int = 0,
        num_single_blocks: int = 0,
        inject_blocks: str = "middle_late",
        alpha_init: float = 0.0,
        pool_size: int = 4,
        sr_patch_size: int = 3,
        sr_in_channels: int = 3,
        use_scale_embedding: bool = True,
        task: str = "super_resolution",
        use_temporal_embedding: bool = True,
        temporal_offset_scale: float = 1.0,
        inpaint_dilations: tuple[int, ...] = (1,),
        inpaint_min_visible_fraction: float = 0.5,
        inpaint_encoder_max_size: int = 0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.ctrl_dim = int(ctrl_dim)
        self.pool_size = int(pool_size)
        self.sr_patch_size = int(sr_patch_size)
        self.use_scale_embedding = bool(use_scale_embedding)
        self.task = str(task)
        self.temporal_offset_scale = float(temporal_offset_scale)
        self.inpaint_dilations = tuple(int(value) for value in inpaint_dilations)
        if not self.inpaint_dilations or any(
            value < 1 for value in self.inpaint_dilations
        ):
            raise ValueError("inpaint_dilations must contain positive integers.")
        self.inpaint_min_visible_fraction = float(inpaint_min_visible_fraction)
        if not 0.0 <= self.inpaint_min_visible_fraction <= 1.0:
            raise ValueError(
                "inpaint_min_visible_fraction must be between zero and one."
            )
        self.inpaint_encoder_max_size = int(inpaint_encoder_max_size)
        if self.inpaint_encoder_max_size < 0:
            raise ValueError("inpaint_encoder_max_size must be non-negative.")
        self.condition_in_channels = int(sr_in_channels)

        self.condition_encoder = SimpleConditionEncoder(sr_in_channels, self.ctrl_dim)
        self.scale_embedding = (
            ScalarEmbedding(self.ctrl_dim) if self.use_scale_embedding else None
        )
        self.temporal_embedding = (
            ScalarEmbedding(self.ctrl_dim)
            if self.task == "frame_prediction" and use_temporal_embedding
            else None
        )

        self.global_q = nn.Linear(self.hidden_dim, self.ctrl_dim)
        self.global_k = nn.Linear(self.ctrl_dim, self.ctrl_dim)
        self.global_v = nn.Linear(self.ctrl_dim, self.ctrl_dim)
        self.global_out = nn.Linear(self.ctrl_dim, self.hidden_dim)

        self.local_q = nn.Linear(self.hidden_dim, self.ctrl_dim)
        self.local_k = nn.Linear(self.ctrl_dim, self.ctrl_dim)
        self.local_v = nn.Linear(self.ctrl_dim, self.ctrl_dim)
        self.local_out = nn.Linear(self.ctrl_dim, self.hidden_dim)

        self.control_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.dual_indices = _selected_indices(num_dual_blocks, inject_blocks)
        self.single_indices = _selected_indices(num_single_blocks, inject_blocks)
        self.dual_alpha = nn.ParameterDict(
            {
                str(index): nn.Parameter(torch.tensor([float(alpha_init)]))
                for index in sorted(self.dual_indices)
            }
        )
        self.single_alpha = nn.ParameterDict(
            {
                str(index): nn.Parameter(torch.tensor([float(alpha_init)]))
                for index in sorted(self.single_indices)
            }
        )

    @classmethod
    def from_config(cls, transformer, config: dict) -> "GLoRControl":
        glor_config = config.get("glor_control", config)
        hidden_dim = (
            int(transformer.config.attention_head_dim)
            * int(transformer.config.num_attention_heads)
        )
        task = glor_config.get("task", "super_resolution")
        task_config = glor_config.get("tasks", {}).get(task, {})
        global_config = glor_config.get("global_memory", {})
        local_config = glor_config.get("local_retrieval", {})
        return cls(
            hidden_dim=hidden_dim,
            ctrl_dim=glor_config.get("ctrl_dim", 512),
            num_dual_blocks=len(transformer.transformer_blocks),
            num_single_blocks=len(transformer.single_transformer_blocks),
            inject_blocks=glor_config.get("inject_blocks", "middle_late"),
            alpha_init=glor_config.get("alpha_init", 0.0),
            pool_size=global_config.get("pool_size", 4),
            sr_patch_size=local_config.get(
                "patch_size", local_config.get("sr_patch_size", 3)
            ),
            sr_in_channels=task_config.get("in_channels", 3),
            use_scale_embedding=task_config.get(
                "use_scale_embedding", task == "super_resolution"
            ),
            task=task,
            use_temporal_embedding=task_config.get(
                "use_temporal_embedding", True
            ),
            temporal_offset_scale=task_config.get("temporal_offset_scale", 1.0),
            inpaint_dilations=local_config.get("dilations", [1]),
            inpaint_min_visible_fraction=local_config.get(
                "min_visible_fraction", 0.5
            ),
            inpaint_encoder_max_size=task_config.get("encoder_max_size", 0),
        )

    def build_context(
        self,
        batch: dict,
        task: str,
        target_latent_shape: tuple[int, int],
    ) -> dict:
        if task != self.task:
            raise ValueError(
                f"GLoR was configured for {self.task!r}, but received {task!r}."
            )

        if task == "super_resolution":
            image = batch.get("lr_image", batch.get("condition_0"))
            if image is None:
                raise KeyError("GLoR-Control requires lr_image.")
            images = [image]
            temporal_offsets = None
            inpaint_masks = None
        elif task == "frame_prediction":
            images = batch.get("context_images")
            if images is None:
                images = [
                    batch.get("condition_0"),
                    batch.get("condition_1"),
                ]
            if len(images) != 2 or any(image is None for image in images):
                raise KeyError(
                    "Frame-prediction GLoR requires exactly two context images."
                )
            temporal_offsets = batch.get("temporal_offsets")
            inpaint_masks = None
        elif task == "inpainting":
            image = batch.get("inpaint_image", batch.get("condition_0"))
            mask = batch.get("inpaint_mask")
            if image is None or mask is None:
                raise KeyError(
                    "Inpainting GLoR requires inpaint_image and inpaint_mask."
                )
            images = [image]
            inpaint_masks = [mask]
            temporal_offsets = None
        else:
            raise ValueError(f"Unsupported GLoR task: {task}")

        global_memories = []
        local_memories = []
        global_valid_masks = []
        local_valid_masks = []
        target_masks = []
        dtype = next(self.condition_encoder.parameters()).dtype
        for index, image in enumerate(images):
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if task == "inpainting":
                mask = inpaint_masks[index]
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                if mask.ndim != 4 or mask.shape[1] != 1:
                    raise ValueError(
                        "inpaint_mask must have shape [B, 1, H, W]."
                    )
                if image.ndim != 4 or image.shape[1] != self.condition_in_channels:
                    raise ValueError(
                        "inpaint_image channel count must match the configured "
                        f"in_channels={self.condition_in_channels}."
                    )
                if (
                    mask.shape[0] != image.shape[0]
                    or mask.shape[-2:] != image.shape[-2:]
                ):
                    raise ValueError(
                        "inpaint_mask and inpaint_image must have matching batch "
                        "and spatial dimensions."
                    )
                mask = mask.to(device=image.device, dtype=torch.float32).clamp_(0, 1)
                if (
                    self.inpaint_encoder_max_size
                    and max(image.shape[-2:]) > self.inpaint_encoder_max_size
                ):
                    resize_scale = self.inpaint_encoder_max_size / max(
                        image.shape[-2:]
                    )
                    encoder_size = tuple(
                        max(1, round(size * resize_scale))
                        for size in image.shape[-2:]
                    )
                    rgb = F.interpolate(
                        image[:, :-1].float(),
                        size=encoder_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    mask = F.interpolate(mask, size=encoder_size, mode="area")
                    image = torch.cat([rgb, mask], dim=1)
            feat = self.condition_encoder(image.to(dtype=dtype))

            if task == "super_resolution" and self.scale_embedding is not None:
                scale = batch.get("scale", batch.get("position_scale", 1.0))
                if not torch.is_tensor(scale):
                    scale = torch.tensor(scale, device=feat.device)
                scale = scale.to(feat.device)
                if scale.numel() == 1 and feat.shape[0] > 1:
                    scale = scale.expand(feat.shape[0])
                feat = feat + self.scale_embedding(scale).to(feat.dtype)[:, :, None, None]

            if task == "frame_prediction" and self.temporal_embedding is not None:
                if temporal_offsets is None:
                    offset = -1.0 if index == 0 else 1.0
                elif torch.is_tensor(temporal_offsets):
                    offset = (
                        temporal_offsets[:, index]
                        if temporal_offsets.ndim > 1
                        else temporal_offsets
                    )
                else:
                    offset = temporal_offsets[index]
                if not torch.is_tensor(offset):
                    offset = torch.tensor(offset, device=feat.device)
                offset = offset.to(feat.device).reshape(-1)
                if offset.numel() == 1 and feat.shape[0] > 1:
                    offset = offset.expand(feat.shape[0])
                if offset.numel() != feat.shape[0]:
                    raise ValueError(
                        "Temporal offsets must provide one value per batch item."
                    )
                offset = offset * self.temporal_offset_scale
                temporal_embedding = self.temporal_embedding(offset).to(feat.dtype)
                feat = feat + temporal_embedding[:, :, None, None]

            if task == "inpainting":
                mask = mask.to(device=feat.device, dtype=torch.float32).clamp_(0, 1)
                visible = 1.0 - mask
                visible_for_feat = visible.to(feat.dtype)
                pooled_visible = F.adaptive_avg_pool2d(
                    visible, output_size=(self.pool_size, self.pool_size)
                )
                pooled_features = F.adaptive_avg_pool2d(
                    feat * visible_for_feat,
                    output_size=(self.pool_size, self.pool_size),
                )
                visible_denominator = pooled_visible.to(feat.dtype).clamp_min(
                    1.0e-6
                )
                global_tokens = pooled_features / visible_denominator
                global_valid_masks.append(
                    pooled_visible.flatten(2).squeeze(1) > 0
                )
            else:
                global_tokens = F.adaptive_avg_pool2d(
                    feat, output_size=(self.pool_size, self.pool_size)
                )
            global_memories.append(
                global_tokens.flatten(2).transpose(1, 2).contiguous()
            )
            aligned = F.interpolate(
                feat,
                size=target_latent_shape,
                mode="bilinear",
                align_corners=False,
            )
            if task == "inpainting":
                aligned_mask = F.interpolate(
                    mask,
                    size=target_latent_shape,
                    mode="area",
                )
                aligned_visible = 1.0 - aligned_mask
                feature_patches = []
                validity_patches = []
                for dilation in self.inpaint_dilations:
                    feature_patches.append(
                        unfold_local_patches(
                            aligned,
                            self.sr_patch_size,
                            dilation=dilation,
                        )
                    )
                    validity_patches.append(
                        unfold_local_patches(
                            aligned_visible,
                            self.sr_patch_size,
                            dilation=dilation,
                        )[..., 0]
                        >= self.inpaint_min_visible_fraction
                    )
                local_memories.append(
                    torch.cat(feature_patches, dim=2).contiguous()
                )
                local_valid_masks.append(
                    torch.cat(validity_patches, dim=2).contiguous()
                )
                target_masks.append(
                    aligned_mask.flatten(2).transpose(1, 2).contiguous()
                )
            else:
                local_memories.append(
                    unfold_local_patches(aligned, self.sr_patch_size).contiguous()
                )

        context = {
            "global_tokens": torch.cat(global_memories, dim=1),
            "local_tokens": torch.cat(local_memories, dim=2),
        }
        if task == "inpainting":
            global_valid_mask = torch.cat(global_valid_masks, dim=1)
            context.update(
                {
                    "global_valid_mask": global_valid_mask,
                    "local_valid_mask": torch.cat(local_valid_masks, dim=2),
                    "target_mask": torch.stack(target_masks, dim=0).amax(dim=0),
                    "has_visible_context": global_valid_mask.any(
                        dim=1, keepdim=True
                    ).unsqueeze(-1),
                }
            )
        return context

    def forward(self, hidden_states: torch.Tensor, context: dict) -> torch.Tensor:
        global_tokens = context["global_tokens"].to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        local_tokens = context["local_tokens"].to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        if hidden_states.shape[:2] != local_tokens.shape[:2]:
            raise ValueError(
                "GLoR local tokens must match hidden state batch/token dimensions: "
                f"{tuple(local_tokens.shape)} vs {tuple(hidden_states.shape)}."
            )

        q_global = self.global_q(hidden_states)
        k_global = self.global_k(global_tokens)
        v_global = self.global_v(global_tokens)
        global_attn = torch.einsum(
            "bnd,bmd->bnm", q_global.float(), k_global.float()
        )
        global_attn = global_attn / math.sqrt(self.ctrl_dim)
        global_valid_mask = context.get("global_valid_mask")
        if global_valid_mask is not None:
            global_attn = _masked_softmax(
                global_attn,
                global_valid_mask[:, None, :],
            )
        else:
            global_attn = global_attn.softmax(dim=-1)
        global_attn = global_attn.to(v_global.dtype)
        global_ctrl = torch.einsum("bnm,bmd->bnd", global_attn, v_global)
        global_ctrl = self.global_out(global_ctrl)
        if global_valid_mask is not None:
            global_ctrl = global_ctrl * global_valid_mask.any(
                dim=-1, keepdim=True
            )[:, None, :].to(global_ctrl.dtype)

        local_query = hidden_states + global_ctrl
        q_local = self.local_q(local_query)
        k_local = self.local_k(local_tokens)
        v_local = self.local_v(local_tokens)
        local_attn = torch.einsum(
            "bnd,bnkd->bnk", q_local.float(), k_local.float()
        )
        local_attn = local_attn / math.sqrt(self.ctrl_dim)
        local_valid_mask = context.get("local_valid_mask")
        if local_valid_mask is not None:
            local_attn = _masked_softmax(local_attn, local_valid_mask)
        else:
            local_attn = local_attn.softmax(dim=-1)
        local_attn = local_attn.to(v_local.dtype)
        local_ctrl = torch.einsum("bnk,bnkd->bnd", local_attn, v_local)
        local_ctrl = self.local_out(local_ctrl)
        if local_valid_mask is not None:
            local_ctrl = local_ctrl * local_valid_mask.any(
                dim=-1, keepdim=True
            ).to(local_ctrl.dtype)

        return self.control_mlp(
            torch.cat([hidden_states, global_ctrl, local_ctrl], dim=-1)
        )

    def inject(
        self,
        hidden_states: torch.Tensor,
        context: dict | None,
        block_kind: str,
        block_index: int,
    ) -> torch.Tensor:
        if context is None:
            return hidden_states
        if block_kind == "dual":
            alphas = self.dual_alpha
        elif block_kind == "single":
            alphas = self.single_alpha
        else:
            raise ValueError(f"Unknown GLoR block kind: {block_kind}")

        key = str(block_index)
        if key not in alphas:
            return hidden_states
        delta = self(hidden_states, context)
        target_mask = context.get("target_mask")
        if target_mask is not None:
            if target_mask.shape[:2] != hidden_states.shape[:2]:
                raise ValueError(
                    "GLoR target mask must match hidden state batch/token dimensions: "
                    f"{tuple(target_mask.shape)} vs {tuple(hidden_states.shape)}."
                )
            delta = delta * target_mask.to(
                device=delta.device,
                dtype=delta.dtype,
            )
        has_visible_context = context.get("has_visible_context")
        if has_visible_context is not None:
            delta = delta * has_visible_context.to(
                device=delta.device,
                dtype=delta.dtype,
            )
        return hidden_states + alphas[key].to(hidden_states.dtype) * delta
