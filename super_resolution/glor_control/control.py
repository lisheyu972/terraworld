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


def _resolve_indices(
    total: int,
    mode: str,
    explicit_indices: list[int] | tuple[int, ...] | None,
) -> set[int]:
    if explicit_indices is None:
        return _selected_indices(total, mode)
    indices = {int(index) for index in explicit_indices}
    invalid = sorted(index for index in indices if not 0 <= index < total)
    if invalid:
        raise ValueError(
            f"GLoR block indices {invalid} are outside the valid range "
            f"[0, {total - 1}]"
        )
    return indices


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
        inject_dual_indices: list[int] | tuple[int, ...] | None = None,
        inject_single_indices: list[int] | tuple[int, ...] | None = None,
        fusion_dim: int | None = None,
        sr_feature_mode: str = "rgb",
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.ctrl_dim = int(ctrl_dim)
        self.pool_size = int(pool_size)
        self.sr_patch_size = int(sr_patch_size)
        self.use_scale_embedding = bool(use_scale_embedding)
        self.task = str(task)
        self.temporal_offset_scale = float(temporal_offset_scale)
        self.sr_feature_mode = str(sr_feature_mode)
        self.sr_in_channels = int(sr_in_channels)

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

        fusion_dim = self.hidden_dim if fusion_dim is None else int(fusion_dim)
        if fusion_dim <= 0:
            raise ValueError("GLoR fusion_dim must be positive")
        self.fusion_dim = fusion_dim
        self.control_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.fusion_dim),
            nn.SiLU(),
            nn.Linear(self.fusion_dim, self.hidden_dim),
        )

        self.dual_indices = _resolve_indices(
            num_dual_blocks, inject_blocks, inject_dual_indices
        )
        self.single_indices = _resolve_indices(
            num_single_blocks, inject_blocks, inject_single_indices
        )
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
            inject_dual_indices=glor_config.get("inject_dual_indices"),
            inject_single_indices=glor_config.get("inject_single_indices"),
            fusion_dim=glor_config.get("fusion_dim"),
            sr_feature_mode=task_config.get("feature_mode", "rgb"),
        )

    def _sr_condition_features(self, image: torch.Tensor) -> torch.Tensor:
        if self.sr_feature_mode == "rgb":
            features = image
        else:
            luminance = (
                0.299 * image[:, 0:1]
                + 0.587 * image[:, 1:2]
                + 0.114 * image[:, 2:3]
            )
            if self.sr_feature_mode == "rgb_y":
                features = torch.cat([image, luminance], dim=1)
            elif self.sr_feature_mode == "rgb_y_edge":
                kernel_x = image.new_tensor(
                    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
                ).view(1, 1, 3, 3)
                kernel_y = kernel_x.transpose(-1, -2)
                edge_x = F.conv2d(luminance, kernel_x, padding=1)
                edge_y = F.conv2d(luminance, kernel_y, padding=1)
                edge = torch.sqrt(
                    edge_x.float().square()
                    + edge_y.float().square()
                    + 1.0e-12
                ).to(image.dtype)
                edge = edge.div(4.0).clamp_(0.0, 1.0)
                features = torch.cat([image, luminance, edge], dim=1)
            else:
                raise ValueError(
                    f"Unsupported super-resolution GLoR feature mode: "
                    f"{self.sr_feature_mode!r}"
                )
        if features.shape[1] != self.sr_in_channels:
            raise ValueError(
                f"GLoR feature mode {self.sr_feature_mode!r} produced "
                f"{features.shape[1]} channels, but the condition encoder "
                f"expects {self.sr_in_channels}"
            )
        return features

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
        else:
            raise ValueError(f"Unsupported GLoR task: {task}")

        global_memories = []
        local_memories = []
        dtype = next(self.condition_encoder.parameters()).dtype
        for index, image in enumerate(images):
            if image.ndim == 3:
                image = image.unsqueeze(0)
            image = image.to(dtype=dtype)
            if task == "super_resolution":
                image = self._sr_condition_features(image)
            feat = self.condition_encoder(image)

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
                feat = feat + self.temporal_embedding(offset).to(feat.dtype)[:, :, None, None]

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
            local_memories.append(
                unfold_local_patches(aligned, self.sr_patch_size).contiguous()
            )

        return {
            "global_tokens": torch.cat(global_memories, dim=1),
            "local_tokens": torch.cat(local_memories, dim=2),
        }

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
        if "_global_k" not in context:
            context["_global_k"] = self.global_k(global_tokens)
            context["_global_v"] = self.global_v(global_tokens)
            context["_local_k"] = self.local_k(local_tokens)
            context["_local_v"] = self.local_v(local_tokens)
        k_global = context["_global_k"]
        v_global = context["_global_v"]
        global_attn = torch.einsum(
            "bnd,bmd->bnm", q_global.float(), k_global.float()
        )
        global_attn = (global_attn / math.sqrt(self.ctrl_dim)).softmax(dim=-1)
        global_attn = global_attn.to(v_global.dtype)
        global_ctrl = torch.einsum("bnm,bmd->bnd", global_attn, v_global)
        global_ctrl = self.global_out(global_ctrl)

        local_query = hidden_states + global_ctrl
        q_local = self.local_q(local_query)
        k_local = context["_local_k"]
        v_local = context["_local_v"]
        local_attn = torch.einsum(
            "bnd,bnkd->bnk", q_local.float(), k_local.float()
        )
        local_attn = (local_attn / math.sqrt(self.ctrl_dim)).softmax(dim=-1)
        local_attn = local_attn.to(v_local.dtype)
        local_ctrl = torch.einsum("bnk,bnkd->bnd", local_attn, v_local)
        local_ctrl = self.local_out(local_ctrl)

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
        return hidden_states + alphas[key].to(hidden_states.dtype) * delta
