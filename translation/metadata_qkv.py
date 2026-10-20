"""Metadata-injected Q/K modulation for OminiControl MM-Attention."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class MetadataQKVDeltas:
    """Metadata deltas.

    Target tensors have shape ``[B,H,1,D_head]``. Condition tensors use the
    same shape for one condition or ``[B,C,H,1,D_head]`` for multiple image
    conditions.
    """

    target_q: Tensor
    relation_q: Tensor
    condition_k: Tensor
    condition_v: Tensor


class MetadataQKVInjector(nn.Module):
    """Inject compact metadata deltas into existing MM-Attention Q/K/V tensors."""

    def __init__(
        self,
        *,
        condition_metadata_dim: int,
        target_metadata_dim: int,
        metadata_hidden_dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        target_to_q: bool = True,
        condition_to_k: bool = True,
        condition_to_v: bool = False,
        pair_to_q: bool = False,
        gate_init: float = 0.0,
        projection_init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        dimensions = {
            "condition_metadata_dim": condition_metadata_dim,
            "target_metadata_dim": target_metadata_dim,
            "metadata_hidden_dim": metadata_hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }
        invalid = {name: value for name, value in dimensions.items() if value <= 0}
        if invalid:
            raise ValueError(f"Metadata-QKV dimensions must be positive: {invalid}")
        if projection_init_std <= 0:
            raise ValueError("projection_init_std must be positive")

        self.condition_metadata_dim = int(condition_metadata_dim)
        self.target_metadata_dim = int(target_metadata_dim)
        self.metadata_hidden_dim = int(metadata_hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.target_to_q = bool(target_to_q)
        self.condition_to_k = bool(condition_to_k)
        self.condition_to_v = bool(condition_to_v)
        self.pair_to_q = bool(pair_to_q)

        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_metadata_dim, self.metadata_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.metadata_hidden_dim, self.metadata_hidden_dim),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(self.target_metadata_dim, self.metadata_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.metadata_hidden_dim, self.metadata_hidden_dim),
        )
        self.relation_encoder = nn.Sequential(
            nn.Linear(4 * self.metadata_hidden_dim, self.metadata_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.metadata_hidden_dim, self.metadata_hidden_dim),
        )

        self.target_q_projection = nn.Linear(
            self.metadata_hidden_dim, self.inner_dim
        )
        self.relation_q_projection = nn.Linear(
            self.metadata_hidden_dim, self.inner_dim
        )
        self.condition_k_projection = nn.Linear(
            self.metadata_hidden_dim, self.inner_dim
        )
        self.condition_v_projection = nn.Linear(
            self.metadata_hidden_dim, self.inner_dim
        )

        gate_shape = (self.num_layers, self.num_heads)
        self.target_q_gate = nn.Parameter(torch.full(gate_shape, gate_init))
        self.relation_q_gate = nn.Parameter(torch.full(gate_shape, gate_init))
        self.condition_k_gate = nn.Parameter(torch.full(gate_shape, gate_init))
        self.condition_v_gate = nn.Parameter(torch.full(gate_shape, gate_init))

        for projection in (
            self.target_q_projection,
            self.relation_q_projection,
            self.condition_k_projection,
            self.condition_v_projection,
        ):
            nn.init.normal_(
                projection.weight,
                mean=0.0,
                std=float(projection_init_std),
            )
            nn.init.zeros_(projection.bias)

        self._configure_trainable_branches()

    @staticmethod
    def _set_trainable(module_or_parameter, enabled: bool) -> None:
        if isinstance(module_or_parameter, nn.Parameter):
            module_or_parameter.requires_grad_(enabled)
        else:
            module_or_parameter.requires_grad_(enabled)

    def _configure_trainable_branches(self) -> None:
        """Freeze switches that are disabled by the MIQ-MMA stage config.

        DDP expects every parameter marked trainable to contribute to the
        training graph. Stage A intentionally disables relation-to-Q and
        condition-to-V, so their parameters must not remain trainable.
        """

        self.requires_grad_(False)

        condition_encoder_enabled = (
            self.condition_to_k or self.condition_to_v or self.pair_to_q
        )
        target_encoder_enabled = self.target_to_q or self.pair_to_q
        self._set_trainable(
            self.condition_encoder,
            condition_encoder_enabled,
        )
        self._set_trainable(self.target_encoder, target_encoder_enabled)

        self._set_trainable(self.target_q_projection, self.target_to_q)
        self._set_trainable(self.target_q_gate, self.target_to_q)
        self._set_trainable(self.relation_encoder, self.pair_to_q)
        self._set_trainable(self.relation_q_projection, self.pair_to_q)
        self._set_trainable(self.relation_q_gate, self.pair_to_q)
        self._set_trainable(
            self.condition_k_projection,
            self.condition_to_k,
        )
        self._set_trainable(self.condition_k_gate, self.condition_to_k)
        self._set_trainable(
            self.condition_v_projection,
            self.condition_to_v,
        )
        self._set_trainable(self.condition_v_gate, self.condition_to_v)

    def trainable_parameters(self):
        """Return only parameters used by the configured injection branches."""

        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    @classmethod
    def from_transformer(
        cls,
        transformer,
        config: dict[str, Any],
    ) -> "MetadataQKVInjector":
        dual_layers = len(transformer.transformer_blocks)
        single_layers = len(transformer.single_transformer_blocks)
        return cls(
            condition_metadata_dim=int(config["condition_metadata_dim"]),
            target_metadata_dim=int(config["target_metadata_dim"]),
            metadata_hidden_dim=int(config.get("metadata_hidden_dim", 256)),
            num_layers=dual_layers + single_layers,
            num_heads=int(transformer.config.num_attention_heads),
            head_dim=int(transformer.config.attention_head_dim),
            target_to_q=config.get("target_to_q", True),
            condition_to_k=config.get("condition_to_k", True),
            condition_to_v=config.get("condition_to_v", False),
            pair_to_q=config.get("pair_to_q", False),
            gate_init=float(config.get("gate_init", 0.0)),
            projection_init_std=float(
                config.get("projection_init_std", 1e-3)
            ),
        )

    def _reshape_delta(self, value: Tensor) -> Tensor:
        if value.ndim == 2:
            return value.view(
                value.shape[0],
                self.num_heads,
                1,
                self.head_dim,
            )
        if value.ndim == 3:
            return value.view(
                value.shape[0],
                value.shape[1],
                self.num_heads,
                1,
                self.head_dim,
            )
        raise ValueError("Projected metadata must have shape [B,D] or [B,C,D]")

    def prepare_deltas(
        self,
        condition_metadata: Tensor,
        target_metadata: Tensor,
    ) -> MetadataQKVDeltas:
        if condition_metadata.ndim not in {2, 3}:
            raise ValueError(
                "condition_metadata must have shape [B,D] or [B,C,D]"
            )
        if target_metadata.ndim != 2:
            raise ValueError("target_metadata must have shape [B, D]")
        if condition_metadata.shape[0] != target_metadata.shape[0]:
            raise ValueError("Condition and target metadata batch sizes differ")
        if condition_metadata.shape[-1] != self.condition_metadata_dim:
            raise ValueError(
                "Unexpected condition metadata width: "
                f"{condition_metadata.shape[-1]} != "
                f"{self.condition_metadata_dim}"
            )
        if target_metadata.shape[1] != self.target_metadata_dim:
            raise ValueError(
                "Unexpected target metadata width: "
                f"{target_metadata.shape[1]} != {self.target_metadata_dim}"
            )

        condition_embedding = self.condition_encoder(condition_metadata)
        # Relation-to-Q is optional and needs one condition-side summary. Mean
        # pooling is order-independent while condition-to-K below retains each
        # condition frame's own metadata delta.
        relation_condition_embedding = (
            condition_embedding.mean(dim=1)
            if condition_embedding.ndim == 3
            else condition_embedding
        )
        target_embedding = self.target_encoder(target_metadata)
        relation_embedding = self.relation_encoder(
            torch.cat(
                (
                    target_embedding,
                    relation_condition_embedding,
                    target_embedding - relation_condition_embedding,
                    target_embedding * relation_condition_embedding,
                ),
                dim=-1,
            )
        )
        return MetadataQKVDeltas(
            target_q=self._reshape_delta(
                self.target_q_projection(target_embedding)
            ),
            relation_q=self._reshape_delta(
                self.relation_q_projection(relation_embedding)
            ),
            condition_k=self._reshape_delta(
                self.condition_k_projection(condition_embedding)
            ),
            condition_v=self._reshape_delta(
                self.condition_v_projection(condition_embedding)
            ),
        )

    def _gate(self, table: Tensor, layer_index: int, reference: Tensor) -> Tensor:
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"layer_index={layer_index} is outside [0, {self.num_layers})"
            )
        return table[layer_index].to(
            device=reference.device,
            dtype=reference.dtype,
        )[None, :, None, None]

    def inject(
        self,
        *,
        q_main: Tensor,
        k_condition: Tensor,
        v_condition: Tensor,
        deltas: MetadataQKVDeltas,
        layer_index: int,
        condition_slot: int | None = None,
        inject_main_query: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        expected_heads = self.num_heads
        expected_head_dim = self.head_dim
        for name, tensor in {
            "q_main": q_main,
            "k_condition": k_condition,
            "v_condition": v_condition,
        }.items():
            if tensor.ndim != 4:
                raise ValueError(f"{name} must have shape [B,H,N,D]")
            if tensor.shape[1] != expected_heads:
                raise ValueError(
                    f"{name} has {tensor.shape[1]} heads, expected {expected_heads}"
                )
            if tensor.shape[-1] != expected_head_dim:
                raise ValueError(
                    f"{name} head dim is {tensor.shape[-1]}, "
                    f"expected {expected_head_dim}"
                )

        condition_k_delta = deltas.condition_k
        condition_v_delta = deltas.condition_v
        if condition_k_delta.ndim == 5:
            slot = 0 if condition_slot is None else int(condition_slot)
            if not 0 <= slot < condition_k_delta.shape[1]:
                raise IndexError(
                    f"condition_slot={slot} is outside "
                    f"[0, {condition_k_delta.shape[1]})"
                )
            condition_k_delta = condition_k_delta[:, slot]
            condition_v_delta = condition_v_delta[:, slot]
        elif condition_slot not in {None, 0}:
            raise IndexError(
                "A nonzero condition_slot requires metadata shaped [B,C,D]"
            )

        if inject_main_query and self.target_to_q:
            q_main = q_main + (
                self._gate(self.target_q_gate, layer_index, q_main)
                * deltas.target_q.to(device=q_main.device, dtype=q_main.dtype)
            )
        if inject_main_query and self.pair_to_q:
            q_main = q_main + (
                self._gate(self.relation_q_gate, layer_index, q_main)
                * deltas.relation_q.to(device=q_main.device, dtype=q_main.dtype)
            )
        if self.condition_to_k:
            k_condition = k_condition + (
                self._gate(self.condition_k_gate, layer_index, k_condition)
                * condition_k_delta.to(
                    device=k_condition.device,
                    dtype=k_condition.dtype,
                )
            )
        if self.condition_to_v:
            v_condition = v_condition + (
                self._gate(self.condition_v_gate, layer_index, v_condition)
                * condition_v_delta.to(
                    device=v_condition.device,
                    dtype=v_condition.dtype,
                )
            )
        return q_main, k_condition, v_condition

    def gate_statistics(self) -> dict[str, float]:
        statistics = {}
        for name in (
            "target_q_gate",
            "relation_q_gate",
            "condition_k_gate",
            "condition_v_gate",
        ):
            values = getattr(self, name).detach().float()
            statistics[f"{name}_abs_mean"] = values.abs().mean().item()
            statistics[f"{name}_abs_max"] = values.abs().max().item()
        return statistics

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "config": {
                "condition_metadata_dim": self.condition_metadata_dim,
                "target_metadata_dim": self.target_metadata_dim,
                "metadata_hidden_dim": self.metadata_hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "target_to_q": self.target_to_q,
                "condition_to_k": self.condition_to_k,
                "condition_to_v": self.condition_to_v,
                "pair_to_q": self.pair_to_q,
            },
            "state_dict": self.state_dict(),
        }


def configure_metadata_qkv(
    transformer,
    config: dict[str, Any] | None,
    *,
    device,
    dtype,
    checkpoint_dir: str | None = None,
) -> MetadataQKVInjector | None:
    config = config or {}
    if not config.get("enabled", False):
        return None
    if float(config.get("gate_init", 0.0)) != 0.0:
        raise ValueError(
            "This MIQ-MMA experiment requires gate_init=0.0 for exact "
            "initial baseline parity"
        )

    injector = MetadataQKVInjector.from_transformer(transformer, config)
    checkpoint_path = (
        Path(checkpoint_dir) / "metadata_qkv.pt"
        if checkpoint_dir
        else None
    )
    if checkpoint_path is not None and checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu")
        injector.load_state_dict(payload["state_dict"], strict=True)
        print(f"Loaded Metadata-QKV module from {checkpoint_path}")
    injector.to(device=device, dtype=dtype)
    return injector
