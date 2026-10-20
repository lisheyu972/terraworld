"""Metadata-injected Q/K/V modulation for OminiControl MM-Attention."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class MetadataQKVDeltas:
    target_q: Tensor
    relation_q: Tensor
    condition_k: Tensor
    condition_v: Tensor


class MetadataQKVInjector(nn.Module):
    """Broadcast compact metadata vectors into existing attention Q/K/V."""

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
    ):
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
            raise ValueError("projection_init_std must be positive.")

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

        def encoder(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, self.metadata_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.metadata_hidden_dim, self.metadata_hidden_dim),
            )

        self.condition_encoder = encoder(self.condition_metadata_dim)
        self.target_encoder = encoder(self.target_metadata_dim)
        self.relation_encoder = encoder(4 * self.metadata_hidden_dim)
        self.target_q_projection = nn.Linear(self.metadata_hidden_dim, self.inner_dim)
        self.relation_q_projection = nn.Linear(self.metadata_hidden_dim, self.inner_dim)
        self.condition_k_projection = nn.Linear(self.metadata_hidden_dim, self.inner_dim)
        self.condition_v_projection = nn.Linear(self.metadata_hidden_dim, self.inner_dim)

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
            nn.init.normal_(projection.weight, std=float(projection_init_std))
            nn.init.zeros_(projection.bias)
        self._configure_trainable_branches()

    def _configure_trainable_branches(self):
        self.requires_grad_(False)
        self.condition_encoder.requires_grad_(
            self.condition_to_k or self.condition_to_v or self.pair_to_q
        )
        self.target_encoder.requires_grad_(self.target_to_q or self.pair_to_q)
        for module, enabled in (
            (self.target_q_projection, self.target_to_q),
            (self.relation_encoder, self.pair_to_q),
            (self.relation_q_projection, self.pair_to_q),
            (self.condition_k_projection, self.condition_to_k),
            (self.condition_v_projection, self.condition_to_v),
        ):
            module.requires_grad_(enabled)
        for parameter, enabled in (
            (self.target_q_gate, self.target_to_q),
            (self.relation_q_gate, self.pair_to_q),
            (self.condition_k_gate, self.condition_to_k),
            (self.condition_v_gate, self.condition_to_v),
        ):
            parameter.requires_grad_(enabled)

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    @classmethod
    def from_transformer(cls, transformer, config: dict[str, Any]):
        return cls(
            condition_metadata_dim=int(config["condition_metadata_dim"]),
            target_metadata_dim=int(config["target_metadata_dim"]),
            metadata_hidden_dim=int(config.get("metadata_hidden_dim", 256)),
            num_layers=len(transformer.transformer_blocks)
            + len(transformer.single_transformer_blocks),
            num_heads=int(transformer.config.num_attention_heads),
            head_dim=int(transformer.config.attention_head_dim),
            target_to_q=config.get("target_to_q", True),
            condition_to_k=config.get("condition_to_k", True),
            condition_to_v=config.get("condition_to_v", False),
            pair_to_q=config.get("pair_to_q", False),
            gate_init=float(config.get("gate_init", 0.0)),
            projection_init_std=float(config.get("projection_init_std", 1e-3)),
        )

    def _reshape_delta(self, value):
        if value.ndim == 2:
            return value.view(value.shape[0], self.num_heads, 1, self.head_dim)
        if value.ndim == 3:
            return value.view(
                value.shape[0], value.shape[1], self.num_heads, 1, self.head_dim
            )
        raise ValueError("Projected metadata must have shape [B,D] or [B,C,D].")

    def prepare_deltas(self, condition_metadata, target_metadata):
        if condition_metadata.ndim not in {2, 3}:
            raise ValueError("condition_metadata must have shape [B,D] or [B,C,D].")
        if target_metadata.ndim != 2:
            raise ValueError("target_metadata must have shape [B,D].")
        if condition_metadata.shape[0] != target_metadata.shape[0]:
            raise ValueError("Condition and target metadata batch sizes differ.")
        if condition_metadata.shape[-1] != self.condition_metadata_dim:
            raise ValueError("Unexpected condition metadata width.")
        if target_metadata.shape[-1] != self.target_metadata_dim:
            raise ValueError("Unexpected target metadata width.")

        condition_embedding = self.condition_encoder(condition_metadata)
        relation_condition = (
            condition_embedding.mean(dim=1)
            if condition_embedding.ndim == 3
            else condition_embedding
        )
        target_embedding = self.target_encoder(target_metadata)
        relation_embedding = self.relation_encoder(
            torch.cat(
                (
                    target_embedding,
                    relation_condition,
                    target_embedding - relation_condition,
                    target_embedding * relation_condition,
                ),
                dim=-1,
            )
        )
        return MetadataQKVDeltas(
            target_q=self._reshape_delta(self.target_q_projection(target_embedding)),
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

    def _gate(self, table, layer_index, reference):
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(f"Invalid metadata layer index: {layer_index}")
        return table[layer_index].to(reference)[None, :, None, None]

    def inject(
        self,
        *,
        q_main,
        k_condition,
        v_condition,
        deltas,
        layer_index,
        condition_slot=None,
        inject_main_query=True,
    ):
        condition_k = deltas.condition_k
        condition_v = deltas.condition_v
        if condition_k.ndim == 5:
            slot = 0 if condition_slot is None else int(condition_slot)
            condition_k = condition_k[:, slot]
            condition_v = condition_v[:, slot]
        if inject_main_query and self.target_to_q:
            q_main = q_main + self._gate(
                self.target_q_gate, layer_index, q_main
            ) * deltas.target_q.to(q_main)
        if inject_main_query and self.pair_to_q:
            q_main = q_main + self._gate(
                self.relation_q_gate, layer_index, q_main
            ) * deltas.relation_q.to(q_main)
        if self.condition_to_k:
            k_condition = k_condition + self._gate(
                self.condition_k_gate, layer_index, k_condition
            ) * condition_k.to(k_condition)
        if self.condition_to_v:
            v_condition = v_condition + self._gate(
                self.condition_v_gate, layer_index, v_condition
            ) * condition_v.to(v_condition)
        return q_main, k_condition, v_condition

    def gate_statistics(self):
        result = {}
        for name in (
            "target_q_gate",
            "relation_q_gate",
            "condition_k_gate",
            "condition_v_gate",
        ):
            values = getattr(self, name).detach().float()
            result[f"{name}_abs_mean"] = values.abs().mean().item()
            result[f"{name}_abs_max"] = values.abs().max().item()
        return result

    def checkpoint_payload(self):
        return {"version": 1, "state_dict": self.state_dict()}


def configure_metadata_qkv(
    transformer,
    config: dict[str, Any] | None,
    *,
    device,
    dtype,
    checkpoint_dir: str | None = None,
):
    config = config or {}
    if not config.get("enabled", False):
        return None
    if float(config.get("gate_init", 0.0)) != 0.0:
        raise ValueError("MIQ-MMA requires gate_init=0.0 for baseline parity.")
    injector = MetadataQKVInjector.from_transformer(transformer, config)
    checkpoint_path = Path(checkpoint_dir) / "metadata_qkv.pt" if checkpoint_dir else None
    if checkpoint_path is not None and checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        injector.load_state_dict(payload["state_dict"], strict=True)
        print(f"Loaded Metadata-QKV module from {checkpoint_path}")
    injector.to(device=device, dtype=dtype)
    return injector
