"""Residual injection for TOA-Control fields."""

from __future__ import annotations

import torch
from torch import nn


def _selected_indices(total: int, mode: str) -> set[int]:
    if total <= 0:
        return set()
    if mode == "all":
        return set(range(total))
    if mode == "late":
        return set(range(total // 2, total))
    if mode == "middle_late":
        return set(range(total // 3, total))
    if mode == "middle":
        return set(range(total // 3, (2 * total) // 3))
    raise ValueError(f"Unsupported TOA inject_blocks mode: {mode}")


class TOAControlInjector(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ctrl_dim: int,
        num_dual_blocks: int,
        num_single_blocks: int,
        inject_blocks: str = "middle_late",
        alpha_init: float = 0.0,
    ):
        super().__init__()
        self.ctrl_dim = int(ctrl_dim)
        self.hidden_dim = int(hidden_dim)
        self.dual_indices = _selected_indices(int(num_dual_blocks), inject_blocks)
        self.single_indices = _selected_indices(int(num_single_blocks), inject_blocks)
        self.dual_mlps = nn.ModuleDict(
            {
                str(index): self._make_mlp()
                for index in sorted(self.dual_indices)
            }
        )
        self.single_mlps = nn.ModuleDict(
            {
                str(index): self._make_mlp()
                for index in sorted(self.single_indices)
            }
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

    def _make_mlp(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.hidden_dim + self.ctrl_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def inject(
        self,
        hidden_states: torch.Tensor,
        ctrl_field: torch.Tensor | None,
        block_kind: str,
        block_index: int,
    ) -> torch.Tensor:
        if ctrl_field is None:
            return hidden_states
        if hidden_states.shape[:2] != ctrl_field.shape[:2]:
            raise ValueError(
                "TOA ctrl_field must match hidden state batch/token dimensions: "
                f"{tuple(ctrl_field.shape)} vs {tuple(hidden_states.shape)}."
            )
        if block_kind == "dual":
            mlps, alphas = self.dual_mlps, self.dual_alpha
        elif block_kind == "single":
            mlps, alphas = self.single_mlps, self.single_alpha
        else:
            raise ValueError(f"Unknown TOA block kind: {block_kind}")

        key = str(block_index)
        if key not in mlps:
            return hidden_states
        ctrl_field = ctrl_field.to(device=hidden_states.device, dtype=hidden_states.dtype)
        delta = mlps[key](torch.cat([hidden_states, ctrl_field], dim=-1))
        return hidden_states + alphas[key].to(hidden_states.dtype) * delta


def configure_toa_control(transformer, toa_config: dict, device=None, dtype=None):
    """Attach a TOA injector to a FLUX transformer and return it."""

    hidden_dim = (
        int(transformer.config.attention_head_dim)
        * int(transformer.config.num_attention_heads)
    )
    injector = TOAControlInjector(
        hidden_dim=hidden_dim,
        ctrl_dim=toa_config.get("ctrl_dim", 512),
        num_dual_blocks=len(transformer.transformer_blocks),
        num_single_blocks=len(transformer.single_transformer_blocks),
        inject_blocks=toa_config.get("inject_blocks", "middle_late"),
        alpha_init=toa_config.get("alpha_init", 0.0),
    )
    if device is None:
        device = next(transformer.parameters()).device
    if dtype is None:
        dtype = next(transformer.parameters()).dtype
    injector.to(device=device, dtype=dtype)
    transformer.toa_control_injector = injector
    return injector
