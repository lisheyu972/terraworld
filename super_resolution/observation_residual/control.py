"""Zero-initialized gated residuals for FLUX condition branches."""

from __future__ import annotations

import torch
from torch import nn


def _selected_indices(total: int, mode: str, custom) -> set[int]:
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
    if mode == "custom":
        indices = {int(index) for index in (custom or [])}
        invalid = sorted(index for index in indices if index < 0 or index >= total)
        if invalid:
            raise ValueError(
                f"Observation Residual block indices out of range for {total} "
                f"blocks: {invalid}"
            )
        return indices
    raise ValueError(f"Unsupported Observation Residual blocks mode: {mode}")


def configure_observation_residual(transformer, config: dict, device=None, dtype=None):
    """Attach one trainable scalar gate to each selected FLUX block."""
    config = config or {}
    if not config.get("enable", False):
        return []
    mode = config.get("mode", "scalar_gate")
    if mode != "scalar_gate":
        raise ValueError(
            "The first Observation Residual implementation supports only "
            "mode=scalar_gate."
        )
    init = config.get("init", "zero")
    if init not in {"zero", "one"}:
        raise ValueError("observation_residual.init must be zero or one.")
    init_value = 0.0 if init == "zero" else 1.0
    if device is None:
        device = next(transformer.parameters()).device
    if dtype is None:
        dtype = next(transformer.parameters()).dtype

    block_mode = config.get("blocks", "all")
    dual_indices = _selected_indices(
        len(transformer.transformer_blocks),
        block_mode,
        config.get("custom_dual_blocks"),
    )
    single_indices = _selected_indices(
        len(transformer.single_transformer_blocks),
        block_mode,
        config.get("custom_single_blocks"),
    )
    params = []
    for index in sorted(dual_indices):
        block = transformer.transformer_blocks[index]
        block.obs_res_alpha = nn.Parameter(
            torch.tensor([init_value], device=device, dtype=dtype)
        )
        block.obs_res_name = f"transformer_blocks.{index}"
        params.append(block.obs_res_alpha)
    for index in sorted(single_indices):
        block = transformer.single_transformer_blocks[index]
        block.obs_res_alpha = nn.Parameter(
            torch.tensor([init_value], device=device, dtype=dtype)
        )
        block.obs_res_name = f"single_transformer_blocks.{index}"
        params.append(block.obs_res_alpha)

    transformer.observation_residual_enabled = True
    transformer.observation_residual_blocks = block_mode
    return params


def apply_observation_residual(block, output, identity):
    """Apply the configured gated shortcut without changing baseline at alpha=0."""
    alpha = getattr(block, "obs_res_alpha", None)
    if alpha is None:
        return output
    if output.shape != identity.shape:
        raise ValueError(
            "Observation Residual shape mismatch: "
            f"output={tuple(output.shape)}, identity={tuple(identity.shape)}"
        )
    return output + alpha.to(dtype=output.dtype) * identity


def observation_residual_state_dict(transformer) -> dict:
    state = {}
    for name, module in transformer.named_modules():
        if hasattr(module, "obs_res_alpha"):
            state[f"{name}.obs_res_alpha"] = module.obs_res_alpha.detach().cpu()
    return state


def load_observation_residual_state_dict(
    transformer, state_dict: dict, strict: bool = False
):
    expected = {}
    for name, module in transformer.named_modules():
        if hasattr(module, "obs_res_alpha"):
            expected[f"{name}.obs_res_alpha"] = module

    loaded = set()
    unexpected = []
    for raw_key, value in state_dict.items():
        key = raw_key
        if key.startswith("observation_residual."):
            key = key[len("observation_residual.") :]
        module = expected.get(key)
        if module is None:
            unexpected.append(key)
            continue
        module.obs_res_alpha.data.copy_(
            value.to(module.obs_res_alpha.device, module.obs_res_alpha.dtype)
        )
        loaded.add(key)

    missing = sorted(set(expected) - loaded)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Missing Observation Residual keys: {missing}; "
            f"unexpected keys: {unexpected}"
        )
    return missing, unexpected
