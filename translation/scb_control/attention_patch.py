"""Attach trainable SCB gamma parameters to FLUX attention modules."""

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
    if mode == "middle":
        return set(range(total // 3, (2 * total) // 3))
    if mode == "middle_late":
        return set(range(total // 3, total))
    raise ValueError(f"Unsupported SCB inject_blocks mode: {mode}")


def _attention_targets(transformer, inject_blocks: str):
    dual_indices = _selected_indices(len(transformer.transformer_blocks), inject_blocks)
    single_indices = _selected_indices(
        len(transformer.single_transformer_blocks), inject_blocks
    )
    for index in sorted(dual_indices):
        yield f"transformer_blocks.{index}.attn", transformer.transformer_blocks[index].attn
    for index in sorted(single_indices):
        yield (
            f"single_transformer_blocks.{index}.attn",
            transformer.single_transformer_blocks[index].attn,
        )


def configure_scb_control(transformer, scb_config: dict, device=None, dtype=None):
    if device is None:
        device = next(transformer.parameters()).device
    if dtype is None:
        dtype = next(transformer.parameters()).dtype
    inject_blocks = scb_config.get("inject_blocks", "all")
    gamma_init = float(scb_config.get("gamma_init", 0.0))
    params = []
    for name, attention in _attention_targets(transformer, inject_blocks):
        parameter = nn.Parameter(torch.tensor([gamma_init], device=device, dtype=dtype))
        attention.scb_gamma = parameter
        attention.scb_name = name
        params.append(parameter)
    transformer.scb_control_enabled = True
    transformer.scb_inject_blocks = inject_blocks
    return params


def scb_state_dict(transformer) -> dict:
    state = {}
    for name, module in transformer.named_modules():
        if hasattr(module, "scb_gamma"):
            state[f"{name}.scb_gamma"] = module.scb_gamma.detach().cpu()
    return state


def load_scb_state_dict(transformer, state_dict: dict, strict: bool = False):
    missing, unexpected = [], []
    expected = {}
    for name, module in transformer.named_modules():
        if hasattr(module, "scb_gamma"):
            expected[f"{name}.scb_gamma"] = module

    for key, value in state_dict.items():
        if key.startswith("scb_control."):
            key = key[len("scb_control.") :]
        module = expected.get(key)
        if module is None:
            unexpected.append(key)
            continue
        module.scb_gamma.data.copy_(value.to(module.scb_gamma.device, module.scb_gamma.dtype))

    for key in expected:
        raw_key = key
        prefixed_key = f"scb_control.{key}"
        if raw_key not in state_dict and prefixed_key not in state_dict:
            missing.append(key)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Missing SCB keys: {missing}; unexpected SCB keys: {unexpected}")
    return missing, unexpected
