"""Condition-to-target flow-matching utilities."""

import re

import torch


C2TFM_OUTPUT_HEAD_MODULES = ("norm_out.linear", "proj_out")


def add_c2tfm_output_head_targets(target_modules):
    """Include the FLUX final normalization and projection in a LoRA target."""

    if isinstance(target_modules, str):
        missing = [
            module
            for module in C2TFM_OUTPUT_HEAD_MODULES
            if re.fullmatch(target_modules, module) is None
        ]
        if not missing:
            return target_modules
        alternatives = "|".join(re.escape(module) for module in missing)
        if target_modules.startswith("(") and target_modules.endswith(")"):
            return f"{target_modules[:-1]}|{alternatives})"
        return f"({target_modules}|{alternatives})"

    if isinstance(target_modules, (list, tuple)):
        targets = list(target_modules)
        for module in C2TFM_OUTPUT_HEAD_MODULES:
            if module not in targets:
                targets.append(module)
        return targets

    raise TypeError("LoRA target_modules must be a regex string, list, or tuple.")


def _broadcast_time(t, reference):
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, device=reference.device, dtype=reference.dtype)
    else:
        t = t.to(device=reference.device, dtype=reference.dtype)
    if t.ndim == 0:
        t = t.expand(reference.shape[0])
    if t.ndim != 1 or t.shape[0] != reference.shape[0]:
        raise ValueError(
            "t must be a scalar or contain one value per batch item; "
            f"got {tuple(t.shape)} for batch {reference.shape[0]}."
        )
    return t.reshape(t.shape[0], *([1] * (reference.ndim - 1)))


def build_c2tfm_path(target, source, t):
    """Return z_t=(1-t)z_target+t*z_source and source-target velocity."""

    if target.shape != source.shape:
        raise ValueError(
            f"C2TFM endpoint shapes differ: {tuple(target.shape)} vs "
            f"{tuple(source.shape)}."
        )
    t_view = _broadcast_time(t, target)
    trajectory = (1.0 - t_view) * target + t_view * source
    velocity = source - target
    return trajectory, velocity


def recover_c2tfm_target(trajectory, velocity, t):
    """Recover the t=0 endpoint as z_t-t*v."""

    if trajectory.shape != velocity.shape:
        raise ValueError(
            f"C2TFM trajectory/velocity shapes differ: {tuple(trajectory.shape)} "
            f"vs {tuple(velocity.shape)}."
        )
    return trajectory - _broadcast_time(t, trajectory) * velocity
