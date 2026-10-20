"""Shared inference helpers for paired PNG-to-optical tasks."""

import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from diffusers import FluxPipeline, FluxTransformer2DModel


def initialize_distributed(device):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if str(device).startswith("cuda") else "gloo"
        # Independent inference shards may finish hours apart on resume.
        # Allow slow workers to finish before the final manifest barriers.
        timeout_seconds = int(os.environ.get("INFERENCE_DIST_TIMEOUT_SECONDS", "21600"))
        if timeout_seconds <= 0:
            raise ValueError("INFERENCE_DIST_TIMEOUT_SECONDS must be positive.")
        dist.init_process_group(
            backend=backend, timeout=timedelta(seconds=timeout_seconds)
        )
    if str(device).startswith("cuda"):
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        elif ":" in str(device):
            torch.cuda.set_device(torch.device(device))
    print(
        f"Distributed inference: rank={rank}/{world_size}, "
        f"local_rank={local_rank}, device={device}"
    )
    return rank, world_size, device


def shard_samples(samples, rank, world_size):
    return list(enumerate(samples))[rank::world_size]


def write_distributed_manifest(output_dir, records, rank, world_size):
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if world_size == 1:
        manifest_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest_path, records

    part_path = output_dir / f"manifest.rank{rank:05d}.json"
    part_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dist.barrier()
    merged = None
    if rank == 0:
        merged = []
        for part_rank in range(world_size):
            part_path = output_dir / f"manifest.rank{part_rank:05d}.json"
            with part_path.open("r", encoding="utf-8") as stream:
                merged.extend(json.load(stream))
        merged.sort(key=lambda record: int(record["sample_index"]))
        manifest_path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    dist.barrier()
    return manifest_path, merged


def close_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def active_adapter_names(config):
    adapter_names = config.get("model", {}).get("adapter_names")
    if adapter_names is None:
        return ["default"]
    if not isinstance(adapter_names, list):
        raise ValueError("model.adapter_names must be a list.")
    active_names = []
    for adapter_name in adapter_names:
        if adapter_name is None:
            continue
        if not isinstance(adapter_name, str) or not adapter_name:
            raise ValueError(
                "Each model.adapter_names entry must be null or a non-empty string."
            )
        if adapter_name not in active_names:
            active_names.append(adapter_name)
    if not active_names:
        raise ValueError("model.adapter_names must contain at least one active adapter.")
    return active_names


def find_latest_checkpoint(runs_dir, preset):
    runs_dir = Path(runs_dir)
    # Also accept a concrete checkpoint directory. This makes RUNS_DIR/CLI
    # mistakes fail less mysteriously while preserving the normal run-root
    # discovery behavior.
    if runs_dir.is_dir() and runs_dir.name.isdigit():
        try:
            config_path = find_run_config(runs_dir)
        except FileNotFoundError:
            config_path = None
        if config_path is not None:
            config_text = config_path.read_text(encoding="utf-8")
            config = yaml.safe_load(config_text)
            config_preset = (
                "expanded" if "time_text_embed" in config_text else "default"
            )
            adapter_names = active_adapter_names(config)
            if config_preset == preset and all(
                (runs_dir / f"{name}.safetensors").is_file()
                for name in adapter_names
            ):
                return runs_dir

    candidates = []
    for config_path in runs_dir.glob("*/config.yaml"):
        config_text = config_path.read_text(encoding="utf-8")
        config = yaml.safe_load(config_text)
        config_preset = "expanded" if "time_text_embed" in config_text else "default"
        if config_preset != preset:
            continue
        adapter_names = active_adapter_names(config)
        for checkpoint in (config_path.parent / "ckpt").glob("*"):
            if (
                checkpoint.is_dir()
                and checkpoint.name.isdigit()
                and all(
                    (checkpoint / f"{name}.safetensors").is_file()
                    for name in adapter_names
                )
            ):
                candidates.append(
                    (config_path.parent.name, int(checkpoint.name), checkpoint)
                )
    if not candidates:
        raise FileNotFoundError(
            f"No {preset!r} LoRA checkpoints found below {runs_dir}."
        )
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def find_run_config(checkpoint):
    checkpoint = checkpoint.resolve()
    start = checkpoint if checkpoint.is_dir() else checkpoint.parent
    for parent in (start, *start.parents):
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find config.yaml above checkpoint {checkpoint}."
    )


def load_pipeline(config, checkpoint, device):
    dtype_name = config.get("dtype", "bfloat16")
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype in config: {dtype_name}")
    dtype = getattr(torch, dtype_name)
    kwargs = {"torch_dtype": dtype}
    transformer_path = config.get("transformer_path")
    if transformer_path:
        kwargs["transformer"] = FluxTransformer2DModel.from_pretrained(
            transformer_path, torch_dtype=dtype
        )
    pipe = FluxPipeline.from_pretrained(config["flux_path"], **kwargs).to(device)
    adapter_names = active_adapter_names(config)
    checkpoint = checkpoint.resolve()
    if checkpoint.is_file():
        if len(adapter_names) != 1:
            raise ValueError(
                "A single LoRA file cannot provide multiple adapters; pass the "
                f"checkpoint directory containing: {adapter_names}."
            )
        weight_dir = checkpoint.parent
        weights = [(adapter_names[0], checkpoint.name)]
    else:
        weight_dir = checkpoint
        weights = [
            (adapter_name, f"{adapter_name}.safetensors")
            for adapter_name in adapter_names
        ]
    for adapter_name, weight_name in weights:
        weight_path = weight_dir / weight_name
        if not weight_path.is_file():
            raise FileNotFoundError(weight_path)
        pipe.load_lora_weights(
            str(weight_dir), weight_name=weight_name, adapter_name=adapter_name
        )
    pipe.set_adapters(adapter_names)
    print(f"Loaded LoRA adapters: {', '.join(adapter_names)}")
    return pipe


def checkpoint_dir(checkpoint):
    checkpoint = checkpoint.resolve()
    return checkpoint.parent if checkpoint.is_file() else checkpoint


def find_glor_checkpoint(lora_checkpoint):
    weight_dir = checkpoint_dir(lora_checkpoint)
    for name in ("glor_control.pt", "glor_control.pth", "glor_control.safetensors"):
        path = weight_dir / name
        if path.is_file():
            return path
    return None


def load_module_state(path):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def load_glor_weights(path, glor_control):
    state_dict = load_module_state(path)
    if "glor_control" in state_dict:
        state_dict = state_dict["glor_control"]
    elif any(key.startswith("glor_control.") for key in state_dict):
        state_dict = {
            key[len("glor_control.") :]: value
            for key, value in state_dict.items()
            if key.startswith("glor_control.")
        }
    return glor_control.load_state_dict(state_dict, strict=False)


def target_token_shape(pipe, target_height, target_width):
    token_stride = int(pipe.vae_scale_factor) * 2
    if target_height % token_stride or target_width % token_stride:
        raise ValueError(
            f"Target size {target_width}x{target_height} is not divisible by "
            f"the packed FLUX token stride {token_stride}."
        )
    return target_height // token_stride, target_width // token_stride
