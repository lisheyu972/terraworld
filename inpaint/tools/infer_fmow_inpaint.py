#!/usr/bin/env python3
"""Run dynamic-mask fMOW inpainting inference on the independent test set."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
import yaml
from PIL import Image

from glor_control import GLoRControl
from metadata_qkv import configure_metadata_qkv
from omini.pipeline.rivg import (
    RIVGConfig,
    fuse_rivg_information_maps,
    prepare_rivg_reference,
)
from omini.train_flux.fmow_inpaint_data import (
    FMOW_MIQ_METADATA_DIM,
    MASK_TYPES,
    apply_mask,
    build_inpaint_prompts,
    build_fmow_miq_metadata,
    collect_fmow_frames,
    mask_coverage,
    read_annotation,
    read_center_cropped_rgb,
    pil_to_float_tensor,
    sample_inpaint_mask,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inpaint dynamically masked fMOW test images with OminiControl FLUX."
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--glor-checkpoint", type=Path)
    parser.add_argument("--lora-preset", choices=("default", "expanded"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/fmow_inpaint_flux_base"))
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--transformer-path",
        type=Path,
        help="Override config.transformer_path with a Diffusers transformer.",
    )
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--test-index", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument(
        "--frame-mode",
        choices=("first", "random", "all"),
        default="first",
        help="Select one frame per location or evaluate every unique frame.",
    )
    parser.add_argument("--mask-type", choices=("random", *MASK_TYPES), default="random")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-random-glor",
        action="store_true",
        help="Allow an untrained GLoR module for wiring smoke tests only.",
    )
    rivg = parser.add_argument_group("training-free RIVG guidance")
    rivg.add_argument(
        "--rivg-method",
        choices=("stable_rivg", "deficit_rivg"),
        default=None,
    )
    rivg.add_argument("--weak-kernel-size", type=int, default=9)
    rivg.add_argument("--weak-sigma", type=float, default=2.0)
    rivg.add_argument("--info-lower-quantile", type=float, default=0.40)
    rivg.add_argument("--info-upper-quantile", type=float, default=0.95)
    rivg.add_argument("--rivg-info-fusion", choices=("max", "mean"), default="max")
    rivg.add_argument("--rivg-lambda-max", type=float, default=0.25)
    rivg.add_argument("--rivg-time-power", type=float, default=2.0)
    rivg.add_argument("--rivg-start-fraction", type=float, default=0.70)
    rivg.add_argument("--residual-norm-ratio", type=float, default=0.50)
    rivg.add_argument("--absorption-start-fraction", type=float, default=0.70)
    rivg.add_argument("--absorption-temperature", type=float, default=0.25)
    rivg.add_argument("--information-power", type=float, default=1.0)
    rivg.add_argument("--deficit-power", type=float, default=1.5)
    rivg.add_argument("--active-quantile", type=float, default=0.70)
    rivg.add_argument("--max-deficit-weight", type=float, default=2.0)
    rivg.add_argument("--ema-decay", type=float, default=0.80)
    rivg.add_argument("--save-rivg-debug", action="store_true")
    return parser.parse_args()


def build_rivg_config(args):
    if args.rivg_method is None:
        return None
    return RIVGConfig(
        method=args.rivg_method,
        lambda_max=args.rivg_lambda_max,
        time_power=args.rivg_time_power,
        guidance_start_fraction=args.rivg_start_fraction,
        absorption_start_fraction=args.absorption_start_fraction,
        absorption_temperature=args.absorption_temperature,
        information_power=args.information_power,
        deficit_power=args.deficit_power,
        active_quantile=args.active_quantile,
        max_deficit_weight=args.max_deficit_weight,
        ema_decay=args.ema_decay,
        residual_norm_ratio=args.residual_norm_ratio,
    )


def information_map_image(value):
    array = (
        value.detach()
        .float()
        .squeeze()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="L")


def active_adapter_names(config):
    adapter_names = config.get("model", {}).get("adapter_names")
    if not isinstance(adapter_names, list):
        raise ValueError("model.adapter_names must be a list.")
    active = []
    for name in adapter_names:
        if name is not None and name not in active:
            active.append(name)
    if not active:
        raise ValueError("No active LoRA adapter names are configured.")
    return active


def find_latest_checkpoint(runs_dir, preset):
    candidates = []
    for config_path in Path(runs_dir).glob("*/config.yaml"):
        config_text = config_path.read_text(encoding="utf-8")
        if ("time_text_embed" in config_text) != (preset == "expanded"):
            continue
        config = yaml.safe_load(config_text)
        adapter_names = active_adapter_names(config)
        glor_enabled = bool(config.get("glor_control", {}).get("enable", False))
        metadata_enabled = bool(config.get("metadata_qkv", {}).get("enabled", False))
        for checkpoint in (config_path.parent / "ckpt").glob("*"):
            if (
                checkpoint.is_dir()
                and checkpoint.name.isdigit()
                and all(
                    (checkpoint / f"{name}.safetensors").is_file()
                    for name in adapter_names
                )
                and (
                    not glor_enabled
                    or (checkpoint / "glor_control.pt").is_file()
                )
                and (
                    not metadata_enabled
                    or (checkpoint / "metadata_qkv.pt").is_file()
                )
            ):
                candidates.append(
                    (config_path.parent.name, int(checkpoint.name), checkpoint)
                )
    if not candidates:
        raise FileNotFoundError(
            f"No {preset!r} fMOW inpainting checkpoints found below {runs_dir}."
        )
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def find_run_config(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    start = checkpoint if checkpoint.is_dir() else checkpoint.parent
    for parent in (start, *start.parents):
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find config.yaml above {checkpoint}.")


def checkpoint_dir(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    return checkpoint.parent if checkpoint.is_file() else checkpoint


def find_glor_checkpoint(lora_checkpoint):
    weight_dir = checkpoint_dir(lora_checkpoint)
    for name in ("glor_control.pt", "glor_control.pth", "glor_control.safetensors"):
        candidate = weight_dir / name
        if candidate.is_file():
            return candidate
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


def target_token_shape(pipe, height, width):
    token_stride = int(pipe.vae_scale_factor) * 2
    if height % token_stride or width % token_stride:
        raise ValueError(
            f"Target size {width}x{height} is not divisible by packed FLUX "
            f"token stride {token_stride}."
        )
    return height // token_stride, width // token_stride


def load_pipeline(config, checkpoint, device):
    from diffusers import FluxPipeline, FluxTransformer2DModel

    dtype_name = config.get("dtype", "bfloat16")
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    dtype = getattr(torch, dtype_name)
    kwargs = {"torch_dtype": dtype}
    if config.get("transformer_path"):
        kwargs["transformer"] = FluxTransformer2DModel.from_pretrained(
            config["transformer_path"], torch_dtype=dtype
        )
    pipe = FluxPipeline.from_pretrained(config["flux_path"], **kwargs).to(device)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir():
        raise ValueError(
            "This model uses text/main/condition adapters; --checkpoint must be a directory."
        )
    adapter_names = active_adapter_names(config)
    for name in adapter_names:
        weight = checkpoint / f"{name}.safetensors"
        if not weight.is_file():
            raise FileNotFoundError(weight)
        pipe.load_lora_weights(
            str(checkpoint), weight_name=weight.name, adapter_name=name
        )
    pipe.set_adapters(adapter_names)
    print(f"Loaded LoRA adapters: {', '.join(adapter_names)}")
    metadata_config = config.get("metadata_qkv", {})
    metadata_injector = None
    if metadata_config.get("enabled", False):
        for key in ("condition_metadata_dim", "target_metadata_dim"):
            if int(metadata_config.get(key, -1)) != FMOW_MIQ_METADATA_DIM:
                raise ValueError(
                    f"metadata_qkv.{key} must be {FMOW_MIQ_METADATA_DIM}."
                )
        metadata_path = checkpoint / "metadata_qkv.pt"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"MIQ-MMA config is enabled but weights are missing: {metadata_path}"
            )
        metadata_injector = configure_metadata_qkv(
            pipe.transformer,
            metadata_config,
            device=device,
            dtype=dtype,
            checkpoint_dir=str(checkpoint),
        )
        metadata_injector.eval().requires_grad_(False)
        print(f"Loaded MIQ-MMA weights: {metadata_path}")
    pipe._omini_metadata_qkv_injector = metadata_injector
    return pipe


def resolve_checkpoint(args):
    if args.checkpoint is None:
        if args.lora_preset is None:
            raise ValueError("Pass either --checkpoint or --lora-preset.")
        args.checkpoint = find_latest_checkpoint(args.runs_dir, args.lora_preset)
        print(f"Automatically selected checkpoint: {args.checkpoint}")


def select_samples(groups, frame_mode, seed):
    if frame_mode == "all":
        return [sample for group in groups for sample in group]
    if frame_mode == "first":
        return [group[0] for group in groups]
    rng = random.Random(seed)
    return [rng.choice(group) for group in groups]


def setup_distributed(args):
    """Initialize one inference process per GPU when launched with torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-process inference requires CUDA GPUs.")
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
        torch.distributed.init_process_group(backend="nccl")
    return rank, world_size


def save_distributed_manifest(output_dir, records, rank, world_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if world_size == 1:
        manifest_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return

    rank_manifest = output_dir / f".manifest.rank{rank}.json"
    rank_manifest.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.distributed.barrier()
    if rank == 0:
        merged_records = []
        rank_manifests = [
            output_dir / f".manifest.rank{worker_rank}.json"
            for worker_rank in range(world_size)
        ]
        for path in rank_manifests:
            merged_records.extend(json.loads(path.read_text(encoding="utf-8")))
        merged_records.sort(key=lambda record: record["sample_index"])
        manifest_path.write_text(
            json.dumps(merged_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for path in rank_manifests:
            path.unlink()
    torch.distributed.barrier()


def main():
    args = parse_args()
    rank, world_size = setup_distributed(args)
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative.")
    if args.save_rivg_debug and args.rivg_method is None:
        raise ValueError("--save-rivg-debug requires --rivg-method.")
    rivg_config = build_rivg_config(args)
    resolve_checkpoint(args)
    config_path = args.config or find_run_config(args.checkpoint)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if args.transformer_path is not None:
        if not (args.transformer_path / "config.json").is_file():
            raise FileNotFoundError(args.transformer_path / "config.json")
        config["transformer_path"] = str(args.transformer_path)
    glor_config = config.get("glor_control", {})
    glor_enabled = bool(glor_config.get("enable", False))
    if glor_enabled and glor_config.get("task") != "inpainting":
        raise ValueError(
            "fMOW mask-aware inference requires glor_control.task='inpainting'."
        )
    if args.glor_checkpoint is not None and not glor_enabled:
        raise ValueError(
            "--glor-checkpoint requires a config with glor_control.enable=true."
        )
    dataset_config = config["train"]["dataset"]
    adapter_names = config.get("model", {}).get("adapter_names")
    if not isinstance(adapter_names, list) or len(adapter_names) != 3:
        raise ValueError(
            "fMOW inpainting requires model.adapter_names=[text, main, condition]."
        )

    test_root = args.test_root or Path(dataset_config["test_root"])
    test_index = args.test_index or Path(
        dataset_config.get("test_index_path", test_root / "index.jsonl")
    )
    test_splits = dataset_config.get("test_source_splits", ["test"])
    groups = collect_fmow_frames(
        test_root,
        index_path=test_index,
        source_splits=test_splits,
        validate_files=True,
    )
    samples = select_samples(groups, args.frame_mode, args.seed)
    if args.max_samples:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No valid fMOW test samples were found.")
    indexed_samples = list(enumerate(samples))[rank::world_size]
    if world_size > 1:
        print(
            f"[rank {rank}/{world_size}] Processing {len(indexed_samples)} of "
            f"{len(samples)} samples on {args.device}."
        )

    directories = {
        name: args.output_dir / name
        for name in ("generated", "target", "condition", "mask", "prompts")
    }
    if rivg_config is not None and args.save_rivg_debug:
        directories.update(
            {
                "rivg_weak_condition": args.output_dir / "rivg_weak_condition",
                "rivg_information_map": args.output_dir / "rivg_information_map",
            }
        )
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    from omini.pipeline.flux_omini import Condition, generate, seed_everything

    pipe = load_pipeline(config, args.checkpoint, args.device)
    metadata_injector = pipe._omini_metadata_qkv_injector
    width, height = map(int, dataset_config.get("image_size", [512, 512]))
    glor_control = None
    glor_checkpoint = None
    token_shape = None
    if glor_enabled:
        dtype_name = config.get("dtype", "bfloat16")
        if not hasattr(torch, dtype_name):
            raise ValueError(f"Unsupported torch dtype: {dtype_name}")
        dtype = getattr(torch, dtype_name)
        glor_control = GLoRControl.from_config(
            pipe.transformer, {"glor_control": glor_config}
        ).to(device=args.device, dtype=dtype)
        pipe.transformer.glor_control = glor_control
        glor_checkpoint = args.glor_checkpoint or find_glor_checkpoint(
            args.checkpoint
        )
        if glor_checkpoint is None and not args.allow_random_glor:
            raise FileNotFoundError(
                "No GLoR checkpoint found beside the LoRA checkpoint. Pass "
                "--glor-checkpoint or --allow-random-glor for a smoke test."
            )
        if glor_checkpoint is not None:
            load_result = load_glor_weights(glor_checkpoint, glor_control)
            print(
                f"Loaded GLoR-Control from {glor_checkpoint} "
                f"(missing={len(load_result.missing_keys)}, "
                f"unexpected={len(load_result.unexpected_keys)})"
            )
        else:
            print("Warning: using randomly initialized GLoR-Control.")
        glor_control.eval()
        token_shape = target_token_shape(pipe, height, width)

    if rank == 0 and rivg_config is not None:
        guided_steps = sum(
            index / max(args.steps - 1, 1)
            >= rivg_config.guidance_start_fraction
            for index in range(args.steps)
        )
        image_cfg_multiplier = 2 if args.image_guidance_scale != 1.0 else 1
        transformer_nfe = (args.steps + guided_steps) * image_cfg_multiplier
        print(
            "RIVG enabled for the masked Omini reference"
            + (" and mask-aware GLoR context" if glor_control is not None else "")
            + f": {json.dumps(rivg_config.to_dict(), sort_keys=True)}"
        )
        print(
            f"Information-map fusion: {args.rivg_info_fusion}; "
            f"estimated transformer NFE/sample: {transformer_nfe}"
        )

    records = []
    for completed, (index, sample) in enumerate(indexed_samples, start=1):
        sample_seed = args.seed + index
        rng = random.Random(sample_seed)
        target = read_center_cropped_rgb(sample["image_path"], (width, height))
        mask, mask_type = sample_inpaint_mask(
            width, height, rng=rng, mask_type=args.mask_type
        )
        condition_image = apply_mask(target, mask)
        weak_condition_image = None
        information_map = None
        if rivg_config is not None:
            weak_condition_image, information_map = prepare_rivg_reference(
                condition_image,
                weak_kernel_size=args.weak_kernel_size,
                weak_sigma=args.weak_sigma,
                lower_quantile=args.info_lower_quantile,
                upper_quantile=args.info_upper_quantile,
            )
            # Keep the missing region exactly black in both RIVG branches;
            # Gaussian smoothing must not leak known pixels through the mask.
            weak_condition_image = apply_mask(weak_condition_image, mask)
            information_map = fuse_rivg_information_maps(
                [information_map], method=args.rivg_info_fusion
            )
        annotation = read_annotation(str(sample["json_path"]))
        clip_prompt, t5_prompt = build_inpaint_prompts(annotation)
        name = f"{index:06d}__{sample['id']}.png"
        generated_path = directories["generated"] / name
        if not generated_path.exists() or args.overwrite:
            seed_everything(sample_seed)
            generator = torch.Generator(device=args.device).manual_seed(sample_seed)
            glor_context = None
            weak_glor_context = None
            if glor_control is not None:
                dtype = next(glor_control.parameters()).dtype
                masked_rgb = pil_to_float_tensor(condition_image).unsqueeze(0)
                mask_tensor = torch.from_numpy(
                    np.asarray(mask, dtype=np.float32).copy()
                ).unsqueeze(0).unsqueeze(0).div_(255.0)
                masked_rgb = masked_rgb.to(device=args.device, dtype=dtype)
                mask_tensor = mask_tensor.to(device=args.device, dtype=dtype)
                with torch.no_grad():
                    glor_context = glor_control.build_context(
                        {
                            "inpaint_image": torch.cat(
                                [masked_rgb, mask_tensor], dim=1
                            ),
                            "inpaint_mask": mask_tensor,
                        },
                        task="inpainting",
                        target_latent_shape=token_shape,
                    )
                    if rivg_config is not None:
                        weak_masked_rgb = pil_to_float_tensor(
                            weak_condition_image
                        ).unsqueeze(0).to(device=args.device, dtype=dtype)
                        weak_glor_context = glor_control.build_context(
                            {
                                "inpaint_image": torch.cat(
                                    [weak_masked_rgb, mask_tensor], dim=1
                                ),
                                "inpaint_mask": mask_tensor,
                            },
                            task="inpainting",
                            target_latent_shape=token_shape,
                        )
            transformer_kwargs = {}
            if metadata_injector is not None:
                metadata_dtype = next(metadata_injector.parameters()).dtype
                frame_metadata = build_fmow_miq_metadata(annotation).unsqueeze(0)
                metadata_deltas = metadata_injector.prepare_deltas(
                    frame_metadata.to(device=args.device, dtype=metadata_dtype),
                    frame_metadata.to(device=args.device, dtype=metadata_dtype),
                )
                transformer_kwargs = {
                    "metadata_qkv_injector": metadata_injector,
                    "metadata_qkv_deltas": metadata_deltas,
                }
            condition_posterior = "mode" if rivg_config is not None else "sample"
            image = generate(
                pipe,
                prompt=clip_prompt,
                prompt_2=t5_prompt,
                conditions=[
                    Condition(
                        condition_image,
                        adapter_names[2],
                        position_delta=(0.0, 0.0),
                        position_scale=1.0,
                        posterior=condition_posterior,
                    )
                ],
                height=height,
                width=width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                image_guidance_scale=args.image_guidance_scale,
                generator=generator,
                glor_control_context=glor_context,
                rivg_config=rivg_config,
                rivg_weak_conditions=(
                    [
                        Condition(
                            weak_condition_image,
                            adapter_names[2],
                            position_delta=(0.0, 0.0),
                            position_scale=1.0,
                            posterior="mode",
                        )
                    ]
                    if rivg_config is not None
                    else None
                ),
                rivg_information_map=information_map,
                rivg_weak_glor_control_context=weak_glor_context,
                main_adapter=adapter_names[1],
                text_adapter=adapter_names[0],
                main_adapter_on_text=False,
                transformer_kwargs=transformer_kwargs,
            ).images[0]
            image.save(generated_path)
        saved_paths = {
            "target": directories["target"] / name,
            "condition": directories["condition"] / name,
            "mask": directories["mask"] / name,
        }
        for image, path in (
            (target, saved_paths["target"]),
            (condition_image, saved_paths["condition"]),
            (mask, saved_paths["mask"]),
        ):
            if not path.exists() or args.overwrite:
                image.save(path)
        if rivg_config is not None and args.save_rivg_debug:
            weak_path = directories["rivg_weak_condition"] / name
            map_path = directories["rivg_information_map"] / name
            if not weak_path.exists() or args.overwrite:
                weak_condition_image.save(weak_path)
            if not map_path.exists() or args.overwrite:
                information_map_image(information_map).save(map_path)
        prompt_path = directories["prompts"] / name.replace(".png", ".json")
        if not prompt_path.exists() or args.overwrite:
            prompt_path.write_text(
                json.dumps(
                    {"clip_prompt": clip_prompt, "t5_prompt": t5_prompt},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        records.append(
            {
                "sample_index": index,
                "sample_id": sample["id"],
                "source_group": sample["group"],
                "input": str(sample["image_path"]),
                "generated": str(generated_path),
                "target": str(saved_paths["target"]),
                "condition": str(saved_paths["condition"]),
                "mask": str(saved_paths["mask"]),
                "mask_type": mask_type,
                "mask_coverage": round(mask_coverage(mask), 6),
                "seed": sample_seed,
                "checkpoint": str(args.checkpoint),
                "glor_checkpoint": (
                    str(glor_checkpoint) if glor_checkpoint is not None else None
                ),
                "metadata_qkv_checkpoint": (
                    str(checkpoint_dir(args.checkpoint) / "metadata_qkv.pt")
                    if metadata_injector is not None
                    else None
                ),
                "config": str(config_path),
                "rivg": rivg_config.to_dict() if rivg_config is not None else None,
                "rivg_info_fusion": (
                    args.rivg_info_fusion if rivg_config is not None else None
                ),
            }
        )
        prefix = f"rank {rank} " if world_size > 1 else ""
        print(
            f"[{prefix}{completed}/{len(indexed_samples)}] "
            f"{sample['id']}: {mask_type}"
        )

    save_distributed_manifest(args.output_dir, records, rank, world_size)
    if rank == 0:
        print(f"Saved {len(samples)} fMOW inpainting result(s) to {args.output_dir}")
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
