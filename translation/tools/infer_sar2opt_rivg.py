#!/usr/bin/env python3
"""Training-free RIVG inference for paired SAR/NIR-to-optical translation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import yaml
from PIL import Image

from glor_control import GLoRControl
from paired_to_optical_infer_utils import (
    close_distributed,
    find_glor_checkpoint,
    find_latest_checkpoint,
    find_run_config,
    initialize_distributed,
    load_glor_weights,
    load_pipeline,
    shard_samples,
    target_token_shape,
    write_distributed_manifest,
)
from omini.pipeline.flux_omini import Condition, generate, seed_everything
from omini.pipeline.rivg import (
    RIVGConfig, prepare_rivg_reference, validate_information_weights,
)
from omini.train_flux.nir_to_optical_data import (
    build_nir_to_optical_prompts,
    collect_nir_to_optical_pairs,
    load_pair_images as load_nir_pair_images,
)
from omini.train_flux.sar_to_optical_data import (
    build_sar_to_optical_prompts,
    collect_sar_to_optical_pairs,
    load_pair_images as load_sar_pair_images,
    pil_to_float_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("sar", "nir"), default="sar")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--glor-checkpoint", type=Path, default=None)
    parser.add_argument("--lora-preset", choices=("default", "expanded"))
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--glor-control", action="store_true")
    parser.add_argument("--keep-omini-condition", action="store_true")
    parser.add_argument(
        "--rivg-method",
        choices=("stable_rivg", "deficit_rivg"),
        required=True,
    )
    parser.add_argument("--weak-kernel-size", type=int, default=9)
    parser.add_argument("--weak-sigma", type=float, default=2.0)
    parser.add_argument("--info-lower-quantile", type=float, default=0.40)
    parser.add_argument("--info-upper-quantile", type=float, default=0.95)
    parser.add_argument(
        "--rivg-info-weights", type=float, nargs=3, default=(0.5, 0.3, 0.2),
        metavar=("GRADIENT", "VARIANCE", "COHERENCE"),
        help="Non-negative information-map weights summing to 1.",
    )
    parser.add_argument("--rivg-lambda-max", type=float, default=0.25)
    parser.add_argument("--rivg-time-power", type=float, default=2.0)
    parser.add_argument("--rivg-start-fraction", type=float, default=0.70)
    parser.add_argument("--residual-norm-ratio", type=float, default=0.50)
    parser.add_argument("--absorption-start-fraction", type=float, default=0.70)
    parser.add_argument("--absorption-temperature", type=float, default=0.25)
    parser.add_argument("--information-power", type=float, default=1.0)
    parser.add_argument("--deficit-power", type=float, default=1.5)
    parser.add_argument("--active-quantile", type=float, default=0.70)
    parser.add_argument("--max-deficit-weight", type=float, default=2.0)
    parser.add_argument("--ema-decay", type=float, default=0.80)
    parser.add_argument("--save-rivg-debug", action="store_true")
    args = parser.parse_args()
    try:
        args.rivg_info_weights = validate_information_weights(args.rivg_info_weights)
    except ValueError as error:
        parser.error(str(error))
    return args


def resolve_checkpoint(args: argparse.Namespace) -> None:
    if args.checkpoint is None:
        if args.lora_preset is None:
            raise ValueError("Pass either --checkpoint or --lora-preset.")
        args.checkpoint = find_latest_checkpoint(args.runs_dir, args.lora_preset)
        print(f"Automatically selected checkpoint: {args.checkpoint}")


def information_map_image(value: torch.Tensor) -> Image.Image:
    array = (
        value.detach().float().squeeze().clamp(0.0, 1.0).mul(255.0).round()
        .byte().cpu().numpy()
    )
    return Image.fromarray(array, mode="L")


def modality_spec(modality):
    if modality == "sar":
        return {
            "collect": collect_sar_to_optical_pairs,
            "load_pair": load_sar_pair_images,
            "build_prompts": build_sar_to_optical_prompts,
            "input_key": "sar_path",
            "input_dir": "sar",
            "weak_input_dir": "rivg_weak_sar",
            "output_prefix": "sar2opt",
            "source": "grsm_opt_sar",
            "label": "SAR",
        }
    return {
        "collect": collect_nir_to_optical_pairs,
        "load_pair": load_nir_pair_images,
        "build_prompts": build_nir_to_optical_prompts,
        "input_key": "nir_path",
        "input_dir": "nir",
        "weak_input_dir": "rivg_weak_nir",
        "output_prefix": "nir2opt",
        "source": "grsm_opt_nir_whu",
        "label": "NIR",
    }


def output_paths(
    output_dir: Path,
    sample_id: str,
    spec: dict,
) -> dict[str, Path]:
    name = f"{spec['output_prefix']}__{sample_id}.png"
    return {
        "output": output_dir / name,
        "original": output_dir / "originals" / name,
        "input": output_dir / spec["input_dir"] / name,
        "weak_input": output_dir / spec["weak_input_dir"] / name,
        "information_map": output_dir / "rivg_information_map" / name,
    }


def build_rivg_config(args: argparse.Namespace) -> RIVGConfig:
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


def build_glor_context(
    glor_control,
    image,
    *,
    device,
    dtype,
    task,
    token_shape,
):
    tensor = pil_to_float_tensor(image).unsqueeze(0).to(device=device, dtype=dtype)
    return glor_control.build_context(
        {
            "lr_image": tensor,
            "scale": torch.ones(1, device=device, dtype=dtype),
        },
        task=task,
        target_latent_shape=token_shape,
    )


def main() -> None:
    args = parse_args()
    rank, world_size, args.device = initialize_distributed(args.device)
    spec = modality_spec(args.modality)
    if args.max_samples < 0 or args.steps <= 0:
        raise ValueError("--max-samples must be non-negative and --steps positive.")
    if args.rivg_method == "deficit_rivg" and not args.keep_omini_condition:
        raise ValueError(
            "Deficit RIVG needs --keep-omini-condition for the aligned "
            "reference latent."
        )
    resolve_checkpoint(args)
    config_path = args.config or find_run_config(args.checkpoint)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    adapter_names = config.get("model", {}).get("adapter_names")
    if not isinstance(adapter_names, list) or len(adapter_names) != 3:
        raise ValueError(
            f"RIVG {spec['label']} inference requires "
            "[text, main, condition] adapters."
        )

    dataset_config = dict(config["train"]["dataset"])
    if args.data_root is not None:
        dataset_config["root"] = str(args.data_root.expanduser().resolve())
    image_size = tuple(map(int, dataset_config.get("image_size", [512, 512])))
    center_crop = bool(dataset_config.get("center_crop", False))
    samples = spec["collect"](dataset_config["root"], [args.split])
    if args.max_samples:
        samples = samples[: args.max_samples]
    total_samples = len(samples)
    local_samples = shard_samples(samples, rank, world_size)
    print(
        f"Rank {rank}: assigned {len(local_samples)} of "
        f"{total_samples} samples."
    )
    rivg_config = build_rivg_config(args)

    directories = [
        args.output_dir,
        args.output_dir / "originals",
        args.output_dir / spec["input_dir"],
    ]
    if args.save_rivg_debug:
        directories.extend(
            [
                args.output_dir / spec["weak_input_dir"],
                args.output_dir / "rivg_information_map",
            ]
        )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    pipe = (
        load_pipeline(config, args.checkpoint, args.device)
        if local_samples
        else None
    )
    dtype_name = config.get("dtype", "bfloat16")
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    dtype = getattr(torch, dtype_name)
    glor_control = None
    glor_checkpoint = None
    glor_task = None
    token_shape = None
    if args.glor_control and local_samples:
        glor_config = config.get("glor_control", {})
        if not glor_config.get("enable", False):
            raise ValueError(f"{config_path} does not enable glor_control.")
        glor_task = glor_config.get("task", "super_resolution")
        if glor_task != "super_resolution":
            raise ValueError(
                f"{spec['label']} RIVG requires GLoR task=super_resolution."
            )
        glor_control = GLoRControl.from_config(
            pipe.transformer, {"glor_control": glor_config}
        ).to(device=args.device, dtype=dtype)
        pipe.transformer.glor_control = glor_control
        glor_checkpoint = args.glor_checkpoint or find_glor_checkpoint(args.checkpoint)
        if glor_checkpoint is None:
            raise FileNotFoundError("No glor_control.pt beside the LoRA checkpoint.")
        load_result = load_glor_weights(glor_checkpoint, glor_control)
        print(
            f"Loaded GLoR-Control from {glor_checkpoint} "
            f"(missing={len(load_result.missing_keys)}, "
            f"unexpected={len(load_result.unexpected_keys)})"
        )
        glor_control.eval()
        token_shape = target_token_shape(pipe, image_size[1], image_size[0])

    guided_steps = sum(
        index / max(args.steps - 1, 1) >= rivg_config.guidance_start_fraction
        for index in range(args.steps)
    )
    image_cfg_multiplier = 2 if args.image_guidance_scale != 1.0 else 1
    transformer_nfe = (args.steps + guided_steps) * image_cfg_multiplier
    print(f"Dataset root: {dataset_config['root']}")
    print(f"Modality: {spec['label']} -> optical")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"RIVG: {json.dumps(rivg_config.to_dict(), sort_keys=True)}")
    print(
        f"Weak reference: Gaussian {args.weak_kernel_size}x"
        f"{args.weak_kernel_size}, sigma={args.weak_sigma}"
    )
    print(f"Estimated transformer NFE per sample: {transformer_nfe}")

    width, height = image_size
    records = []
    for local_index, (global_index, sample) in enumerate(local_samples):
        paths = output_paths(args.output_dir, sample["id"], spec)
        if args.modality == "sar":
            input_image, optical_image = spec["load_pair"](
                sample,
                image_size,
                center_crop=center_crop,
            )
        else:
            input_image, optical_image = spec["load_pair"](sample, image_size)
        weak_input, information_map = prepare_rivg_reference(
            input_image,
            weak_kernel_size=args.weak_kernel_size,
            weak_sigma=args.weak_sigma,
            lower_quantile=args.info_lower_quantile,
            upper_quantile=args.info_upper_quantile,
            information_weights=args.rivg_info_weights,
        )
        elapsed_seconds = None
        peak_memory_bytes = None
        if not paths["output"].exists() or args.overwrite:
            clip_prompt, t5_prompt = spec["build_prompts"](sample["metadata"])
            sample_seed = args.seed + global_index
            seed_everything(sample_seed)
            generator = torch.Generator(device=args.device).manual_seed(sample_seed)
            full_condition = Condition(
                input_image,
                adapter_names[2],
                position_delta=(0.0, 0.0),
                position_scale=1.0,
                posterior="mode",
            )
            weak_condition = Condition(
                weak_input,
                adapter_names[2],
                position_delta=(0.0, 0.0),
                position_scale=1.0,
                posterior="mode",
            )
            conditions = []
            weak_conditions = []
            if not args.glor_control or args.keep_omini_condition:
                conditions = [full_condition]
                weak_conditions = [weak_condition]

            glor_context = None
            weak_glor_context = None
            if glor_control is not None:
                with torch.no_grad():
                    glor_context = build_glor_context(
                        glor_control,
                        input_image,
                        device=args.device,
                        dtype=dtype,
                        task=glor_task,
                        token_shape=token_shape,
                    )
                    weak_glor_context = build_glor_context(
                        glor_control,
                        weak_input,
                        device=args.device,
                        dtype=dtype,
                        task=glor_task,
                        token_shape=token_shape,
                    )
            if str(args.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(torch.device(args.device))
                torch.cuda.synchronize(torch.device(args.device))
            started = time.perf_counter()
            result = generate(
                pipe,
                prompt=clip_prompt,
                prompt_2=t5_prompt,
                conditions=conditions,
                height=height,
                width=width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                image_guidance_scale=args.image_guidance_scale,
                generator=generator,
                main_adapter=adapter_names[1],
                text_adapter=adapter_names[0],
                main_adapter_on_text=False,
                glor_control_context=glor_context,
                rivg_config=rivg_config,
                rivg_weak_conditions=weak_conditions,
                rivg_information_map=information_map,
                rivg_weak_glor_control_context=weak_glor_context,
            ).images[0]
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize(torch.device(args.device))
                peak_memory_bytes = torch.cuda.max_memory_allocated(
                    torch.device(args.device)
                )
            elapsed_seconds = time.perf_counter() - started
            result.save(paths["output"])
        if not paths["original"].exists() or args.overwrite:
            optical_image.save(paths["original"])
        if not paths["input"].exists() or args.overwrite:
            input_image.save(paths["input"])
        if args.save_rivg_debug:
            if not paths["weak_input"].exists() or args.overwrite:
                weak_input.save(paths["weak_input"])
            if not paths["information_map"].exists() or args.overwrite:
                information_map_image(information_map).save(paths["information_map"])

        records.append(
            {
                "sample_id": sample["id"],
                "source": spec["source"],
                "input": str(sample[spec["input_key"]]),
                "output": str(paths["output"]),
                "original": str(paths["original"]),
                spec["input_dir"]: str(paths["input"]),
                "image_size": [width, height],
                "center_crop": center_crop,
                "seed": args.seed + global_index,
                "sample_index": global_index,
                "rank": rank,
                "checkpoint": str(args.checkpoint),
                "glor_checkpoint": (
                    str(glor_checkpoint) if glor_checkpoint is not None else None
                ),
                "config": str(config_path),
                "glor_control": args.glor_control,
                # Do not read the loop-local ``conditions`` here: it is only
                # constructed when a sample is generated. On a resumed run an
                # existing output is skipped, so that local may not have been
                # assigned yet. This expression is the configuration-level
                # equivalent of the condition-list construction above.
                "keep_omini_condition": (
                    not args.glor_control or args.keep_omini_condition
                ),
                "training_free_velocity_guidance": rivg_config.to_dict(),
                "weak_reference": {
                    "method": "gaussian",
                    "kernel_size": args.weak_kernel_size,
                    "sigma": args.weak_sigma,
                },
                "information_map": {
                    "weights": args.rivg_info_weights,
                    "method": "speckle_robust",
                    "lower_quantile": args.info_lower_quantile,
                    "upper_quantile": args.info_upper_quantile,
                    "debug_path": (
                        str(paths["information_map"])
                        if args.save_rivg_debug
                        else None
                    ),
                },
                "transformer_nfe": transformer_nfe,
                "elapsed_seconds": elapsed_seconds,
                "peak_memory_bytes": peak_memory_bytes,
            }
        )
        print(
            f"[rank {rank}: {local_index + 1}/{len(local_samples)}; "
            f"global {global_index + 1}/{total_samples}] {sample['id']}: "
            f"{rivg_config.method}, elapsed={elapsed_seconds}"
        )

    manifest_path, merged = write_distributed_manifest(
        args.output_dir,
        records,
        rank,
        world_size,
    )
    if rank == 0:
        result_count = len(merged) if merged is not None else len(records)
        print(
            f"Saved {result_count} RIVG result(s) to {args.output_dir}; "
            f"manifest: {manifest_path}"
        )
    close_distributed()


if __name__ == "__main__":
    main()
