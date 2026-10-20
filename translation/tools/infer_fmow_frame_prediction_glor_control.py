#!/usr/bin/env python3
"""Infer fMoW frame-prediction samples with temporal GLoR-Control."""

from __future__ import annotations

import argparse
import json
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

from glor_control import GLoRControl
from infer_fmow_frame_prediction import (
    build_rivg_config,
    close_distributed,
    collect_examples,
    find_latest_checkpoint,
    information_map_image,
    initialize_distributed,
    load_pipeline,
    load_test_list,
    open_resized_rgb,
    shard_examples,
    uniformly_subsample,
    write_distributed_manifest,
)
from omini.pipeline.flux_omini import Condition, generate, seed_everything
from omini.pipeline.rivg import (
    fuse_rivg_information_maps,
    prepare_rivg_reference,
    validate_information_weights,
)
from omini.train_flux.train_fmow_frame_prediction import (
    _frame_record,
    _group_id,
    build_fmow_miq_metadata,
    build_prompts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--glor-checkpoint", type=Path, default=None)
    parser.add_argument("--lora-preset", choices=("default", "expanded"))
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs/fmow_frame_prediction_glor_control"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("train/config/fmow_frame_prediction_glor_control.yaml"),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--target-slot",
        choices=("all", "0", "1", "2"),
        default="all",
        help="Target slot in each chronological triplet.",
    )
    parser.add_argument(
        "--test-list",
        type=Path,
        default=None,
        help="Legacy explicit sample list. Omit it to infer a dataset partition.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--uniform-sampling", action="store_true")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-random-glor", action="store_true")
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
    rivg.add_argument(
        "--rivg-info-weights", type=float, nargs=3, default=(0.5, 0.3, 0.2),
        metavar=("GRADIENT", "VARIANCE", "COHERENCE"),
        help="Non-negative information-map weights summing to 1.",
    )
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
    condition_group = parser.add_mutually_exclusive_group()
    condition_group.add_argument(
        "--keep-omini-condition",
        dest="keep_omini_condition",
        action="store_true",
        help="Use both OminiControl condition branches in addition to GLoR-Control.",
    )
    condition_group.add_argument(
        "--disable-omini-condition",
        dest="keep_omini_condition",
        action="store_false",
        help="Use GLoR-Control without the original OminiControl image branches.",
    )
    parser.set_defaults(keep_omini_condition=None)
    args = parser.parse_args()
    try:
        args.rivg_info_weights = validate_information_weights(args.rivg_info_weights)
    except ValueError as error:
        parser.error(str(error))
    return args


def pil_to_unit_tensor(image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(255.0)


def checkpoint_dir(checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    return checkpoint.parent if checkpoint.is_file() else checkpoint


def find_glor_checkpoint(lora_checkpoint: Path) -> Path | None:
    weight_dir = checkpoint_dir(lora_checkpoint)
    for name in ("glor_control.pt", "glor_control.pth", "glor_control.safetensors"):
        path = weight_dir / name
        if path.is_file():
            return path
    return None


def load_module_state(path: Path):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def load_glor_weights(path: Path, glor_control: GLoRControl):
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


def target_token_shape(pipe, target_height: int, target_width: int) -> tuple[int, int]:
    token_stride = int(pipe.vae_scale_factor) * 2
    if target_height % token_stride or target_width % token_stride:
        raise ValueError(
            f"Target size {target_width}x{target_height} is not divisible by "
            f"the packed FLUX token stride {token_stride}."
        )
    return target_height // token_stride, target_width // token_stride


def adapter_names(config: dict, keep_omini_condition: bool) -> tuple[str, str, str]:
    configured = config.get("model", {}).get(
        "adapter_names", ["text", "main", "condition", "condition"]
    )
    if not isinstance(configured, list) or len(configured) < 2:
        raise ValueError("model.adapter_names must contain at least text and main.")
    if keep_omini_condition and len(configured) < 3:
        raise ValueError(
            "Keeping Omini conditions requires a condition adapter in model.adapter_names."
        )
    condition = configured[2] if len(configured) >= 3 else "condition"
    return configured[0], configured[1], condition


def load_partition_examples(
    dataset_config: dict,
    partition: str,
    target_slot: str,
    metadata_enabled: bool,
) -> list[dict]:
    """Expand every valid triplet in one partition into inference examples."""

    root, rows = collect_examples(dataset_config, partition, target_slot)
    examples = []
    for row, current_target_slot in rows:
        frames = row["frames"]
        context_slots = [
            slot for slot in range(3) if slot != current_target_slot
        ]
        contexts = [frames[slot] for slot in context_slots]
        target = frames[current_target_slot]
        context_indices = [
            int(frame["timestamp_index"]) for frame in contexts
        ]
        target_index = int(target["timestamp_index"])
        clip_prompt, t5_prompt = build_prompts(
            row,
            current_target_slot,
            root=root,
        )
        if target_index < min(context_indices):
            selection_type = "backward"
        elif target_index > max(context_indices):
            selection_type = "forward"
        else:
            selection_type = "interpolation"
        example = {
            "id": f"{row.get('sequence_id', 'sequence')}_t{target_index}",
            "sequence_id": row.get("sequence_id", "unknown"),
            "location_id": _group_id(row),
            "selection_type": selection_type,
            "context_paths": [
                root / frame["image_path"] for frame in contexts
            ],
            "target_path": root / target["image_path"],
            "context_indices": context_indices,
            "target_index": target_index,
            "temporal_offsets": [
                index - target_index for index in context_indices
            ],
            "clip_prompt": clip_prompt,
            "t5_prompt": t5_prompt,
        }
        if metadata_enabled:
            example["condition_metadata"] = torch.stack(
                [
                    build_fmow_miq_metadata(_frame_record(frame, root))
                    for frame in contexts
                ]
            )
            example["target_metadata"] = build_fmow_miq_metadata(
                _frame_record(target, root)
            )
        examples.append(example)
    return examples


def main() -> None:
    args = parse_args()
    rank, world_size, args.device = initialize_distributed(args.device)
    if args.checkpoint is None and args.lora_preset is None:
        raise ValueError("Pass either --checkpoint or --lora-preset.")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative.")
    if args.save_rivg_debug and args.rivg_method is None:
        raise ValueError("--save-rivg-debug requires --rivg-method.")
    rivg_config = build_rivg_config(args)

    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    metadata_enabled = bool(
        config.get("metadata_qkv", {}).get("enabled", False)
    )
    if args.checkpoint is None:
        args.checkpoint = find_latest_checkpoint(
            args.runs_dir,
            args.lora_preset,
            require_metadata=metadata_enabled,
        )
        print(f"Automatically selected checkpoint: {args.checkpoint}")
    glor_config = config.get("glor_control", {})
    if not glor_config.get("enable", False):
        raise RuntimeError(f"{args.config} does not enable glor_control.")
    if glor_config.get("task") != "frame_prediction":
        raise RuntimeError("GLoR-Control must be configured for frame_prediction.")

    if args.keep_omini_condition is None:
        args.keep_omini_condition = not bool(
            glor_config.get("disable_omini_condition", True)
        )
    if metadata_enabled and not args.keep_omini_condition:
        raise ValueError(
            "MIQ-MMA condition-to-K requires the two Omini condition "
            "branches; do not pass --disable-omini-condition."
        )
    text_adapter, main_adapter, condition_adapter = adapter_names(
        config, args.keep_omini_condition
    )
    print(
        "Inference controls: temporal GLoR-Control + "
        f"Omini conditions={args.keep_omini_condition}"
    )

    dataset_config = config["train"]["dataset"]
    image_size = dataset_config.get("image_size", [512, 512])
    if args.test_list is not None:
        examples = load_test_list(args.test_list, dataset_config)
        source_description = f"explicit test list {args.test_list}"
    else:
        examples = load_partition_examples(
            dataset_config,
            args.split,
            args.target_slot,
            metadata_enabled,
        )
        source_description = (
            f"dataset partition={args.split}, target_slot={args.target_slot}"
        )
    if args.max_samples:
        examples = (
            uniformly_subsample(examples, args.max_samples)
            if args.uniform_sampling
            else examples[: args.max_samples]
        )
    if not examples:
        raise RuntimeError(f"No valid samples found in {source_description}.")
    total_examples = len(examples)
    local_examples = shard_examples(examples, rank, world_size)
    print(f"Inference source: {source_description}; total={total_examples}.")
    print(
        f"Rank {rank}: assigned {len(local_examples)} of "
        f"{total_examples} examples."
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    directories = {
        name: args.output_dir / name
        for name in ("generated", "target", "context_0", "context_1", "prompts")
    }
    if rivg_config is not None and args.save_rivg_debug:
        directories.update(
            {
                "rivg_weak_context_0": args.output_dir / "rivg_weak_context_0",
                "rivg_weak_context_1": args.output_dir / "rivg_weak_context_1",
                "rivg_information_map_0": args.output_dir / "rivg_information_map_0",
                "rivg_information_map_1": args.output_dir / "rivg_information_map_1",
                "rivg_information_map": args.output_dir / "rivg_information_map",
            }
        )
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    args.checkpoint = args.checkpoint.resolve()
    dtype_name = config.get("dtype", "bfloat16")
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    dtype = getattr(torch, dtype_name)
    glor_checkpoint = args.glor_checkpoint or find_glor_checkpoint(args.checkpoint)
    if glor_checkpoint is None and not args.allow_random_glor:
        raise FileNotFoundError(
            "No GLoR-Control checkpoint found beside the LoRA checkpoint. Pass "
            "--glor-checkpoint or --allow-random-glor for a wiring smoke test."
        )
    glor_checkpoint = (
        glor_checkpoint.resolve() if glor_checkpoint is not None else None
    )

    pipe = None
    glor_control = None
    token_shape = None
    if local_examples:
        pipe = load_pipeline(config, args.checkpoint, args.device)
        metadata_injector = pipe._omini_metadata_qkv_injector
        if metadata_enabled and metadata_injector is None:
            raise RuntimeError(
                "MIQ-MMA is enabled but the injector was not loaded."
            )
        glor_control = GLoRControl.from_config(
            pipe.transformer, {"glor_control": glor_config}
        ).to(device=args.device, dtype=dtype)
        pipe.transformer.glor_control = glor_control
    else:
        metadata_injector = None

    if glor_checkpoint is not None and local_examples:
        result = load_glor_weights(glor_checkpoint, glor_control)
        print(
            f"Loaded GLoR-Control from {glor_checkpoint} "
            f"(missing={len(result.missing_keys)}, "
            f"unexpected={len(result.unexpected_keys)})"
        )
    elif local_examples:
        print("Warning: using randomly initialized GLoR-Control modules.")
    if glor_control is not None:
        glor_control.eval()

    width, height = map(int, image_size)
    if pipe is not None:
        token_shape = target_token_shape(pipe, height, width)
    effective_image_guidance = (
        args.image_guidance_scale if args.keep_omini_condition else 1.0
    )
    if not args.keep_omini_condition and args.image_guidance_scale != 1.0:
        print(
            "Image guidance is disabled because original Omini conditions are disabled."
        )
    if rank == 0 and rivg_config is not None:
        guided_steps = sum(
            index / max(args.steps - 1, 1)
            >= rivg_config.guidance_start_fraction
            for index in range(args.steps)
        )
        print(
            "RIVG enabled for temporal GLoR and Omini references: "
            f"{json.dumps(rivg_config.to_dict(), sort_keys=True)}"
        )
        print(
            f"Information-map fusion: {args.rivg_info_fusion}; "
            f"estimated transformer NFE/sample: {args.steps + guided_steps}"
        )

    records = []
    for local_index, (global_index, example) in enumerate(local_examples):
        sample_id = example["id"]
        generated_path = directories["generated"] / f"{sample_id}.png"
        contexts = [
            open_resized_rgb(path, image_size) for path in example["context_paths"]
        ]
        weak_contexts = []
        context_information_maps = []
        fused_information_map = None
        if rivg_config is not None:
            for context in contexts:
                weak_context, information_map = prepare_rivg_reference(
                    context,
                    weak_kernel_size=args.weak_kernel_size,
                    weak_sigma=args.weak_sigma,
                    lower_quantile=args.info_lower_quantile,
                    upper_quantile=args.info_upper_quantile,
                    information_weights=args.rivg_info_weights,
                )
                weak_contexts.append(weak_context)
                context_information_maps.append(information_map)
            fused_information_map = fuse_rivg_information_maps(
                context_information_maps,
                method=args.rivg_info_fusion,
            )
        if not generated_path.exists() or args.overwrite:
            context_tensors = [
                pil_to_unit_tensor(image)
                .unsqueeze(0)
                .to(device=args.device, dtype=dtype)
                for image in contexts
            ]
            temporal_offsets = torch.tensor(
                [example["temporal_offsets"]], device=args.device, dtype=dtype
            )
            weak_context_tensors = [
                pil_to_unit_tensor(image)
                .unsqueeze(0)
                .to(device=args.device, dtype=dtype)
                for image in weak_contexts
            ]
            with torch.no_grad():
                glor_context = glor_control.build_context(
                    {
                        "context_images": context_tensors,
                        "temporal_offsets": temporal_offsets,
                    },
                    task="frame_prediction",
                    target_latent_shape=token_shape,
                )
                weak_glor_context = None
                if rivg_config is not None:
                    weak_glor_context = glor_control.build_context(
                        {
                            "context_images": weak_context_tensors,
                            "temporal_offsets": temporal_offsets,
                        },
                        task="frame_prediction",
                        target_latent_shape=token_shape,
                    )
                transformer_kwargs = {}
                if metadata_injector is not None:
                    metadata_dtype = next(
                        metadata_injector.parameters()
                    ).dtype
                    metadata_deltas = metadata_injector.prepare_deltas(
                        example["condition_metadata"]
                        .unsqueeze(0)
                        .to(device=args.device, dtype=metadata_dtype),
                        example["target_metadata"]
                        .unsqueeze(0)
                        .to(device=args.device, dtype=metadata_dtype),
                    )
                    transformer_kwargs = {
                        "metadata_qkv_injector": metadata_injector,
                        "metadata_qkv_deltas": metadata_deltas,
                    }

            conditions = []
            if args.keep_omini_condition:
                condition_posterior = (
                    "mode" if rivg_config is not None else "sample"
                )
                conditions = [
                    Condition(
                        image,
                        condition_adapter,
                        position_delta=(0.0, 0.0),
                        posterior=condition_posterior,
                    )
                    for image in contexts
                ]
            weak_conditions = []
            if rivg_config is not None and args.keep_omini_condition:
                weak_conditions = [
                    Condition(
                        image,
                        condition_adapter,
                        position_delta=(0.0, 0.0),
                        posterior="mode",
                    )
                    for image in weak_contexts
                ]
            sample_seed = args.seed + global_index
            seed_everything(sample_seed)
            generator = torch.Generator(device=args.device).manual_seed(sample_seed)
            result_image = generate(
                pipe,
                prompt=example["clip_prompt"],
                prompt_2=example["t5_prompt"],
                conditions=conditions,
                main_adapter=main_adapter,
                text_adapter=text_adapter,
                main_adapter_on_text=False,
                height=height,
                width=width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                image_guidance_scale=effective_image_guidance,
                generator=generator,
                glor_control_context=glor_context,
                transformer_kwargs=transformer_kwargs,
                rivg_config=rivg_config,
                rivg_weak_conditions=(
                    weak_conditions if rivg_config is not None else None
                ),
                rivg_information_map=fused_information_map,
                rivg_weak_glor_control_context=weak_glor_context,
            ).images[0]
            result_image.save(generated_path)

        saved_contexts = []
        for context_index, image in enumerate(contexts):
            path = directories[f"context_{context_index}"] / f"{sample_id}.png"
            if not path.exists() or args.overwrite:
                image.save(path)
            saved_contexts.append(str(path))
        if rivg_config is not None and args.save_rivg_debug:
            for context_index, weak_context in enumerate(weak_contexts):
                weak_path = directories[
                    f"rivg_weak_context_{context_index}"
                ] / f"{sample_id}.png"
                map_path = directories[
                    f"rivg_information_map_{context_index}"
                ] / f"{sample_id}.png"
                if not weak_path.exists() or args.overwrite:
                    weak_context.save(weak_path)
                if not map_path.exists() or args.overwrite:
                    information_map_image(
                        context_information_maps[context_index]
                    ).save(map_path)
            fused_map_path = (
                directories["rivg_information_map"] / f"{sample_id}.png"
            )
            if not fused_map_path.exists() or args.overwrite:
                information_map_image(fused_information_map).save(fused_map_path)
        target = open_resized_rgb(example["target_path"], image_size)
        target_path = directories["target"] / f"{sample_id}.png"
        if not target_path.exists() or args.overwrite:
            target.save(target_path)

        prompt_path = directories["prompts"] / f"{sample_id}.json"
        if not prompt_path.exists() or args.overwrite:
            prompt_path.write_text(
                json.dumps(
                    {
                        "clip_prompt": example["clip_prompt"],
                        "t5_prompt": example["t5_prompt"],
                        "context_indices": example["context_indices"],
                        "target_index": example["target_index"],
                        "temporal_offsets": example["temporal_offsets"],
                        "condition_metadata": (
                            example["condition_metadata"].tolist()
                            if metadata_enabled
                            else None
                        ),
                        "target_metadata": (
                            example["target_metadata"].tolist()
                            if metadata_enabled
                            else None
                        ),
                        "rivg": (
                            rivg_config.to_dict()
                            if rivg_config is not None
                            else None
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        records.append(
            {
                "sample_id": sample_id,
                "sequence_id": example["sequence_id"],
                "location_id": example.get("location_id"),
                "selection_type": example.get("selection_type"),
                "context_indices": example["context_indices"],
                "target_index": example["target_index"],
                "temporal_offsets": example["temporal_offsets"],
                "contexts": [str(path) for path in example["context_paths"]],
                "saved_contexts": saved_contexts,
                "target": str(example["target_path"]),
                "saved_target": str(target_path),
                "generated": str(generated_path),
                "seed": args.seed + global_index,
                "sample_index": global_index,
                "rank": rank,
                "checkpoint": str(args.checkpoint),
                "glor_checkpoint": (
                    str(glor_checkpoint) if glor_checkpoint is not None else None
                ),
                "keep_omini_condition": args.keep_omini_condition,
                "miq_mma": metadata_enabled,
                "training_free_velocity_guidance": (
                    rivg_config.to_dict() if rivg_config is not None else None
                ),
                "rivg_information_map_fusion": (
                    args.rivg_info_fusion if rivg_config is not None else None
                ),
                "rivg_information_weights": (
                    args.rivg_info_weights if rivg_config is not None else None
                ),
            }
        )
        print(
            f"[rank {rank}: {local_index + 1}/{len(local_examples)}; "
            f"global {global_index + 1}/{total_examples}] {sample_id}: offsets "
            f"{example['temporal_offsets']} -> {generated_path}"
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
            f"Saved {result_count} GLoR-Control result(s) to "
            f"{args.output_dir}; manifest: {manifest_path}"
        )
    close_distributed()


if __name__ == "__main__":
    main()
