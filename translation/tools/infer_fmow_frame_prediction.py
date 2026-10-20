#!/usr/bin/env python3
"""Infer an fMoW frame from any two temporal context frames."""

import argparse
import json
import os
from datetime import timedelta
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
import yaml
from diffusers import (
    FlowMatchHeunDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from PIL import Image

from metadata_qkv import configure_metadata_qkv
from omini.pipeline.flux_omini import Condition, generate, seed_everything
from omini.pipeline.rivg import (
    RIVGConfig,
    fuse_rivg_information_maps,
    prepare_rivg_reference,
)
from omini.train_flux.train_fmow_frame_prediction import (
    FMOW_MIQ_METADATA_DIM,
    _frame_record,
    _group_id,
    _partition_for,
    _read_jsonl,
    _validate_frames,
    build_fmow_miq_metadata,
    build_prompts,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the missing fMoW frame from two context frames."
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--lora-preset", choices=("default", "expanded"))
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs/fmow_frame_prediction")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("train/config/fmow_frame_prediction.yaml"),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--target-slot",
        choices=("all", "0", "1", "2"),
        default="all",
        help="Target slot in each chronological triplet.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--uniform-sampling", action="store_true")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--scheduler",
        choices=("default", "flowmatch_heun"),
        default="default",
        help=(
            "Sampling scheduler. 'default' preserves the FLUX FlowMatch Euler "
            "scheduler; 'flowmatch_heun' uses second-order predictor/corrector "
            "updates."
        ),
    )

    custom = parser.add_argument_group("custom two-frame inference")
    custom.add_argument("--frame-a", type=Path)
    custom.add_argument("--frame-b", type=Path)
    custom.add_argument("--frame-a-index", type=int, default=0)
    custom.add_argument("--frame-b-index", type=int, default=1)
    custom.add_argument("--target-index", type=int, default=2)
    custom.add_argument("--scene", default="remote sensing scene")
    custom.add_argument("--frame-a-datetime")
    custom.add_argument("--frame-b-datetime")
    custom.add_argument("--target-datetime")
    custom.add_argument("--center-lat", type=float)
    custom.add_argument("--center-lon", type=float)

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


def find_latest_checkpoint(runs_dir, preset, require_metadata=False):
    candidates = []
    for config_path in runs_dir.glob("*/config.yaml"):
        config_text = config_path.read_text(encoding="utf-8")
        config_preset = "expanded" if "time_text_embed" in config_text else "default"
        if config_preset != preset:
            continue
        for checkpoint in (config_path.parent / "ckpt").glob("*"):
            if not checkpoint.is_dir() or not checkpoint.name.isdigit():
                continue
            if all(
                (checkpoint / f"{name}.safetensors").is_file()
                for name in ("text", "main", "condition")
            ) and (
                not require_metadata
                or (checkpoint / "metadata_qkv.pt").is_file()
            ):
                candidates.append(
                    (config_path.parent.name, int(checkpoint.name), checkpoint)
                )
    if not candidates:
        raise FileNotFoundError(
            f"No {preset!r} fMoW checkpoints found below {runs_dir}."
        )
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def open_resized_rgb(path, image_size):
    width, height = map(int, image_size)
    previous_max_image_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as image:
            image.draft("RGB", (width, height))
            image = image.convert("RGB")
            if image.size != (width, height):
                image = image.resize(
                    (width, height),
                    Image.Resampling.LANCZOS,
                )
            return image.copy()
    finally:
        Image.MAX_IMAGE_PIXELS = previous_max_image_pixels


def uniformly_subsample(samples, count):
    if count <= 0 or count >= len(samples):
        return samples
    if count == 1:
        return [samples[len(samples) // 2]]
    last = len(samples) - 1
    return [samples[(index * last) // (count - 1)] for index in range(count)]


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


def shard_examples(examples, rank, world_size):
    indexed = list(enumerate(examples))
    return indexed[rank::world_size]


def write_distributed_manifest(output_dir, records, rank, world_size):
    output_dir = Path(output_dir)
    if world_size == 1:
        manifest_path = output_dir / "manifest.json"
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
    manifest_path = output_dir / "manifest.json"
    if rank == 0:
        merged = []
        for part_rank in range(world_size):
            path = output_dir / f"manifest.rank{part_rank:05d}.json"
            with path.open("r", encoding="utf-8") as stream:
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


def collect_examples(dataset_config, partition, target_slot):
    root = Path(dataset_config.get("root") or Path(dataset_config["index_path"]).parent)
    source_splits = set(dataset_config.get("source_splits", ["train"]))
    split_ratios = dataset_config.get("split_ratios") or {
        "train": 0.98,
        "val": 0.01,
        "test": 0.01,
    }
    split_seed = dataset_config.get("split_seed", 42)
    target_slots = range(3) if target_slot == "all" else [int(target_slot)]
    examples = []
    for row in _read_jsonl(dataset_config["index_path"]):
        if row.get("split", "train") not in source_splits:
            continue
        if _partition_for(_group_id(row), split_ratios, split_seed) != partition:
            continue
        frames = _validate_frames(row)
        if frames is None:
            continue
        if not all((root / frame["image_path"]).is_file() for frame in frames):
            continue
        row = dict(row)
        row["frames"] = frames
        for slot in target_slots:
            examples.append((row, slot))
    return root, examples


def load_test_list(test_list, dataset_config):
    test_list = Path(test_list)
    with test_list.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{test_list} does not contain a samples list.")
    root = Path(
        dataset_config.get("root")
        or manifest.get("dataset_root")
        or Path(dataset_config["index_path"]).parent
    )
    examples = []
    for sample in samples:
        contexts = sample.get("input_frames")
        target = sample.get("output_frame")
        if not isinstance(contexts, list) or len(contexts) != 2:
            raise ValueError(
                f"{sample.get('sample_id')} must contain two input_frames."
            )
        if not isinstance(target, dict):
            raise ValueError(
                f"{sample.get('sample_id')} has no output_frame."
            )
        frames = sorted(
            [*contexts, target],
            key=lambda frame: int(frame["timestamp_index"]),
        )
        target_path_text = target["image_path"]
        target_slot = next(
            index
            for index, frame in enumerate(frames)
            if frame["image_path"] == target_path_text
        )
        row = {
            "sequence_id": sample.get("sequence_id", "unknown"),
            "frames": frames,
        }
        clip_prompt, t5_prompt = build_prompts(
            row,
            target_slot,
            root=root,
        )
        context_indices = [
            int(frame["timestamp_index"]) for frame in contexts
        ]
        target_index = int(target["timestamp_index"])
        paths = [root / frame["image_path"] for frame in contexts]
        target_path = root / target_path_text
        missing = [
            str(path) for path in [*paths, target_path] if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{sample.get('sample_id')} has missing images: {missing}"
            )
        examples.append(
            {
                "id": sample["sample_id"],
                "sequence_id": sample.get("sequence_id", "unknown"),
                "location_id": sample.get("location_id"),
                "selection_type": sample.get("selection_type"),
                "context_paths": paths,
                "target_path": target_path,
                "context_indices": context_indices,
                "target_index": target_index,
                "temporal_offsets": [
                    index - target_index for index in context_indices
                ],
                "clip_prompt": clip_prompt,
                "t5_prompt": t5_prompt,
                "condition_metadata": torch.stack(
                    [
                        build_fmow_miq_metadata(
                            _frame_record(frame, root)
                        )
                        for frame in contexts
                    ]
                ),
                "target_metadata": build_fmow_miq_metadata(
                    _frame_record(target, root)
                ),
            }
        )
    return examples


def configure_inference_scheduler(pipe, scheduler_name):
    if scheduler_name == "default":
        return pipe.scheduler
    if scheduler_name == "flowmatch_heun":
        source_config = pipe.scheduler.config
        pipe.scheduler = FlowMatchHeunDiscreteScheduler(
            num_train_timesteps=int(source_config.num_train_timesteps),
            shift=float(source_config.shift),
        )
        return pipe.scheduler
    raise ValueError(f"Unsupported inference scheduler: {scheduler_name}")


def load_pipeline(config, checkpoint, device, scheduler_name="default"):
    dtype_name = config.get("dtype", "bfloat16")
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    dtype = getattr(torch, dtype_name)
    kwargs = {"torch_dtype": dtype}
    if config.get("transformer_path"):
        kwargs["transformer"] = FluxTransformer2DModel.from_pretrained(
            config["transformer_path"], torch_dtype=dtype
        )
    pipe = FluxPipeline.from_pretrained(config["flux_path"], **kwargs).to(device)
    scheduler = configure_inference_scheduler(pipe, scheduler_name)
    print(
        f"Inference scheduler: {scheduler.__class__.__name__} "
        f"(order={scheduler.order})"
    )
    if not checkpoint.is_dir():
        raise ValueError(
            "This model uses text/main/condition adapters; --checkpoint must be a directory."
        )
    adapter_names = []
    for name in config.get("model", {}).get(
        "adapter_names", ["text", "main", "condition", "condition"]
    ):
        if name is not None and name not in adapter_names:
            adapter_names.append(name)
    for name in adapter_names:
        weight = checkpoint / f"{name}.safetensors"
        if not weight.is_file():
            raise FileNotFoundError(weight)
        pipe.load_lora_weights(
            str(checkpoint), weight_name=weight.name, adapter_name=name
        )
    pipe.set_adapters(adapter_names)
    metadata_config = config.get("metadata_qkv", {})
    metadata_injector = None
    if metadata_config.get("enabled", False):
        for key in ("condition_metadata_dim", "target_metadata_dim"):
            if int(metadata_config.get(key, -1)) != FMOW_MIQ_METADATA_DIM:
                raise ValueError(
                    f"metadata_qkv.{key} must be {FMOW_MIQ_METADATA_DIM} "
                    "for fMoW inference."
                )
        metadata_path = checkpoint / "metadata_qkv.pt"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"MIQ-MMA config is enabled but weights are missing: "
                f"{metadata_path}"
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


def _metadata_record(acquisition_datetime, center_lat, center_lon):
    return {
        "metadata": {
            "acquisition_datetime": acquisition_datetime,
            "center_lat": center_lat,
            "center_lon": center_lon,
        }
    }


def custom_example(args):
    if (args.frame_a is None) != (args.frame_b is None):
        raise ValueError("Pass both --frame-a and --frame-b, or neither.")
    if args.frame_a is None:
        return None
    indices = [args.frame_a_index, args.frame_b_index, args.target_index]
    if len(set(indices)) != 3:
        raise ValueError("Custom frame and target indices must be distinct.")
    direction = (
        "backward temporal prediction"
        if args.target_index < min(args.frame_a_index, args.frame_b_index)
        else (
            "forward temporal prediction"
            if args.target_index > max(args.frame_a_index, args.frame_b_index)
            else "temporal interpolation"
        )
    )
    clip_prompt = f"Optical remote sensing temporal prediction of {args.scene}."
    t5_prompt = (
        "Task: optical remote sensing frame prediction. Given context frames at "
        f"temporal indices {args.frame_a_index} and {args.frame_b_index}, generate "
        f"the missing frame at temporal index {args.target_index}. Prediction mode: "
        f"{direction}. Scene: {args.scene}. Preserve the geographic layout while "
        "predicting time-dependent appearance changes."
    )
    ordered = sorted(
        [
            (args.frame_a_index, args.frame_a, args.frame_a_datetime),
            (args.frame_b_index, args.frame_b, args.frame_b_datetime),
        ]
    )
    return {
        "id": "custom",
        "context_paths": [ordered[0][1], ordered[1][1]],
        "target_path": None,
        "clip_prompt": clip_prompt,
        "t5_prompt": t5_prompt,
        "target_index": args.target_index,
        "sequence_id": "custom",
        "condition_metadata": torch.stack(
            [
                build_fmow_miq_metadata(
                    _metadata_record(
                        item[2],
                        args.center_lat,
                        args.center_lon,
                    )
                )
                for item in ordered
            ]
        ),
        "target_metadata": build_fmow_miq_metadata(
            _metadata_record(
                args.target_datetime,
                args.center_lat,
                args.center_lon,
            )
        ),
    }


def main():
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
    dataset_config = config["train"]["dataset"]
    image_size = dataset_config.get("image_size", [512, 512])

    if args.checkpoint is None:
        args.checkpoint = find_latest_checkpoint(
            args.runs_dir,
            args.lora_preset,
            require_metadata=metadata_enabled,
        )
        print(f"Automatically selected checkpoint: {args.checkpoint}")
    custom = custom_example(args)
    if custom is not None:
        examples = [custom]
    else:
        root, rows = collect_examples(dataset_config, args.split, args.target_slot)
        examples = []
        for row, target_slot in rows:
            frames = row["frames"]
            context_slots = [slot for slot in range(3) if slot != target_slot]
            target_index = int(frames[target_slot]["timestamp_index"])
            clip_prompt, t5_prompt = build_prompts(row, target_slot, root=root)
            example = {
                "id": f"{row.get('sequence_id', 'sequence')}_t{target_index}",
                "context_paths": [
                    root / frames[slot]["image_path"] for slot in context_slots
                ],
                "target_path": root / frames[target_slot]["image_path"],
                "clip_prompt": clip_prompt,
                "t5_prompt": t5_prompt,
                "target_index": target_index,
                "sequence_id": row.get("sequence_id", "unknown"),
            }
            if metadata_enabled:
                example["condition_metadata"] = torch.stack(
                    [
                        build_fmow_miq_metadata(
                            _frame_record(frames[slot], root)
                        )
                        for slot in context_slots
                    ]
                )
                example["target_metadata"] = build_fmow_miq_metadata(
                    _frame_record(frames[target_slot], root)
                )
            examples.append(example)
        if args.max_samples:
            examples = (
                uniformly_subsample(examples, args.max_samples)
                if args.uniform_sampling
                else examples[: args.max_samples]
            )
    if not examples:
        raise RuntimeError("No valid inference examples were found.")
    total_examples = len(examples)
    local_examples = shard_examples(examples, rank, world_size)
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

    pipe = None
    metadata_injector = None
    if local_examples:
        pipe = load_pipeline(
            config,
            args.checkpoint.resolve(),
            args.device,
            scheduler_name=args.scheduler,
        )
        metadata_injector = pipe._omini_metadata_qkv_injector
        if metadata_enabled and metadata_injector is None:
            raise RuntimeError(
                "MIQ-MMA is enabled but the injector was not loaded."
            )
    width, height = map(int, image_size)
    if rank == 0 and rivg_config is not None:
        guided_steps = sum(
            index / max(args.steps - 1, 1)
            >= rivg_config.guidance_start_fraction
            for index in range(args.steps)
        )
        image_cfg_multiplier = 2 if args.image_guidance_scale != 1.0 else 1
        transformer_nfe = (args.steps + guided_steps) * image_cfg_multiplier
        print(
            "RIVG enabled for both temporal contexts: "
            f"{json.dumps(rivg_config.to_dict(), sort_keys=True)}"
        )
        print(
            f"Information-map fusion: {args.rivg_info_fusion}; "
            f"estimated transformer NFE/sample: {transformer_nfe}"
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
                )
                weak_contexts.append(weak_context)
                context_information_maps.append(information_map)
            fused_information_map = fuse_rivg_information_maps(
                context_information_maps,
                method=args.rivg_info_fusion,
            )
        if not generated_path.exists() or args.overwrite:
            seed = args.seed + global_index
            seed_everything(seed)
            generator = torch.Generator(device=args.device).manual_seed(seed)
            transformer_kwargs = {}
            if metadata_injector is not None:
                metadata_dtype = next(metadata_injector.parameters()).dtype
                with torch.no_grad():
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
            condition_posterior = "mode" if rivg_config is not None else "sample"
            full_conditions = [
                Condition(
                    context,
                    "condition",
                    position_delta=(0.0, 0.0),
                    posterior=condition_posterior,
                )
                for context in contexts
            ]
            weak_conditions = [
                Condition(
                    context,
                    "condition",
                    position_delta=(0.0, 0.0),
                    posterior="mode",
                )
                for context in weak_contexts
            ]
            result = generate(
                pipe,
                prompt=example["clip_prompt"],
                prompt_2=example["t5_prompt"],
                conditions=full_conditions,
                main_adapter="main",
                text_adapter="text",
                height=height,
                width=width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                image_guidance_scale=args.image_guidance_scale,
                generator=generator,
                transformer_kwargs=transformer_kwargs,
                rivg_config=rivg_config,
                rivg_weak_conditions=(
                    weak_conditions if rivg_config is not None else None
                ),
                rivg_information_map=fused_information_map,
            ).images[0]
            result.save(generated_path)
        for context_index, image in enumerate(contexts):
            path = directories[f"context_{context_index}"] / f"{sample_id}.png"
            if not path.exists() or args.overwrite:
                image.save(path)
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
        saved_target = None
        if example["target_path"] is not None:
            target = open_resized_rgb(example["target_path"], image_size)
            target_path = directories["target"] / f"{sample_id}.png"
            if not target_path.exists() or args.overwrite:
                target.save(target_path)
            saved_target = str(target_path)
        prompt_path = directories["prompts"] / f"{sample_id}.json"
        if not prompt_path.exists() or args.overwrite:
            prompt_path.write_text(
                json.dumps(
                    {
                        "clip_prompt": example["clip_prompt"],
                        "t5_prompt": example["t5_prompt"],
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
                        "scheduler": args.scheduler,
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
                "target_index": example["target_index"],
                "contexts": [str(path) for path in example["context_paths"]],
                "target": str(example["target_path"]) if example["target_path"] else None,
                "generated": str(generated_path),
                "saved_target": saved_target,
                "seed": args.seed + global_index,
                "sample_index": global_index,
                "rank": rank,
                "miq_mma": metadata_enabled,
                "scheduler": args.scheduler,
                "training_free_velocity_guidance": (
                    rivg_config.to_dict() if rivg_config is not None else None
                ),
                "rivg_information_map_fusion": (
                    args.rivg_info_fusion if rivg_config is not None else None
                ),
            }
        )
        print(
            f"[rank {rank}: {local_index + 1}/{len(local_examples)}; "
            f"global {global_index + 1}/{total_examples}] "
            f"{sample_id} -> {generated_path}"
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
            f"Saved {result_count} result(s) to {args.output_dir}; "
            f"manifest: {manifest_path}"
        )
    close_distributed()


if __name__ == "__main__":
    main()
