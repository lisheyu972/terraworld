import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

EXPANDED_LORA_TARGET_MODULES = (
    r"(.*x_embedder"
    r"|.*time_text_embed\.(timestep_embedder|guidance_embedder|text_embedder)"
    r"\.linear_[12]"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.norm1\.linear"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_k"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_q"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_v"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_out\.0"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.ff\.net\.0\.proj"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.ff\.net\.2"
    r"|.*single_transformer_blocks\.[0-9]+\.norm\.linear"
    r"|.*single_transformer_blocks\.[0-9]+\.proj_mlp"
    r"|.*single_transformer_blocks\.[0-9]+\.proj_out"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.to_k"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.to_q"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.to_v"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.to_out)"
)


def _read_rgb_png(path):
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.div_(255.0).clamp_(0.0, 1.0)


def _frame_cloud(frame):
    value = frame.get("cloud_cover_percent")
    return "unknown" if value is None else f"{float(value):.2f}"


def _normalize_sample(sample):
    if "input_frames" in sample and "target_frame" in sample:
        return sample
    if not all(key in sample for key in ("frame0", "target", "frame1")):
        raise KeyError(
            "sample must contain either input_frames/target_frame or "
            "frame0/target/frame1."
        )

    time_indices = sample.get("time_indices") or [None, None, None]
    if len(time_indices) != 3:
        time_indices = [None, None, None]
    sequence = sample.get("sequence", "unknown")
    sequence_id = Path(sequence).name if sequence else "unknown"
    return {
        "id": sample.get(
            "id",
            f"{sequence_id}_{Path(sample['target']).stem}_S{sample.get('stride', 1)}",
        ),
        "sequence_id": sample.get("sequence_id", sequence_id),
        "sequence": sequence,
        "task": sample.get("task", "frame_interpolation"),
        "stride": sample.get("stride", 1),
        "interpolation_time": sample.get("interpolation_time", 0.5),
        "input_frames": {
            "previous": {
                "timestamp_index": time_indices[0],
                "image_path": sample["frame0"],
                "cloud_cover_percent": sample.get("frame0_cloud_cover_percent"),
            },
            "next": {
                "timestamp_index": time_indices[2],
                "image_path": sample["frame1"],
                "cloud_cover_percent": sample.get("frame1_cloud_cover_percent"),
            },
        },
        "target_frame": {
            "timestamp_index": time_indices[1],
            "image_path": sample["target"],
            "cloud_cover_percent": sample.get("target_cloud_cover_percent"),
        },
    }


def _build_prompts(sample, metadata):
    sample = _normalize_sample(sample)
    previous = sample["input_frames"]["previous"]
    next_frame = sample["input_frames"]["next"]
    target = sample["target_frame"]

    sensor = metadata.get("sensor", "Sentinel-2")
    clip_prompt = f"Temporal middle frame of a {sensor} RGB remote sensing sequence."
    image_size = metadata.get("image_size", [128, 128])
    t5_prompt = (
        "Task: optical remote sensing frame interpolation. "
        f"Generate the temporal middle {sensor} RGB image between the "
        "previous and next context frames. The target frame lies halfway "
        "between the two input frames. "
        f"Sensor: {sensor}; "
        f"modality: {metadata.get('modality', 'optical')}; "
        f"ground sample distance: {metadata.get('gsd_m', 'unknown')} meters; "
        f"image size: {image_size[0]} by {image_size[1]} pixels; "
        f"sequence id: {sample.get('sequence_id', 'unknown')}; "
        f"stride: {sample.get('stride', 1)}; "
        f"interpolation time: {sample.get('interpolation_time', 0.5)}. "
        f"Previous timestamp index: {previous.get('timestamp_index', 'unknown')}; "
        f"target timestamp index: {target.get('timestamp_index', 'unknown')}; "
        f"next timestamp index: {next_frame.get('timestamp_index', 'unknown')}. "
        f"Previous cloud cover: {_frame_cloud(previous)} percent; "
        f"target cloud cover: {_frame_cloud(target)} percent; "
        f"next cloud cover: {_frame_cloud(next_frame)} percent."
    )
    return clip_prompt, t5_prompt


class EarthNetFrameInterpolationDataset(Dataset):
    """EarthNet previous/next frame to temporal middle frame dataset."""

    def __init__(
        self,
        index_path,
        include_splits=("train",),
        root=None,
        drop_text_prob=0.1,
        drop_image_prob=0.05,
        random_flip=True,
        metadata=None,
        max_samples=None,
        validate_files=False,
    ):
        self.index_path = Path(index_path)
        self.root = Path(root) if root is not None else self.index_path.parent
        self.drop_text_prob = float(drop_text_prob)
        self.drop_image_prob = float(drop_image_prob)
        self.random_flip = bool(random_flip)
        self.metadata = metadata or {}
        self.validate_files = bool(validate_files)

        with open(self.index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        samples = []
        split_root = index.get("splits", index)
        for split in include_splits:
            split_samples = split_root.get(split, [])
            if not isinstance(split_samples, list):
                raise TypeError(f"Index split '{split}' must be a list.")
            samples.extend(split_samples)

        valid_samples = []
        for sample in samples:
            try:
                sample = _normalize_sample(sample)
                previous = sample["input_frames"]["previous"]
                next_frame = sample["input_frames"]["next"]
                target = sample["target_frame"]
            except KeyError:
                continue
            if self.validate_files:
                paths = [
                    self.root / previous["image_path"],
                    self.root / next_frame["image_path"],
                    self.root / target["image_path"],
                ]
                if not all(path.is_file() for path in paths):
                    continue
            valid_samples.append(sample)

        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples < 1:
                raise ValueError("max_samples must be positive when set.")
            valid_samples = valid_samples[:max_samples]

        if not valid_samples:
            raise ValueError(
                f"No valid frame interpolation samples found in {self.index_path}."
            )
        self.samples = valid_samples
        print(
            "EarthNet frame interpolation samples: "
            f"{len(self.samples)} from splits {list(include_splits)}."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        previous_frame = sample["input_frames"]["previous"]
        next_frame = sample["input_frames"]["next"]
        target_frame = sample["target_frame"]

        previous = _read_rgb_png(self.root / previous_frame["image_path"])
        next_image = _read_rgb_png(self.root / next_frame["image_path"])
        target = _read_rgb_png(self.root / target_frame["image_path"])

        if self.random_flip:
            if random.random() < 0.5:
                previous = previous.flip(-1)
                next_image = next_image.flip(-1)
                target = target.flip(-1)
            if random.random() < 0.5:
                previous = previous.flip(-2)
                next_image = next_image.flip(-2)
                target = target.flip(-2)

        clip_prompt, t5_prompt = _build_prompts(sample, self.metadata)
        if random.random() < self.drop_text_prob:
            clip_prompt = ""
            t5_prompt = ""
        if random.random() < self.drop_image_prob:
            previous = torch.zeros_like(previous)
            next_image = torch.zeros_like(next_image)

        return {
            "image": target,
            "condition_0": previous,
            "condition_type_0": "previous_frame",
            "position_delta_0": np.array([0.0, 0.0], dtype=np.float32),
            "position_scale_0": 1.0,
            "condition_1": next_image,
            "condition_type_1": "next_frame",
            "position_delta_1": np.array([0.0, 0.0], dtype=np.float32),
            "position_scale_1": 1.0,
            "description": clip_prompt,
            "description_2": t5_prompt,
        }


def _apply_integer_overrides(training_config):
    integer_overrides = {
        "OMINI_BATCH_SIZE": "batch_size",
        "OMINI_ACCUMULATE_GRAD_BATCHES": "accumulate_grad_batches",
        "OMINI_MAX_STEPS": "max_steps",
        "OMINI_SAVE_INTERVAL": "save_interval",
        "OMINI_PRINT_EVERY_N_STEPS": "print_every_n_steps",
    }
    for environment_name, config_name in integer_overrides.items():
        value = os.environ.get(environment_name)
        if value is None:
            continue
        value = int(value)
        if value < 1:
            raise ValueError(f"{environment_name} must be a positive integer.")
        training_config[config_name] = value
        print(f"Training override: {config_name}={value}")


def main():
    from .trainer import OminiModel, get_config, train

    config = get_config()
    training_config = config["train"]
    dataset_config = training_config["dataset"]
    _apply_integer_overrides(training_config)
    lora_path = os.environ.get("OMINI_LORA_PATH") or training_config.get("lora_path")
    if lora_path:
        print(f"Training override: lora_path={lora_path}")

    lora_target_preset = os.environ.get("OMINI_LORA_TARGET_PRESET", "default")
    if lora_target_preset == "expanded":
        training_config["lora_config"][
            "target_modules"
        ] = EXPANDED_LORA_TARGET_MODULES
        print("LoRA target preset: expanded")
    elif lora_target_preset != "default":
        raise ValueError(
            "OMINI_LORA_TARGET_PRESET must be either 'default' or 'expanded'."
        )

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dataset = EarthNetFrameInterpolationDataset(
        index_path=dataset_config["index_path"],
        root=dataset_config.get("root"),
        include_splits=dataset_config.get("include_splits", ["train"]),
        drop_text_prob=dataset_config.get("drop_text_prob", 0.1),
        drop_image_prob=dataset_config.get("drop_image_prob", 0.05),
        random_flip=dataset_config.get("random_flip", True),
        metadata=dataset_config.get("metadata", {}),
        max_samples=dataset_config.get("max_samples"),
        validate_files=dataset_config.get("validate_files", False),
    )

    if dataset_config.get("separate_condition_adapters", False):
        adapter_names = [None, None, "previous", "next"]
    else:
        adapter_names = [None, None, "default", "default"]

    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        transformer_path=config.get("transformer_path"),
        lora_path=lora_path,
        lora_config=training_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        adapter_names=adapter_names,
    )
    train(dataset, trainable_model, config, test_function=None)


if __name__ == "__main__":
    main()
