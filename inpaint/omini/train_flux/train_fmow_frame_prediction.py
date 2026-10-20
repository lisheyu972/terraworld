"""Train fMoW any-two-frames-to-the-third-frame prediction."""

import hashlib
import json
import os
import random
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .train_frame_interp import EXPANDED_LORA_TARGET_MODULES


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def _group_id(row):
    """Return the original fMoW location id shared by related triplets."""
    frames = row.get("frames") or []
    if frames and frames[0].get("image_path"):
        parts = Path(frames[0]["image_path"]).stem.rsplit("_", 2)
        if len(parts) == 3:
            return parts[0]
    return str(row.get("sequence_id", ""))


def _partition_for(sequence_id, split_ratios, split_seed):
    names = ("train", "val", "test")
    ratios = [float(split_ratios.get(name, 0.0)) for name in names]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("split_ratios cannot contain negative values.")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("At least one split ratio must be positive.")
    value = int.from_bytes(
        hashlib.sha256(f"{split_seed}:{sequence_id}".encode()).digest()[:8],
        "big",
    ) / float(2**64)
    cumulative = 0.0
    for name, ratio in zip(names, ratios):
        cumulative += ratio / total
        if value < cumulative:
            return name
    return names[-1]


def _validate_frames(row):
    frames = row.get("frames")
    if not isinstance(frames, list) or len(frames) != 3:
        return None
    frames = sorted(frames, key=lambda frame: int(frame["timestamp_index"]))
    if len({int(frame["timestamp_index"]) for frame in frames}) != 3:
        return None
    if not all(frame.get("image_path") for frame in frames):
        return None
    return frames


def _read_resized_rgb(path, image_size):
    width, height = map(int, image_size)
    with Image.open(path) as image:
        # JPEG draft decoding avoids materializing the full 8K fMoW image before
        # reducing it to the model resolution.
        image.draft("RGB", (width, height))
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(255.0)


def _scene_from_path(image_path):
    stem = Path(image_path).stem
    parts = stem.rsplit("_", 3)
    return parts[0].replace("_", " ") if len(parts) == 4 else "unknown scene"


def _direction(target_index, context_indices):
    if target_index < min(context_indices):
        return "backward temporal prediction"
    if target_index > max(context_indices):
        return "forward temporal prediction"
    return "temporal interpolation"


@lru_cache(maxsize=16384)
def _read_frame_record(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _frame_record(frame, root):
    json_path = frame.get("json_path")
    if root is None or not json_path:
        return {}
    path = Path(root) / json_path
    return _read_frame_record(str(path)) if path.is_file() else {}


def _format_number(value, digits=6):
    if value is None:
        return "unknown"
    return f"{float(value):.{digits}f}"


def _format_list(value):
    return ", ".join(map(str, value)) if value else "unknown"


def _days_from_target(context_time, target_time):
    if not context_time or not target_time:
        return "unknown"
    try:
        context = datetime.fromisoformat(str(context_time).replace("Z", "+00:00"))
        target = datetime.fromisoformat(str(target_time).replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    return f"{(context - target).total_seconds() / 86400.0:+.1f}"


def _describe_context(frame, record, target_time, label):
    metadata = record.get("metadata") or {}
    acquisition_time = metadata.get("acquisition_datetime")
    acquisition_text = acquisition_time or "unknown"
    timestamp_index = frame.get("timestamp_index", "unknown")
    sensor = metadata.get("sensor", "unknown")
    gsd = _format_number(metadata.get("gsd_m"), 3)
    latitude = _format_number(metadata.get("center_lat"))
    longitude = _format_number(metadata.get("center_lon"))
    cloud = _format_number(metadata.get("cloud_cover_percent"), 2)
    offset = _days_from_target(acquisition_time, target_time)
    return (
        f"{label}: temporal index {timestamp_index}; acquisition time "
        f"{acquisition_text}; offset from target {offset} days; "
        f"sensor {sensor}; GSD {gsd} meters; center latitude {latitude}; "
        f"center longitude {longitude}; cloud cover {cloud} percent"
    )


def build_prompts(row, target_slot, root=None):
    frames = _validate_frames(row)
    if frames is None:
        raise ValueError("A sample must contain exactly three valid frames.")
    target = frames[target_slot]
    contexts = [frame for index, frame in enumerate(frames) if index != target_slot]
    target_record = _frame_record(target, root)
    context_records = [_frame_record(frame, root) for frame in contexts]
    target_metadata = target_record.get("metadata") or {}
    target_description = target_record.get("description") or {}
    target_index = int(target["timestamp_index"])
    context_indices = [int(frame["timestamp_index"]) for frame in contexts]
    scene = target_description.get("scene") or _scene_from_path(target["image_path"])
    objects = target_description.get("objects") or [scene]
    content = target_description.get("content_sentence") or "unknown"
    target_time = target_metadata.get("acquisition_datetime")
    clip_target_time = target_time or "the requested target time"
    target_time_text = target_time or "unknown"
    modality = target_metadata.get("modality", "optical")
    direction = _direction(target_index, context_indices)
    sequence_id = row.get("sequence_id", "unknown")
    latitude = _format_number(target_metadata.get("center_lat"))
    longitude = _format_number(target_metadata.get("center_lon"))
    sensor = target_metadata.get("sensor", "unknown")
    gsd = _format_number(target_metadata.get("gsd_m"), 3)
    bands = _format_list(target_metadata.get("bands"))
    patch_size = _format_list(target_metadata.get("patch_size_px"))
    object_text = _format_list(objects)
    clip_prompt = (
        f"{modality.capitalize()} remote sensing image of {scene} at "
        f"{clip_target_time}, latitude {latitude}, "
        f"longitude {longitude}."
    )
    context_0 = _describe_context(
        contexts[0], context_records[0], target_time, "Context frame 0"
    )
    context_1 = _describe_context(
        contexts[1], context_records[1], target_time, "Context frame 1"
    )
    t5_prompt = (
        "Task: optical remote sensing frame prediction. "
        f"Given context frames at temporal indices {context_indices[0]} and "
        f"{context_indices[1]}, generate the missing frame at temporal index "
        f"{target_index}. Prediction mode: {direction}. Sequence id: {sequence_id}; "
        f"scene class: {scene}; objects: {object_text}; scene description: {content}; "
        f"modality: {modality}; bands: {bands}. Target specification: acquisition "
        f"time {target_time_text}; center latitude {latitude}; center "
        f"longitude {longitude}; sensor {sensor}; GSD {gsd} meters; original patch "
        f"size {patch_size} pixels. {context_0}. {context_1}. Preserve the geographic "
        "layout and scene identity while predicting time-dependent appearance changes."
    )
    return clip_prompt, t5_prompt


class FMOWAnyTwoFramesDataset(Dataset):
    """Expand every fMoW triplet into all choices of one target and two inputs."""

    def __init__(
        self,
        index_path,
        root=None,
        include_partitions=("train",),
        source_splits=("train",),
        split_ratios=None,
        split_seed=42,
        image_size=(512, 512),
        target_mode="all",
        drop_text_prob=0.1,
        drop_image_prob=0.05,
        random_flip=True,
        validate_files=False,
        max_sequences=None,
    ):
        self.index_path = Path(index_path)
        self.root = Path(root) if root is not None else self.index_path.parent
        self.image_size = tuple(map(int, image_size))
        self.target_mode = str(target_mode)
        self.drop_text_prob = float(drop_text_prob)
        self.drop_image_prob = float(drop_image_prob)
        self.random_flip = bool(random_flip)
        if self.target_mode not in {"all", "random", "middle"}:
            raise ValueError("target_mode must be one of: all, random, middle.")
        if len(self.image_size) != 2 or any(size <= 0 for size in self.image_size):
            raise ValueError("image_size must contain two positive integers.")
        if not 0 <= self.drop_text_prob <= 1 or not 0 <= self.drop_image_prob <= 1:
            raise ValueError("Drop probabilities must be between 0 and 1.")

        include_partitions = set(include_partitions)
        source_splits = set(source_splits)
        split_ratios = split_ratios or {"train": 0.98, "val": 0.01, "test": 0.01}
        samples = []
        for row in _read_jsonl(self.index_path):
            if row.get("split", "train") not in source_splits:
                continue
            split_key = _group_id(row)
            partition = _partition_for(split_key, split_ratios, split_seed)
            if partition not in include_partitions:
                continue
            frames = _validate_frames(row)
            if frames is None:
                continue
            if validate_files and not all(
                (self.root / frame["image_path"]).is_file() for frame in frames
            ):
                continue
            row = dict(row)
            row["frames"] = frames
            row["partition"] = partition
            samples.append(row)
            if max_sequences is not None and len(samples) >= int(max_sequences):
                break

        if not samples:
            raise ValueError(
                f"No fMoW sequences found in {self.index_path} for partitions "
                f"{sorted(include_partitions)}."
            )
        self.samples = samples
        multiplier = 3 if self.target_mode == "all" else 1
        print(
            f"fMoW sequences: {len(samples)}; target mode: {self.target_mode}; "
            f"effective examples: {len(samples) * multiplier}."
        )

    def __len__(self):
        return len(self.samples) * (3 if self.target_mode == "all" else 1)

    def _row_and_target_slot(self, index):
        if self.target_mode == "all":
            return self.samples[index // 3], index % 3
        row = self.samples[index]
        if self.target_mode == "random":
            return row, random.randrange(3)
        return row, 1

    def __getitem__(self, index):
        row, target_slot = self._row_and_target_slot(index)
        frames = row["frames"]
        context_slots = [slot for slot in range(3) if slot != target_slot]
        target = _read_resized_rgb(
            self.root / frames[target_slot]["image_path"], self.image_size
        )
        contexts = [
            _read_resized_rgb(self.root / frames[slot]["image_path"], self.image_size)
            for slot in context_slots
        ]

        if self.random_flip:
            if random.random() < 0.5:
                target = target.flip(-1)
                contexts = [image.flip(-1) for image in contexts]
            if random.random() < 0.5:
                target = target.flip(-2)
                contexts = [image.flip(-2) for image in contexts]

        clip_prompt, t5_prompt = build_prompts(row, target_slot, root=self.root)
        if random.random() < self.drop_text_prob:
            clip_prompt = t5_prompt = ""
        if random.random() < self.drop_image_prob:
            contexts = [torch.zeros_like(image) for image in contexts]

        return {
            "image": target,
            "condition_0": contexts[0],
            "condition_type_0": "temporal_context",
            "position_delta_0": np.array([0.0, 0.0], dtype=np.float32),
            "position_scale_0": 1.0,
            "condition_1": contexts[1],
            "condition_type_1": "temporal_context",
            "position_delta_1": np.array([0.0, 0.0], dtype=np.float32),
            "position_scale_1": 1.0,
            "description": clip_prompt,
            "description_2": t5_prompt,
        }


def _apply_overrides(training_config):
    for environment_name, config_name in {
        "OMINI_BATCH_SIZE": "batch_size",
        "OMINI_ACCUMULATE_GRAD_BATCHES": "accumulate_grad_batches",
        "OMINI_MAX_STEPS": "max_steps",
        "OMINI_SAVE_INTERVAL": "save_interval",
        "OMINI_PRINT_EVERY_N_STEPS": "print_every_n_steps",
    }.items():
        value = os.environ.get(environment_name)
        if value is None:
            continue
        value = int(value)
        if value < 1:
            raise ValueError(f"{environment_name} must be positive.")
        training_config[config_name] = value
        print(f"Training override: {config_name}={value}")


def main():
    from .trainer import OminiModel, get_config, train

    config = get_config()
    training_config = config["train"]
    dataset_config = training_config["dataset"]
    _apply_overrides(training_config)
    if os.environ.get("OMINI_LORA_TARGET_PRESET", "default") == "expanded":
        training_config["lora_config"]["target_modules"] = (
            EXPANDED_LORA_TARGET_MODULES
        )
        print("LoRA target preset: expanded")

    dataset = FMOWAnyTwoFramesDataset(
        index_path=dataset_config["index_path"],
        root=dataset_config.get("root"),
        include_partitions=dataset_config.get("include_partitions", ["train"]),
        source_splits=dataset_config.get("source_splits", ["train"]),
        split_ratios=dataset_config.get("split_ratios"),
        split_seed=dataset_config.get("split_seed", 42),
        image_size=dataset_config.get("image_size", [512, 512]),
        target_mode=dataset_config.get("target_mode", "all"),
        drop_text_prob=dataset_config.get("drop_text_prob", 0.1),
        drop_image_prob=dataset_config.get("drop_image_prob", 0.05),
        random_flip=dataset_config.get("random_flip", True),
        validate_files=dataset_config.get("validate_files", False),
        max_sequences=dataset_config.get("max_sequences"),
    )

    adapter_names = config.get("model", {}).get(
        "adapter_names", ["text", "main", "condition", "condition"]
    )
    if len(adapter_names) != 4:
        raise ValueError("model.adapter_names must have four entries.")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    model = OminiModel(
        flux_pipe_id=config["flux_path"],
        transformer_path=config.get("transformer_path"),
        lora_path=os.environ.get("OMINI_LORA_PATH") or training_config.get("lora_path"),
        lora_config=training_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        adapter_names=adapter_names,
    )
    train(dataset, model, config, test_function=None)


if __name__ == "__main__":
    main()
