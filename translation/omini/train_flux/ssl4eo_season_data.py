"""Dataset utilities for paired SSL4EO seasonal image translation."""

import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


def build_season_prompts(
    source_season,
    target_season,
    center_lat,
    center_lon,
):
    location = (
        f"latitude {float(center_lat):.6f}, "
        f"longitude {float(center_lon):.6f}"
    )
    clip_prompt = (
        f"Sentinel-2 RGB remote sensing image at {location} in {target_season}."
    )
    t5_prompt = (
        "Task: seasonal remote sensing image translation. Given a co-registered "
        f"Sentinel-2 RGB image acquired in {source_season}, generate the same "
        f"location in {target_season}. The location center is at {location}. "
        "Preserve roads, buildings, terrain, water boundaries, field boundaries, "
        "and spatial layout. Change only season-dependent vegetation, snow, soil "
        "appearance, and illumination. Produce a realistic, cloud-free optical "
        "satellite image at 10 meter ground sample distance."
    )
    return clip_prompt, t5_prompt


def load_coordinates(metadata_path):
    with metadata_path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    metadata = record.get("metadata") or {}
    center_lat = metadata.get("center_lat")
    center_lon = metadata.get("center_lon")
    if not isinstance(center_lat, (int, float)) or not -90 <= center_lat <= 90:
        raise ValueError(f"Invalid center_lat in {metadata_path}: {center_lat}")
    if not isinstance(center_lon, (int, float)) or not -180 <= center_lon <= 180:
        raise ValueError(f"Invalid center_lon in {metadata_path}: {center_lon}")
    return float(center_lat), float(center_lon)


def load_frame_metadata(metadata_path):
    with metadata_path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    metadata = record.get("metadata") or {}
    acquisition_datetime = metadata.get("acquisition_datetime")
    month = day = None
    if acquisition_datetime:
        try:
            parsed = datetime.fromisoformat(
                str(acquisition_datetime).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid acquisition_datetime in {metadata_path}: "
                f"{acquisition_datetime}"
            ) from exc
        month, day = parsed.month, parsed.day
    return {
        "month": month,
        "day": day,
        "center_lat": metadata.get("center_lat"),
        "center_lon": metadata.get("center_lon"),
    }


def build_miq_metadata(month, day, center_lat, center_lon):
    """Encode month/day periodically and lat/lon by fixed physical ranges."""

    values = []
    observed = []
    for value, period, name in (
        (month, 12.0, "month"),
        (day, 31.0, "day"),
    ):
        is_observed = value is not None
        if is_observed:
            value = float(value)
            if not 1 <= value <= period:
                raise ValueError(f"{name}={value} is outside [1, {period:g}]")
            angle = 2.0 * math.pi * value / period
            values.extend((math.sin(angle), math.cos(angle)))
        else:
            values.extend((0.0, 0.0))
        observed.append(float(is_observed))

    for value, scale, name in (
        (center_lat, 90.0, "center_lat"),
        (center_lon, 180.0, "center_lon"),
    ):
        is_observed = value is not None
        if is_observed:
            value = float(value)
            if not -scale <= value <= scale:
                raise ValueError(
                    f"{name}={value} is outside [{-scale:g}, {scale:g}]"
                )
            values.append(value / scale)
        else:
            values.append(0.0)
        observed.append(float(is_observed))

    # Layout:
    # month sin/cos, day sin/cos, normalized latitude/longitude,
    # then observed masks for month, day, latitude, and longitude.
    return torch.tensor([*values, *observed], dtype=torch.float32)


def load_index(
    index_path,
    root=None,
    validate_files=False,
    max_samples=None,
    include_metadata=False,
):
    index_path = Path(index_path)
    root = Path(root) if root is not None else index_path.parent
    with index_path.open("r", encoding="utf-8") as stream:
        index = json.load(stream)
    if index.get("task") != "season_translation":
        raise ValueError(f"Expected season_translation index: {index_path}")
    fields = index.get("pair_fields")
    expected_fields = [
        "sequence_id",
        "source_frame_index",
        "target_frame_index",
    ]
    if fields != expected_fields:
        raise ValueError(f"Unexpected pair_fields in {index_path}: {fields}")
    seasons = {int(key): value for key, value in index["season_by_frame_index"].items()}
    image_root = root / index.get("image_root", "images/sequence")
    pattern = index.get(
        "image_pattern", "{sequence_id}_F{frame_index:03d}.png"
    )
    metadata_root = root / index.get("metadata_root", "json")
    metadata_pattern = index.get(
        "metadata_pattern", "{sequence_id}_F{frame_index:03d}.json"
    )
    indexed_coordinates = index.get("coordinates_by_sequence") or {}
    indexed_frame_metadata = index.get("metadata_by_sequence") or {}
    coordinate_cache = {}
    frame_metadata_cache = {}
    samples = []
    for sequence_id, source_index, target_index in index.get("pairs", []):
        if source_index == target_index:
            raise ValueError(f"Source equals target for {sequence_id}")
        try:
            source_season = seasons[source_index]
            target_season = seasons[target_index]
        except KeyError as exc:
            raise ValueError(f"Unknown season index for {sequence_id}") from exc
        source_path = image_root / pattern.format(
            sequence_id=sequence_id, frame_index=source_index
        )
        target_path = image_root / pattern.format(
            sequence_id=sequence_id, frame_index=target_index
        )
        coordinates = indexed_coordinates.get(sequence_id)
        if coordinates is not None:
            if (
                not isinstance(coordinates, (list, tuple))
                or len(coordinates) != 2
            ):
                raise ValueError(
                    f"Invalid indexed coordinates for {sequence_id}: {coordinates}"
                )
            center_lat, center_lon = map(float, coordinates)
        else:
            if sequence_id not in coordinate_cache:
                metadata_path = metadata_root / metadata_pattern.format(
                    sequence_id=sequence_id,
                    frame_index=source_index,
                )
                coordinate_cache[sequence_id] = load_coordinates(metadata_path)
            center_lat, center_lon = coordinate_cache[sequence_id]
        if not -90 <= center_lat <= 90 or not -180 <= center_lon <= 180:
            raise ValueError(
                f"Coordinates out of range for {sequence_id}: "
                f"{center_lat}, {center_lon}"
            )
        source_frame_metadata = target_frame_metadata = None
        if include_metadata:
            sequence_metadata = indexed_frame_metadata.get(sequence_id) or {}

            def resolve_frame_metadata(frame_index):
                embedded = sequence_metadata.get(str(frame_index))
                if embedded is not None:
                    if isinstance(embedded, dict):
                        frame_metadata = dict(embedded)
                    elif isinstance(embedded, (list, tuple)) and len(embedded) == 4:
                        frame_metadata = {
                            "month": embedded[0],
                            "day": embedded[1],
                            "center_lat": embedded[2],
                            "center_lon": embedded[3],
                        }
                    else:
                        raise ValueError(
                            f"Invalid metadata for {sequence_id} frame "
                            f"{frame_index}: {embedded}"
                        )
                else:
                    cache_key = (sequence_id, frame_index)
                    if cache_key not in frame_metadata_cache:
                        metadata_path = metadata_root / metadata_pattern.format(
                            sequence_id=sequence_id,
                            frame_index=frame_index,
                        )
                        frame_metadata_cache[cache_key] = load_frame_metadata(
                            metadata_path
                        )
                    frame_metadata = frame_metadata_cache[cache_key]
                if frame_metadata.get("center_lat") is None:
                    frame_metadata["center_lat"] = center_lat
                if frame_metadata.get("center_lon") is None:
                    frame_metadata["center_lon"] = center_lon
                return frame_metadata

            source_frame_metadata = resolve_frame_metadata(source_index)
            target_frame_metadata = resolve_frame_metadata(target_index)
        if validate_files and (not source_path.is_file() or not target_path.is_file()):
            raise FileNotFoundError(
                f"Incomplete seasonal pair {sequence_id}: {source_path}, {target_path}"
            )
        samples.append(
            {
                "id": f"{sequence_id}_{source_season}_to_{target_season}",
                "sequence_id": sequence_id,
                "source_index": source_index,
                "target_index": target_index,
                "source_season": source_season,
                "target_season": target_season,
                "source_path": source_path,
                "target_path": target_path,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "source_metadata": source_frame_metadata,
                "target_metadata": target_frame_metadata,
            }
        )
        if max_samples is not None and len(samples) >= int(max_samples):
            break
    if not samples:
        raise ValueError(f"No seasonal pairs found in {index_path}")
    return samples


def load_center_cropped_pair(sample, image_size=(256, 256)):
    width, height = map(int, image_size)

    def load(path):
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.width < width or image.height < height:
                raise ValueError(
                    f"Image {image.size} is smaller than requested crop "
                    f"{(width, height)}: {path}"
                )
            left = (image.width - width) // 2
            top = (image.height - height) // 2
            return image.crop((left, top, left + width, top + height))

    return load(sample["source_path"]), load(sample["target_path"])


def pil_to_float_tensor(image):
    return pil_to_tensor(image).float().div_(255.0)


class SSL4EOSeasonDataset(Dataset):
    """One source-season condition paired with a target-season image."""

    def __init__(
        self,
        index_path,
        root=None,
        image_size=(256, 256),
        drop_text_prob=0.0,
        drop_image_prob=0.05,
        random_flip=True,
        validate_files=False,
        max_samples=None,
        include_metadata=False,
    ):
        self.image_size = tuple(map(int, image_size))
        self.drop_text_prob = float(drop_text_prob)
        self.drop_image_prob = float(drop_image_prob)
        self.random_flip = bool(random_flip)
        self.include_metadata = bool(include_metadata)
        if not 0 <= self.drop_text_prob <= 1 or not 0 <= self.drop_image_prob <= 1:
            raise ValueError("Drop probabilities must be between 0 and 1")
        self.samples = load_index(
            index_path=index_path,
            root=root,
            validate_files=validate_files,
            max_samples=max_samples,
            include_metadata=self.include_metadata,
        )
        print(
            f"SSL4EO seasonal pairs: {len(self.samples)}; "
            f"center crop: {self.image_size[0]}x{self.image_size[1]}."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        source_image, target_image = load_center_cropped_pair(
            sample, self.image_size
        )
        source = pil_to_float_tensor(source_image)
        target = pil_to_float_tensor(target_image)
        if self.random_flip:
            if random.random() < 0.5:
                source, target = source.flip(-1), target.flip(-1)
            if random.random() < 0.5:
                source, target = source.flip(-2), target.flip(-2)
        clip_prompt, t5_prompt = build_season_prompts(
            sample["source_season"],
            sample["target_season"],
            sample["center_lat"],
            sample["center_lon"],
        )
        if random.random() < self.drop_text_prob:
            clip_prompt = t5_prompt = ""
        if random.random() < self.drop_image_prob:
            source = torch.zeros_like(source)
        result = {
            "image": target,
            "condition_0": source,
            "condition_type_0": "season_translation",
            "position_delta_0": np.array([0.0, 0.0], dtype=np.float32),
            "position_scale_0": 1.0,
            "description": clip_prompt,
            "description_2": t5_prompt,
        }
        if self.include_metadata:
            source_metadata = sample["source_metadata"]
            target_metadata = sample["target_metadata"]
            result["condition_metadata"] = build_miq_metadata(
                source_metadata["month"],
                source_metadata["day"],
                source_metadata["center_lat"],
                source_metadata["center_lon"],
            )
            result["target_metadata"] = build_miq_metadata(
                target_metadata["month"],
                target_metadata["day"],
                target_metadata["center_lat"],
                target_metadata["center_lon"],
            )
        return result
