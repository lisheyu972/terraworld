"""Dataset utilities for paired SSL4EO seasonal image translation."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


def build_season_prompts(source_season, target_season):
    clip_prompt = (
        f"Sentinel-2 RGB remote sensing image of the same location in "
        f"{target_season}."
    )
    t5_prompt = (
        "Task: seasonal remote sensing image translation. Given a co-registered "
        f"Sentinel-2 RGB image acquired in {source_season}, generate the same "
        f"location in {target_season}. Preserve roads, buildings, terrain, water "
        "boundaries, field boundaries, and spatial layout. Change only "
        "season-dependent vegetation, snow, soil appearance, and illumination. "
        "Produce a realistic, cloud-free optical satellite image at 10 meter "
        "ground sample distance."
    )
    return clip_prompt, t5_prompt


def load_index(index_path, root=None, validate_files=False, max_samples=None):
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
    ):
        self.image_size = tuple(map(int, image_size))
        self.drop_text_prob = float(drop_text_prob)
        self.drop_image_prob = float(drop_image_prob)
        self.random_flip = bool(random_flip)
        if not 0 <= self.drop_text_prob <= 1 or not 0 <= self.drop_image_prob <= 1:
            raise ValueError("Drop probabilities must be between 0 and 1")
        self.samples = load_index(
            index_path=index_path,
            root=root,
            validate_files=validate_files,
            max_samples=max_samples,
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
            sample["source_season"], sample["target_season"]
        )
        if random.random() < self.drop_text_prob:
            clip_prompt = t5_prompt = ""
        if random.random() < self.drop_image_prob:
            source = torch.zeros_like(source)
        return {
            "image": target,
            "condition_0": source,
            "condition_type_0": "season_translation",
            "position_delta_0": np.array([0.0, 0.0], dtype=np.float32),
            "position_scale_0": 1.0,
            "description": clip_prompt,
            "description_2": t5_prompt,
        }
