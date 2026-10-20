"""Dataset utilities for paired SAR-to-optical translation."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


def build_sar_to_optical_prompts(metadata):
    description = metadata.get("description") or {}
    sensor = (metadata.get("metadata") or {}).get("sensor", "optical satellite")
    scene = description.get("scene") or "remote sensing scene"
    objects = description.get("objects") or []
    layout = description.get("layout") or "Preserve the spatial layout."
    texture = description.get("texture") or "Render realistic optical textures."
    content = description.get("content_sentence") or f"An optical image of {scene}."
    object_text = ", ".join(map(str, objects)) if objects else "scene structures"
    clip_prompt = f"Optical remote sensing image of {scene}."
    t5_prompt = (
        "Task: SAR-to-optical image translation. Given a co-registered SAR "
        "observation, generate the corresponding natural-color optical remote "
        f"sensing image of {scene}. Target sensor: {sensor}. Visible objects: "
        f"{object_text}. Target content: {content} Layout: {layout} Texture: "
        f"{texture} Preserve geometry and object locations from the SAR condition "
        "while synthesizing realistic optical color and appearance."
    )
    return clip_prompt, t5_prompt


def collect_sar_to_optical_pairs(root, splits):
    root = Path(root)
    splits = set(splits)
    samples = []
    for json_path in sorted((root / "json").glob("OPT_*.json")):
        with json_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("split") not in splits:
            continue
        optical_id = str(metadata.get("image_id", json_path.stem))
        suffix = optical_id.removeprefix("OPT_")
        sar_path = root / "images" / "sar" / f"SAR_{suffix}.png"
        optical_path = root / metadata.get(
            "image_path", f"images/optical/OPT_{suffix}.png"
        )
        if not sar_path.is_file() or not optical_path.is_file():
            raise FileNotFoundError(
                f"Incomplete SAR/optical pair {suffix}: {sar_path}, {optical_path}"
            )
        samples.append(
            {
                "id": suffix,
                "split": metadata.get("split"),
                "sar_path": sar_path,
                "optical_path": optical_path,
                "json_path": json_path,
                "metadata": metadata,
            }
        )
    if not samples:
        raise ValueError(f"No SAR/optical pairs found for splits {sorted(splits)}.")
    return samples


def load_pair_images(sample, image_size=(256, 256)):
    image_size = tuple(map(int, image_size))
    with Image.open(sample["sar_path"]) as image:
        if image.size != image_size:
            raise ValueError(
                f"Expected SAR size {image_size}, got {image.size}: "
                f"{sample['sar_path']}"
            )
        sar = image.convert("RGB")
    with Image.open(sample["optical_path"]) as image:
        if image.size != image_size:
            raise ValueError(
                f"Expected optical size {image_size}, got {image.size}: "
                f"{sample['optical_path']}"
            )
        optical = image.convert("RGB")
    return sar, optical


def pil_to_float_tensor(image):
    return pil_to_tensor(image).float().div_(255.0)


class SARToOpticalDataset(Dataset):
    """Paired 256x256 SAR condition to optical target dataset."""

    def __init__(
        self,
        root,
        include_splits=("train",),
        image_size=(256, 256),
        drop_text_prob=0.1,
        drop_image_prob=0.05,
        random_flip=True,
    ):
        self.image_size = tuple(map(int, image_size))
        if self.image_size != (256, 256):
            raise ValueError(
                f"SAR-to-optical training requires image_size=(256, 256), got "
                f"{self.image_size}."
            )
        self.drop_text_prob = float(drop_text_prob)
        self.drop_image_prob = float(drop_image_prob)
        self.random_flip = bool(random_flip)
        if not 0 <= self.drop_text_prob <= 1 or not 0 <= self.drop_image_prob <= 1:
            raise ValueError("Drop probabilities must be between 0 and 1.")
        self.samples = collect_sar_to_optical_pairs(root, include_splits)
        print(
            f"SAR-to-optical pairs: {len(self.samples)} from splits "
            f"{sorted(set(include_splits))}; image size: 256x256."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        sar_image, optical_image = load_pair_images(sample, self.image_size)
        sar = pil_to_float_tensor(sar_image)
        optical = pil_to_float_tensor(optical_image)

        if self.random_flip:
            if random.random() < 0.5:
                sar, optical = sar.flip(-1), optical.flip(-1)
            if random.random() < 0.5:
                sar, optical = sar.flip(-2), optical.flip(-2)

        clip_prompt, t5_prompt = build_sar_to_optical_prompts(sample["metadata"])
        if random.random() < self.drop_text_prob:
            clip_prompt = t5_prompt = ""
        if random.random() < self.drop_image_prob:
            sar = torch.zeros_like(sar)

        return {
            "image": optical,
            "condition_0": sar,
            "condition_type_0": "sar_to_optical",
            "position_delta_0": np.array([0.0, 0.0], dtype=np.float32),
            "position_scale_0": 1.0,
            "description": clip_prompt,
            "description_2": t5_prompt,
        }
