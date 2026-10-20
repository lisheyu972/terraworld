"""Dataset and mask utilities for dynamic fMOW image inpainting."""

import json
import math
import random
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from torch.utils.data import Dataset


# The source fMOW JPEGs can be much larger than Pillow's default safety limit.
Image.MAX_IMAGE_PIXELS = None

MASK_TYPES = (
    "irregular_brush",
    "rectangle_polygon",
    "smooth_cloud",
    "stripe_scanline",
)
MASK_WEIGHTS = (35, 20, 20, 10)
_FRAME_NAME = re.compile(r"^(.+)_([0-9]+)_rgb$")


def read_jsonl(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def source_group(image_path):
    """Remove the final temporal index from train and test fMOW frame names."""
    match = _FRAME_NAME.fullmatch(Path(image_path).stem)
    if match is None:
        raise ValueError(f"Unexpected fMOW frame name: {image_path}")
    return match.group(1)


def collect_fmow_frames(root, index_path=None, source_splits=None, validate_files=True):
    root = Path(root)
    index_path = Path(index_path) if index_path is not None else root / "index.jsonl"
    source_splits = set(source_splits or [])
    frames_by_group = defaultdict(dict)
    for row in read_jsonl(index_path):
        if source_splits and row.get("split") not in source_splits:
            continue
        for frame in row.get("frames") or []:
            image_key = frame.get("image_path")
            json_key = frame.get("json_path")
            if not image_key or not json_key:
                continue
            image_path = root / image_key
            json_path = root / json_key
            if validate_files and (not image_path.is_file() or not json_path.is_file()):
                raise FileNotFoundError(f"Incomplete fMOW frame: {image_path} / {json_path}")
            group = source_group(image_key)
            frames_by_group[group][image_key] = {
                "id": Path(image_key).stem,
                "group": group,
                "image_path": image_path,
                "json_path": json_path,
                "split": row.get("split"),
                "sequence_id": row.get("sequence_id"),
                "timestamp_index": frame.get("timestamp_index"),
            }
    groups = [
        sorted(frames.values(), key=lambda item: str(item["image_path"]))
        for _, frames in sorted(frames_by_group.items())
    ]
    if not groups:
        raise ValueError(f"No fMOW frames found in {index_path}.")
    return groups


@lru_cache(maxsize=32768)
def read_annotation(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def build_inpaint_prompts(annotation):
    description = annotation.get("description") or {}
    metadata = annotation.get("metadata") or {}
    scene = str(description.get("scene") or "remote sensing scene").replace("_", " ")
    acquisition_time = metadata.get("acquisition_datetime", "unknown time")
    latitude = metadata.get("center_lat")
    longitude = metadata.get("center_lon")
    location = ""
    if latitude is not None and longitude is not None:
        location = f", centered at latitude {float(latitude):.6f}, longitude {float(longitude):.6f}"
    clip_prompt = f"Restored optical satellite image of {scene}."
    t5_prompt = (
        "Task: optical satellite image inpainting. Restore every black missing "
        f"region in this image of {scene}, acquired at {acquisition_time}{location}. "
        "Reconstruct realistic content consistent with the surrounding spatial "
        "context while preserving the geographic layout, structures, viewpoint, "
        "colors, and every visible pixel. Output one complete satellite image."
    )
    return clip_prompt, t5_prompt


def read_center_cropped_rgb(path, image_size):
    width, height = map(int, image_size)
    with Image.open(path) as image:
        image.draft("RGB", (width, height))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(
            image,
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        return image.copy()


def pil_to_float_tensor(image):
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(255.0)


def mask_coverage(mask):
    return mask.histogram()[255] / float(mask.width * mask.height)


def _irregular_brush_mask(width, height, rng):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    short_side = min(width, height)
    for _ in range(rng.randint(2, 6)):
        x, y = rng.randint(0, width - 1), rng.randint(0, height - 1)
        points = [(x, y)]
        for _ in range(rng.randint(3, 10)):
            angle = rng.uniform(0, math.tau)
            length = rng.uniform(0.04, 0.22) * short_side
            x = min(width - 1, max(0, round(x + math.cos(angle) * length)))
            y = min(height - 1, max(0, round(y + math.sin(angle) * length)))
            points.append((x, y))
        brush_width = rng.randint(max(3, short_side // 40), max(4, short_side // 9))
        draw.line(points, fill=255, width=brush_width, joint="curve")
        radius = brush_width // 2
        for point_x, point_y in points:
            draw.ellipse(
                (point_x - radius, point_y - radius, point_x + radius, point_y + radius),
                fill=255,
            )
    return mask


def _rectangle_polygon_mask(width, height, rng):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for _ in range(rng.randint(1, 4)):
        center_x, center_y = rng.randint(0, width - 1), rng.randint(0, height - 1)
        shape_width = rng.randint(max(8, width // 10), max(9, width // 2))
        shape_height = rng.randint(max(8, height // 10), max(9, height // 2))
        if rng.random() < 0.5:
            draw.rectangle(
                (
                    center_x - shape_width // 2,
                    center_y - shape_height // 2,
                    center_x + shape_width // 2,
                    center_y + shape_height // 2,
                ),
                fill=255,
            )
        else:
            points = []
            point_count = rng.randint(4, 8)
            for index in range(point_count):
                angle = math.tau * index / point_count + rng.uniform(-0.35, 0.35)
                points.append(
                    (
                        round(center_x + math.cos(angle) * shape_width * rng.uniform(0.3, 0.55)),
                        round(center_y + math.sin(angle) * shape_height * rng.uniform(0.3, 0.55)),
                    )
                )
            draw.polygon(points, fill=255)
    return mask


def _smooth_cloud_mask(width, height, rng):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    center_x, center_y = rng.randint(0, width - 1), rng.randint(0, height - 1)
    spread_x, spread_y = rng.uniform(0.12, 0.4) * width, rng.uniform(0.12, 0.4) * height
    for _ in range(rng.randint(8, 24)):
        x, y = rng.gauss(center_x, spread_x), rng.gauss(center_y, spread_y)
        radius_x, radius_y = rng.uniform(0.04, 0.18) * width, rng.uniform(0.04, 0.18) * height
        draw.ellipse((x - radius_x, y - radius_y, x + radius_x, y + radius_y), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(rng.uniform(0.012, 0.045) * min(width, height)))
    threshold = rng.randint(70, 150)
    return mask.point(lambda value: 255 if value >= threshold else 0)


def _stripe_scanline_mask(width, height, rng):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    orientation = rng.choice(("horizontal", "vertical"))
    extent = height if orientation == "horizontal" else width
    cross_extent = width if orientation == "horizontal" else height
    for _ in range(rng.randint(1, 5)):
        start = rng.randint(0, extent - 1)
        thickness = rng.randint(max(2, extent // 100), max(3, extent // 14))
        if orientation == "horizontal":
            draw.rectangle((0, start, cross_extent, min(extent, start + thickness)), fill=255)
        else:
            draw.rectangle((start, 0, min(extent, start + thickness), cross_extent), fill=255)
    if rng.random() < 0.6:
        spacing = rng.randint(max(8, extent // 30), max(9, extent // 10))
        line_width = rng.randint(1, max(2, extent // 100))
        for start in range(rng.randrange(spacing), extent, spacing):
            if orientation == "horizontal":
                draw.rectangle((0, start, cross_extent, min(extent, start + line_width)), fill=255)
            else:
                draw.rectangle((start, 0, min(extent, start + line_width), cross_extent), fill=255)
    return mask


_MASK_GENERATORS = {
    "irregular_brush": _irregular_brush_mask,
    "rectangle_polygon": _rectangle_polygon_mask,
    "smooth_cloud": _smooth_cloud_mask,
    "stripe_scanline": _stripe_scanline_mask,
}
_MASK_COVERAGE_RANGES = {
    "irregular_brush": (0.05, 0.60),
    "rectangle_polygon": (0.06, 0.65),
    "smooth_cloud": (0.08, 0.65),
    "stripe_scanline": (0.02, 0.40),
}


def sample_inpaint_mask(width, height, rng=None, mask_type=None):
    if width <= 0 or height <= 0:
        raise ValueError("Mask dimensions must be positive.")
    rng = rng or random
    if mask_type is None or mask_type == "random":
        mask_type = rng.choices(MASK_TYPES, weights=MASK_WEIGHTS, k=1)[0]
    if mask_type not in _MASK_GENERATORS:
        raise ValueError(f"Unsupported mask type: {mask_type}")
    minimum, maximum = _MASK_COVERAGE_RANGES[mask_type]
    generator = _MASK_GENERATORS[mask_type]
    for _ in range(12):
        mask = generator(width, height, rng)
        if minimum <= mask_coverage(mask) <= maximum:
            break
    return mask, mask_type


def apply_mask(image, mask):
    image_array = np.asarray(image, dtype=np.uint8).copy()
    image_array[np.asarray(mask, dtype=np.uint8) > 0] = 0
    return Image.fromarray(image_array, mode="RGB")


class FMOWInpaintDataset(Dataset):
    """One dynamic masked-RGB inpainting example per source location per pass."""

    def __init__(
        self,
        root,
        index_path=None,
        source_splits=("train",),
        image_size=(512, 512),
        drop_text_prob=0.1,
        drop_image_prob=0.05,
        random_flip=True,
        validate_files=True,
        max_groups=None,
    ):
        self.root = Path(root)
        self.image_size = tuple(map(int, image_size))
        self.drop_text_prob = float(drop_text_prob)
        self.drop_image_prob = float(drop_image_prob)
        self.random_flip = bool(random_flip)
        if len(self.image_size) != 2 or any(size <= 0 or size % 16 for size in self.image_size):
            raise ValueError("image_size must contain two positive multiples of 16.")
        if not 0 <= self.drop_text_prob <= 1 or not 0 <= self.drop_image_prob <= 1:
            raise ValueError("Drop probabilities must be between 0 and 1.")
        self.groups = collect_fmow_frames(
            self.root,
            index_path=index_path,
            source_splits=source_splits,
            validate_files=validate_files,
        )
        if max_groups is not None:
            if int(max_groups) < 1:
                raise ValueError("max_groups must be positive when set.")
            self.groups = self.groups[: int(max_groups)]
        frame_count = sum(map(len, self.groups))
        print(
            f"fMOW inpainting: {len(self.groups)} source locations, "
            f"{frame_count} candidate frames; one random frame per location per pass."
        )

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        sample = random.choice(self.groups[index])
        target_image = read_center_cropped_rgb(sample["image_path"], self.image_size)
        mask, _ = sample_inpaint_mask(*target_image.size)
        condition_image = apply_mask(target_image, mask)
        target = pil_to_float_tensor(target_image)
        condition = pil_to_float_tensor(condition_image)

        if self.random_flip:
            if random.random() < 0.5:
                target, condition = target.flip(-1), condition.flip(-1)
            if random.random() < 0.5:
                target, condition = target.flip(-2), condition.flip(-2)

        clip_prompt, t5_prompt = build_inpaint_prompts(read_annotation(str(sample["json_path"])))
        if random.random() < self.drop_text_prob:
            clip_prompt = t5_prompt = ""
        if random.random() < self.drop_image_prob:
            condition = torch.zeros_like(condition)
        return {
            "image": target,
            "condition_0": condition,
            "condition_type_0": "masked_rgb_inpaint",
            "position_delta_0": np.array([0.0, 0.0], dtype=np.float32),
            "position_scale_0": 1.0,
            "description": clip_prompt,
            "description_2": t5_prompt,
        }
