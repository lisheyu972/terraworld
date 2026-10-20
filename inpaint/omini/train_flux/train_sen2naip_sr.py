import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .sen2naip_prompt import build_prompts
from .trainer import OminiModel, get_config, train

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


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


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _top_left_padding(height, width, target_height, target_width):
    if height > target_height or width > target_width:
        raise ValueError(
            f"Cannot pad {height}x{width} to the smaller size "
            f"{target_height}x{target_width}."
        )
    return 0, target_width - width, 0, target_height - height


def _pad_aligned_pair(lr, hr, condition_size, target_size, scale):
    condition_width, condition_height = condition_size
    target_width, target_height = target_size
    _, lr_height, lr_width = lr.shape
    _, hr_height, hr_width = hr.shape

    lr_padding = _top_left_padding(
        lr_height, lr_width, condition_height, condition_width
    )
    hr_padding = tuple(int(round(value * scale)) for value in lr_padding)
    expected_hr_width = hr_width + hr_padding[0] + hr_padding[1]
    expected_hr_height = hr_height + hr_padding[2] + hr_padding[3]
    if (expected_hr_width, expected_hr_height) != (target_width, target_height):
        raise ValueError(
            "LR padding cannot be mapped exactly to the requested HR size: "
            f"LR padding {lr_padding} at {scale}x gives HR "
            f"{expected_hr_width}x{expected_hr_height}, expected "
            f"{target_width}x{target_height}."
        )
    return (
        F.pad(lr, lr_padding, mode="constant", value=0.0),
        F.pad(hr, hr_padding, mode="constant", value=0.0),
    )


def _read_tiff(path, divisor):
    with rasterio.open(path) as src:
        image = src.read()
    if image.shape[0] < 3:
        raise ValueError(f"{path} has only {image.shape[0]} band(s); RGB is required.")
    image = np.ascontiguousarray(image[:3], dtype=np.float32)
    return torch.from_numpy(image).div_(divisor).clamp_(0.0, 1.0)


class SEN2NAIPSuperResolutionDataset(Dataset):
    """Joint dataset with top-left aligned zero padding and no resizing."""

    def __init__(
        self,
        real_root,
        synthetic_root,
        include_splits=("train", "val", "test"),
        real_repeat=4,
        condition_size=(128, 128),
        target_size=(512, 512),
        position_scale=4.0,
        real_lr_divisor=10000.0,
        hr_divisor=255.0,
        drop_text_prob=0.1,
        drop_image_prob=0.05,
        random_flip=True,
        include_metadata_prompt=True,
    ):
        self.condition_size = tuple(condition_size)
        self.target_size = tuple(target_size)
        self.position_scale = float(position_scale)
        self.real_lr_divisor = float(real_lr_divisor)
        self.hr_divisor = float(hr_divisor)
        self.drop_text_prob = float(drop_text_prob)
        self.drop_image_prob = float(drop_image_prob)
        self.random_flip = bool(random_flip)
        self.include_metadata_prompt = bool(include_metadata_prompt)

        include_splits = set(include_splits)
        real_samples = self._load_real_samples(
            Path(real_root), include_splits, self.include_metadata_prompt
        )
        synthetic_samples = self._load_synthetic_samples(
            Path(synthetic_root), include_splits, self.include_metadata_prompt
        )

        if real_repeat < 1:
            raise ValueError("real_repeat must be at least 1.")
        self.samples = synthetic_samples + real_samples * int(real_repeat)

        print(
            "SEN2NAIP pairs: "
            f"{len(real_samples)} unique real x {real_repeat}, "
            f"{len(synthetic_samples)} synthetic, "
            f"{len(self.samples)} effective training samples."
        )

    @staticmethod
    def _load_real_samples(root, include_splits, include_metadata_prompt):
        rows = _read_jsonl(root / "index.jsonl")
        samples = []
        seen_pairs = set()
        for row in rows:
            if row.get("split") not in include_splits:
                continue
            if "input" not in row or "target" not in row:
                continue
            lr_path = root / row["input"]["image_path"]
            hr_path = root / row["target"]["image_path"]
            pair = (str(lr_path), str(hr_path))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if not lr_path.is_file() or not hr_path.is_file():
                continue
            lr_json = root / row["input"]["json_path"]
            hr_json = root / row["target"]["json_path"]
            clip_prompt, t5_prompt = build_prompts(
                lr_json, hr_json, include_metadata_prompt
            )
            samples.append(
                {
                    "source": "real",
                    "lr_path": lr_path,
                    "hr_path": hr_path,
                    "clip_prompt": clip_prompt,
                    "t5_prompt": t5_prompt,
                }
            )
        return samples

    @staticmethod
    def _load_synthetic_samples(root, include_splits, include_metadata_prompt):
        rows = _read_jsonl(root / "index.jsonl")
        samples = []
        for row in rows:
            if row.get("split") not in include_splits:
                continue
            if "lr_image_path" not in row or "hr_image_path" not in row:
                continue
            lr_path = root / row["lr_image_path"]
            hr_path = root / row["hr_image_path"]
            if not lr_path.is_file() or not hr_path.is_file():
                continue
            lr_json = root / row["lr_json_path"]
            hr_json = root / row["hr_json_path"]
            clip_prompt, t5_prompt = build_prompts(
                lr_json, hr_json, include_metadata_prompt
            )
            samples.append(
                {
                    "source": "synthetic",
                    "lr_path": lr_path,
                    "hr_path": hr_path,
                    "clip_prompt": clip_prompt,
                    "t5_prompt": t5_prompt,
                }
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_pair(self, sample):
        if sample["source"] == "real":
            lr = _read_tiff(sample["lr_path"], self.real_lr_divisor)
        else:
            lr = torch.load(
                sample["lr_path"], map_location="cpu", weights_only=True
            )
            if not isinstance(lr, torch.Tensor):
                raise TypeError(f"{sample['lr_path']} does not contain a tensor.")
            lr = lr[:3].float().clamp_(0.0, 1.0)

        hr = _read_tiff(sample["hr_path"], self.hr_divisor)
        if lr.ndim != 3 or hr.ndim != 3:
            raise ValueError(
                f"Expected CHW tensors, got LR {tuple(lr.shape)}, HR {tuple(hr.shape)}."
            )
        return lr, hr

    def __getitem__(self, index):
        sample = self.samples[index]
        lr, hr = self._load_pair(sample)

        if self.random_flip:
            if random.random() < 0.5:
                lr, hr = lr.flip(-1), hr.flip(-1)
            if random.random() < 0.5:
                lr, hr = lr.flip(-2), hr.flip(-2)

        lr, hr = _pad_aligned_pair(
            lr,
            hr,
            self.condition_size,
            self.target_size,
            self.position_scale,
        )

        clip_prompt = sample["clip_prompt"]
        t5_prompt = sample["t5_prompt"]
        if random.random() < self.drop_text_prob:
            clip_prompt = ""
            t5_prompt = ""
        if random.random() < self.drop_image_prob:
            lr = torch.zeros_like(lr)

        return {
            "image": hr,
            "condition_0": lr,
            "condition_type_0": "sr",
            "position_delta_0": np.array([0.0, 0.0], dtype=np.float32),
            "position_scale_0": self.position_scale,
            "description": clip_prompt,
            "description_2": t5_prompt,
        }


def main():
    config = get_config()
    training_config = config["train"]
    dataset_config = training_config["dataset"]

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

    keep_omini_condition = os.environ.get("OMINI_KEEP_OMINI_CONDITION")
    if keep_omini_condition is not None:
        normalized = keep_omini_condition.strip().lower()
        if normalized not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError(
                "OMINI_KEEP_OMINI_CONDITION must be one of "
                "0, 1, false, true, no, or yes."
            )
        keep_omini = normalized in {"1", "true", "yes"}
        if config.get("toa_control", {}).get("enable", False):
            config["toa_control"]["disable_omini_condition"] = not keep_omini
        if config.get("glor_control", {}).get("enable", False):
            config["glor_control"]["disable_omini_condition"] = not keep_omini
        if (
            config.get("toa_control", {}).get("enable", False)
            or config.get("glor_control", {}).get("enable", False)
        ):
            print(f"Training override: keep_omini_condition={keep_omini}")

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

    dataset = SEN2NAIPSuperResolutionDataset(
        real_root=dataset_config["real_root"],
        synthetic_root=dataset_config["synthetic_root"],
        include_splits=dataset_config.get(
            "include_splits", ["train", "val", "test"]
        ),
        real_repeat=dataset_config.get("real_repeat", 4),
        condition_size=dataset_config.get("condition_size", [128, 128]),
        target_size=dataset_config.get("target_size", [512, 512]),
        position_scale=dataset_config.get("position_scale", 4.0),
        real_lr_divisor=dataset_config.get("real_lr_divisor", 10000.0),
        hr_divisor=dataset_config.get("hr_divisor", 255.0),
        drop_text_prob=dataset_config.get("drop_text_prob", 0.1),
        drop_image_prob=dataset_config.get("drop_image_prob", 0.05),
        random_flip=dataset_config.get("random_flip", True),
        include_metadata_prompt=dataset_config.get(
            "include_metadata_prompt", True
        ),
    )

    toa_enabled = config.get("toa_control", {}).get("enable", False)
    glor_enabled = config.get("glor_control", {}).get("enable", False)
    control_enabled = toa_enabled or glor_enabled
    control_disables_omini_condition = (
        (toa_enabled and config.get("toa_control", {}).get("disable_omini_condition", True))
        or (
            glor_enabled
            and config.get("glor_control", {}).get("disable_omini_condition", True)
        )
    )
    if control_enabled:
        adapter_names = (
            ["default", "default"]
            if control_disables_omini_condition
            else ["default", "default", "default"]
        )
    else:
        adapter_names = [None, None, "default"]

    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        transformer_path=config.get("transformer_path"),
        lora_config=training_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        toa_control_config=config.get("toa_control", {}),
        glor_control_config=config.get("glor_control", {}),
        scb_control_config=config.get("scb_control", {}),
        toa_task="super_resolution",
        glor_task="super_resolution",
        scb_task="super_resolution",
        adapter_names=adapter_names,
    )

    train(dataset, trainable_model, config, test_function=None)


if __name__ == "__main__":
    main()
