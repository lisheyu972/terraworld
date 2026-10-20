import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .sen2naip_prompt import build_prompts
from .train_sen2naip_sr import (
    EXPANDED_LORA_TARGET_MODULES,
    _read_jsonl,
    _read_tiff,
)
from .trainer import OminiModel, get_config, train


class AIDSuperResolutionCondition512Dataset(Dataset):
    """AID SR dataset with each raw LR condition upsampled to 512x512."""

    def __init__(
        self,
        root,
        include_splits=("train", "val"),
        condition_size=(128, 128),
        upsampled_condition_size=(512, 512),
        target_size=(512, 512),
        position_scale=1.0,
        lr_divisor=255.0,
        hr_divisor=255.0,
        drop_text_prob=0.1,
        drop_image_prob=0.05,
        random_flip=True,
        include_metadata_prompt=True,
    ):
        self.root = Path(root)
        self.condition_size = tuple(condition_size)
        self.upsampled_condition_size = tuple(upsampled_condition_size)
        if self.upsampled_condition_size != (512, 512):
            raise ValueError(
                "upsampled_condition_size must be (512, 512), got "
                f"{self.upsampled_condition_size}."
            )
        self.target_size = tuple(target_size)
        self.position_scale = float(position_scale)
        if self.upsampled_condition_size != self.target_size:
            raise ValueError(
                "upsampled_condition_size and target_size must match, got "
                f"{self.upsampled_condition_size} and {self.target_size}."
            )
        if self.position_scale != 1.0:
            raise ValueError(
                "Aligned 512x512 condition and target require position_scale=1.0, "
                f"got {self.position_scale}."
            )
        self.lr_divisor = float(lr_divisor)
        self.hr_divisor = float(hr_divisor)
        self.drop_text_prob = float(drop_text_prob)
        self.drop_image_prob = float(drop_image_prob)
        self.random_flip = bool(random_flip)

        splits = set(include_splits)
        self.samples = []
        for row in _read_jsonl(self.root / "index.jsonl"):
            if row.get("split") not in splits:
                continue
            input_info = row.get("input", {})
            target_info = row.get("target", {})
            required = (
                input_info.get("image_path"),
                input_info.get("json_path"),
                target_info.get("image_path"),
                target_info.get("json_path"),
            )
            if not all(required):
                continue
            lr_path, lr_json, hr_path, hr_json = (
                self.root / value for value in required
            )
            if not all(path.is_file() for path in (lr_path, lr_json, hr_path, hr_json)):
                continue
            clip_prompt, t5_prompt = build_prompts(
                lr_json, hr_json, include_metadata_prompt
            )
            self.samples.append(
                (lr_path, hr_path, clip_prompt, t5_prompt)
            )

        if not self.samples:
            raise ValueError(
                f"No AID pairs found in {self.root} for splits {sorted(splits)}."
            )
        print(
            f"AID pairs: {len(self.samples)} from splits {sorted(splits)}."
        )
        print(
            f"Condition preprocessing: {self.condition_size[0]}x"
            f"{self.condition_size[1]} -> {self.upsampled_condition_size[0]}x"
            f"{self.upsampled_condition_size[1]} (bicubic), "
            f"position_scale={self.position_scale:g}."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        lr_path, hr_path, clip_prompt, t5_prompt = self.samples[index]
        lr = _read_tiff(lr_path, self.lr_divisor)
        hr = _read_tiff(hr_path, self.hr_divisor)

        if tuple(lr.shape[-2:]) != tuple(reversed(self.condition_size)):
            raise ValueError(
                f"Expected LR size {self.condition_size}, got "
                f"{tuple(reversed(lr.shape[-2:]))} at {lr_path}."
            )
        if tuple(hr.shape[-2:]) != tuple(reversed(self.target_size)):
            raise ValueError(
                f"Expected HR size {self.target_size}, got "
                f"{tuple(reversed(hr.shape[-2:]))} at {hr_path}."
            )

        if self.random_flip:
            if random.random() < 0.5:
                lr, hr = lr.flip(-1), hr.flip(-1)
            if random.random() < 0.5:
                lr, hr = lr.flip(-2), hr.flip(-2)
        if random.random() < self.drop_text_prob:
            clip_prompt = t5_prompt = ""
        if random.random() < self.drop_image_prob:
            lr = torch.zeros_like(lr)

        lr = F.interpolate(
            lr.unsqueeze(0),
            size=tuple(reversed(self.upsampled_condition_size)),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0).clamp_(0.0, 1.0)

        return {
            "image": hr,
            "condition_0": lr,
            "condition_type_0": "sr",
            "position_delta_0": np.array([0.0, 0.0], dtype=np.float32),
            "position_scale_0": self.position_scale,
            "description": clip_prompt,
            "description_2": t5_prompt,
        }


def _apply_overrides(config):
    training_config = config["train"]
    for environment_name, config_name in {
        "OMINI_BATCH_SIZE": "batch_size",
        "OMINI_ACCUMULATE_GRAD_BATCHES": "accumulate_grad_batches",
        "OMINI_MAX_STEPS": "max_steps",
        "OMINI_SAVE_INTERVAL": "save_interval",
        "OMINI_PRINT_EVERY_N_STEPS": "print_every_n_steps",
    }.items():
        value = os.environ.get(environment_name)
        if value is not None:
            value = int(value)
            if value < 1:
                raise ValueError(f"{environment_name} must be positive.")
            training_config[config_name] = value
            print(f"Training override: {config_name}={value}")

    save_path = os.environ.get("OMINI_SAVE_PATH")
    if save_path:
        training_config["save_path"] = save_path
        print(f"Training override: save_path={save_path}")

    ignore_transformer_path = os.environ.get("OMINI_IGNORE_TRANSFORMER_PATH")
    if ignore_transformer_path is not None:
        normalized = ignore_transformer_path.strip().lower()
        if normalized not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError("Invalid OMINI_IGNORE_TRANSFORMER_PATH value.")
        if normalized in {"1", "true", "yes"}:
            config.pop("transformer_path", None)
            print("Model override: using the transformer bundled with flux_path")

    preset = os.environ.get("OMINI_LORA_TARGET_PRESET", "default")
    if preset == "expanded":
        training_config["lora_config"]["target_modules"] = (
            EXPANDED_LORA_TARGET_MODULES
        )
        print("LoRA target preset: expanded")
    elif preset != "default":
        raise ValueError("OMINI_LORA_TARGET_PRESET must be default or expanded.")

    keep_condition = os.environ.get("OMINI_KEEP_OMINI_CONDITION")
    if keep_condition is not None and config.get("glor_control", {}).get("enable"):
        normalized = keep_condition.strip().lower()
        if normalized not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError("Invalid OMINI_KEEP_OMINI_CONDITION value.")
        config["glor_control"]["disable_omini_condition"] = normalized not in {
            "1", "true", "yes"
        }


def main():
    config = get_config()
    _apply_overrides(config)
    training_config = config["train"]
    dataset_config = training_config["dataset"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    dataset = AIDSuperResolutionCondition512Dataset(
        root=dataset_config["root"],
        include_splits=dataset_config.get("include_splits", ["train", "val"]),
        condition_size=dataset_config.get("condition_size", [128, 128]),
        upsampled_condition_size=dataset_config.get(
            "upsampled_condition_size", [512, 512]
        ),
        target_size=dataset_config.get("target_size", [512, 512]),
        position_scale=dataset_config.get("position_scale", 1.0),
        lr_divisor=dataset_config.get("lr_divisor", 255.0),
        hr_divisor=dataset_config.get("hr_divisor", 255.0),
        drop_text_prob=dataset_config.get("drop_text_prob", 0.1),
        drop_image_prob=dataset_config.get("drop_image_prob", 0.05),
        random_flip=dataset_config.get("random_flip", True),
        include_metadata_prompt=dataset_config.get("include_metadata_prompt", True),
    )

    glor_config = config.get("glor_control", {})
    glor_enabled = glor_config.get("enable", False)
    disable_condition = glor_enabled and glor_config.get(
        "disable_omini_condition", True
    )
    adapter_names = (
        ["default", "default"]
        if disable_condition
        else (
            ["default", "default", "default"]
            if glor_enabled
            else [None, "default", "default"]
        )
    )
    configured_adapter_names = config.get("model", {}).get("adapter_names")
    if configured_adapter_names is not None:
        if not isinstance(configured_adapter_names, list) or len(
            configured_adapter_names
        ) != len(adapter_names):
            raise ValueError(
                "model.adapter_names must be a list matching the active "
                f"branch count ({len(adapter_names)})."
            )
        adapter_names = configured_adapter_names
        print(f"Adapter branch override: {adapter_names}")
    model = OminiModel(
        flux_pipe_id=config["flux_path"],
        transformer_path=config.get("transformer_path"),
        lora_config=training_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        glor_control_config=glor_config,
        scb_control_config=config.get("scb_control", {}),
        observation_residual_config=config.get("observation_residual", {}),
        glor_task="super_resolution",
        scb_task="super_resolution",
        adapter_names=adapter_names,
    )
    train(dataset, model, config, test_function=None)


if __name__ == "__main__":
    main()
