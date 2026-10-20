"""Train expanded SEN2NAIP LoRA with a bicubic LR-derived AnchorFlow start."""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from .train_sen2naip_sr import (
    EXPANDED_LORA_TARGET_MODULES,
    SEN2NAIPSuperResolutionDataset,
    _pad_aligned_pair,
    _top_left_padding,
)
from .trainer import OminiModel, get_config, train


ANCHOR_FLOW_IMAGE_LORA_TARGET_MODULES = (
    EXPANDED_LORA_TARGET_MODULES[:-1]
    + r"|norm_out\.linear|proj_out)"
)

ANCHOR_FLOW_TEXT_LORA_TARGET_MODULES = (
    ANCHOR_FLOW_IMAGE_LORA_TARGET_MODULES[:-1]
    + r"|context_embedder"
    + r"|.*(?<!single_)transformer_blocks\.[0-9]+\.norm1_context\.linear"
    + r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.add_q_proj"
    + r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.add_k_proj"
    + r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.add_v_proj"
    + r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_add_out"
    + r"|.*(?<!single_)transformer_blocks\.[0-9]+\.ff_context\.net\.0\.proj"
    + r"|.*(?<!single_)transformer_blocks\.[0-9]+\.ff_context\.net\.2)"
)


class SEN2NAIPAnchorFlowDataset(SEN2NAIPSuperResolutionDataset):
    """Add a bicubic native-HR anchor, padded exactly like the HR target."""

    def __getitem__(self, index):
        sample = self.samples[index]
        lr, hr = self._load_pair(sample)

        if self.random_flip:
            if random.random() < 0.5:
                lr, hr = lr.flip(-1), hr.flip(-1)
            if random.random() < 0.5:
                lr, hr = lr.flip(-2), hr.flip(-2)

        # Construct the anchor before padding: 121x121 LR -> 484x484 HR.
        anchor = F.interpolate(
            lr.unsqueeze(0),
            size=hr.shape[-2:],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        anchor = anchor.clamp_(0.0, 1.0)
        anchor_padding = _top_left_padding(
            anchor.shape[-2],
            anchor.shape[-1],
            self.target_size[1],
            self.target_size[0],
        )
        anchor = F.pad(anchor, anchor_padding, mode="constant", value=0.0)

        lr, hr = _pad_aligned_pair(
            lr,
            hr,
            self.condition_size,
            self.target_size,
            self.position_scale,
        )
        if anchor.shape != hr.shape:
            raise ValueError(
                f"Anchor shape {tuple(anchor.shape)} does not match "
                f"target shape {tuple(hr.shape)}."
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
            "anchor_image": anchor,
            "condition_0": lr,
            "condition_type_0": "sr",
            "position_delta_0": np.array([0.0, 0.0], dtype=np.float32),
            "position_scale_0": self.position_scale,
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
    config = get_config()
    training_config = config["train"]
    dataset_config = training_config["dataset"]
    anchor_flow_config = config.get("anchor_flow", {})
    _apply_integer_overrides(training_config)
    train_main_adapter = bool(
        anchor_flow_config.get("train_main_adapter", False)
    )
    train_text_adapter = bool(
        anchor_flow_config.get("train_text_adapter", False)
    )
    if anchor_flow_config.get("enable", False) and not train_main_adapter:
        raise ValueError(
            "AnchorFlow requires anchor_flow.train_main_adapter=true. "
            "The changed anchor-to-target trajectory must be learned by the "
            "main image branch."
        )

    training_config["lora_config"][
        "target_modules"
    ] = (
        ANCHOR_FLOW_TEXT_LORA_TARGET_MODULES
        if train_text_adapter
        else ANCHOR_FLOW_IMAGE_LORA_TARGET_MODULES
    )
    print(
        "LoRA target preset: expanded + main output head"
        + (" + text branch" if train_text_adapter else "")
    )
    if anchor_flow_config.get("enable", False):
        print(
            "AnchorFlow: enabled, "
            f"sigma={anchor_flow_config.get('anchor_noise_sigma', 0.4)}, "
            f"dropout={anchor_flow_config.get('anchor_dropout', 0.1)}"
        )
    else:
        print("AnchorFlow: disabled; using the original noise-to-target path.")

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dataset = SEN2NAIPAnchorFlowDataset(
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
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        transformer_path=config.get("transformer_path"),
        lora_config=training_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        anchor_flow_config=anchor_flow_config,
        adapter_names=[
            "default" if train_text_adapter else None,
            "default" if train_main_adapter else None,
            "default",
        ],
    )
    train(dataset, trainable_model, config, test_function=None)


if __name__ == "__main__":
    main()
