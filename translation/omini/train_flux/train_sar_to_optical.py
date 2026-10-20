"""Train paired 256x256 SAR-to-optical translation."""

import os

import torch

from .sar_to_optical_data import SARToOpticalDataset
from .condition_shift import RandomConditionShiftDataset
from .train_frame_interp import EXPANDED_LORA_TARGET_MODULES
from .trainer import OminiModel, get_config, train


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
        if value is None:
            continue
        value = int(value)
        if value < 1:
            raise ValueError(f"{environment_name} must be positive.")
        training_config[config_name] = value
        print(f"Training override: {config_name}={value}")

    save_path = os.environ.get("OMINI_SAVE_PATH")
    if save_path:
        training_config["save_path"] = save_path
        print(f"Training override: save_path={save_path}")

    data_root = os.environ.get("OMINI_DATA_ROOT")
    if data_root:
        training_config["dataset"]["root"] = data_root
        print(f"Training override: dataset.root={data_root}")

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
            "1",
            "true",
            "yes",
        }
        print(
            "Keep original Omini condition: "
            f"{not config['glor_control']['disable_omini_condition']}"
        )

    psnr_weight = os.environ.get("OMINI_PSNR_LOSS_WEIGHT")
    if psnr_weight is not None:
        psnr_weight = float(psnr_weight)
        if psnr_weight < 0:
            raise ValueError("OMINI_PSNR_LOSS_WEIGHT must be non-negative.")
        psnr_config = training_config.setdefault("psnr_loss", {})
        psnr_config["enable"] = psnr_weight > 0
        psnr_config["weight"] = psnr_weight
        print(f"Training override: psnr_loss.weight={psnr_weight}")


def main():
    config = get_config()
    _apply_overrides(config)
    transformer_path = config.get("transformer_path")
    if transformer_path:
        print(f"Pretrained DiT transformer: {transformer_path}")
    else:
        print("Base model: original FLUX transformer bundled with flux_path")

    training_config = config["train"]
    dataset_config = training_config["dataset"]
    dataset = SARToOpticalDataset(
        root=dataset_config["root"],
        include_splits=dataset_config.get("include_splits", ["train"]),
        image_size=dataset_config.get("image_size", [256, 256]),
        drop_text_prob=dataset_config.get("drop_text_prob", 0.1),
        drop_image_prob=dataset_config.get("drop_image_prob", 0.05),
        random_flip=dataset_config.get("random_flip", True),
        center_crop=dataset_config.get("center_crop", False),
    )
    shift_pixels = int(dataset_config.get("condition_shift_pixels", 0))
    if shift_pixels:
        dataset = RandomConditionShiftDataset(dataset, shift_pixels)
        print(f"Unknown SAR condition translation: up to {shift_pixels} pixels.")

    glor_config = config.get("glor_control", {})
    glor_enabled = glor_config.get("enable", False)
    glor_task = glor_config.get("task", "super_resolution")
    if glor_enabled and glor_task != "super_resolution":
        raise ValueError(
            "SAR-to-optical GLoR-Control must use the single-condition "
            "'super_resolution' task."
        )
    disable_condition = glor_enabled and glor_config.get(
        "disable_omini_condition", True
    )
    expected_adapter_names = (
        ["text", "main"] if disable_condition else ["text", "main", "condition"]
    )
    configured_adapter_names = config.get("model", {}).get("adapter_names")
    if configured_adapter_names is None:
        adapter_names = expected_adapter_names
    elif not isinstance(configured_adapter_names, list):
        raise ValueError("model.adapter_names must be a list.")
    elif len(configured_adapter_names) == len(expected_adapter_names):
        adapter_names = configured_adapter_names
    elif disable_condition and len(configured_adapter_names) == 3:
        adapter_names = configured_adapter_names[:2]
        print(f"Disabled SAR condition adapter branch: {adapter_names}")
    else:
        raise ValueError(
            "model.adapter_names must match the active SAR-to-optical branch "
            f"count ({len(expected_adapter_names)})."
        )

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    model = OminiModel(
        flux_pipe_id=config["flux_path"],
        transformer_path=transformer_path,
        lora_path=os.environ.get("OMINI_LORA_PATH")
        or training_config.get("lora_path"),
        lora_config=training_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        glor_control_config=glor_config,
        glor_task=glor_task,
        adapter_names=adapter_names,
        psnr_loss_config=training_config.get("psnr_loss"),
    )
    train(dataset, model, config, test_function=None)


if __name__ == "__main__":
    main()
