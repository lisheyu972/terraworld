"""Train OminiControl FLUX LoRA for dynamic fMOW inpainting."""

import os

import torch

from .fmow_inpaint_data import FMOWInpaintDataset
from .train_frame_interp import EXPANDED_LORA_TARGET_MODULES


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
    preset = os.environ.get("OMINI_LORA_TARGET_PRESET", "default")
    if preset == "expanded":
        training_config["lora_config"]["target_modules"] = EXPANDED_LORA_TARGET_MODULES
        print("LoRA target preset: expanded")
    elif preset != "default":
        raise ValueError("OMINI_LORA_TARGET_PRESET must be default or expanded.")


def main():
    from .trainer import OminiModel, get_config, train

    config = get_config()
    _apply_overrides(config)
    training_config = config["train"]
    dataset_config = training_config["dataset"]
    dataset = FMOWInpaintDataset(
        root=dataset_config["root"],
        index_path=dataset_config.get("index_path"),
        source_splits=dataset_config.get("source_splits", ["train"]),
        image_size=dataset_config.get("image_size", [512, 512]),
        drop_text_prob=dataset_config.get("drop_text_prob", 0.1),
        drop_image_prob=dataset_config.get("drop_image_prob", 0.05),
        random_flip=dataset_config.get("random_flip", True),
        validate_files=dataset_config.get("validate_files", True),
        max_groups=dataset_config.get("max_groups"),
    )

    adapter_names = config.get("model", {}).get(
        "adapter_names", ["text", "main", "condition"]
    )
    if not isinstance(adapter_names, list) or len(adapter_names) != 3:
        raise ValueError(
            "model.adapter_names must be [text, main, condition] for fMOW inpainting."
        )
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
