"""Train paired SSL4EO seasonal translation with one image condition."""

import os

import torch

from .ssl4eo_season_data import SSL4EOSeasonDataset
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
            raise ValueError(f"{environment_name} must be positive")
        training_config[config_name] = value
        print(f"Training override: {config_name}={value}")

    save_path = os.environ.get("OMINI_SAVE_PATH")
    if save_path:
        training_config["save_path"] = save_path
        print(f"Training override: save_path={save_path}")

    preset = os.environ.get("OMINI_LORA_TARGET_PRESET", "default")
    if preset == "expanded":
        training_config["lora_config"]["target_modules"] = (
            EXPANDED_LORA_TARGET_MODULES
        )
        print("LoRA target preset: expanded")
    elif preset != "default":
        raise ValueError("OMINI_LORA_TARGET_PRESET must be default or expanded")

    keep_condition = os.environ.get("OMINI_KEEP_OMINI_CONDITION")
    glor_config = config.get("glor_control", {})
    if keep_condition is not None and glor_config.get("enable", False):
        normalized = keep_condition.strip().lower()
        if normalized not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError("Invalid OMINI_KEEP_OMINI_CONDITION value")
        glor_config["disable_omini_condition"] = normalized not in {
            "1",
            "true",
            "yes",
        }
        print(
            "Keep original Omini source-season condition: "
            f"{not glor_config['disable_omini_condition']}"
        )


def main():
    config = get_config()
    _apply_overrides(config)
    if config.get("transformer_path"):
        raise ValueError(
            "SSL4EO seasonal training uses the transformer bundled with "
            "flux_path; remove transformer_path from the config."
        )
    print("Base model: original FLUX transformer bundled with flux_path")

    training_config = config["train"]
    dataset_config = training_config["dataset"]
    dataset = SSL4EOSeasonDataset(
        index_path=dataset_config["index_path"],
        root=dataset_config.get("root"),
        image_size=dataset_config.get("image_size", [256, 256]),
        drop_text_prob=dataset_config.get("drop_text_prob", 0.0),
        drop_image_prob=dataset_config.get("drop_image_prob", 0.05),
        random_flip=dataset_config.get("random_flip", True),
        validate_files=dataset_config.get("validate_files", False),
        max_samples=dataset_config.get("max_samples"),
    )

    glor_config = config.get("glor_control", {})
    disable_condition = glor_config.get("enable", False) and glor_config.get(
        "disable_omini_condition", True
    )
    expected_adapter_names = (
        ["text", "main"]
        if disable_condition
        else ["text", "main", "condition"]
    )
    adapter_names = config.get("model", {}).get(
        "adapter_names", expected_adapter_names
    )
    if (
        not isinstance(adapter_names, list)
        or len(adapter_names) != len(expected_adapter_names)
    ):
        raise ValueError(
            "model.adapter_names must match the active SSL4EO branch count "
            f"({len(expected_adapter_names)})"
        )

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    model = OminiModel(
        flux_pipe_id=config["flux_path"],
        transformer_path=None,
        lora_path=os.environ.get("OMINI_LORA_PATH")
        or training_config.get("lora_path"),
        lora_config=training_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        glor_control_config=glor_config,
        glor_task="super_resolution",
        adapter_names=adapter_names,
    )
    train(dataset, model, config, test_function=None)


if __name__ == "__main__":
    main()
