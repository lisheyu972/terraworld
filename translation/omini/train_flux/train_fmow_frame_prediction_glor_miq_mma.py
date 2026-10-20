"""Train combined temporal GLoR-Control and MIQ-MMA for fMoW prediction."""

import os

import torch

from .train_fmow_frame_prediction import (
    EXPANDED_LORA_TARGET_MODULES,
    FMOW_MIQ_METADATA_DIM,
    FMOWAnyTwoFramesDataset,
    _apply_overrides,
)
from .train_fmow_frame_prediction_glor_control import (
    FMOWFramePredictionGLoRModel,
)
from .trainer import get_config, train


def _enabled_from_environment(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError(f"Invalid {name} value: {value}")
    return normalized in {"1", "true", "yes"}


def main():
    config = get_config()
    training_config = config["train"]
    dataset_config = training_config["dataset"]
    _apply_overrides(training_config)

    transformer_path = os.environ.get("OMINI_TRANSFORMER_PATH")
    if transformer_path:
        transformer_path = os.path.abspath(transformer_path)
        if not os.path.isdir(transformer_path):
            raise FileNotFoundError(transformer_path)
        config["transformer_path"] = transformer_path
        print(f"Training override: transformer_path={transformer_path}")

    preset = os.environ.get("OMINI_LORA_TARGET_PRESET", "default")
    if preset == "expanded":
        training_config["lora_config"]["target_modules"] = (
            EXPANDED_LORA_TARGET_MODULES
        )
        print("LoRA target preset: expanded")
    elif preset != "default":
        raise ValueError(
            "OMINI_LORA_TARGET_PRESET must be default or expanded."
        )

    glor_config = config.get("glor_control", {})
    if not glor_config.get("enable", False):
        raise ValueError(
            "This entry point requires glor_control.enable=true."
        )
    if glor_config.get("task") != "frame_prediction":
        raise ValueError(
            "glor_control.task must be 'frame_prediction'."
        )

    metadata_config = config.get("metadata_qkv", {})
    if not metadata_config.get("enabled", False):
        raise ValueError(
            "This entry point requires metadata_qkv.enabled=true."
        )
    for key in ("condition_metadata_dim", "target_metadata_dim"):
        if int(metadata_config.get(key, -1)) != FMOW_MIQ_METADATA_DIM:
            raise ValueError(
                f"metadata_qkv.{key} must be {FMOW_MIQ_METADATA_DIM} "
                "for the fMoW metadata encoder."
            )
    if not dataset_config.get("include_metadata", False):
        raise ValueError(
            "Combined GLoR+MIQ training requires "
            "train.dataset.include_metadata=true."
        )
    if not dataset_config.get("include_temporal_offsets", False):
        raise ValueError(
            "Combined GLoR+MIQ training requires "
            "train.dataset.include_temporal_offsets=true."
        )

    keep_conditions = _enabled_from_environment(
        "OMINI_KEEP_OMINI_CONDITION",
        default=not bool(glor_config.get("disable_omini_condition", False)),
    )
    if not keep_conditions:
        raise ValueError(
            "MIQ-MMA condition-to-K requires both Omini image-condition "
            "branches; OMINI_KEEP_OMINI_CONDITION must remain true."
        )
    glor_config["disable_omini_condition"] = False
    print("Keep original Omini temporal conditions: True (required by MIQ-MMA)")

    dataset = FMOWAnyTwoFramesDataset(
        index_path=dataset_config["index_path"],
        root=dataset_config.get("root"),
        include_partitions=dataset_config.get(
            "include_partitions", ["train"]
        ),
        source_splits=dataset_config.get("source_splits", ["train"]),
        split_ratios=dataset_config.get("split_ratios"),
        split_seed=dataset_config.get("split_seed", 42),
        image_size=dataset_config.get("image_size", [512, 512]),
        target_mode=dataset_config.get("target_mode", "all"),
        drop_text_prob=dataset_config.get("drop_text_prob", 0.1),
        drop_image_prob=dataset_config.get("drop_image_prob", 0.05),
        random_flip=dataset_config.get("random_flip", True),
        validate_files=dataset_config.get("validate_files", False),
        max_sequences=dataset_config.get("max_sequences"),
        include_metadata=True,
        include_temporal_offsets=True,
    )

    adapter_names = config.get("model", {}).get(
        "adapter_names", ["text", "main", "condition", "condition"]
    )
    if not isinstance(adapter_names, list) or len(adapter_names) != 4:
        raise ValueError(
            "Combined fMoW GLoR+MIQ requires model.adapter_names="
            "[text, main, condition, condition]."
        )

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    model = FMOWFramePredictionGLoRModel(
        flux_pipe_id=config["flux_path"],
        transformer_path=config.get("transformer_path"),
        lora_path=os.environ.get("OMINI_LORA_PATH")
        or training_config.get("lora_path"),
        lora_config=training_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get(
            "gradient_checkpointing", False
        ),
        glor_control_config=glor_config,
        glor_task="frame_prediction",
        metadata_qkv_config=metadata_config,
        adapter_names=adapter_names,
    )
    train(dataset, model, config, test_function=None)


if __name__ == "__main__":
    main()
