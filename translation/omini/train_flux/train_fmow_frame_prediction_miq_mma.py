"""Train MIQ-MMA OminiControl for fMoW any-two-frames prediction."""

import os

import torch

from .train_fmow_frame_prediction import (
    EXPANDED_LORA_TARGET_MODULES,
    FMOW_MIQ_METADATA_DIM,
    FMOWAnyTwoFramesDataset,
    _apply_overrides,
)
from .trainer import OminiModel, get_config, train


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
            "fMoW MIQ-MMA requires train.dataset.include_metadata=true."
        )

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
    )

    adapter_names = config.get("model", {}).get(
        "adapter_names", ["text", "main", "condition", "condition"]
    )
    if not isinstance(adapter_names, list) or len(adapter_names) != 4:
        raise ValueError(
            "fMoW MIQ-MMA requires model.adapter_names="
            "[text, main, condition, condition]."
        )

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    model = OminiModel(
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
        metadata_qkv_config=metadata_config,
        adapter_names=adapter_names,
    )
    train(dataset, model, config, test_function=None)


if __name__ == "__main__":
    main()
