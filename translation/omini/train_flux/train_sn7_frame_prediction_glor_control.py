"""Train SN7 t0,t1 -> t2 using the unchanged fMOW model and image reader."""

import os
from contextlib import redirect_stdout
from io import StringIO

import torch

from .train_fmow_frame_prediction_glor_control import (
    FMOWAnyTwoFramesGLoRDataset,
    FMOWFramePredictionGLoRModel,
    _apply_overrides,
)
from .train_frame_interp import EXPANDED_LORA_TARGET_MODULES
from .condition_shift import RandomConditionShiftDataset
from .trainer import get_config, train


class SN7LastFrameDataset(FMOWAnyTwoFramesGLoRDataset):
    """Reuse the fMOW sample reader, choosing only the final frame as target."""

    def __init__(self, *args, target_mode="last", **kwargs):
        if target_mode != "last":
            raise ValueError("SN7 frame prediction requires target_mode='last'.")
        # The base loader accepts 'middle' for one sample per triplet. Its
        # __getitem__ calls our target-slot method below, so t2 is the target.
        with redirect_stdout(StringIO()):
            super().__init__(*args, target_mode="middle", **kwargs)
        self.target_mode = "last"
        print(f"SN7 sequences: {len(self.samples)}; target mode: last; effective examples: {len(self.samples)}.")

    def _row_and_target_slot(self, index):
        return self.samples[index], 2


def main():
    config = get_config()
    training_config = config["train"]
    dataset_config = training_config["dataset"]
    _apply_overrides(training_config)
    if os.environ.get("OMINI_LORA_TARGET_PRESET", "default") == "expanded":
        training_config["lora_config"]["target_modules"] = EXPANDED_LORA_TARGET_MODULES
        print("LoRA target preset: expanded")

    glor_config = config.get("glor_control", {})
    if not glor_config.get("enable", False):
        raise ValueError("This entry point requires glor_control.enable=true.")
    keep_condition = os.environ.get("OMINI_KEEP_OMINI_CONDITION", "true")
    normalized = keep_condition.strip().lower()
    if normalized not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError("Invalid OMINI_KEEP_OMINI_CONDITION value.")
    glor_config["disable_omini_condition"] = normalized not in {"1", "true", "yes"}
    print(f"Keep original Omini temporal conditions: {not glor_config['disable_omini_condition']}")

    dataset = SN7LastFrameDataset(
        index_path=dataset_config["index_path"],
        root=dataset_config.get("root"),
        include_partitions=dataset_config.get("include_partitions", ["train"]),
        source_splits=dataset_config.get("source_splits", ["train"]),
        split_ratios=dataset_config.get("split_ratios"),
        split_seed=dataset_config.get("split_seed", 42),
        image_size=dataset_config.get("image_size", [512, 512]),
        target_mode=dataset_config.get("target_mode", "last"),
        drop_text_prob=dataset_config.get("drop_text_prob", 0.1),
        drop_image_prob=dataset_config.get("drop_image_prob", 0.05),
        random_flip=dataset_config.get("random_flip", True),
        validate_files=dataset_config.get("validate_files", False),
        max_sequences=dataset_config.get("max_sequences"),
    )
    shift_pixels = int(dataset_config.get("condition_shift_pixels", 0))
    if shift_pixels:
        dataset = RandomConditionShiftDataset(dataset, shift_pixels)
        print(f"Unknown context translations: up to {shift_pixels} pixels each.")

    expected_adapter_names = (
        ["text", "main"] if glor_config["disable_omini_condition"]
        else ["text", "main", "condition", "condition"]
    )
    adapter_names = config.get("model", {}).get("adapter_names", expected_adapter_names)
    if len(adapter_names) != len(expected_adapter_names):
        raise ValueError(
            "model.adapter_names must match the active branch count "
            f"({len(expected_adapter_names)})."
        )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    model = FMOWFramePredictionGLoRModel(
        flux_pipe_id=config["flux_path"],
        transformer_path=config.get("transformer_path"),
        lora_path=os.environ.get("OMINI_LORA_PATH") or training_config.get("lora_path"),
        lora_config=training_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        glor_control_config=glor_config,
        glor_task="frame_prediction",
        adapter_names=adapter_names,
    )
    train(dataset, model, config, test_function=None)


if __name__ == "__main__":
    main()
