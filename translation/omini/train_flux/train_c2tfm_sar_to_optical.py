"""Train C2TFM-Omini for paired 256x256 SAR-to-optical translation."""

import os

import torch

from .c2tfm import add_c2tfm_output_head_targets
from .sar_to_optical_data import SARToOpticalDataset
from .train_sar_to_optical import _apply_overrides
from .trainer import OminiModel, get_config, train


def main():
    config = get_config()
    _apply_overrides(config)
    c2tfm_config = config.get("c2tfm", {})
    if not c2tfm_config.get("enable", False):
        raise ValueError("This entry point requires c2tfm.enable=true.")
    transformer_path = config.get("transformer_path")
    if not transformer_path:
        raise ValueError(
            "C2TFM SAR-to-optical training requires transformer_path."
        )
    print(f"Pretrained DiT transformer: {transformer_path}")
    training_config = config["train"]
    lora_config = training_config["lora_config"]
    lora_config["target_modules"] = add_c2tfm_output_head_targets(
        lora_config["target_modules"]
    )
    print("C2TFM LoRA targets: expanded layers + norm_out.linear + proj_out")

    glor_config = config.get("glor_control", {})
    glor_enabled = bool(glor_config.get("enable", False))
    glor_task = glor_config.get("task", "super_resolution")
    if glor_enabled:
        if glor_task != "super_resolution":
            raise ValueError(
                "C2TFM SAR-to-optical GLoR-Control requires the "
                "'super_resolution' single-condition path."
            )
        if glor_config.get("disable_omini_condition", False):
            raise ValueError(
                "C2TFM-Omini keeps the original SAR condition tokens; set "
                "glor_control.disable_omini_condition=false."
            )
        print("GLoR-Control: enabled with the original SAR condition branch")

    dataset_config = training_config["dataset"]
    if float(dataset_config.get("drop_image_prob", 0.0)) != 0.0:
        raise ValueError(
            "C2TFM requires dataset.drop_image_prob=0 because the SAR image is "
            "the main flow source endpoint."
        )
    dataset = SARToOpticalDataset(
        root=dataset_config["root"],
        include_splits=dataset_config.get("include_splits", ["train"]),
        image_size=dataset_config.get("image_size", [256, 256]),
        drop_text_prob=dataset_config.get("drop_text_prob", 0.1),
        drop_image_prob=dataset_config.get("drop_image_prob", 0.0),
        random_flip=dataset_config.get("random_flip", True),
    )

    adapter_names = config.get("model", {}).get(
        "adapter_names", ["text", "main", "condition"]
    )
    if not isinstance(adapter_names, list) or len(adapter_names) != 3:
        raise ValueError(
            "C2TFM SAR-to-optical training requires model.adapter_names="
            "[text, main, condition]."
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
        c2tfm_config=c2tfm_config,
        glor_control_config=glor_config,
        glor_task=glor_task,
        adapter_names=adapter_names,
    )
    train(dataset, model, config, test_function=None)


if __name__ == "__main__":
    main()
