#!/usr/bin/env python3
"""Run one-step FluxSR inference on paired AID samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from diffusers import FluxPipeline, FluxTransformer2DModel

from glor_control import GLoRControl
from omini.pipeline.flux_omini import transformer_forward
from omini.train_flux.aid_fluxsr_ftd import (
    AIDFluxSRDataset,
    decode_packed_to_rgb,
    encode_rgb_to_packed,
    make_comparison_panel,
    tensor_to_pil,
)


DEFAULT_MODEL = PROJECT_ROOT.parent / "assets/models/flux"
DEFAULT_DATASET = PROJECT_ROOT.parent / "assets/datasets/aid"


def parse_args():
    parser = argparse.ArgumentParser(description="Run FluxSR inference on AID image pairs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "train/config/aid_fluxsr_ftd_rsbackbone_glor.yaml",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--transformer-path",
        type=Path,
        default=None,
        help=(
            "Optional custom FLUX transformer directory. If omitted, use "
            "transformer_path from the YAML config when present."
        ),
    )
    parser.add_argument(
        "--glor-checkpoint",
        type=Path,
        default=None,
        help="Optional glor_control.pt override for a GLoR-enabled config.",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/aid_fluxsr_ftd_test",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--visualization-count",
        type=int,
        default=10,
        help="Number of LR/prediction/HR panels to save (default: 10).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Process the first N pairs. Use 0 for the entire split.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.visualization_count < 0 or args.max_samples < 0:
        parser.error("--visualization-count and --max-samples must be non-negative")
    return args


def load_lora(pipe: FluxPipeline, checkpoint: Path):
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.is_dir():
        weight_path = checkpoint / "default.safetensors"
    else:
        weight_path = checkpoint
    if not weight_path.is_file():
        raise FileNotFoundError(f"LoRA weights do not exist: {weight_path}")
    pipe.load_lora_weights(
        str(weight_path.parent),
        weight_name=weight_path.name,
        adapter_name="default",
    )
    pipe.set_adapters(["default"])


def load_glor_control(
    pipe: FluxPipeline,
    config: dict,
    checkpoint: Path,
    override: Path | None,
    device: str,
    dtype: torch.dtype,
) -> GLoRControl | None:
    glor_config = config.get("glor_control", {})
    if not glor_config.get("enable", False):
        if override is not None:
            raise ValueError(
                "--glor-checkpoint was supplied, but the config does not "
                "enable glor_control"
            )
        return None

    checkpoint = checkpoint.expanduser().resolve()
    checkpoint_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
    glor_path = (
        override.expanduser().resolve()
        if override is not None
        else checkpoint_dir / "glor_control.pt"
    )
    if not glor_path.is_file():
        raise FileNotFoundError(
            f"GLoR checkpoint not found: {glor_path}. A GLoR-enabled LoRA "
            "requires its matching glor_control.pt."
        )
    control = GLoRControl.from_config(
        pipe.transformer, {"glor_control": glor_config}
    ).to(device=device, dtype=dtype)
    try:
        payload = torch.load(glor_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(glor_path, map_location="cpu")
    state_dict = (
        payload["glor_control"]
        if isinstance(payload, dict) and "glor_control" in payload
        else payload
    )
    control.load_state_dict(state_dict, strict=True)
    control.eval()
    pipe.transformer.glor_control = control
    print(f"Loaded GLoR-Control: {glor_path}")
    return control


def predict_one(
    pipe: FluxPipeline,
    item: dict,
    t_l: float,
    guidance_scale: float,
    device: str,
    glor_control: GLoRControl | None = None,
) -> torch.Tensor:
    lr_up = item["lr_up"].unsqueeze(0).to(device)
    prompt = [item["prompt"]]
    z_l, latent_ids = encode_rgb_to_packed(pipe, lr_up)
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=256,
        lora_scale=None,
    )
    timestep = torch.full(
        (1,), t_l, device=device, dtype=z_l.dtype
    )
    guidance = (
        torch.full_like(timestep, guidance_scale)
        if pipe.transformer.config.guidance_embeds
        else None
    )
    glor_context = None
    if glor_control is not None:
        glor_dtype = next(glor_control.parameters()).dtype
        lr = item["lr"].unsqueeze(0).to(device=device, dtype=glor_dtype)
        token_height = int(latent_ids[:, 1].max().item()) + 1
        token_width = int(latent_ids[:, 2].max().item()) + 1
        scale = torch.tensor(
            [item["hr"].shape[-1] / item["lr"].shape[-1]],
            device=device,
            dtype=glor_dtype,
        )
        glor_context = glor_control.build_context(
            {"lr_image": lr, "scale": scale},
            task="super_resolution",
            target_latent_shape=(token_height, token_width),
        )
    velocity = transformer_forward(
        pipe.transformer,
        image_features=[z_l],
        text_features=[prompt_embeds],
        img_ids=[latent_ids],
        txt_ids=[text_ids],
        timesteps=[timestep, timestep],
        pooled_projections=[pooled_prompt_embeds, pooled_prompt_embeds],
        guidances=[guidance, guidance],
        adapters=["default", "default"],
        glor_control_context=glor_context,
        return_dict=False,
    )[0]
    z_hat = z_l - t_l * velocity
    return decode_packed_to_rgb(pipe, z_hat, 512, 512)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    t_l = float(config["fluxsr"]["t_l"])
    guidance_scale = float(config["fluxsr"].get("guidance_scale", 3.5))
    dtype = getattr(torch, config.get("dtype", "bfloat16"))
    transformer_path = args.transformer_path
    if transformer_path is None and config.get("transformer_path"):
        transformer_path = PROJECT_ROOT / Path(config["transformer_path"]).expanduser()

    dataset_config = config.get("train", {}).get("dataset", {})
    dataset_root = args.dataset_root or (
        PROJECT_ROOT / Path(dataset_config.get("root", DEFAULT_DATASET)).expanduser()
    )
    model_path = args.model_path or (
        PROJECT_ROOT / Path(config.get("flux_path", DEFAULT_MODEL)).expanduser()
    )
    dataset = AIDFluxSRDataset(
        root=dataset_root,
        splits=[args.split],
        lr_size=int(dataset_config.get("lr_size", 128)),
        hr_size=int(dataset_config.get("hr_size", 512)),
        random_flip=False,
        lr_source=dataset_config.get("lr_source", "paired"),
        downsample_mode=dataset_config.get("downsample_mode", "bicubic"),
    )
    if args.max_samples:
        dataset.samples = dataset.samples[: args.max_samples]
    scale = dataset.hr_size // dataset.lr_size
    print(f"Running x{scale} inference on {len(dataset)} {args.split} pairs.")

    output_dir = args.output_dir.expanduser().resolve()
    prediction_dir = output_dir / "predictions"
    visualization_dir = output_dir / "visualizations"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    if args.visualization_count:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    pipeline_kwargs = {"torch_dtype": dtype}
    if transformer_path is not None:
        transformer_path = transformer_path.expanduser().resolve()
        if not transformer_path.is_dir():
            raise FileNotFoundError(
                f"Custom FLUX transformer directory does not exist: "
                f"{transformer_path}"
            )
        pipeline_kwargs["transformer"] = (
            FluxTransformer2DModel.from_pretrained(
                str(transformer_path),
                torch_dtype=dtype,
            )
        )
        print(f"Using custom FLUX transformer: {transformer_path}")
    else:
        print("Using transformer bundled with the FLUX base pipeline")

    pipe = FluxPipeline.from_pretrained(
        str(model_path.expanduser().resolve()),
        **pipeline_kwargs,
    ).to(args.device)
    load_lora(pipe, args.checkpoint)
    glor_control = load_glor_control(
        pipe,
        config,
        args.checkpoint,
        args.glor_checkpoint,
        args.device,
        dtype,
    )
    pipe.transformer.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()

    manifest = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            sample_id = item["sample_id"]
            output_path = prediction_dir / f"{sample_id}.png"
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Prediction already exists: {output_path}. Pass --overwrite to replace it."
                )

            prediction = predict_one(
                pipe,
                item,
                t_l,
                guidance_scale,
                args.device,
                glor_control=glor_control,
            )[0].float().cpu()
            tensor_to_pil(prediction).save(output_path)
            if index < args.visualization_count:
                panel = make_comparison_panel(item["lr"], prediction, item["hr"])
                panel.save(visualization_dir / f"{index + 1:02d}_{sample_id}.png")

            manifest.append(
                {
                    "sample_id": sample_id,
                    "input": item["lr_path"],
                    "target": item["hr_path"],
                    "prediction": str(output_path),
                }
            )
            print(f"[{index + 1}/{len(dataset)}] Saved {sample_id}")

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Predictions: {prediction_dir}")


if __name__ == "__main__":
    main()
