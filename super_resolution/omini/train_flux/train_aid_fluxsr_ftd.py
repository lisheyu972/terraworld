"""Train one-step AID super-resolution with FluxSR FTD and SwanLab."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import lightning as L
import swanlab
import torch
import yaml
from torch.utils.data import DataLoader

from .aid_fluxsr_ftd import (
    AIDFluxSRDataset,
    AIDFluxSRFTDModel,
    FLUXSR_LORA_TARGET_MODULES,
    make_comparison_panel,
)
from .trainer import get_config, get_rank


SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY")


def _collate_single(item):
    batch = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = [value]
    return batch


class FluxSRCheckpointCallback(L.Callback):
    def __init__(
        self,
        run_dir: Path,
        preview_dataset: AIDFluxSRDataset,
        save_interval: int,
        log_interval: int,
        sample_count: int,
        use_swanlab: bool,
    ):
        self.run_dir = run_dir
        self.preview_dataset = preview_dataset
        self.save_interval = int(save_interval)
        self.log_interval = int(log_interval)
        self.sample_count = int(sample_count)
        self.use_swanlab = use_swanlab
        self.last_seen_step = -1
        self.last_saved_step = -1

    def _save_checkpoint_and_samples(self, trainer, model, step: int):
        checkpoint_dir = self.run_dir / "ckpt" / str(step)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_lora(str(checkpoint_dir))
        state = {
            "global_step": step,
            "epoch": int(trainer.current_epoch),
            "t_l": model.t_l,
            "method": "FluxSR Flow Trajectory Distillation",
            "checkpoint_type": (
                "LoRA + GLoR-Control"
                if model.glor_control is not None
                else "LoRA"
            ),
            "glor_control_enabled": model.glor_control is not None,
            "scale_factor": int(
                self.preview_dataset.hr_size / self.preview_dataset.lr_size
            ),
            "lr_source": self.preview_dataset.lr_source,
        }
        with (checkpoint_dir / "fluxsr_state.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        sample_dir = self.run_dir / "samples" / f"step_{step:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        swanlab_images = []
        was_training = model.training
        model.eval()
        try:
            for index in range(min(self.sample_count, len(self.preview_dataset))):
                item = self.preview_dataset[index]
                batch = _collate_single(item)
                prediction = model.generate_one_step(batch)[0].cpu()
                panel = make_comparison_panel(
                    item["lr"], prediction, item["hr"]
                )
                output_path = sample_dir / f"{item['sample_id']}.png"
                panel.save(output_path)
                if self.use_swanlab:
                    swanlab_images.append(
                        swanlab.Image(
                            str(output_path),
                            caption=f"step={step}, sample={item['sample_id']}",
                        )
                    )
        finally:
            if was_training:
                model.train()
                model.lpips_model.eval()

        if self.use_swanlab:
            swanlab.log(
                {
                    "checkpoint/step": step,
                    "checkpoint/example_images": swanlab_images,
                },
                step=step,
            )
        print(f"Saved FluxSR checkpoint and samples: {checkpoint_dir}")
        self.last_saved_step = step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        if step <= 0 or step == self.last_seen_step:
            return
        self.last_seen_step = step
        if self.use_swanlab and step % self.log_interval == 0:
            optimizer_groups = trainer.optimizers[0].param_groups
            learning_rate = optimizer_groups[0]["lr"]
            metrics = dict(pl_module.last_loss_components)
            metrics.update(
                {
                    "train/epoch": float(trainer.current_epoch),
                    "train/learning_rate": float(learning_rate),
                }
            )
            for index, group in enumerate(optimizer_groups):
                group_name = str(group.get("name", f"group_{index}"))
                metrics[f"train/learning_rate_{group_name}"] = float(
                    group["lr"]
                )
            swanlab.log(metrics, step=step)
        if step % self.save_interval == 0:
            self._save_checkpoint_and_samples(trainer, pl_module, step)

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        if step > 0 and step != self.last_saved_step:
            self._save_checkpoint_and_samples(trainer, pl_module, step)
        if self.use_swanlab:
            swanlab.finish()


def apply_environment_overrides(config: dict):
    train_config = config["train"]
    integer_options = {
        "OMINI_BATCH_SIZE": "batch_size",
        "OMINI_ACCUMULATE_GRAD_BATCHES": "accumulate_grad_batches",
        "OMINI_MAX_STEPS": "max_steps",
        "OMINI_SAVE_INTERVAL": "save_interval",
        "OMINI_DATALOADER_WORKERS": "dataloader_workers",
        "OMINI_NUM_DEVICES": "devices",
    }
    for environment_name, config_name in integer_options.items():
        value = os.environ.get(environment_name)
        if value is not None:
            train_config[config_name] = int(value)
    boolean_options = {
        "OMINI_GRADIENT_CHECKPOINTING": "gradient_checkpointing",
        "OMINI_ALLOW_TF32": "allow_tf32",
        "OMINI_CUDNN_BENCHMARK": "cudnn_benchmark",
    }
    for environment_name, config_name in boolean_options.items():
        value = os.environ.get(environment_name)
        if value is None:
            continue
        normalized = value.strip().lower()
        if normalized not in {"1", "0", "true", "false", "yes", "no"}:
            raise ValueError(
                f"{environment_name} must be one of 1/0/true/false/yes/no"
            )
        train_config[config_name] = normalized in {"1", "true", "yes"}
    if os.environ.get("OMINI_SAVE_PATH"):
        train_config["save_path"] = os.environ["OMINI_SAVE_PATH"]
    if os.environ.get("OMINI_INIT_LORA_PATH"):
        train_config["init_lora_path"] = os.environ["OMINI_INIT_LORA_PATH"]
    if os.environ.get("OMINI_TRANSFORMER_PATH"):
        config["transformer_path"] = os.environ["OMINI_TRANSFORMER_PATH"]


def main():
    config = get_config()
    apply_environment_overrides(config)
    train_config = config["train"]
    dataset_config = train_config["dataset"]
    rank = get_rank()
    stage_name = str(train_config.get("stage", "stage1"))
    transformer_path = config.get("transformer_path")
    if transformer_path:
        transformer_dir = Path(transformer_path).expanduser()
        if not transformer_dir.is_dir():
            raise FileNotFoundError(
                f"Custom FLUX transformer directory does not exist: "
                f"{transformer_dir}"
            )
        transformer_path = str(transformer_dir.resolve())
        config["transformer_path"] = transformer_path
    init_lora_path = train_config.get("init_lora_path")
    if train_config.get("require_init_lora", False) and not init_lora_path:
        raise ValueError(
            "Stage-2 training requires OMINI_INIT_LORA_PATH or "
            "train.init_lora_path pointing to a stage-1 LoRA checkpoint."
        )
    if init_lora_path:
        checkpoint_path = Path(init_lora_path).expanduser()
        weight_path = (
            checkpoint_path / "default.safetensors"
            if checkpoint_path.is_dir()
            else checkpoint_path
        )
        if not weight_path.is_file():
            raise FileNotFoundError(
                f"Initial LoRA checkpoint does not exist: {weight_path}"
            )
        init_lora_path = str(checkpoint_path.resolve())
        train_config["init_lora_path"] = init_lora_path

    optimizer_params = train_config.get("optimizer", {}).get("params", {})
    gradient_clip_val = float(train_config.get("gradient_clip_val", 0.0))
    if optimizer_params.get("fused", False) and gradient_clip_val > 0.0:
        if rank == 0:
            print(
                "Compatibility notice: disabling fused AdamW because Lightning "
                "AMP gradient clipping is enabled."
            )
        optimizer_params["fused"] = False

    torch.cuda.set_device(rank)
    torch.set_float32_matmul_precision(
        train_config.get("matmul_precision", "high")
    )
    torch.backends.cuda.matmul.allow_tf32 = bool(
        train_config.get("allow_tf32", True)
    )
    torch.backends.cudnn.allow_tf32 = bool(
        train_config.get("allow_tf32", True)
    )
    torch.backends.cudnn.benchmark = bool(
        train_config.get("cudnn_benchmark", True)
    )

    train_dataset = AIDFluxSRDataset(
        root=dataset_config["root"],
        splits=dataset_config.get("train_splits", ["train"]),
        lr_size=dataset_config.get("lr_size", 128),
        hr_size=dataset_config.get("hr_size", 512),
        random_flip=dataset_config.get("random_flip", True),
        lr_source=dataset_config.get("lr_source", "paired"),
        downsample_mode=dataset_config.get("downsample_mode", "bicubic"),
    )
    preview_dataset = AIDFluxSRDataset(
        root=dataset_config["root"],
        splits=dataset_config.get("preview_splits", ["test"]),
        lr_size=dataset_config.get("lr_size", 128),
        hr_size=dataset_config.get("hr_size", 512),
        random_flip=False,
        lr_source=dataset_config.get("lr_source", "paired"),
        downsample_mode=dataset_config.get("downsample_mode", "bicubic"),
    )
    if rank == 0:
        print(
            f"AID split summary: train={len(train_dataset)}, "
            f"preview/test={len(preview_dataset)}"
        )
        print(
            f"SR degradation: source={train_dataset.lr_source}, "
            f"downsample={train_dataset.downsample_mode}, "
            f"scale=x{train_dataset.hr_size // train_dataset.lr_size}, "
            f"LR={train_dataset.lr_size}, HR={train_dataset.hr_size}"
        )
        print(f"FLUX base pipeline: {config['flux_path']}")
        print(
            "FLUX transformer: "
            f"{transformer_path or 'bundled transformer from flux_path'}"
        )
    worker_count = int(train_config.get("dataloader_workers", 8))
    loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": int(train_config.get("batch_size", 1)),
        "shuffle": True,
        "num_workers": worker_count,
        "pin_memory": True,
        "persistent_workers": worker_count > 0,
        "drop_last": True,
    }
    if worker_count > 0:
        loader_kwargs["prefetch_factor"] = int(
            train_config.get("prefetch_factor", 4)
        )
    loader = DataLoader(
        **loader_kwargs,
    )

    train_config["lora_config"]["target_modules"] = FLUXSR_LORA_TARGET_MODULES
    model = AIDFluxSRFTDModel(
        flux_pipe_id=config["flux_path"],
        transformer_path=transformer_path,
        lora_path=init_lora_path,
        lora_config=None if init_lora_path else train_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config.get("dtype", "bfloat16")),
        optimizer_config=train_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        adapter_names=["default", "default"],
        fluxsr_config=config["fluxsr"],
        glor_control_config=config.get("glor_control", {}),
        glor_task="super_resolution",
        image_size=dataset_config.get("hr_size", 512),
    )

    run_prefix = train_config.get("run_name_prefix", f"aid-fluxsr-ftd-{stage_name}")
    run_name = os.environ.get(
        "OMINI_RUN_NAME", time.strftime(f"{run_prefix}-%Y%m%d-%H%M%S")
    )
    run_dir = Path(train_config.get("save_path", "runs/aid_fluxsr_ftd")) / run_name
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=False)
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        if SWANLAB_API_KEY:
            swanlab.login(api_key=SWANLAB_API_KEY)
            swanlab.init(
                project=train_config.get("swanlab", {}).get(
                    "project", "AID-FluxSR-FTD"
                ),
                experiment_name=run_name,
                config=config,
                logdir=str(run_dir / "swanlog"),
                mode="online",
            )
        callbacks = [
            FluxSRCheckpointCallback(
                run_dir=run_dir,
                preview_dataset=preview_dataset,
                save_interval=int(train_config.get("save_interval", 5000)),
                log_interval=int(train_config.get("log_interval", 10)),
                sample_count=int(train_config.get("sample_count", 4)),
                use_swanlab=bool(SWANLAB_API_KEY),
            )
        ]
    else:
        callbacks = []

    devices = int(train_config.get("devices", 2))
    strategy = (
        train_config.get("strategy", "ddp_find_unused_parameters_false")
        if devices > 1
        else "auto"
    )
    if rank == 0:
        effective_batch = (
            int(train_config.get("batch_size", 1))
            * devices
            * int(train_config.get("accumulate_grad_batches", 1))
        )
        print(
            f"Training topology: devices={devices}, strategy={strategy}, "
            f"effective_global_batch={effective_batch}, "
            f"gradient_checkpointing={train_config.get('gradient_checkpointing', False)}"
        )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        precision=train_config.get("precision", "bf16-mixed"),
        max_steps=int(train_config.get("max_steps", 20000)),
        max_epochs=-1,
        accumulate_grad_batches=int(
            train_config.get("accumulate_grad_batches", 1)
        ),
        gradient_clip_val=float(train_config.get("gradient_clip_val", 1.0)),
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=False,
        log_every_n_steps=int(train_config.get("log_interval", 10)),
    )
    trainer.fit(model, loader)


if __name__ == "__main__":
    main()
