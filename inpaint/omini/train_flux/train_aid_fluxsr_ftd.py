"""Train one-step AID x4 super-resolution with FluxSR FTD and SwanLab."""

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


SWANLAB_API_KEY = "5wQrgI0ZQg0OeeqwHU5o4"


def _collate_single(item):
    batch = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = [value]
    return batch


class FluxSRSwanLabCallback(L.Callback):
    def __init__(
        self,
        run_dir: Path,
        preview_dataset: AIDFluxSRDataset,
        save_interval: int,
        log_interval: int,
        sample_count: int,
    ):
        self.run_dir = run_dir
        self.preview_dataset = preview_dataset
        self.save_interval = int(save_interval)
        self.log_interval = int(log_interval)
        self.sample_count = int(sample_count)
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
            "checkpoint_type": "LoRA",
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
        if step % self.log_interval == 0:
            learning_rate = trainer.optimizers[0].param_groups[0]["lr"]
            metrics = dict(pl_module.last_loss_components)
            metrics.update(
                {
                    "train/epoch": float(trainer.current_epoch),
                    "train/learning_rate": float(learning_rate),
                }
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
        swanlab.finish()


def apply_environment_overrides(config: dict):
    train_config = config["train"]
    integer_options = {
        "OMINI_BATCH_SIZE": "batch_size",
        "OMINI_ACCUMULATE_GRAD_BATCHES": "accumulate_grad_batches",
        "OMINI_MAX_STEPS": "max_steps",
        "OMINI_SAVE_INTERVAL": "save_interval",
        "OMINI_DATALOADER_WORKERS": "dataloader_workers",
    }
    for environment_name, config_name in integer_options.items():
        value = os.environ.get(environment_name)
        if value is not None:
            train_config[config_name] = int(value)
    if os.environ.get("OMINI_SAVE_PATH"):
        train_config["save_path"] = os.environ["OMINI_SAVE_PATH"]


def main():
    config = get_config()
    apply_environment_overrides(config)
    train_config = config["train"]
    dataset_config = train_config["dataset"]
    rank = get_rank()
    torch.cuda.set_device(rank)

    train_dataset = AIDFluxSRDataset(
        root=dataset_config["root"],
        splits=dataset_config.get("train_splits", ["train"]),
        lr_size=dataset_config.get("lr_size", 128),
        hr_size=dataset_config.get("hr_size", 512),
        random_flip=dataset_config.get("random_flip", True),
    )
    preview_dataset = AIDFluxSRDataset(
        root=dataset_config["root"],
        splits=dataset_config.get("preview_splits", ["test"]),
        lr_size=dataset_config.get("lr_size", 128),
        hr_size=dataset_config.get("hr_size", 512),
        random_flip=False,
    )
    if rank == 0:
        print(
            f"AID split summary: train={len(train_dataset)}, "
            f"preview/test={len(preview_dataset)}"
        )
    loader = DataLoader(
        train_dataset,
        batch_size=int(train_config.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(train_config.get("dataloader_workers", 4)),
        pin_memory=True,
        persistent_workers=int(train_config.get("dataloader_workers", 4)) > 0,
        drop_last=True,
    )

    train_config["lora_config"]["target_modules"] = FLUXSR_LORA_TARGET_MODULES
    model = AIDFluxSRFTDModel(
        flux_pipe_id=config["flux_path"],
        transformer_path=None,
        lora_config=train_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config.get("dtype", "bfloat16")),
        optimizer_config=train_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        adapter_names=["default", "default"],
        fluxsr_config=config["fluxsr"],
        image_size=dataset_config.get("hr_size", 512),
    )

    run_name = os.environ.get(
        "OMINI_RUN_NAME", time.strftime("aid-fluxsr-ftd-%Y%m%d-%H%M%S")
    )
    run_dir = Path(train_config.get("save_path", "runs/aid_fluxsr_ftd")) / run_name
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=False)
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        swanlab.login(api_key=SWANLAB_API_KEY)
        swanlab.init(
            project=train_config.get("swanlab", {}).get(
                "project", "AID-FluxSR-FTD"
            ),
            experiment_name=run_name,
            description=(
                "FLUX.1-dev one-step AID x4 super-resolution using paired-data "
                "Flow Trajectory Distillation, TV-LPIPS, and ADL."
            ),
            config=config,
            logdir=str(run_dir / "swanlog"),
            mode="online",
        )
        callbacks = [
            FluxSRSwanLabCallback(
                run_dir=run_dir,
                preview_dataset=preview_dataset,
                save_interval=int(train_config.get("save_interval", 5000)),
                log_interval=int(train_config.get("log_interval", 10)),
                sample_count=int(train_config.get("sample_count", 4)),
            )
        ]
    else:
        callbacks = []

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
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
