import lightning as L
from diffusers import FluxTransformer2DModel
from diffusers.pipelines import FluxPipeline
import torch
import torch.nn.functional as F
import wandb
import os
import yaml
from peft import LoraConfig, get_peft_model_state_dict
from torch.utils.data import DataLoader
import time

from typing import List

import prodigyopt

from ..pipeline.flux_omini import transformer_forward, encode_images
from toa_control import TargetObservationControlField, configure_toa_control
from glor_control import GLoRControl
from scb_control import (
    SoftCorrespondenceBiasBuilder,
    configure_scb_control,
    scb_state_dict,
)
from observation_residual import (
    configure_observation_residual,
    observation_residual_state_dict,
)
from metadata_qkv import configure_metadata_qkv


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("OMINI_CONFIG")
    assert config_path is not None, "Please set the OMINI_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        transformer_path: str = None,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        adapter_names: List[str] = [None, None, "default"],
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        anchor_flow_config: dict = None,
        toa_control_config: dict = None,
        glor_control_config: dict = None,
        scb_control_config: dict = None,
        observation_residual_config: dict = None,
        metadata_qkv_config: dict = None,
        toa_task: str = "super_resolution",
        glor_task: str = "super_resolution",
        scb_task: str = "super_resolution",
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.anchor_flow_config = anchor_flow_config or {}
        self.toa_control_config = toa_control_config or {}
        self.glor_control_config = glor_control_config or {}
        self.scb_control_config = scb_control_config or {}
        self.observation_residual_config = observation_residual_config or {}
        self.metadata_qkv_config = metadata_qkv_config or {}
        self.toa_task = toa_task
        self.glor_task = glor_task
        self.scb_task = scb_task
        glor_task_config = self.glor_control_config.get("tasks", {}).get(
            self.glor_task, {}
        )
        self.inpaint_mask_loss_weight = float(
            glor_task_config.get("mask_loss_weight", 0.0)
        )
        if self.inpaint_mask_loss_weight < 0:
            raise ValueError("mask_loss_weight must be non-negative.")
        self.toa_disable_omini_condition = bool(
            self.toa_control_config.get("disable_omini_condition", True)
        )
        self.glor_disable_omini_condition = bool(
            self.glor_control_config.get("disable_omini_condition", True)
        )

        # Load the Flux pipeline
        pipeline_kwargs = {"torch_dtype": dtype}
        if transformer_path:
            pipeline_kwargs["transformer"] = FluxTransformer2DModel.from_pretrained(
                transformer_path, torch_dtype=dtype
            )
        self.flux_pipe: FluxPipeline = FluxPipeline.from_pretrained(
            flux_pipe_id,
            **pipeline_kwargs,
        ).to(device)
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()
        self.adapter_names = adapter_names
        self.adapter_set = set([each for each in adapter_names if each is not None])

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)
        self.toa_control_field = None
        self.toa_control_injector = None
        if self.toa_control_config.get("enable", False):
            self.toa_control_field = TargetObservationControlField.from_config(
                {"toa_control": self.toa_control_config}
            )
            self.toa_control_injector = configure_toa_control(
                self.transformer, self.toa_control_config, device=device, dtype=dtype
            )
        self.glor_control = None
        if self.glor_control_config.get("enable", False):
            self.glor_control = GLoRControl.from_config(
                self.transformer, {"glor_control": self.glor_control_config}
            )
            self.glor_control.to(device=device, dtype=dtype)
            self.transformer.glor_control = self.glor_control
        self.scb_bias_builder = None
        self.scb_gamma_params = []
        if self.scb_control_config.get("enable", False):
            self.scb_bias_builder = SoftCorrespondenceBiasBuilder(
                self.scb_control_config
            )
            self.scb_gamma_params = configure_scb_control(
                self.transformer, self.scb_control_config, device=device, dtype=dtype
            )
        self.observation_residual_params = configure_observation_residual(
            self.transformer,
            self.observation_residual_config,
            device=device,
            dtype=dtype,
        )
        metadata_checkpoint_dir = (
            lora_path if lora_path and os.path.isdir(lora_path) else None
        )
        self.metadata_qkv_injector = configure_metadata_qkv(
            self.transformer,
            self.metadata_qkv_config,
            device=device,
            dtype=dtype,
            checkpoint_dir=metadata_checkpoint_dir,
        )

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            if not self.adapter_set:
                raise ValueError("lora_path was provided, but no adapter names are active.")
            lora_path = os.path.abspath(lora_path)
            for adapter_name in self.adapter_set:
                if os.path.isdir(lora_path):
                    weight_name = f"{adapter_name}.safetensors"
                    if not os.path.isfile(os.path.join(lora_path, weight_name)):
                        if adapter_name == "default" and os.path.isfile(
                            os.path.join(lora_path, "default.safetensors")
                        ):
                            weight_name = "default.safetensors"
                        else:
                            raise FileNotFoundError(os.path.join(lora_path, weight_name))
                    self.flux_pipe.load_lora_weights(
                        lora_path,
                        weight_name=weight_name,
                        adapter_name=adapter_name,
                    )
                else:
                    if len(self.adapter_set) != 1:
                        raise ValueError(
                            "A single LoRA file can only be loaded when exactly one "
                            "adapter is active. Use a checkpoint directory for multiple adapters."
                        )
                    self.flux_pipe.load_lora_weights(
                        os.path.dirname(lora_path),
                        weight_name=os.path.basename(lora_path),
                        adapter_name=adapter_name,
                    )
            self.flux_pipe.set_adapters(sorted(self.adapter_set))
            lora_layers = [
                param
                for name, param in self.transformer.named_parameters()
                if "lora_" in name
            ]
            if not lora_layers:
                raise RuntimeError(f"No LoRA parameters were loaded from {lora_path}.")
            print(
                f"Loaded LoRA from {lora_path} for adapters: "
                f"{', '.join(sorted(self.adapter_set))}"
            )
        else:
            for adapter_name in self.adapter_set:
                self.transformer.add_adapter(
                    LoraConfig(**lora_config), adapter_name=adapter_name
                )
            if self.adapter_set:
                self.flux_pipe.set_adapters(sorted(self.adapter_set))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        for adapter_name in self.adapter_set:
            FluxPipeline.save_lora_weights(
                save_directory=path,
                weight_name=f"{adapter_name}.safetensors",
                transformer_lora_layers=get_peft_model_state_dict(
                    self.transformer, adapter_name=adapter_name
                ),
                safe_serialization=True,
            )
        if self.toa_control_field is not None and self.toa_control_injector is not None:
            os.makedirs(path, exist_ok=True)
            torch.save(
                {
                    "control_field": self.toa_control_field.state_dict(),
                    "injector": self.toa_control_injector.state_dict(),
                },
                os.path.join(path, "toa_control.pt"),
            )
        if self.glor_control is not None:
            os.makedirs(path, exist_ok=True)
            torch.save(
                {"glor_control": self.glor_control.state_dict()},
                os.path.join(path, "glor_control.pt"),
            )
        if self.scb_bias_builder is not None:
            os.makedirs(path, exist_ok=True)
            torch.save(
                {"scb_control": scb_state_dict(self.transformer)},
                os.path.join(path, "scb_control.pt"),
            )
        if self.observation_residual_params:
            os.makedirs(path, exist_ok=True)
            torch.save(
                {
                    "observation_residual": observation_residual_state_dict(
                        self.transformer
                    )
                },
                os.path.join(path, "observation_residual.pt"),
            )
        if self.metadata_qkv_injector is not None:
            os.makedirs(path, exist_ok=True)
            torch.save(
                self.metadata_qkv_injector.checkpoint_payload(),
                os.path.join(path, "metadata_qkv.pt"),
            )

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = list(self.lora_layers)
        if self.toa_control_field is not None:
            self.trainable_params.extend(self.toa_control_field.parameters())
        if self.toa_control_injector is not None:
            self.trainable_params.extend(self.toa_control_injector.parameters())
        if self.glor_control is not None:
            self.trainable_params.extend(self.glor_control.parameters())
        self.trainable_params.extend(self.scb_gamma_params)
        self.trainable_params.extend(self.observation_residual_params)
        if self.metadata_qkv_injector is not None:
            self.trainable_params.extend(
                self.metadata_qkv_injector.trainable_parameters()
            )

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError("Optimizer not implemented.")
        return optimizer

    def _target_latent_shape(self, img_ids):
        height = int(img_ids[:, 1].max().item()) + 1
        width = int(img_ids[:, 2].max().item()) + 1
        return height, width

    def _build_toa_control_field(self, batch, conditions, img_ids, image_latent_mask):
        if self.toa_control_field is None:
            return None
        if not conditions:
            raise KeyError("TOA-Control requires at least one image condition.")
        dtype = next(self.toa_control_field.parameters()).dtype
        scale = batch.get("position_scale_0", 1.0)
        if torch.is_tensor(scale):
            scale = scale.to(self.device)
        ctrl_field = self.toa_control_field(
            {
                "lr_image": conditions[0].to(device=self.device, dtype=dtype),
                "scale": scale,
            },
            task=self.toa_task,
            target_latent_shape=self._target_latent_shape(img_ids),
        )
        if image_latent_mask is not None:
            ctrl_field = ctrl_field[:, image_latent_mask[0]]
        return ctrl_field

    def _build_glor_control_context(self, batch, conditions, img_ids, image_latent_mask):
        if self.glor_control is None:
            return None
        if not conditions:
            raise KeyError("GLoR-Control requires at least one image condition.")
        dtype = next(self.glor_control.parameters()).dtype
        scale = batch.get("position_scale_0", 1.0)
        if torch.is_tensor(scale):
            scale = scale.to(self.device)
        control_batch = {
            "lr_image": conditions[0].to(device=self.device, dtype=dtype),
            "scale": scale,
        }
        if self.glor_task == "inpainting":
            inpaint_mask = batch.get("inpaint_mask")
            if inpaint_mask is None:
                raise KeyError("Inpainting GLoR requires batch['inpaint_mask'].")
            inpaint_mask = inpaint_mask.to(device=self.device, dtype=dtype)
            control_batch.update(
                {
                    "inpaint_image": torch.cat(
                        [control_batch["lr_image"], inpaint_mask], dim=1
                    ),
                    "inpaint_mask": inpaint_mask,
                }
            )
        context = self.glor_control.build_context(
            control_batch,
            task=self.glor_task,
            target_latent_shape=self._target_latent_shape(img_ids),
        )
        if image_latent_mask is not None:
            context = {
                key: (
                    value[:, image_latent_mask[0]]
                    if key in {"local_tokens", "local_valid_mask", "target_mask"}
                    else value
                )
                for key, value in context.items()
            }
        return context

    def _build_inpaint_loss_mask(self, batch, img_ids, image_latent_mask):
        if self.glor_task != "inpainting" or self.inpaint_mask_loss_weight <= 0:
            return None
        inpaint_mask = batch.get("inpaint_mask")
        if inpaint_mask is None:
            raise KeyError("Mask-weighted inpainting loss requires inpaint_mask.")
        token_mask = F.interpolate(
            inpaint_mask.to(device=self.device, dtype=torch.float32),
            size=self._target_latent_shape(img_ids),
            mode="area",
        ).flatten(2).transpose(1, 2).contiguous()
        if image_latent_mask is not None:
            token_mask = token_mask[:, image_latent_mask[0]]
        return token_mask

    def _condition_latent_shape(self, condition):
        token_stride = int(self.flux_pipe.vae_scale_factor) * 2
        height = int(condition.shape[-2]) // token_stride
        width = int(condition.shape[-1]) // token_stride
        if height < 1 or width < 1:
            raise ValueError(
                f"Condition image is too small for packed latent stride {token_stride}."
            )
        return height, width

    def _build_scb_bias(self, batch, conditions, img_ids):
        if self.scb_bias_builder is None:
            return None
        if not conditions:
            raise KeyError("SCB-Control requires at least one image condition.")
        scale = batch.get("position_scale_0", 1.0)
        if torch.is_tensor(scale):
            scale = scale.to(self.device)
        return self.scb_bias_builder(
            task=self.scb_task,
            target_hw=self._target_latent_shape(img_ids),
            condition_hw=self._condition_latent_shape(conditions[0]),
            batch={
                "scale": scale,
                "position_scale": scale,
                "batch_size": int(conditions[0].shape[0]),
            },
            device=self.device,
            dtype=self.dtype,
        )

    def training_step(self, batch, batch_idx):
        imgs, prompts = batch["image"], batch["description"]
        prompts_2 = batch.get("description_2")
        image_latent_mask = batch.get("image_latent_mask", None)

        # Get the conditions and position deltas from the batch
        conditions, position_deltas, position_scales, latent_masks = [], [], [], []
        for i in range(1000):
            if f"condition_{i}" not in batch:
                break
            conditions.append(batch[f"condition_{i}"])
            position_deltas.append(batch.get(f"position_delta_{i}", [[0, 0]]))
            position_scales.append(batch.get(f"position_scale_{i}", [1.0])[0])
            latent_masks.append(batch.get(f"condition_latent_mask_{i}", None))

        use_toa_control = self.toa_control_field is not None
        use_glor_control = self.glor_control is not None
        disable_omini_condition = (
            (use_toa_control and self.toa_disable_omini_condition)
            or (use_glor_control and self.glor_disable_omini_condition)
        )
        model_conditions = (
            [] if disable_omini_condition else conditions
        )
        model_position_deltas = (
            [] if disable_omini_condition else position_deltas
        )
        model_position_scales = (
            [] if disable_omini_condition else position_scales
        )
        model_latent_masks = (
            [] if disable_omini_condition else latent_masks
        )

        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images(self.flux_pipe, imgs)
            inpaint_loss_mask = self._build_inpaint_loss_mask(
                batch, img_ids, image_latent_mask
            )

            # Prepare text input
            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = self.flux_pipe.encode_prompt(
                prompt=prompts,
                prompt_2=prompts_2,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=self.flux_pipe.device,
                num_images_per_prompt=1,
                max_sequence_length=self.model_config.get("max_sequence_length", 512),
                lora_scale=None,
            )

            # Prepare t and x_t. This repository uses t=0 for the target and
            # t=1 for the starting point, the reverse parameterization of the
            # AnchorFlow README but the same interpolation path.
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            noise = torch.randn_like(x_0).to(self.device)
            if self.anchor_flow_config.get("enable", False):
                anchor_imgs = batch.get("anchor_image")
                if anchor_imgs is None:
                    raise KeyError(
                        "AnchorFlow is enabled, but the batch has no 'anchor_image'."
                    )
                anchor_latents, anchor_ids = encode_images(
                    self.flux_pipe, anchor_imgs
                )
                if anchor_latents.shape != x_0.shape:
                    raise ValueError(
                        "Anchor and target latent shapes differ: "
                        f"{tuple(anchor_latents.shape)} vs {tuple(x_0.shape)}."
                    )
                if anchor_ids.shape != img_ids.shape:
                    raise ValueError(
                        "Anchor and target latent ID shapes differ: "
                        f"{tuple(anchor_ids.shape)} vs {tuple(img_ids.shape)}."
                    )
                sigma = float(
                    self.anchor_flow_config.get("anchor_noise_sigma", 0.4)
                )
                dropout = float(self.anchor_flow_config.get("anchor_dropout", 0.1))
                if sigma < 0:
                    raise ValueError("anchor_noise_sigma must be non-negative.")
                if not 0 <= dropout <= 1:
                    raise ValueError("anchor_dropout must be between 0 and 1.")
                x_1 = anchor_latents + sigma * noise
                if torch.rand((), device=self.device) < dropout:
                    x_1 = noise
            else:
                x_1 = noise
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)
            if image_latent_mask is not None:
                x_0 = x_0[:, image_latent_mask[0]]
                x_1 = x_1[:, image_latent_mask[0]]
                x_t = x_t[:, image_latent_mask[0]]
                img_ids = img_ids[image_latent_mask[0]]

            # Prepare conditions
            condition_latents, condition_ids = [], []
            for cond, p_delta, p_scale, latent_mask in zip(
                model_conditions,
                model_position_deltas,
                model_position_scales,
                model_latent_masks,
            ):
                # Prepare conditions
                c_latents, c_ids = encode_images(self.flux_pipe, cond)
                # Scale the position (see OminiConrtol2)
                if p_scale != 1.0:
                    scale_bias = (p_scale - 1.0) / 2
                    c_ids[:, 1:] *= p_scale
                    c_ids[:, 1:] += scale_bias
                # Add position delta (see OminiControl)
                c_ids[:, 1] += p_delta[0][0]
                c_ids[:, 2] += p_delta[0][1]
                if len(p_delta) > 1:
                    p_delta_tensor = torch.as_tensor(p_delta)
                    if not torch.all(p_delta_tensor == p_delta_tensor[0]):
                        print(
                            "Warning: position deltas differ within the batch; "
                            "only the first position delta is used."
                        )
                # Append to the list
                if latent_mask is not None:
                    c_latents, c_ids = c_latents[latent_mask], c_ids[latent_mask[0]]
                condition_latents.append(c_latents)
                condition_ids.append(c_ids)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        branch_n = 2 + len(condition_latents)
        group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool).to(self.device)
        # Disable the attention cross different condition branches
        group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(condition_latents)))
        # Disable the attention from condition branches to image branch and text branch
        if self.model_config.get("independent_condition", False):
            group_mask[2:, :2] = False

        ctrl_field = self._build_toa_control_field(
            batch, conditions, img_ids, image_latent_mask
        )
        glor_context = self._build_glor_control_context(
            batch, conditions, img_ids, image_latent_mask
        )
        scb_bias = self._build_scb_bias(batch, conditions, img_ids)
        if image_latent_mask is not None and scb_bias is not None:
            scb_bias = scb_bias[:, :, image_latent_mask[0]]

        metadata_qkv_deltas = None
        if self.metadata_qkv_injector is not None:
            if not condition_latents:
                raise ValueError(
                    "Metadata-QKV requires an active image-condition branch."
                )
            condition_metadata = batch.get("condition_metadata")
            target_metadata = batch.get("target_metadata")
            if condition_metadata is None or target_metadata is None:
                raise KeyError(
                    "Metadata-QKV requires condition_metadata and target_metadata."
                )
            metadata_dtype = next(self.metadata_qkv_injector.parameters()).dtype
            metadata_qkv_deltas = self.metadata_qkv_injector.prepare_deltas(
                condition_metadata.to(device=self.device, dtype=metadata_dtype),
                target_metadata.to(device=self.device, dtype=metadata_dtype),
            )

        # Forward pass
        transformer_out = transformer_forward(
            self.transformer,
            image_features=[x_t, *(condition_latents)],
            text_features=[prompt_embeds],
            img_ids=[img_ids, *(condition_ids)],
            txt_ids=[text_ids],
            # There are three timesteps for the three branches
            # (text, image, and the condition)
            timesteps=[t, t] + [torch.zeros_like(t)] * len(condition_latents),
            # Same as above
            pooled_projections=[pooled_prompt_embeds] * branch_n,
            guidances=[guidance] * branch_n,
            # The LoRA adapter names of each branch
            adapters=self.adapter_names,
            return_dict=False,
            group_mask=group_mask,
            toa_control_field=ctrl_field,
            glor_control_context=glor_context,
            scb_bias=scb_bias,
            metadata_qkv_injector=self.metadata_qkv_injector,
            metadata_qkv_deltas=metadata_qkv_deltas,
        )
        pred = transformer_out[0]

        # Compute loss
        squared_error = (pred.float() - (x_1 - x_0).float()).square()
        if inpaint_loss_mask is None:
            step_loss = squared_error.mean()
        else:
            loss_weights = 1.0 + (
                self.inpaint_mask_loss_weight * inpaint_loss_mask
            )
            step_loss = (squared_error * loss_weights).sum() / (
                loss_weights.sum() * squared_error.shape[-1]
            )
        if not torch.isfinite(step_loss):
            raise FloatingPointError(
                f"Non-finite training loss at batch {batch_idx}; aborting before "
                "the optimizer can corrupt LoRA/control weights."
            )
        self.last_t = t.mean().item()

        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def generate_a_sample(self):
        raise NotImplementedError("Generate a sample not implemented.")


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}, test_function=None):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0
        self.test_function = test_function
        self.step_start_time = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.step_start_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step_time = (
            time.perf_counter() - self.step_start_time
            if self.step_start_time is not None
            else float("nan")
        )
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        observation_residual_params = getattr(
            pl_module, "observation_residual_params", []
        )
        observation_residual_stats = None
        if observation_residual_params:
            alpha_values = torch.cat(
                [param.detach().float().reshape(-1) for param in observation_residual_params]
            )
            observation_residual_stats = {
                "mean": alpha_values.mean().item(),
                "abs_mean": alpha_values.abs().mean().item(),
                "min": alpha_values.min().item(),
                "max": alpha_values.max().item(),
            }
        metadata_qkv_stats = None
        metadata_qkv_injector = getattr(
            pl_module, "metadata_qkv_injector", None
        )
        if metadata_qkv_injector is not None:
            metadata_qkv_stats = metadata_qkv_injector.gate_statistics()

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
                "step_time_seconds": step_time,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            if observation_residual_stats is not None:
                report_dict.update(
                    {
                        f"observation_residual/{key}": value
                        for key, value in observation_residual_stats.items()
                    }
                )
            if metadata_qkv_stats is not None:
                report_dict.update(
                    {
                        f"metadata_qkv/{key}": value
                        for key, value in metadata_qkv_stats.items()
                    }
                )
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            message = (
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, "
                f"Batch: {batch_idx}, Step time: {step_time:.3f}s, "
                f"Loss: {pl_module.log_loss:.4f}, "
                f"Gradient size: {gradient_size:.4f}, "
                f"Max gradient size: {max_gradient_size:.4f}"
            )
            if observation_residual_stats is not None:
                message += (
                    ", ObsRes alpha: "
                    f"mean={observation_residual_stats['mean']:.6f}, "
                    f"abs_mean={observation_residual_stats['abs_mean']:.6f}, "
                    f"range=[{observation_residual_stats['min']:.6f}, "
                    f"{observation_residual_stats['max']:.6f}]"
                )
            if metadata_qkv_stats is not None:
                message += (
                    ", MIQ gates: target-Q="
                    f"{metadata_qkv_stats['target_q_gate_abs_mean']:.6e}, "
                    "condition-K="
                    f"{metadata_qkv_stats['condition_k_gate_abs_mean']:.6e}"
                )
            print(message)

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0 and self.test_function:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            pl_module.eval()
            self.test_function(
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
            )
            pl_module.train()


def train(dataset, trainable_model, config, test_function):
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    # config = get_config()

    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    # Initialize dataloader
    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 1),
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    # Callbacks for testing and saving checkpoints
    if is_main_process:
        callbacks = [TrainingCallback(run_name, training_config, test_function)]

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=callbacks if is_main_process else [],
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    setattr(trainer, "training_config", training_config)
    setattr(trainable_model, "training_config", training_config)

    # Save the training config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader)
