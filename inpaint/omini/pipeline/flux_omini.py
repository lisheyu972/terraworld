import torch
from typing import List, Union, Optional, Dict, Any, Callable, Type, Tuple

from diffusers.pipelines import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    FluxTransformer2DModel,
    calculate_shift,
    retrieve_timesteps,
    np,
)
from diffusers.models.attention_processor import Attention, F
from diffusers.models.embeddings import apply_rotary_emb
from transformers import pipeline

from peft.tuners.tuners_utils import BaseTunerLayer
from accelerate.utils import is_torch_version
from observation_residual import apply_observation_residual
from omini.pipeline.rivg import (
    RIVGConfig,
    RIVGState,
    apply_rivg_velocity,
    merge_rivg_condition_latents,
    resize_map_to_tokens,
)

from contextlib import contextmanager

import cv2

from PIL import Image, ImageFilter


def seed_everything(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)


def clip_hidden_states(hidden_states: torch.FloatTensor) -> torch.FloatTensor:
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return hidden_states


def encode_images(
    pipeline: FluxPipeline,
    images: torch.Tensor,
    posterior: str = "sample",
):
    """
    Encodes the images into tokens and ids for FLUX pipeline.
    """
    images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    latent_dist = pipeline.vae.encode(images).latent_dist
    if posterior == "sample":
        images = latent_dist.sample()
    elif posterior == "mode":
        images = latent_dist.mode()
    elif posterior == "mean":
        images = latent_dist.mean
    else:
        raise ValueError("posterior must be one of: sample, mode, mean.")
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    return images_tokens, images_ids


depth_pipe = None


def convert_to_condition(
    condition_type: str,
    raw_img: Union[Image.Image, torch.Tensor],
    blur_radius: Optional[int] = 5,
) -> Union[Image.Image, torch.Tensor]:
    if condition_type == "depth":
        global depth_pipe
        depth_pipe = depth_pipe or pipeline(
            task="depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device="cpu",  # Use "cpu" to enable parallel processing
        )
        source_image = raw_img.convert("RGB")
        condition_img = depth_pipe(source_image)["depth"].convert("RGB")
        return condition_img
    elif condition_type == "canny":
        img = np.array(raw_img)
        edges = cv2.Canny(img, 100, 200)
        edges = Image.fromarray(edges).convert("RGB")
        return edges
    elif condition_type == "coloring":
        return raw_img.convert("L").convert("RGB")
    elif condition_type == "deblurring":
        condition_image = (
            raw_img.convert("RGB")
            .filter(ImageFilter.GaussianBlur(blur_radius))
            .convert("RGB")
        )
        return condition_image
    else:
        print("Warning: Returning the raw image.")
        return raw_img.convert("RGB")


class Condition(object):
    def __init__(
        self,
        condition: Union[Image.Image, torch.Tensor],
        adapter_setting: Union[str, dict],
        position_delta=None,
        position_scale=1.0,
        latent_mask=None,
        is_complement=False,
        posterior="sample",
    ) -> None:
        self.condition = condition
        self.adapter = adapter_setting
        self.position_delta = position_delta
        self.position_scale = position_scale
        self.latent_mask = (
            latent_mask.T.reshape(-1) if latent_mask is not None else None
        )
        self.is_complement = is_complement
        if posterior not in {"sample", "mode", "mean"}:
            raise ValueError("Condition posterior must be sample, mode, or mean.")
        self.posterior = posterior

    def encode(
        self, pipe: FluxPipeline, empty: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        condition_empty = Image.new("RGB", self.condition.size, (0, 0, 0))
        tokens, ids = encode_images(
            pipe,
            condition_empty if empty else self.condition,
            posterior=self.posterior,
        )

        if self.position_delta is not None:
            ids[:, 1] += self.position_delta[0]
            ids[:, 2] += self.position_delta[1]

        if self.position_scale != 1.0:
            scale_bias = (self.position_scale - 1.0) / 2
            ids[:, 1:] *= self.position_scale
            ids[:, 1:] += scale_bias

        if self.latent_mask is not None:
            tokens = tokens[:, self.latent_mask]
            ids = ids[self.latent_mask]

        return tokens, ids


@contextmanager
def specify_lora(lora_modules: List[BaseTunerLayer], specified_lora):
    # Filter valid lora modules
    valid_lora_modules = [m for m in lora_modules if isinstance(m, BaseTunerLayer)]
    # Save original scales
    original_scales = [
        {
            adapter: module.scaling[adapter]
            for adapter in module.active_adapters
            if adapter in module.scaling
        }
        for module in valid_lora_modules
    ]
    # Enter context: adjust scaling
    for module in valid_lora_modules:
        for adapter in module.active_adapters:
            if adapter in module.scaling:
                module.scaling[adapter] = 1 if adapter == specified_lora else 0
    try:
        yield
    finally:
        # Exit context: restore original scales
        for module, scales in zip(valid_lora_modules, original_scales):
            for adapter in module.active_adapters:
                if adapter in module.scaling:
                    module.scaling[adapter] = scales[adapter]


def attn_forward(
    attn: Attention,
    hidden_states: List[torch.FloatTensor],
    adapters: List[str],
    hidden_states2: Optional[List[torch.FloatTensor]] = [],
    position_embs: Optional[List[torch.Tensor]] = None,
    group_mask: Optional[torch.Tensor] = None,
    cache_mode: Optional[str] = None,
    # to determine whether to cache the keys and values for this branch
    to_cache: Optional[List[torch.Tensor]] = None,
    cache_storage: Optional[List[torch.Tensor]] = None,
    scb_bias: Optional[torch.Tensor] = None,
    scb_query_index: Optional[int] = None,
    scb_condition_start_index: Optional[int] = None,
    metadata_qkv_injector=None,
    metadata_qkv_deltas=None,
    metadata_layer_index: Optional[int] = None,
    metadata_query_index: Optional[int] = None,
    metadata_condition_indices: Optional[List[int]] = None,
    **kwargs: dict,
) -> torch.FloatTensor:
    bs, _, _ = hidden_states[0].shape
    h2_n = len(hidden_states2)

    queries, keys, values = [], [], []

    # Prepare query, key, value for each encoder hidden state (text branch)
    for i, hidden_state in enumerate(hidden_states2):
        with specify_lora(
            (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj), adapters[i]
        ):
            query = attn.add_q_proj(hidden_state)
            key = attn.add_k_proj(hidden_state)
            value = attn.add_v_proj(hidden_state)

        head_dim = key.shape[-1] // attn.heads
        reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)

        query, key, value = map(reshape_fn, (query, key, value))

        queries.append(query)
        keys.append(key)
        values.append(value)

    # Prepare query, key, value for each hidden state (image branch)
    for i, hidden_state in enumerate(hidden_states):
        with specify_lora((attn.to_q, attn.to_k, attn.to_v), adapters[i + h2_n]):
            query = attn.to_q(hidden_state)
            key = attn.to_k(hidden_state)
            value = attn.to_v(hidden_state)

        head_dim = key.shape[-1] // attn.heads
        reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)

        query, key, value = map(reshape_fn, (query, key, value))

        queries.append(query)
        keys.append(key)
        values.append(value)

    if (metadata_qkv_injector is None) != (metadata_qkv_deltas is None):
        raise ValueError(
            "metadata_qkv_injector and metadata_qkv_deltas must be provided together"
        )
    if metadata_qkv_injector is not None:
        if metadata_layer_index is None:
            raise ValueError("Metadata-QKV requires metadata_layer_index.")
        query_index = h2_n if metadata_query_index is None else metadata_query_index
        condition_indices = list(
            metadata_condition_indices
            if metadata_condition_indices is not None
            else [h2_n + 1]
        )
        if not 0 <= query_index < len(queries):
            raise IndexError(f"Invalid metadata query index: {query_index}")
        if not condition_indices:
            raise ValueError("Metadata-QKV requires an image-condition branch.")
        condition_delta = metadata_qkv_deltas.condition_k
        if condition_delta.ndim == 5 and condition_delta.shape[1] != len(
            condition_indices
        ):
            raise ValueError("Metadata count does not match condition branches.")
        for slot, condition_index in enumerate(condition_indices):
            if not 0 <= condition_index < len(keys):
                raise IndexError(
                    f"Invalid metadata condition index: {condition_index}"
                )
            queries[query_index], keys[condition_index], values[condition_index] = (
                metadata_qkv_injector.inject(
                    q_main=queries[query_index],
                    k_condition=keys[condition_index],
                    v_condition=values[condition_index],
                    deltas=metadata_qkv_deltas,
                    layer_index=metadata_layer_index,
                    condition_slot=slot if condition_delta.ndim == 5 else None,
                    inject_main_query=slot == 0,
                )
            )

    # MIQ-MMA deltas are injected before the original Q/K normalization and RoPE.
    for index in range(len(queries)):
        if index < h2_n:
            queries[index] = attn.norm_added_q(queries[index])
            keys[index] = attn.norm_added_k(keys[index])
        else:
            queries[index] = attn.norm_q(queries[index])
            keys[index] = attn.norm_k(keys[index])

    # Apply rotary embedding
    if position_embs is not None:
        queries = [apply_rotary_emb(q, position_embs[i]) for i, q in enumerate(queries)]
        keys = [apply_rotary_emb(k, position_embs[i]) for i, k in enumerate(keys)]

    if cache_mode == "write":
        for i, (k, v) in enumerate(zip(keys, values)):
            if to_cache[i]:
                cache_storage[attn.cache_idx][0].append(k)
                cache_storage[attn.cache_idx][1].append(v)

    attn_outputs = []
    if scb_query_index is None:
        scb_query_index = h2_n
    if scb_condition_start_index is None:
        scb_condition_start_index = h2_n + 1
    for i, query in enumerate(queries):
        keys_, values_ = [], []
        bias_segments = []
        scb_cursor = 0
        apply_scb = (
            scb_bias is not None
            and hasattr(attn, "scb_gamma")
            and i == scb_query_index
        )
        # Add keys and values from other branches
        for j, (k, v) in enumerate(zip(keys, values)):
            if (group_mask is not None) and not (group_mask[i][j].item()):
                continue
            keys_.append(k)
            values_.append(v)
            if apply_scb:
                key_len = k.shape[2]
                if j >= scb_condition_start_index:
                    segment = scb_bias[:, :, :, scb_cursor : scb_cursor + key_len]
                    scb_cursor += key_len
                    if segment.shape[-1] != key_len:
                        segment = torch.zeros(
                            query.shape[0],
                            1,
                            query.shape[2],
                            key_len,
                            device=query.device,
                            dtype=query.dtype,
                        )
                else:
                    segment = torch.zeros(
                        query.shape[0],
                        1,
                        query.shape[2],
                        key_len,
                        device=query.device,
                        dtype=query.dtype,
                    )
                bias_segments.append(segment)
        if cache_mode == "read":
            keys_.extend(cache_storage[attn.cache_idx][0])
            values_.extend(cache_storage[attn.cache_idx][1])
            if apply_scb:
                for cached_key in cache_storage[attn.cache_idx][0]:
                    bias_segments.append(
                        torch.zeros(
                            query.shape[0],
                            1,
                            query.shape[2],
                            cached_key.shape[2],
                            device=query.device,
                            dtype=query.dtype,
                        )
                    )
        # Add keys and values from cache TODO
        # Attention computation
        key_tensor = torch.cat(keys_, dim=2)
        value_tensor = torch.cat(values_, dim=2)
        if apply_scb:
            logits = torch.matmul(query, key_tensor.transpose(-2, -1))
            logits = logits / (query.shape[-1] ** 0.5)
            bias = torch.cat(bias_segments, dim=-1).to(
                device=query.device, dtype=query.dtype
            )
            logits = logits + attn.scb_gamma.to(query.dtype) * bias
            attn_probs = logits.softmax(dim=-1)
            attn_output = torch.matmul(attn_probs, value_tensor).to(query.dtype)
        else:
            attn_output = F.scaled_dot_product_attention(
                query, key_tensor, value_tensor
            ).to(query.dtype)
        attn_output = attn_output.transpose(1, 2).reshape(bs, -1, attn.heads * head_dim)
        attn_outputs.append(attn_output)

    # Reshape attention output to match the original hidden states
    h_out, h2_out = [], []

    for i, hidden_state in enumerate(hidden_states2):
        with specify_lora((attn.to_add_out,), adapters[i]):
            h2_out.append(attn.to_add_out(attn_outputs[i]))

    for i, hidden_state in enumerate(hidden_states):
        h = attn_outputs[i + h2_n]
        if getattr(attn, "to_out", None) is not None:
            with specify_lora((attn.to_out[0],), adapters[i + h2_n]):
                h = attn.to_out[0](h)
        h_out.append(h)

    return (h_out, h2_out) if h2_n else h_out


def block_forward(
    self,
    image_hidden_states: List[torch.FloatTensor],
    text_hidden_states: List[torch.FloatTensor],
    tembs: List[torch.FloatTensor],
    adapters: List[str],
    position_embs=None,
    attn_forward=attn_forward,
    **kwargs: dict,
):
    txt_n = len(text_hidden_states)

    img_variables, txt_variables = [], []

    for i, text_h in enumerate(text_hidden_states):
        with specify_lora((self.norm1_context.linear,), adapters[i]):
            txt_variables.append(self.norm1_context(text_h, emb=tembs[i]))

    for i, image_h in enumerate(image_hidden_states):
        with specify_lora((self.norm1.linear,), adapters[i + txt_n]):
            img_variables.append(self.norm1(image_h, emb=tembs[i + txt_n]))

    # Attention.
    img_attn_output, txt_attn_output = attn_forward(
        self.attn,
        hidden_states=[each[0] for each in img_variables],
        hidden_states2=[each[0] for each in txt_variables],
        position_embs=position_embs,
        adapters=adapters,
        **kwargs,
    )

    text_out = []
    for i in range(len(text_hidden_states)):
        _, gate_msa, shift_mlp, scale_mlp, gate_mlp = txt_variables[i]
        text_h = text_hidden_states[i] + txt_attn_output[i] * gate_msa.unsqueeze(1)
        norm_h = (
            self.norm2_context(text_h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        with specify_lora(
            (self.ff_context.net[0].proj, self.ff_context.net[2]), adapters[i]
        ):
            text_h = self.ff_context(norm_h) * gate_mlp.unsqueeze(1) + text_h
        text_out.append(clip_hidden_states(text_h))

    image_out = []
    for i in range(len(image_hidden_states)):
        _, gate_msa, shift_mlp, scale_mlp, gate_mlp = img_variables[i]
        image_h = (
            image_hidden_states[i] + img_attn_output[i] * gate_msa.unsqueeze(1)
        ).to(image_hidden_states[i].dtype)
        if i >= 1:
            image_h = apply_observation_residual(
                self, image_h, image_hidden_states[i]
            )
        norm_h = self.norm2(image_h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        with specify_lora(
            (self.ff.net[0].proj, self.ff.net[2]), adapters[i + txt_n]
        ):
            image_h = image_h + self.ff(norm_h) * gate_mlp.unsqueeze(1)
        image_out.append(clip_hidden_states(image_h))
    return image_out, text_out


def single_block_forward(
    self,
    hidden_states: List[torch.FloatTensor],
    tembs: List[torch.FloatTensor],
    adapters: List[str],
    position_embs=None,
    attn_forward=attn_forward,
    observation_condition_start_index=None,
    **kwargs: dict,
):
    mlp_hidden_states, gates = [[None for _ in hidden_states] for _ in range(2)]

    hidden_state_norm = []
    for i, hidden_state in enumerate(hidden_states):
        # [NOTE]!: This function's output is slightly DIFFERENT from the original
        # FLUX version. In the original implementation, the gates were computed using
        # the combined hidden states from both the image and text branches. Here, each
        # branch computes its gate using only its own hidden state.
        with specify_lora((self.norm.linear, self.proj_mlp), adapters[i]):
            h_norm, gates[i] = self.norm(hidden_state, emb=tembs[i])
            mlp_hidden_states[i] = self.act_mlp(self.proj_mlp(h_norm))
        hidden_state_norm.append(h_norm)

    attn_outputs = attn_forward(
        self.attn, hidden_state_norm, adapters, position_embs=position_embs, **kwargs
    )

    h_out = []
    for i in range(len(hidden_states)):
        with specify_lora((self.proj_out,), adapters[i]):
            h = torch.cat([attn_outputs[i], mlp_hidden_states[i]], dim=2)
            h = gates[i].unsqueeze(1) * self.proj_out(h) + hidden_states[i]
            if (
                observation_condition_start_index is not None
                and i >= observation_condition_start_index
            ):
                h = apply_observation_residual(self, h, hidden_states[i])
            h_out.append(clip_hidden_states(h))

    return h_out


def transformer_forward(
    transformer: FluxTransformer2DModel,
    image_features: List[torch.Tensor],
    text_features: List[torch.Tensor] = None,
    img_ids: List[torch.Tensor] = None,
    txt_ids: List[torch.Tensor] = None,
    pooled_projections: List[torch.Tensor] = None,
    timesteps: List[torch.LongTensor] = None,
    guidances: List[torch.Tensor] = None,
    adapters: List[str] = None,
    # Assign the function to be used for the forward pass
    single_block_forward=single_block_forward,
    block_forward=block_forward,
    attn_forward=attn_forward,
    **kwargs: dict,
):
    self = transformer
    txt_n = len(text_features) if text_features is not None else 0
    return_adl_loss = bool(kwargs.pop("return_adl_loss", False))
    toa_control_field = kwargs.pop("toa_control_field", None)
    toa_injector = getattr(self, "toa_control_injector", None)
    glor_control_context = kwargs.pop("glor_control_context", None)
    glor_control = getattr(self, "glor_control", None)

    def inject_glor(hidden_states, block_kind, block_index):
        if glor_control is None or glor_control_context is None:
            return hidden_states
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                glor_control.inject,
                hidden_states,
                glor_control_context,
                block_kind=block_kind,
                block_index=block_index,
                use_reentrant=False,
            )
        return glor_control.inject(
            hidden_states,
            glor_control_context,
            block_kind=block_kind,
            block_index=block_index,
        )

    adapters = adapters or [None] * (txt_n + len(image_features))
    assert len(adapters) == len(timesteps)

    # Preprocess the image_features
    image_hidden_states = []
    for i, image_feature in enumerate(image_features):
        with specify_lora((self.x_embedder,), adapters[i + txt_n]):
            image_hidden_states.append(self.x_embedder(image_feature))

    # Preprocess the text_features
    text_hidden_states = []
    for i, text_feature in enumerate(text_features):
        with specify_lora((self.context_embedder,), adapters[i]):
            text_hidden_states.append(self.context_embedder(text_feature))

    # Prepare embeddings of (timestep, guidance, pooled_projections)
    assert len(timesteps) == len(image_features) + len(text_features)

    def get_temb(timestep, guidance, pooled_projection, adapter):
        timestep = timestep.to(image_hidden_states[0].dtype) * 1000
        with specify_lora(tuple(self.time_text_embed.modules()), adapter):
            if guidance is not None:
                guidance = guidance.to(image_hidden_states[0].dtype) * 1000
                return self.time_text_embed(timestep, guidance, pooled_projection)
            else:
                return self.time_text_embed(timestep, pooled_projection)

    tembs = [
        get_temb(*each, adapter)
        for each, adapter in zip(
            zip(timesteps, guidances, pooled_projections), adapters
        )
    ]

    # Prepare position embeddings for each token
    position_embs = [self.pos_embed(each) for each in (*txt_ids, *img_ids)]

    # Prepare the gradient checkpointing kwargs
    gckpt_kwargs: Dict[str, Any] = (
        {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
    )

    adl_terms = []

    def append_adl_term(features):
        if not return_adl_loss:
            return
        mean_feature = features.mean(dim=1, keepdim=True)
        cosine = F.cosine_similarity(
            features,
            mean_feature.expand_as(features),
            dim=-1,
            eps=1.0e-8,
        )
        adl_terms.append(cosine.float().mean())

    # dual branch blocks
    for block_index, block in enumerate(self.transformer_blocks):
        block_kwargs = {
            "self": block,
            "image_hidden_states": image_hidden_states,
            "text_hidden_states": text_hidden_states,
            "tembs": tembs,
            "position_embs": position_embs,
            "adapters": adapters,
            "attn_forward": attn_forward,
            "scb_query_index": txt_n,
            "scb_condition_start_index": txt_n + 1,
            "metadata_query_index": txt_n,
            "metadata_condition_indices": list(
                range(txt_n + 1, txt_n + len(image_features))
            ),
            "metadata_layer_index": block_index,
            **kwargs,
        }
        if self.training and self.gradient_checkpointing:
            image_hidden_states, text_hidden_states = torch.utils.checkpoint.checkpoint(
                block_forward, **block_kwargs, **gckpt_kwargs
            )
        else:
            image_hidden_states, text_hidden_states = block_forward(**block_kwargs)
        append_adl_term(image_hidden_states[0])
        if toa_injector is not None and toa_control_field is not None:
            image_hidden_states[0] = toa_injector.inject(
                image_hidden_states[0],
                toa_control_field,
                block_kind="dual",
                block_index=block_index,
            )
        if glor_control is not None and glor_control_context is not None:
            image_hidden_states[0] = inject_glor(
                image_hidden_states[0],
                block_kind="dual",
                block_index=block_index,
            )

    # combine image and text hidden states then pass through the single transformer blocks
    all_hidden_states = [*text_hidden_states, *image_hidden_states]
    for block_index, block in enumerate(self.single_transformer_blocks):
        block_kwargs = {
            "self": block,
            "hidden_states": all_hidden_states,
            "tembs": tembs,
            "position_embs": position_embs,
            "adapters": adapters,
            "attn_forward": attn_forward,
            "scb_query_index": txt_n,
            "scb_condition_start_index": txt_n + 1,
            "metadata_query_index": txt_n,
            "metadata_condition_indices": list(
                range(txt_n + 1, txt_n + len(image_features))
            ),
            "metadata_layer_index": len(self.transformer_blocks) + block_index,
            "observation_condition_start_index": txt_n + 1,
            **kwargs,
        }
        if self.training and self.gradient_checkpointing:
            all_hidden_states = torch.utils.checkpoint.checkpoint(
                single_block_forward, **block_kwargs, **gckpt_kwargs
            )
        else:
            all_hidden_states = single_block_forward(**block_kwargs)
        append_adl_term(all_hidden_states[txt_n])
        if toa_injector is not None and toa_control_field is not None:
            all_hidden_states[txt_n] = toa_injector.inject(
                all_hidden_states[txt_n],
                toa_control_field,
                block_kind="single",
                block_index=block_index,
            )
        if glor_control is not None and glor_control_context is not None:
            all_hidden_states[txt_n] = inject_glor(
                all_hidden_states[txt_n],
                block_kind="single",
                block_index=block_index,
            )

    # The first image branch is the generated/main image branch. Keep its
    # output-head LoRA independently switchable from text and conditions.
    with specify_lora(
        (self.norm_out.linear, self.proj_out), adapters[txt_n]
    ):
        image_hidden_states = self.norm_out(all_hidden_states[txt_n], tembs[txt_n])
        output = self.proj_out(image_hidden_states)

    if return_adl_loss:
        if not adl_terms:
            adl_loss = output.new_zeros((), dtype=torch.float32)
        else:
            adl_loss = torch.stack(adl_terms).mean()
        return output, adl_loss
    return (output,)


@torch.no_grad()
def generate(
    pipeline: FluxPipeline,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    # Condition Parameters (Optional)
    main_adapter: Optional[str] = None,
    text_adapter: Optional[str] = None,
    main_adapter_on_text: bool = True,
    conditions: List[Condition] = [],
    image_guidance_scale: float = 1.0,
    transformer_kwargs: Optional[Dict[str, Any]] = {},
    toa_control_field: Optional[torch.FloatTensor] = None,
    glor_control_context: Optional[Dict[str, torch.FloatTensor]] = None,
    rivg_config: Optional[Union[RIVGConfig, Dict[str, Any]]] = None,
    rivg_weak_conditions: Optional[List[Condition]] = None,
    rivg_information_map: Optional[torch.FloatTensor] = None,
    rivg_weak_glor_control_context: Optional[
        Dict[str, torch.FloatTensor]
    ] = None,
    scb_bias: Optional[torch.FloatTensor] = None,
    kv_cache=False,
    latent_mask=None,
    **params: dict,
):
    self = pipeline

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs

    # Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # Prepare prompt embeddings
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    # Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    if latent_mask is not None:
        latent_mask = latent_mask.T.reshape(-1)
        latents = latents[:, latent_mask]
        latent_image_ids = latent_image_ids[latent_mask]

    # Prepare conditions
    c_latents, uc_latents, c_ids, c_timesteps = ([], [], [], [])
    c_projections, c_guidances, c_adapters = ([], [], [])
    complement_cond = None
    for condition in conditions:
        tokens, ids = condition.encode(self)
        c_latents.append(tokens)  # [batch_size, token_n, token_dim]
        # Empty condition for unconditioned image
        if image_guidance_scale != 1.0:
            uc_latents.append(condition.encode(self, empty=True)[0])
        c_ids.append(ids)  # [token_n, id_dim(3)]
        c_timesteps.append(torch.zeros([1], device=device))
        c_projections.append(pooled_prompt_embeds)
        c_guidances.append(torch.ones([1], device=device))
        c_adapters.append(condition.adapter)
        # This complement_condition will be combined with the original image.
        # See the token integration of OminiControl2 [https://arxiv.org/abs/2503.08280]
        if condition.is_complement:
            complement_cond = (tokens, ids)

    # Prepare the deterministic weak-reference branch once. It is evaluated
    # only during the configured late denoising steps.
    rivg = RIVGConfig.from_value(rivg_config) if rivg_config is not None else None
    rivg_state = RIVGState()
    rivg_information_tokens = None
    rivg_spatial_shape = None
    rivg_weak_latents = []
    rivg_condition_latent = None
    if rivg is not None:
        if kv_cache:
            raise ValueError("RIVG currently requires kv_cache=False.")
        if rivg_information_map is None:
            raise ValueError("RIVG requires rivg_information_map.")
        weak_conditions = list(rivg_weak_conditions or [])
        if len(weak_conditions) != len(conditions):
            raise ValueError(
                "RIVG full and weak condition lists must have equal lengths."
            )
        if not conditions and glor_control_context is None:
            raise ValueError("RIVG requires an Omini or GLoR reference condition.")
        if (
            glor_control_context is not None
            and rivg_weak_glor_control_context is None
        ):
            raise ValueError(
                "RIVG with GLoR-Control requires a weak GLoR context."
            )
        for index, weak_condition in enumerate(weak_conditions):
            weak_tokens, weak_ids = weak_condition.encode(self)
            if weak_tokens.shape != c_latents[index].shape:
                raise ValueError("RIVG full and weak condition token shapes differ.")
            if weak_ids.shape != c_ids[index].shape or not torch.equal(
                weak_ids, c_ids[index]
            ):
                raise ValueError("RIVG full and weak condition positions differ.")
            if weak_condition.adapter != c_adapters[index]:
                raise ValueError("RIVG full and weak condition adapters differ.")
            rivg_weak_latents.append(weak_tokens)
        rivg_condition_latent = merge_rivg_condition_latents(c_latents)
        information_map = rivg_information_map
        if information_map.shape[0] == 1 and latents.shape[0] != 1:
            information_map = information_map.expand(latents.shape[0], -1, -1, -1)
        if information_map.shape[0] != latents.shape[0]:
            raise ValueError("RIVG information-map batch size does not match latents.")
        rivg_information_tokens, rivg_spatial_shape = resize_map_to_tokens(
            information_map,
            latent_image_ids,
            device=device,
            dtype=latents.dtype,
        )
        print(
            f"RIVG enabled: method={rivg.method}, lambda_max={rivg.lambda_max}, "
            f"start={rivg.guidance_start_fraction}, token_grid="
            f"{rivg_spatial_shape[1]}x{rivg_spatial_shape[0]}"
        )

    # Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu
    )
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )
    self._num_timesteps = len(timesteps)

    if kv_cache:
        attn_counter = 0
        for module in self.transformer.modules():
            if isinstance(module, Attention):
                setattr(module, "cache_idx", attn_counter)
                attn_counter += 1
        kv_cond = [[[], []] for _ in range(attn_counter)]
        kv_uncond = [[[], []] for _ in range(attn_counter)]

        def clear_cache():
            for storage in [kv_cond, kv_uncond]:
                for kesy, values in storage:
                    kesy.clear()
                    values.clear()

    branch_n = len(conditions) + 2
    group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool)
    # Disable the attention cross different condition branches
    group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(conditions)))
    # Disable the attention from condition branches to image branch and text branch
    if kv_cache:
        group_mask[2:, :2] = False

    # Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000

            # handle guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=device)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance, c_guidances = None, [None for _ in c_guidances]

            if kv_cache:
                mode = "write" if i == 0 else "read"
                if mode == "write":
                    clear_cache()
            use_cond = not (kv_cache) or mode == "write"

            text_main_adapter = (
                text_adapter
                if text_adapter is not None
                else (main_adapter if main_adapter_on_text else None)
            )
            noise_pred = transformer_forward(
                self.transformer,
                image_features=[latents] + (c_latents if use_cond else []),
                text_features=[prompt_embeds],
                img_ids=[latent_image_ids] + (c_ids if use_cond else []),
                txt_ids=[text_ids],
                timesteps=[timestep, timestep] + (c_timesteps if use_cond else []),
                pooled_projections=[pooled_prompt_embeds] * 2
                + (c_projections if use_cond else []),
                guidances=[guidance] * 2 + (c_guidances if use_cond else []),
                return_dict=False,
                adapters=[text_main_adapter, main_adapter]
                + (c_adapters if use_cond else []),
                cache_mode=mode if kv_cache else None,
                cache_storage=kv_cond if kv_cache else None,
                to_cache=[False, False, *[True] * len(c_latents)],
                group_mask=group_mask,
                toa_control_field=toa_control_field,
                glor_control_context=glor_control_context,
                scb_bias=scb_bias,
                **transformer_kwargs,
            )[0]

            if image_guidance_scale != 1.0:
                unc_pred = transformer_forward(
                    self.transformer,
                    image_features=[latents] + (uc_latents if use_cond else []),
                    text_features=[prompt_embeds],
                    img_ids=[latent_image_ids] + (c_ids if use_cond else []),
                    txt_ids=[text_ids],
                    timesteps=[timestep, timestep] + (c_timesteps if use_cond else []),
                    pooled_projections=[pooled_prompt_embeds] * 2
                    + (c_projections if use_cond else []),
                    guidances=[guidance] * 2 + (c_guidances if use_cond else []),
                    return_dict=False,
                    adapters=[text_main_adapter, main_adapter]
                    + (c_adapters if use_cond else []),
                    cache_mode=mode if kv_cache else None,
                    cache_storage=kv_uncond if kv_cache else None,
                    to_cache=[False, False, *[True] * len(c_latents)],
                    toa_control_field=toa_control_field,
                    glor_control_context=glor_control_context,
                    scb_bias=scb_bias,
                    **transformer_kwargs,
                )[0]

                noise_pred = unc_pred + image_guidance_scale * (noise_pred - unc_pred)

            progress_fraction = i / max(len(timesteps) - 1, 1)
            if (
                rivg is not None
                and progress_fraction >= rivg.guidance_start_fraction
            ):
                weak_pred = transformer_forward(
                    self.transformer,
                    image_features=[latents] + rivg_weak_latents,
                    text_features=[prompt_embeds],
                    img_ids=[latent_image_ids] + c_ids,
                    txt_ids=[text_ids],
                    timesteps=[timestep, timestep] + c_timesteps,
                    pooled_projections=[pooled_prompt_embeds] * 2 + c_projections,
                    guidances=[guidance] * 2 + c_guidances,
                    return_dict=False,
                    adapters=[text_main_adapter, main_adapter] + c_adapters,
                    group_mask=group_mask,
                    toa_control_field=toa_control_field,
                    glor_control_context=rivg_weak_glor_control_context,
                    scb_bias=scb_bias,
                    **transformer_kwargs,
                )[0]
                if image_guidance_scale != 1.0:
                    weak_unc_pred = transformer_forward(
                        self.transformer,
                        image_features=[latents] + uc_latents,
                        text_features=[prompt_embeds],
                        img_ids=[latent_image_ids] + c_ids,
                        txt_ids=[text_ids],
                        timesteps=[timestep, timestep] + c_timesteps,
                        pooled_projections=[pooled_prompt_embeds] * 2
                        + c_projections,
                        guidances=[guidance] * 2 + c_guidances,
                        return_dict=False,
                        adapters=[text_main_adapter, main_adapter] + c_adapters,
                        group_mask=group_mask,
                        toa_control_field=toa_control_field,
                        glor_control_context=rivg_weak_glor_control_context,
                        scb_bias=scb_bias,
                        **transformer_kwargs,
                    )[0]
                    weak_pred = weak_unc_pred + image_guidance_scale * (
                        weak_pred - weak_unc_pred
                    )
                noise_pred, rivg_state, _ = apply_rivg_velocity(
                    latent=latents,
                    velocity_full=noise_pred,
                    velocity_weak=weak_pred,
                    condition_latent=rivg_condition_latent,
                    information_map=rivg_information_tokens,
                    spatial_shape=rivg_spatial_shape,
                    normalized_timestep=timestep,
                    progress_fraction=progress_fraction,
                    config=rivg,
                    state=rivg_state,
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    if latent_mask is not None:
        # Combine the generated latents and the complement condition
        assert complement_cond is not None
        comp_latent, comp_ids = complement_cond
        all_ids = torch.cat([latent_image_ids, comp_ids], dim=0)  # (Ta+Tc,3)
        shape = (all_ids.max(dim=0).values + 1).to(torch.long)  # (3,)
        H, W = shape[1].item(), shape[2].item()
        B, _, C = latents.shape
        # Create a empty canvas
        canvas = latents.new_zeros(B, H * W, C)  # (B,H*W,C)

        # Stash the latents and the complement condition
        def _stash(canvas, tokens, ids, H, W) -> None:
            B, T, C = tokens.shape
            ids = ids.to(torch.long)
            flat_idx = (ids[:, 1] * W + ids[:, 2]).to(torch.long)
            canvas.view(B, -1, C).index_copy_(1, flat_idx, tokens)

        _stash(canvas, latents, latent_image_ids, H, W)
        _stash(canvas, comp_latent, comp_ids, H, W)
        latents = canvas.view(B, H * W, C)

    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return FluxPipelineOutput(images=image)
