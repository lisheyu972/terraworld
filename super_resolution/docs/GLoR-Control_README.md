# GLoR-Control README

## Goal

现有代码已经支持：

```text
Frozen FLUX / OminiControl backbone
+ task-specific LoRA training
```

现在需要新增一个轻量控制模块：

```text
GLoR-Control: Global-Local Observation Retrieval Control
```

目标：

> 让每个目标 latent token 同时吸收局部观测证据和全局场景上下文，而不是只依赖普通 LoRA 或简单条件拼接。

当前支持三个任务：

```text
1. Super-resolution
2. Frame interpolation
3. Optical-to-SAR translation
```

三个任务分开训练，不需要联合训练。

---

## Core Idea

遥感生成中，每个输出像素/token 既依赖局部观测，也依赖全图上下文。

```text
Local evidence:
  精确结构、边缘、纹理、局部对应关系

Global context:
  场景类型、整体风格、传感器/模态状态、全图一致性
```

因此新增两个分支：

```text
Input observation
      ↓
Condition Encoder
      ↓
 ┌────────────────────┬────────────────────┐
 │ Local Observation  │ Global Observation │
 │ Retrieval          │ Memory             │
 └────────────────────┴────────────────────┘
      ↓                         ↓
 local_ctrl                global_ctrl
      ↓                         ↓
        fuse and inject into DiT hidden states
```

---

## Main Module

建议新增目录：

```text
glor_control/
├── __init__.py
├── control.py
├── local_retrieval.py
├── global_memory.py
├── inject.py
└── utils.py
```

主类：

```python
class GLoRControl(nn.Module):
    def forward(
        self,
        hidden_states,          # [B, N, C]
        batch,
        task,
        target_latent_shape,    # (H_latent, W_latent)
    ):
        ...
        return delta            # [B, N, C]
```

在 DiT block 中注入：

```python
if glor_control is not None:
    delta = glor_control(
        hidden_states=hidden_states,
        batch=batch,
        task=task,
        target_latent_shape=(H_latent, W_latent),
    )
    hidden_states = hidden_states + alpha * delta
```

要求：

```python
alpha = nn.Parameter(torch.zeros(1))
```

这样训练初期不会破坏原模型。

---

## Condition Encoder

三个任务先共用一个简单 CNN encoder。

```python
class SimpleConditionEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)
```

输出：

```text
cond_feat: [B, C_ctrl, H_cond, W_cond]
```

---

## Global Observation Memory

从整张条件图提取少量 global tokens，用来提供全图上下文。

第一版建议用 adaptive average pooling：

```python
feat = condition_encoder(cond_img)  # [B, C_ctrl, H, W]

global_tokens = F.adaptive_avg_pool2d(feat, output_size=(4, 4))
global_tokens = global_tokens.flatten(2).transpose(1, 2)
# [B, 16, C_ctrl]
```

作用：

```text
SR:
  提供整体地物类型和纹理风格。

Frame interpolation:
  提供整体时间变化趋势，例如季节、光照、云层状态。

Optical-to-SAR:
  提供整体 SAR 风格、场景类型、speckle 强度和模态上下文。
```

---

## Local Observation Retrieval

对每个目标 latent token，从输入观测的局部邻域检索条件信息。

核心输出：

```text
local_tokens: [B, N_target, K, C_ctrl]
```

其中：

```text
N_target = H_target * W_target
K = local patch size
```

可以用 `torch.nn.functional.unfold` 实现。

### Local Attention

```python
q = q_proj(hidden_states)       # [B, N, D]
k = k_proj(local_tokens)        # [B, N, K, D]
v = v_proj(local_tokens)        # [B, N, K, D]

attn = torch.einsum("bnd,bnkd->bnk", q, k) / math.sqrt(D)
attn = attn.softmax(dim=-1)

local_ctrl = torch.einsum("bnk,bnkd->bnd", attn, v)
# [B, N, D]
```

---

## Global-Guided Local Retrieval

不要只做 local retrieval。先用 global memory 调制 hidden states，再做 local retrieval。

推荐流程：

```python
global_ctrl = global_cross_attention(
    query=hidden_states,
    key=global_tokens,
    value=global_tokens,
)  # [B, N, D]

local_query = hidden_states + global_ctrl

local_ctrl = local_retrieval_attention(
    query=local_query,
    local_tokens=local_tokens,
)  # [B, N, D]

delta = control_mlp(
    torch.cat([hidden_states, global_ctrl, local_ctrl], dim=-1)
)  # [B, N, C]
```

最后注入：

```python
hidden_states = hidden_states + alpha * delta
```

---

# Task 1: Super-Resolution

## Input

```text
LR image
scale factor
target HR latent size
```

## Local Candidates

对于每个 HR target token，找到它对应的 LR parent 位置，再取周围局部 patch。

简单第一版：

```python
lr_feat = condition_encoder(lr_image)  # [B, C, H_lr, W_lr]

# 先 resize 到 target latent grid
feat_aligned = F.interpolate(
    lr_feat,
    size=(H_target, W_target),
    mode="bilinear",
    align_corners=False,
)

# 在 aligned feature 上 unfold local patch
local_tokens = unfold_local_patches(
    feat_aligned,
    kernel_size=3,
)
# [B, N_target, 9, C]
```

## Global Memory

```python
global_tokens = adaptive_pool(lr_feat, 4x4)
```

## Recommended Patch Size

```text
3x3 first version
5x5 optional
```

---

# Task 2: Frame Interpolation

## Input

```text
frame0
frame1
target time ratio tau
```

## Local Candidates

第一版不加光流，默认帧已配准。

```python
feat0 = condition_encoder(frame0)
feat1 = condition_encoder(frame1)

feat0 = resize_to_target(feat0)
feat1 = resize_to_target(feat1)

local0 = unfold_local_patches(feat0, kernel_size=3)
local1 = unfold_local_patches(feat1, kernel_size=3)

local_tokens = torch.cat([local0, local1], dim=2)
# [B, N_target, 18, C] for 2 * 3 * 3
```

加入时间信息：

```python
time_emb = time_embedding(tau)  # [B, C]
global_tokens = global_tokens + time_emb[:, None, :]
```

也可以把 `time_emb` 加到 local tokens。

## Global Memory

```python
global0 = adaptive_pool(feat0, 4x4)
global1 = adaptive_pool(feat1, 4x4)

global_tokens = (1 - tau) * global0 + tau * global1
global_tokens = global_tokens + time_emb[:, None, :]
```

## Recommended Patch Size

```text
2 * 3x3 first version
```

后续可以加入 optical flow / deformable offsets，但第一版不要做复杂。

---

# Task 3: Optical-to-SAR Translation

## Input

```text
optical image
optional SAR metadata:
  - polarization: VV / VH / HH / HV
  - incidence angle
  - orbit direction
```

## Local Candidates

Optical 和 SAR 默认已经地理配准。

每个 SAR target token 从 optical 同位置局部 patch 检索信息。

建议 optical encoder 输入加入边缘：

```python
edge = sobel_or_canny(optical_image)
cond_input = torch.cat([optical_image, edge], dim=1)
```

然后：

```python
opt_feat = condition_encoder(cond_input)
opt_feat = resize_to_target(opt_feat)

local_tokens = unfold_local_patches(
    opt_feat,
    kernel_size=5,
)
# [B, N_target, 25, C]
```

## Global Memory

```python
global_tokens = adaptive_pool(opt_feat, 4x4)
```

加入 SAR metadata embedding：

```python
sar_emb = sar_metadata_embedding(
    polarization=batch.get("polarization", None),
    incidence_angle=batch.get("incidence_angle", None),
    orbit=batch.get("orbit", None),
)
global_tokens = global_tokens + sar_emb[:, None, :]
```

## Recommended Patch Size

```text
5x5 first version
```

Optical-to-SAR 不是 RGB-to-gray，所以 local patch 和 global SAR metadata 都很重要。

---

## Utility: Unfold Local Patches

实现一个通用函数：

```python
def unfold_local_patches(feat, kernel_size):
    # feat: [B, C, H, W]
    B, C, H, W = feat.shape
    padding = kernel_size // 2

    patches = F.unfold(
        feat,
        kernel_size=kernel_size,
        padding=padding,
    )
    # [B, C*K, H*W]

    K = kernel_size * kernel_size
    patches = patches.transpose(1, 2)
    patches = patches.view(B, H * W, K, C)

    return patches
```

---

## Training

训练流程基本不变。

原来：

```text
freeze backbone
train LoRA
```

现在：

```text
freeze backbone
train LoRA + GLoR-Control modules
```

需要训练：

```text
LoRA parameters
Condition encoder
Global attention
Local retrieval attention
Control MLP
Task embeddings / metadata embeddings
alpha parameters
```

不训练：

```text
FLUX / OminiControl backbone
VAE
text encoder
```

---

## Forward Flow

```python
target_latents = encode_vae(batch["target_image"])

noise = torch.randn_like(target_latents)
z_t, target_v = build_flow_training_pair(
    noise,
    target_latents,
    timestep,
)

pred_v = model(
    hidden_states=z_t,
    timestep=timestep,
    text_condition=text_condition,
    batch=batch,
    task=task,
    use_glor_control=True,
)

loss = F.mse_loss(pred_v.float(), target_v.float())
```

---

## DiT Block Injection

在选定 block 中加入：

```python
if self.use_glor_control:
    delta = self.glor_control(
        hidden_states=hidden_states,
        batch=batch,
        task=task,
        target_latent_shape=(H_latent, W_latent),
    )
    hidden_states = hidden_states + self.glor_alpha * delta
```

建议第一版只注入 middle / late blocks。

---

## Config Example

```yaml
glor_control:
  enable: true
  ctrl_dim: 512
  inject_blocks: "middle_late"
  alpha_init: 0.0

  global_memory:
    pool_size: 4

  local_retrieval:
    sr_patch_size: 3
    interpolation_patch_size: 3
    optical_to_sar_patch_size: 5

  tasks:
    super_resolution:
      use_scale_embedding: true

    frame_interpolation:
      use_time_embedding: true
      use_flow: false

    optical_to_sar:
      use_edge_feature: true
      use_sar_metadata: true
```

---

## Minimal Success Criteria

### General

1. `glor_control.enable=false` 时，代码行为和原始 LoRA 一致。
2. `glor_control.enable=true` 时，可以正常训练。
3. `local_tokens.shape == [B, N, K, C_ctrl]`。
4. `global_tokens.shape == [B, M, C_ctrl]`。
5. `delta.shape == hidden_states.shape`。
6. `alpha` 初始化为 0，训练后逐渐变为非零。

### Super-resolution

1. 每个 HR token 能检索到对应 LR/upsampled LR 的局部 patch。
2. 输出结构比普通 LoRA 更稳定。
3. 细节比只用 global condition 更清晰。

### Frame interpolation

1. local tokens 同时来自 frame0 和 frame1。
2. 不同 tau 产生不同 global/time control。
3. tau=0 更接近 frame0，tau=1 更接近 frame1。

### Optical-to-SAR

1. 每个 SAR token 从 optical 局部 patch 检索信息。
2. global memory 能注入 SAR metadata。
3. 加 edge feature 后，SAR 结构边界应更清晰。

---

## Paper Wording

English:

> We propose Global-Local Observation Retrieval Control for remote sensing generation. Each target latent token retrieves condition information from a task-defined local observation neighborhood while being modulated by compact global observation memory. The local branch provides precise structure and texture evidence, while the global branch maintains scene-level consistency and sensor/modality context. This design improves condition absorption for super-resolution, frame interpolation, and optical-to-SAR translation without modifying the frozen DiT backbone.

Chinese:

> 我们提出全局-局部观测检索控制。每个目标 latent token 从任务定义的局部观测邻域中检索条件信息，同时由紧凑的全局观测记忆进行调制。局部分支提供精确的结构和纹理证据，全局分支保证场景级一致性和传感器/模态上下文。该设计在不修改冻结 DiT backbone 的情况下，提升了超分、插帧和光学到 SAR 任务中的条件吸收能力。

---

## Do Not Do in First Version

```text
Do not rewrite the backbone.
Do not jointly train all tasks.
Do not replace all attention globally.
Do not add optical flow in the first version.
Do not inject into every block at first.
Do not make SAR metadata mandatory.
```

First version:

```text
LoRA + GLoR-Control in middle/late blocks
```
