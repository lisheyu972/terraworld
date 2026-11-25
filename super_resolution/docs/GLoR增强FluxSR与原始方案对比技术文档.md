# GLoR 增强 FluxSR 与原始方案对比技术文档

## 1. 文档目的与术语

本文对比两套面向 AID 遥感图像的 4 倍超分方案：

1. 原始方案：遥感预训练 FLUX Transformer + FluxSR-FTD + LoRA；
2. 当前方案：在相同方案中加入超分专用 GLoR-Control。

本文统一使用 **GLoR**（Global-Local Observation Retrieval，全局—局部
观测检索）这一名称。它不是通常所说的 Generalized LoRA 或其他名为
GLoRA 的低秩适配方法。当前模型中 LoRA 与 GLoR 是两个不同部分：

- LoRA 负责参数高效地适配冻结的 FLUX Transformer；
- GLoR 负责从原始 LR 图像显式检索全局与局部观测证据。

## 2. 共同任务与模型基础

两套方案处理完全相同的任务：

```text
输入：128×128 LR 遥感图像
输出：512×512 SR 遥感图像
倍率：×4
数据集：AID
训练划分：JSON 中 split=train
测试划分：JSON 中 split=test，共1970对
```

两套方案都使用：

- FLUX.1-dev 的 VAE、文本编码器、tokenizer 和 scheduler；
- `SR_TRANSFORMER_DIR` 指定的遥感预训练 FLUX Transformer
- rank-16 LoRA；
- FluxSR 的一步 FTD 训练与推理形式；
- 相同损失权重、优化器、学习率和 cosine scheduler。

结构对照的变量是是否加入 GLoR。

## 3. 原始 FluxSR-FTD 方案是怎么做的

### 3.1 数据进入模型的方式

原始方案先把 128×128 LR 图像通过 bicubic 放大到 512×512：

$$

I_{LR}^{up}=\operatorname{Bicubic}(I_{LR},512,512)

$$


然后分别使用冻结的 FLUX VAE 编码 bicubic-LR 和 HR：

$$
z_L=E_{VAE}(I_{LR}^{up}),\qquad
z_0=E_{VAE}(I_{HR})
$$

其中：

- \(z_L\) 是一步超分的起始 latent；
- \(z_0\) 是训练真值 latent；
- 推理时不使用 HR，只使用 \(z_L\)。

### 3.2 FTD 训练

训练时在 \(t\in[t_L,1]\) 上构造从 bicubic-LR latent 到噪声的轨迹：

$$
x_t=
\frac{1-t}{1-t_L}z_L+
\frac{t-t_L}{1-t_L}\epsilon
$$

Transformer 预测速度：

$$
v_\theta(x_t,t,c)
$$

其中 \(c\) 是文本描述等条件。FTD 残差约束为：

$$
\mathcal L_{FTD}=
\left\|
(\epsilon-z_0)-t_Lv_\theta(x_t,t,c)-(\epsilon-z_L)
\right\|_2^2
$$

模型还在 \(t_L=0.4\) 处执行一步重建：

$$
\hat z_0=z_L-t_Lv_\theta(z_L,t_L,c)
$$

$$
\hat I_{SR}=D_{VAE}(\hat z_0)
$$

### 3.3 原始方案如何利用 LR

原始方案中的 LR 信息主要通过以下路径进入：

```text
128×128 LR
    ↓ Bicubic
512×512 LR-up
    ↓ Frozen VAE
z_L
    ↓ 作为一步流恢复的 latent 起点
Frozen RS-FLUX Transformer + trainable LoRA
    ↓
预测速度并恢复 HR latent
```

也就是说，模型确实以 LR 为出发点，但没有一个独立机制让中间层的每个
目标 token 回到原始 LR 空间检索对应的结构证据。

### 3.4 原始方案的优势

- 一步推理，速度明显快于多步扩散超分；
- 直接从配对 AID 数据学习；
- 遥感预训练 Transformer 提供较好的领域先验；
- LoRA 参数量较小，不需要更新完整 FLUX；
- Y-MSE、Y-L1 和 SSIM-Y 目标直接优化亮度结构保真度。

### 3.5 原始方案的局限

1. LR 经过 bicubic 和 VAE 后，细边缘与规则纹理可能被平滑；
2. LR 信息主要隐式存在于 \(z_L\)，缺少显式的跨层观测读取；
3. 中间 latent token 无法只检索其对应 LR 位置附近的证据；
4. 全局场景与局部结构主要依赖冻结主干和 LoRA 自行学习；
5. 一步恢复中若早期结构判断有误，后续没有多步采样进行修正。

## 4. 原始通用 GLoR 思想

原始 GLoR 的目标是让每个目标 latent token 同时获得：

- Local evidence：边缘、纹理、局部空间对应；
- Global context：场景类型、整体布局和全图一致性。

通用流程为：

```text
Condition image
    ↓
Condition encoder
    ├── Global observation memory
    └── Local observation patches
             ↓
Target hidden state先读取全局信息
             ↓
使用全局调制后的query检索局部邻域
             ↓
融合 hidden/global/local
             ↓
alpha门控残差注入DiT
```

其核心不是简单拼接条件，也不是只做全局 cross-attention，而是
“全局先引导、局部再检索”：

$$
g_i=\operatorname{Attn}(Q_g(h_i),K_g(G),V_g(G))
$$

$$
l_i=\operatorname{Attn}
\left(Q_l(h_i+g_i),K_l(P_i),V_l(P_i)\right)
$$

$$
\Delta h_i=\operatorname{MLP}([h_i;g_i;l_i])
$$

$$
h_i'=h_i+\alpha_b\Delta h_i
$$

其中 \(G\) 是全局观测 token，\(P_i\) 是目标 token \(i\) 对应的局部
观测邻域，\(\alpha_b\) 是第 \(b\) 个注入层的可训练门控。

## 5. 当前 FluxSR-GLoR 方案是怎么做的

### 5.1 保留原始 FluxSR 主路径

当前方案没有替换 FluxSR，也没有改变一步推理公式。bicubic-LR 仍然
经过 VAE 得到 \(z_L\)，并作为一步恢复起点：

$$
\hat z_0=z_L-t_Lv_\theta(z_L,t_L,c,C_{GLoR})
$$

新增的 \(C_{GLoR}\) 是从原始 128×128 LR 提取的显式观测上下文。

因此当前模型同时存在两条互补 LR 路径：

```text
路径A：LR → Bicubic → VAE → z_L
作用：提供生成起点和低频内容

路径B：原始LR → GLoR encoder → global/local memory
作用：向中间层显式提供场景、边缘和局部结构证据
```

### 5.2 超分专用五通道观测

原始通用 GLoR 的超分第一版只要求 RGB。当前实现构造：

$$
O=\operatorname{Concat}(R,G,B,Y,E)
$$

其中：

$$
Y=0.299R+0.587G+0.114B
$$

$$
E=\operatorname{clip}
\left(
\frac{\sqrt{(S_x*Y)^2+(S_y*Y)^2+\varepsilon}}{4},
0,1
\right)
$$

- RGB 保存颜色与光谱外观；
- Y 与 PSNR-Y、SSIM-Y 的主要评测协议对齐；
- Sobel edge 强调道路、屋顶、桥梁和建筑轮廓。

五通道观测经过三层 CNN condition encoder，输出512维控制特征。

### 5.3 尺度嵌入

当前任务将 ×4 scale 输入两层 MLP：

$$
e_s=\operatorname{MLP}(4)
$$

然后广播到 condition feature：

$$
F=E_{cond}(O)+e_s
$$

这使控制模块显式知道当前超分倍率，而不是完全依赖数据分布推断。

### 5.4 全局观测记忆

condition feature 经过 \(4\times4\) adaptive average pooling：

$$
G=\operatorname{Flatten}
\left(
\operatorname{AAP}_{4\times4}(F)
\right)
$$

得到16个 global tokens。每个生成 token 都可以读取这些全局 token，
获得场景类别、整体纹理密度和布局信息。

### 5.5 空间对齐的局部检索

condition feature 先插值到目标 packed latent 网格，再使用3×3 unfold：

$$
P_i=\operatorname{Unfold}_{3\times3}
\left(
\operatorname{Resize}(F,H_z,W_z)
\right)_i
$$

每个目标 token 只从与其空间对应的9个局部候选中检索信息，而不是从
整张 LR 图像无约束地搜索。这保持了遥感配准数据的空间对应关系。

### 5.6 全局引导局部检索

当前代码严格采用先全局、后局部的两级检索：

1. hidden state 对16个全局 token 做 attention，得到 \(g_i\)；
2. 使用 \(h_i+g_i\) 作为 local query；
3. 在目标位置的3×3局部邻域中得到 \(l_i\)；
4. 拼接 \(h_i\)、\(g_i\)、\(l_i\)，经过融合 MLP 得到控制残差。

对于 FLUX hidden dimension 3072，当前融合过程为：

```text
[h, global_ctrl, local_ctrl]
        3 × 3072
            ↓
Linear(9216 → 1024) + SiLU
            ↓
Linear(1024 → 3072)
            ↓
delta
```

### 5.7 六点稀疏多深度注入

原始通用建议是注入 middle/late blocks。若对全部中后层注入，会重复
执行大量 attention，增加训练时间与显存。

当前方案使用6个显式注入点：

```yaml
inject_dual_indices: [6, 12, 18]
inject_single_indices: [12, 24, 36]
```

它们分别覆盖 dual-stream 与 single-stream 的中层、后层和末层。
每个注入点有独立门控：

$$
h^{b}_{out}=h^{b}_{in}+\alpha_b\Delta h^{b}
$$

当前门控初始化为：

```text
alpha_init = 0.001
```

较小的非零初值让 GLoR 从训练开始就能接收到梯度，同时避免突然扰乱
遥感预训练主干。

### 5.8 GLoR K/V 复用

同一个 batch 的 LR 观测不会随 Transformer block 改变，因此当前实现
在首次注入时计算并缓存：

```text
global K/V
local K/V
```

后续5个注入位置直接复用。每层 query 仍由该层 hidden state 单独生成，
所以各层仍可产生不同的检索权重。该优化降低了重复线性投影开销。

### 5.9 与 FTD 双前向共享上下文

每个训练 step 包含：

1. 随机 \(t\) 的 FTD 速度预测；
2. \(t_L\) 处的一步重建预测。

当前实现从原始 LR 构建一次 GLoR context，并同时用于两次预测。这保证
轨迹蒸馏约束和最终重建约束使用同一套观测证据。

### 5.10 不再复制 OminiControl 条件图分支

通用 OminiControl 条件生成通常把条件图作为额外图像 token 分支送入
Transformer。当前 FluxSR 已有 \(z_L\) 生成起点，再增加一整套条件图
分支会重复编码 LR，并增加序列长度。

因此当前配置：

```yaml
disable_omini_condition: true
```

GLoR 只向主生成 latent 分支注入观测残差。这样更适合一步配对超分，
也能把实验差异集中在全局—局部检索机制本身。

## 6. 总体流程对比

### 6.1 原始方案

```text
LR 128×128
    ↓ Bicubic
LR-up 512×512
    ↓ Frozen VAE
z_L
    ↓
Frozen RS-FLUX + trainable LoRA
    ↓ one-step velocity
z_hat
    ↓ Frozen VAE decoder
SR 512×512
```

### 6.2 当前方案

```text
                         ┌→ RGB/Y/Edge → Condition Encoder
                         │                    ├→ 4×4 Global Memory
LR 128×128 ──────────────┤                    └→ Aligned 3×3 Local Memory
                         │                                 ↓
                         │                  Global-guided Local Retrieval
                         │                                 ↓
                         │               6-point gated residual injection
                         │                                 ↓
                         └→ Bicubic → Frozen VAE → z_L → RS-FLUX + LoRA
                                                           ↓
                                                   one-step velocity
                                                           ↓
                                                    SR 512×512
```

## 7. 关键差异表

| 对比项 | 原始 FluxSR-FTD + LoRA | 当前 FluxSR-FTD + LoRA + GLoR |
|---|---|---|
| 一步超分主路径 | bicubic-LR latent 起点 | 完全保留 |
| LR条件使用 | 主要隐式存在于 \(z_L\) | \(z_L\) + 原始LR显式检索 |
| 原始LR分辨率输入 | 不直接进入中间控制模块 | 直接使用128×128 LR |
| 观测特征 | RGB经bicubic和VAE | RGB + Y + Sobel edge |
| 全局场景信息 | 依赖主干与LoRA隐式学习 | 16个global memory tokens |
| 局部空间对应 | 无显式检索 | 每个目标token检索对应3×3邻域 |
| 检索次序 | 无 | 全局调制后再局部检索 |
| DiT中间层控制 | 仅LoRA改变映射 | LoRA + GLoR门控残差 |
| 注入位置 | 无GLoR注入 | dual 6/12/18，single 12/24/36 |
| 门控 | 无 | 每点独立 \(\alpha_b\)，初值0.001 |
| 重复计算优化 | 不涉及 | 同一context缓存global/local K/V |
| FTD随机t前向 | 使用 \(z_L\) | 使用 \(z_L+C_{GLoR}\) |
| \(t_L\)重建前向 | 使用 \(z_L\) | 使用相同 \(C_{GLoR}\) |
| Omini条件分支 | 不使用 | 仍不使用，避免重复LR分支 |
| 可训练参数 | rank-16 LoRA | rank-16 LoRA + GLoR |
| 冻结参数 | Transformer、VAE、文本编码器 | 相同 |
| 推理步数 | 1 | 1 |

## 8. 损失函数是否改变

GLoR 配置沿用 FluxSR 基线的损失。

$$
\mathcal L=
0.25\mathcal L_{FTD}
+10\mathcal L_{RGB-MSE}
+\mathcal L_{RGB-L1}
+10\mathcal L_{Y-MSE}
+\mathcal L_{Y-L1}
+0.5(1-SSIM_Y)
+0.02\mathcal L_{perceptual}
$$

其中 Y-MSE、Y-L1 与 SSIM-Y 使用4像素裁边。ADL 权重为0。因此本次
对比不是通过更换损失函数获得优势，而是检验 GLoR 结构本身的作用。

## 9. 结构对照设置

| 实验变量 | 无 GLoR 的基线 | GLoR |
|---|---:|---:|
| Transformer 起点 | 同一遥感预训练权重 | 同一遥感预训练权重 |
| 旧LoRA热启动 | 否 | 否 |
| LoRA初始化 | 新建 | 新建 |
| LoRA rank | 16 | 16 |
| 初始学习率 | \(2\times10^{-5}\) | \(2\times10^{-5}\) |
| Optimizer | AdamW | AdamW |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Precision | bf16 mixed | bf16 mixed |
| 损失权重 | 相同 | 相同 |
| 结构差异 | 无GLoR | 加入GLoR |

## 10. 参数更新与 checkpoint

### 10.1 冻结

- 遥感预训练 FLUX Transformer 的原始参数；
- FLUX VAE；
- CLIP/T5 文本编码器；
- tokenizer 与 scheduler。

### 10.2 训练

- 新初始化的 rank-16 LoRA；
- GLoR condition encoder；
- scale embedding；
- global Q/K/V/out；
- local Q/K/V/out；
- control fusion MLP；
- 6个注入门控 \(\alpha_b\)。

每个 checkpoint 必须包含：

```text
default.safetensors    # LoRA
glor_control.pt        # GLoR
fluxsr_state.json      # 方法和训练状态元数据
```

推理时 LoRA 与 `glor_control.pt` 必须成对加载，不能只加载其中一个。

## 11. 训练与推理

数据准备、训练和推理命令见[超分辨率使用说明](../../README.md#super-resolution-on-aid)。

## 12. 方法创新点与工程优化的边界

### 方法核心

- 每个目标 latent token 同时读取紧凑全局记忆与空间对应局部观测；
- 全局信息先调制 query，再进行局部检索；
- 通过可学习门控残差把观测证据注入冻结 DiT；
- 保持一步 FluxSR 的推理效率。

### 当前超分专用改进

- RGB/Y/Edge 五通道结构观测；
- ×4 scale embedding；
- GLoR 与 FTD 两次前向共享同一原始LR context；
- 适配 FLUX dual/single 两类 block 的六点稀疏注入。

### 工程效率优化

- global/local K/V 在同一 context 中缓存；
- 不引入额外 OminiControl 图像 token 分支；
- 不对全部中后层进行稠密注入。

论文表述时应区分方法贡献与工程加速，避免把 K/V cache 本身描述成
GLoR 的主要算法创新。

## 13. 简要结论

原始 FluxSR 把 bicubic-LR 编码成 \(z_L\)，然后依靠遥感 FLUX 与 LoRA
在一步内预测恢复方向。LR 是生成起点，但没有显式的局部观测检索。

当前 FluxSR-GLoR 保留原始一步主路径，并新增一条从原始 LR 到
Transformer 中间层的控制路径。该路径用 RGB/Y/Edge 表示观测，通过
4×4全局记忆保证场景一致性，通过空间对齐的3×3局部检索补充边缘和
纹理，再在6个中后层位置以小门控残差注入。

因此二者最本质的区别是：

> 原始方法让模型“从 LR latent 出发生成 HR”；当前方法在此基础上，
> 进一步要求每个 HR latent token 在生成过程中“回看原始 LR 的全局
> 场景和对应局部证据”。
