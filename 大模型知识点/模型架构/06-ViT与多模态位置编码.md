---
tags:
  - 面试
  - 大模型
  - 多模态
  - ViT
  - 位置编码
  - MRoPE
created: 2026-05-24
updated: 2026-05-24
---

# ViT 与多模态位置编码：二维位置 / 相对位置 / 多模态对齐

> 核心问题：图像不是 1D 序列，ViT 怎么处理位置？二维位置编码怎么设计？多模态模型如何让文本和图像/视频共享一套位置编码？

---

## 0. 面试怎么答（30 秒电梯版）

> "图像是 2D 结构，但 Transformer 输入是 1D token 序列，所以需要专门设计的位置编码。
>
> **ViT 原版**：把图像切成 patch，按 raster scan 顺序拉成 1D 序列，加一个 **可学习的 1D 位置 embedding**（其实包含 cls token 共 N+1 个）。
>
> **二维位置编码**：把 1D PE 替换为 2D，常见做法是把 $`(x, y)`$ 两个维度分别编码再拼接 / 求和。Swin Transformer 用 **相对位置 bias**。
>
> **RoPE 2D**：把 head_dim 拆成两半，前一半编 x 方向旋转，后一半编 y 方向旋转。
>
> **多模态位置编码**：
> - **简单做法**：图像 patch 拉成 1D 后和文本 token 拼接（如 LLaVA），用一套 1D PE
> - **MRoPE（Qwen2-VL）**：把 RoPE 分成 (time, height, width) 三段，文本 token 在三个维度上同步走，图像/视频 token 在 height/width 上展开
> - **Interleaved MRoPE（Qwen3-VL）**：把 t/h/w 三段交错排列在 head_dim 上，避免 head_dim 末尾位置编码缺失"

---

## 1. ViT 的原版位置编码

### 1.1 输入处理：图像 → patch 序列

```
原图 224×224×3
   │ 切成 16×16 patch
   ↓
14×14 = 196 个 patch
   │ 每 patch 展平 → 16*16*3 = 768 维向量
   ↓
线性投影到 d_model（例如 768）
   ↓
[CLS] + 196 个 patch token → 197 个 token，shape [197, 768]
```

### 1.2 位置 embedding

ViT 原版用**可学习的 1D 位置 embedding**：

```python
self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))  # 包含 cls
x = x + self.pos_embed
```

### 1.3 为什么 1D 也能 work

直觉上 2D 编码应该更好，但 ViT 论文做过实验：

| 编码方式 | ImageNet acc |
|---|---|
| 无位置编码 | 严重掉点 |
| 1D 可学习 | 79.1% |
| 2D 可学习 | 79.0% |
| 相对位置 | 79.4% |

**结论**：差别很小，可能因为模型自己能学到等效的位置关系（patch 数量也不大）。

### 1.4 1D PE 的痛点

- **不能跨分辨率**：训练 224×224（196 patch）就只能推理 224×224
- 需要 **interpolation** 才能换分辨率：

```python
def interpolate_pos_embed(pos_embed_old, num_patches_new):
    # 用双线性插值把旧位置 embedding 调整到新分辨率
    ...
```

DeiT、ViT 在不同分辨率推理时都需要这个 trick。

---

## 2. 二维位置编码

### 2.1 分别编 x 和 y 然后拼接 / 求和

对位置 $`(x, y)`$，把 $`d`$ 维分成两半：
- 前 $`d/2`$ 维编 x：$`PE_x(x)`$
- 后 $`d/2`$ 维编 y：$`PE_y(y)`$
- 拼接：$`PE(x, y) = [PE_x(x);\ PE_y(y)]`$

或者直接相加：

```math
PE(x, y) = PE_x(x) + PE_y(y)
```

(对 Sinusoidal 风格 PE 都成立)

### 2.2 Sinusoidal 2D PE

```python
def get_2d_sincos_pos_embed(d, h, w):
    # 分别为 h 和 w 方向生成 1D sin-cos
    grid_h = torch.arange(h)
    grid_w = torch.arange(w)
    pe_h = sincos_1d(d // 2, grid_h)  # [h, d/2]
    pe_w = sincos_1d(d // 2, grid_w)  # [w, d/2]
    # 外积 / 广播
    pe = torch.cat([pe_h.unsqueeze(1).expand(-1, w, -1),
                    pe_w.unsqueeze(0).expand(h, -1, -1)], dim=-1)
    return pe.reshape(h * w, d)
```

### 2.3 Learned 2D PE

两张可学习表分别表示 h 方向和 w 方向：

```python
self.pos_h = nn.Parameter(torch.zeros(H, d // 2))
self.pos_w = nn.Parameter(torch.zeros(W, d // 2))
```

### 2.4 优势

- **跨分辨率友好**：只要 h、w 不超训练范围都可
- 显式 2D 结构信息

---

## 3. Swin Transformer 的相对位置偏置

### 3.1 局部窗口 attention

Swin 把图像切成不重叠的 **window**（例如 7×7 patch），attention 只在 window 内计算。

### 3.2 相对位置 bias

每个 window 内有 $`M^2 = 49`$ 个 patch，两两位置差 $`(\Delta h, \Delta w)`$ 范围 $`[-(M-1), M-1]^2 = [-6, 6]^2 = 169`$ 种。

学一张 **bias 表** $`B \in \mathbb{R}^{(2M-1)(2M-1)}`$（或 $`\mathbb{R}^{(2M-1) \times (2M-1)}`$），按位置差查表加到 attention score：

```math
\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right) V
```

### 3.3 优点

- 显式 2D 相对位置
- 窗口内查表开销极小
- 窗口移位（shifted window）实现跨窗口连接

### 3.4 用户

Swin Transformer、Swin V2、MViT 等。

---

## 4. RoPE 2D：图像版的 RoPE

### 4.1 思路

对 1D RoPE 来说，把 head_dim 切成 $`d/2`$ 对二维子空间，每对独立旋转。

2D 推广：把 head_dim **切成两半**，前一半的子空间用 x 坐标做旋转，后一半用 y 坐标做旋转：

```
head_dim = d
   │
   ├── d/2 (前半): RoPE_x(x)   ← 对每个二维子空间，旋转角度 = x · θ
   └── d/2 (后半): RoPE_y(y)   ← 对每个二维子空间，旋转角度 = y · θ
```

### 4.2 优势

- 内积只依赖**二维相对位置 $`(\Delta x, \Delta y)`$**
- 跨分辨率泛化（理论上）
- 与 RoPE 各种扩展（NTK/YaRN）兼容

### 4.3 用户

- 多模态模型（Qwen-VL 系列）
- 一些纯视觉模型（如 CogView 2.0+）

---

## 5. 多模态位置编码：让文本 + 图像 / 视频共享坐标系

### 5.1 朴素拼接（LLaVA / MiniGPT-4 早期）

```
[ <text tokens> | <image patch tokens> | <text tokens> ]
       1D 位置 0..n      n+1..n+m         n+m+1..
```

直接当成一维序列，所有 token 用同一套 1D 位置编码。

**问题**：
- 图像内部的 2D 关系完全丢失
- 视频时间维度无法表达
- 长图像/长视频会挤压文本位置范围

### 5.2 MRoPE（Multimodal RoPE，Qwen2-VL）

Qwen2-VL 提出把 RoPE 的 head_dim 分成 **(t, h, w)** 三段：

```
head_dim = d
   │
   ├── d_t (时间维度): RoPE_t
   ├── d_h (高度维度): RoPE_h
   └── d_w (宽度维度): RoPE_w
```

不同模态的 token 在三个维度上的位置 ID 不同：

| Token 类型 | t 位置 | h 位置 | w 位置 |
|---|---|---|---|
| 文本 token | 同步递增 t | 0 | 0 |
| 单图 patch | 同图所有 patch 共享一个 t | patch 行号 | patch 列号 |
| 视频帧 patch | 帧号 t | 帧内行号 | 帧内列号 |

**优势**：
- 一套 RoPE 编码三种模态
- 文本与图像共享时间轴（看时序）
- 图像内部保留 2D 结构

### 5.3 Interleaved MRoPE（Qwen3-VL）

#### MRoPE 的潜在问题

把 head_dim 顺序切成三段：

```
head_dim: [─── t (低维) ───][─── h (中维) ───][─── w (高维) ───]
            高频              中频               低频
```

→ **高频部分只编 t，低频部分只编 w**，不同维度对位置的"分辨能力"不均衡。

#### Interleaved 解决方案

把 t/h/w 三段在 head_dim 上**交错排列**：

```
head_dim: [t][h][w][t][h][w][t][h][w][t][h][w]...
```

每隔三对二维子空间分别给 t/h/w，**三个维度都覆盖从高频到低频的全谱**，位置分辨能力均衡。

### 5.4 多模态位置编码对比

| 方法 | 文本 | 图像 2D | 视频时间 | 模型 |
|---|---|---|---|---|
| 1D PE 拼接 | ✅ | ❌ | ❌ | LLaVA, MiniGPT-4 |
| RoPE 2D（仅图像内）| ✅ | ✅ | ❌ | Qwen-VL v1 |
| MRoPE | ✅ | ✅ | ✅ | Qwen2-VL |
| Interleaved MRoPE | ✅ | ✅ | ✅（频段均衡）| Qwen3-VL |

---

## 6. 多模态位置编码的工程挑战

### 6.1 动态分辨率（Naive Dynamic Resolution）

传统：图像必须 resize 到固定尺寸（224×224 或 336×336）。

Qwen-VL 系列：**保留原始长宽比**，把图像切成可变数量的 patch。位置编码需要支持任意 (h, w)。

→ 这正是 RoPE 2D / MRoPE 的优势。

### 6.2 视频时序对齐

视频不同帧间隔可能不均匀（关键帧采样）。Qwen2.5-VL 提出 **Text-Timestamp Alignment**：

把视频帧的时间戳作为文本注入：`<vision_start>t=0.5s<frame_patches>t=1.0s<frame_patches>...`

→ 把绝对时间戳编进上下文，模型能学习"哪一帧对应几秒"。

### 6.3 跨模态长上下文

文本可能 4k token + 视频可能 1 万帧 patch → 整体长度爆炸。MRoPE + YaRN 组合是当前主流。

---

## 7. ViT 不同代际的位置编码总结

| 模型 | 位置编码 |
|---|---|
| ViT (2020) | 1D learned (with cls) |
| DeiT | 1D learned + token distillation |
| Swin | 相对位置 bias (window 内) |
| ViT 2D-RoPE 变体 | 2D RoPE |
| ConvNeXt | 无 PE（CNN 自带局部 PE） |
| Qwen-VL v1 | 2D RoPE for 图像 |
| Qwen2-VL | MRoPE (t/h/w) |
| Qwen3-VL | Interleaved MRoPE |
| LLaVA | 1D 拼接（继承 LLaMA） |
| GPT-4V / Gemini | 未公开（推测 MRoPE-like） |

---

## 8. 面试高频追问

### Q1：ViT 为什么用 1D PE 就够了？2D 不应该更好吗？
ViT 论文实验显示 1D learned 和 2D learned 差距小（< 0.5%），可能因为：① patch 数量不大（196），自学位置关系不难；② attention 是全局的，不依赖局部空间结构。
但**跨分辨率推理**时 1D 必须 interpolate，2D 天然友好。

### Q2：Swin 的相对位置 bias 和 T5 的有什么区别？
- T5 RB：1D，bucket 化处理远距离
- Swin RB：**2D**，且只在 window 内（最多 7×7），不需要 bucket
本质都是"用一个可学习的标量表，按位置差取值加到 attention score"。

### Q3：为什么多模态模型选 RoPE 而不是 ALiBi？
- ALiBi 的线性衰减假设在 2D / 多模态场景不合理（"左上角"和"右下角"未必应该按欧氏距离衰减）
- RoPE 的旋转结构可以**沿不同维度独立编码**，天然支持 t/h/w 拆分
- 工程上，MRoPE 是 RoPE 的自然推广

### Q4：MRoPE 怎么处理图像 + 文本混排？
每个 token 都有 (t, h, w) 三维位置：
- 文本 token：(当前序列位置, 0, 0)
- 图像 patch：(图像出现时的 t, 在图中的行, 在图中的列)
不同段的 t 是**连续递增的**，让文本和图像在时间维度共享坐标系。

### Q5：Interleaved MRoPE 相比 MRoPE 的好处？
MRoPE 把 head_dim 顺序切成 t/h/w，导致每个维度对应的频率范围不同：t 维度高频、w 维度低频。Interleaved 让三个维度**都覆盖完整频谱**，位置分辨能力均衡，长视频/高分辨率图像表现更稳。

### Q6：动态分辨率下，怎么避免位置编码 OOD？
RoPE 的旋转公式对任意位置都有定义，理论上不存在 OOD（不像 learned PE）。但训练时见过的"最大位置"决定了实际泛化范围，所以**训练数据要包含足够大的分辨率**。

### Q7：CLIP 用的位置编码？
CLIP-ViT 用的就是 ViT 原版的 **1D learned PE**。这也是为什么 CLIP 推理时往往要 resize 到 224。

### Q8：视频模型的位置编码怎么处理"时间"？
两条路：
- **时间位置编码**：把帧号编进 token PE（MRoPE 的 t 维度）
- **时间戳文本注入**：把时间作为字符串放进上下文（Qwen2.5-VL）
最新模型常常**两者都用**。

---

## 关联阅读

- [[04-RoPE-旋转位置编码]] —— RoPE 的数学基础
- [[05-位置编码全景]] —— 1D 各类位置编码
- [[01-MHA-多头注意力]] —— attention 的位置无关性
- [[08-Qwen3-VL核心创新点]] —— Interleaved MRoPE 实战

## 参考资料

- Dosovitskiy et al., *An Image is Worth 16x16 Words (ViT)*, ICLR 2021
- Liu et al., *Swin Transformer*, ICCV 2021
- Wang et al., *Qwen2-VL: Enhancing Vision-Language Models with MRoPE*, 2024
- Qwen Team, *Qwen3-VL Technical Report*, 2025
- Heo et al., *Rotary Position Embedding for Vision Transformer*, ECCV 2024
