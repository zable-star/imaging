# 基于激光距离选通序列的军事三维目标识别与二维平面假目标判别

> 版本：2026-07-06 初稿  
> 写作状态：基于当前仿真、训练和诊断结果形成的论文草稿；文献引用位置用“待补文献”标注，后续需要替换为正式参考文献。

## 摘要

激光距离选通成像能够在不同接收门延迟下获得目标的深度选择性响应，为区分真实三维目标和平面二维假目标提供了不同于普通二维图像分类的观测维度。针对军事三维目标样本获取困难、候选 3D 数据标签噪声较高以及二维平面诱饵容易在单帧图像中伪装真实目标的问题，本文构建了一个基于 Blender 的多 gate 距离选通仿真与识别验证流程。首先，从军事 3D 候选集中人工筛选 44 个有效模型，生成真实三维目标的三门 gate stack；随后，根据平面目标的选通响应模型，将二维假目标表示为同一二维轮廓在不同 gate 中的强度缩放序列。为削弱单帧外观捷径，进一步提出 hard projection 假目标构造方法，即利用真实三维 gate stack 的最大投影生成平面诱饵轮廓，再按平面深度响应生成多门序列。本文同时实现经验 Gaussian response 与矩形激光脉冲-矩形接收门重叠响应两种平面 gate response，其中后者用二者在回波到达时间上的重叠长度描述三角形或梯形选通响应。为检验单门结果是否受曝光差异影响，本文进一步构造 per-gate class-mean exposure matched 数据，将 flat_false 每个 gate 的全图均值匹配到 true3d 对应 gate。实验采用共享 CNN 编码器和 attention_residual 融合网络，在三随机种子下进行军事三分类、小样本迁移、true3d/flat_false 二分类、单 gate 消融和退化鲁棒性测试。结果显示，在 hard projection 经验响应设置下，完整三门 gate stack 的 true3d/flat_false 判别平均最佳验证准确率为 1.0000，而仅输入 gate_0、gate_1、gate_2 时分别为 0.5370、0.6296 和 0.5370；在更物理的 rectangular-overlap 设置下，完整三门仍达到 1.0000，而三组单门结果为 0.5000、0.8704 和 0.5000。经 gate 均值曝光匹配后，完整三门仍为 1.0000，gate_0、gate_1 和 gate_2 单门结果变为 0.5000、0.7222 和 0.5000。上述结果说明判别信息主要来自跨 gate 的响应变化，同时表明峰值 gate 的单帧优势部分来自曝光/亮度差异，部分仍可能来自前景面积或轮廓清晰度等结构线索。此外，曝光匹配后的 flat_false 跨 gate 相关性仍为 0.9698，而 true3d 为 0.3246，进一步验证了平面投影序列与真实三维深度响应序列的物理差异。本文结果表明，距离选通 gate stack 可为军事目标识别与二维平面假目标判别提供可解释的序列判别信息，并可作为后续光电融合和多模光神经网络高速识别研究的仿真基线。

关键词：激光距离选通成像；军事目标识别；二维假目标；gate stack；小样本迁移学习；attention_residual

## 1. 引言

军事目标识别通常面临样本获取受限、成像条件复杂、目标姿态变化大以及诱饵目标干扰等问题。普通二维图像识别方法主要依赖目标在单帧投影图像中的轮廓、纹理和局部结构信息。当平面假目标能够模拟真实目标的外观轮廓时，仅依赖二维投影可能不足以稳定区分真实三维目标与平面二维诱饵。激光距离选通成像通过控制接收门的时间位置，选择性接收不同距离范围内的回波，可在多个 gate 中获得同一目标的深度相关响应。因此，多 gate 图像序列不仅包含目标投影外观，还包含目标沿距离方向的结构响应变化。这一特性为真实三维目标与平面假目标判别提供了物理基础。

现有目标识别研究中，深度图、多视角图像、LiDAR 点云以及多模态融合方法均可用于增强空间结构感知能力（待补文献）。然而，在激光距离选通场景下，网络输入不是显式三维点云，也不是普通 RGB 图像，而是由多个接收门形成的 gate stack。如何构造具有物理含义的 gate stack 数据、如何模拟平面假目标在选通系统中的响应、以及如何证明网络利用了多 gate 序列而不是单帧亮度或轮廓捷径，是当前研究需要解决的问题。

本文围绕上述问题建立了一个从物理仿真、数据筛选、假目标建模、网络训练到结果诊断的完整验证流程。与直接将二维假目标设为异常黑图或局部残缺图不同，本文将平面假目标建模为同一二维轮廓在不同 gate 中的强度缩放序列。进一步地，为避免网络仅利用单帧外观差异，本文提出 hard projection false target：从真实三维目标的 gate stack 中取最大投影作为平面诱饵轮廓，使假目标单帧外观更接近真实目标，再通过平面深度响应生成多 gate 序列。通过该设置，可以更严格地检验完整 gate stack 是否提供了单门图像不具备的判别信息。

本文主要贡献如下：

1. 构建了基于人工筛选军事 3D 模型的距离选通仿真数据流程，形成包含真实三维目标、flat-echo 假目标和 hard projection 假目标的多 gate 数据集。
2. 提出了符合平面目标选通响应的二维假目标建模方式，并进一步提出 hard projection 假目标与矩形脉冲-门重叠响应以削弱单帧外观捷径。
3. 设计了跨 gate 相关性、前景掩膜 IoU 和归一化差分等 gate stack 物理诊断指标，用于量化平面假目标和真实三维目标的序列差异。
4. 在 44 个精选军事模型上验证了小样本迁移训练、true3d/flat_false 判别、单 gate 消融和退化鲁棒性，结果表明完整 gate stack 对 hard projection 与 rectangular-overlap 假目标判别具有关键作用。

## 2. 距离选通成像与平面假目标建模

### 2.1 多 gate 距离选通成像模型

设目标在空间位置处的反射率或散射强度为 \(\rho(x,y,z)\)，距离传播衰减和介质透过项为 \(T(z)\)，第 \(g\) 个接收门的深度响应函数为 \(H_g(z)\)。忽略系统常数后，第 \(g\) 个 gate 的图像可表示为：

```text
I_g(x,y)=∫ rho(x,y,z) T(z) H_g(z) dz + B_g(x,y) + N_g(x,y)
```

其中 \(B_g(x,y)\) 表示背景散射或系统偏置，\(N_g(x,y)\) 表示探测噪声和读出噪声。若激光脉冲函数和接收门函数均为矩形函数，则 \(H_g(z)\) 可理解为发射脉冲与接收门在回波到达时间上的重叠面积。两个矩形函数宽度相同时，响应随延迟呈三角形；宽度不同时，响应呈梯形；在上升沿或下降沿局部近似为线性变化。

对于真实三维目标，目标在深度方向具有一定厚度，不同部件分布于不同 \(z\)。因此不同 gate 会强调不同深度范围内的结构，gate stack 中的目标轮廓、前景区域和局部结构会随 \(g\) 变化。

### 2.2 平面二维假目标响应

对于位于单一深度 \(z_0\) 的平面二维假目标，可将其反射率近似写为：

```text
rho_false(x,y,z)=S(x,y) delta(z-z0)
```

其中 \(S(x,y)\) 为平面假目标的二维轮廓或反射图案。代入距离选通成像模型可得：

```text
I_g^false(x,y)=S(x,y) T(z0) H_g(z0)
```

令：

```text
A_g=T(z0)H_g(z0)
```

则：

```text
I_g^false(x,y)=A_g S(x,y)
```

该式说明，平面假目标在多个 gate 中不应表现为目标局部随机出现或残缺，而应表现为同一完整二维轮廓在不同 gate 中按标量 \(A_g\) 缩放。若三个 gate 采样在三角形或梯形响应的下降沿，则 \(A_g\) 可表现为近似线性衰减；若采样覆盖平台区或峰值附近，则也可能出现先升后降或平台保持。因此，平面假目标的强度序列应由脉冲-门函数重叠响应决定，而不是人为固定为单一线性衰减。

### 2.3 hard projection 假目标

在初始 flat-echo 数据中，平面假目标的整体亮度和前景面积可能与真实三维目标存在较明显差异，网络可能利用单帧亮度或轮廓捷径完成分类。为构造更严格的验证任务，本文提出 hard projection false target。设真实三维目标的三门图像为 \(\{I_0,I_1,I_2\}\)，定义最大投影轮廓：

```text
S_hard(x,y)=max_g I_g(x,y)
```

然后根据随机平面深度对应的 gate response 生成假目标序列：

```text
I_g^hard(x,y)=A_g S_hard(x,y)
```

其中 \(A_g\) 表示平面深度与第 \(g\) 个接收门的响应系数。本文实现两类 \(A_g\)。第一类为经验 Gaussian response，用于构造平滑的强度缩放序列；第二类为矩形脉冲-门重叠响应，用于更直接对应矩形激光函数和矩形门函数。设回波中心为 \(t_0\)，激光脉冲宽度为 \(\tau_p\)，第 \(g\) 个接收门中心为 \(t_g\)，门宽为 \(\tau_g\)，则矩形重叠响应可写为：

```text
A_g = A_min + (1-A_min) * |[t0-tau_p/2,t0+tau_p/2] ∩ [tg-tau_g/2,tg+tau_g/2]| / min(tau_p,tau_g)
```

当 \(\tau_p=\tau_g\) 时，该响应随延迟呈三角形；当二者宽度不等时可出现梯形平台；在响应边缘局部才近似线性衰减。因此，假目标强度不应被固定为单调线性下降，而应由采样 gate 与平面深度之间的脉冲-门重叠关系决定。由于 \(S_hard(x,y)\) 直接来自真实三维 gate stack 的最大投影，该假目标在单帧外观上更接近真实目标；但由于它仍是平面目标，其不同 gate 之间仍保持较高结构相似性。因此，该设置可以用于检验网络是否真正利用跨 gate 响应差异，而不是仅利用单帧图像外观。

## 3. 数据集构建与质量控制

### 3.1 军事 3D 模型筛选

原始军事 3D 候选数据存在标签错误、空模型、类别不符和模型截取异常等问题。因此，本文未直接使用候选池进行训练，而是基于缩略图审查表 `thumbnail_review.csv` 中的 `keep=1` 标记筛选有效模型。最终保留 44 个军事三维模型，包括：

| 类别 | 数量 |
|---|---:|
| 01_Main_Battle_Tank | 12 |
| 02_Fighter_Jet | 20 |
| 03_Attack_Helicopter | 12 |
| 合计 | 44 |

筛选后的模型用于生成真实三维 gate stack，输出目录为 `dataset_new/Military_3D_Gated_Selected44`。每个模型生成三张 gate 图像，共 132 张 gate PNG。对应的 hard projection 假目标和 true3d/flat_false 二分类数据集分别输出到 `dataset_new/Military_3D_HardFlatProjection_Selected44` 和 `dataset_new/Military_TrueFalse_Selected44_hard_projection`。矩形脉冲-门重叠响应版本输出到 `dataset_new/Military_3D_HardFlatRectOverlap_Selected44` 和 `dataset_new/Military_TrueFalse_Selected44_hard_rect_overlap`，同样包含 44 个同源假目标样本和 132 张 false gate PNG。曝光匹配控制数据输出到 `dataset_new/Military_TrueFalse_Selected44_hard_rect_overlap_mean_classgate_matched`，该版本保持 true3d 不变，并对 flat_false 的每个 gate 使用一个类别级缩放因子，使该 gate 的全图平均灰度匹配 true3d。

### 3.2 训练就绪与质量审计

为避免目录结构错误或缺失 gate 图像，本文使用训练就绪检查脚本统计每个类别的 gate PNG 数量、样本分组和有效样本数。真实三维目标、flat-echo 假目标、hard projection 假目标以及 true3d/flat_false 二分类数据均通过 readiness 检查。质量审计进一步统计图像最大灰度、平均灰度、前景像素和低对比图像数量，用于标记可能影响训练的异常样本。

值得注意的是，早期 `flat-echo gain10` 假目标在部分 gate 中偏亮，特别是 gate_0 的平均灰度明显高于真实三维目标，因此不宜作为最终主实验结论。本文将其作为 easy setting 或诊断过程保留，而以 hard projection 作为主要假目标验证设置。进一步的 rectangular-overlap 版本未出现像素饱和，且 false target 前景平均亮度在三个 gate 中分别约为 0.1031、0.1463 和 0.0891，低于 true3d 的 0.1689、0.2365 和 0.1525，因此不属于“假目标过亮导致分类过易”的设置。为进一步排除全局曝光差异，曝光匹配版本将 flat_false 三个 gate 的全图均值调整为 0.0139、0.0501 和 0.0222，基本对齐 true3d 的 0.0139、0.0500 和 0.0221。

## 4. 网络结构与训练设置

### 4.1 多 gate 融合网络

每个样本由三张灰度 gate 图像组成，输入形式为：

```text
[gate_0, gate_1, gate_2]
```

网络首先使用共享 CNN 编码器 `SliceEncoder` 对每张 gate 图像提取特征，得到 gate-level feature。随后通过 `attention_residual` 融合模块计算不同 gate 的判别贡献，并保留残差拼接信息，最后由 MLP 分类器输出类别 logits。需要强调的是，本文中的 attention 是门控切片级判别贡献，不是 Transformer 中的 QKV self-attention，也不是空间显著性图。

训练目标包括两类：

1. 军事三分类：`01_Main_Battle_Tank`、`02_Fighter_Jet`、`03_Attack_Helicopter`。
2. 二分类假目标判别：`true3d` 与 `flat_false`。

### 4.2 数据划分与评估

所有 true3d/flat_false 二分类实验均使用 `split_group_by_sample_id`，确保同一源模型生成的 true3d 和 false target 同时进入训练集或验证集，避免同源模型跨集合造成评估泄漏。所有主要实验使用随机种子 42、332 和 2026 重复运行，并报告平均最佳验证准确率、标准差、最小值和最大值。

## 5. 实验结果

### 5.1 小样本军事三分类

在 44 个精选军事样本上，本文比较了预训练迁移和从零训练。结果如下：

| 方法 | mean best val acc | std | seeds |
|---|---:|---:|---|
| Transfer frozen encoder | 0.7500 | 0.0000 | 42/332/2026 |
| Transfer finetune | 0.7500 | 0.0000 | 42/332/2026 |
| Scratch | 0.7083 | 0.1443 | 42/332/2026 |

结果表明，在当前小样本军事数据条件下，预训练迁移并未显著提高最高准确率，但降低了不同随机种子之间的波动。从零训练最高可达到 0.875，但标准差为 0.1443，说明其对随机初始化和划分更敏感。因此，本文更谨慎地将迁移学习结果解释为小样本稳定性收益，而不是绝对性能提升。

### 5.2 per-gate 归一化 true3d/flat_false 判别

为削弱绝对亮度差异，本文构造了逐 gate 最大值归一化数据集。完整三门输入在三随机种子下平均最佳验证准确率为 1.0000。单 gate 消融结果为：

| 输入 | mean best val acc | std |
|---|---:|---:|
| Full gate stack | 1.0000 | 0.0000 |
| Gate 0 only | 0.9630 | 0.0321 |
| Gate 1 only | 0.9630 | 0.0321 |
| Gate 2 only | 0.8889 | 0.0556 |

该结果说明，即使经过归一化，单帧图像中仍存在一定可学习差异。因此，per-gate 归一化适合作为亮度捷径诊断，但不足以完全证明网络依赖多 gate 序列。

### 5.3 hard projection 与 rectangular-overlap 消融

hard projection 设置用于压低单帧外观差异。为进一步贴近矩形激光脉冲和矩形接收门的物理过程，本文在 hard projection 基础上加入 rectangular-overlap 响应，即以脉冲-门重叠长度决定平面假目标在各 gate 的强度系数。实验结果如下：

| 响应模型 | 输入 | mean best val acc | std | min | max |
|---|---|---:|---:|---:|---:|
| Gaussian/empirical response | Full 3-gate stack | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Gaussian/empirical response | Gate 0 only | 0.5370 | 0.0321 | 0.5000 | 0.5556 |
| Gaussian/empirical response | Gate 1 only | 0.6296 | 0.0849 | 0.5556 | 0.7222 |
| Gaussian/empirical response | Gate 2 only | 0.5370 | 0.0642 | 0.5000 | 0.6111 |
| Rectangular pulse-gate overlap | Full 3-gate stack | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Rectangular pulse-gate overlap | Gate 0 only | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| Rectangular pulse-gate overlap | Gate 1 only | 0.8704 | 0.0849 | 0.7778 | 0.9444 |
| Rectangular pulse-gate overlap | Gate 2 only | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| Rectangular overlap + exposure matched | Full 3-gate stack | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Rectangular overlap + exposure matched | Gate 0 only | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| Rectangular overlap + exposure matched | Gate 1 only | 0.7222 | 0.1667 | 0.5556 | 0.8889 |
| Rectangular overlap + exposure matched | Gate 2 only | 0.5000 | 0.0000 | 0.5000 | 0.5000 |

该结果是本文最关键的消融证据。由于 hard projection 假目标的单帧轮廓来自真实三维 gate stack 的最大投影，单独输入任一 gate 时，经验响应版本接近随机，而输入完整三门 gate stack 时，模型在三随机种子下均达到 1.0000。rectangular-overlap 版本进一步验证了完整 gate stack 的价值：gate_0 与 gate_2 单独输入均为随机水平，完整三门仍稳定达到 1.0000。需要谨慎解释的是，rectangular-overlap 的 gate_1 单独输入达到 0.8704，说明当某一接收门恰好靠近回波峰值时，单帧中仍可能保留前景面积、响应强度或轮廓清晰度线索。曝光匹配后 gate_1 单门下降到 0.7222，而 full stack 仍为 1.0000，说明 gate_1 的原始单门优势部分来自全局曝光差异，但仍存在非曝光的单帧结构线索。因此，本文不将结果表述为“任意单门完全无效”，而是表述为“完整 gate stack 在更严格假目标设置下提供了最稳定、最充分的判别信息”。

为进一步分析 gate_1 残余单帧线索，本文又分别构造了 foreground-mean matched 和 p99 matched 控制集。两者仅改变 flat_false 的类别级强度缩放，使 gate_1 前景均值或 p99 高分位灰度接近 true3d。结果如下：

| gate_1 控制设置 | mean best val acc | std | 解释 |
|---|---:|---:|---|
| Rectangular-overlap raw | 0.8704 | 0.0849 | 峰值 gate 存在较强单帧线索 |
| Mean-all matched | 0.7222 | 0.1667 | 去除全图均值差异后下降 |
| Foreground-mean matched | 0.7222 | 0.1667 | 前景均值匹配后未继续下降 |
| p99 matched | 0.7222 | 0.1667 | 高分位亮度匹配后未继续下降 |
| Full stack, mean-all matched | 1.0000 | 0.0000 | 完整 gate stack 仍最稳定 |

该结果说明，gate_1 的残余单帧可分性不能完全归因于全局曝光、前景均值或高分位亮度，可能与平面最大投影和真实三维切片在局部结构、边缘分布或前景形态上的差异有关。相较之下，完整 gate stack 同时利用跨 gate 结构变化和响应序列，因此在曝光匹配后仍保持稳定判别。

### 5.4 gate stack 物理诊断

为了验证上述分类结果是否符合物理预期，本文计算跨 gate 相关性、前景掩膜 IoU 和归一化差分。结果如下：

| 数据 | 类别 | corr | mask IoU | absdiff |
|---|---|---:|---:|---:|
| per-gate norm | flat_false | 0.9995 | 0.9880 | 0.0055 |
| per-gate norm | true3d | 0.3244 | 0.3174 | 0.1211 |
| hard projection | flat_false | 0.9768 | 0.9301 | 0.0062 |
| hard projection | true3d | 0.3246 | 0.3065 | 0.1210 |
| rectangular-overlap | flat_false | 0.9731 | 0.4299 | 0.0172 |
| rectangular-overlap | true3d | 0.3246 | 0.3065 | 0.1210 |
| rectangular-overlap exposure matched | flat_false | 0.9698 | 0.4274 | 0.0210 |
| rectangular-overlap exposure matched | true3d | 0.3246 | 0.3065 | 0.1210 |

可见，平面假目标在不同 gate 之间保持高相关，而真实三维目标在不同 gate 中的结构差异明显。对于 Gaussian/empirical hard projection，flat_false 同时具有高相关和高前景 IoU，符合“同一轮廓强度缩放”的直观预期。对于 rectangular-overlap，flat_false 的相关性仍保持在 0.9731，但前景 IoU 降低到 0.4299，原因是部分 gate 的重叠响应接近 \(A_min\)，目标前景在固定阈值下被压缩甚至接近消失。曝光匹配后 flat_false 相关性仍为 0.9698，说明类别级曝光校正没有破坏平面假目标“同一轮廓缩放”的核心结构。这一现象不表示平面假目标发生了三维结构变化，而是弱回波 gate 的阈值效应。总体上，该诊断结果与平面假目标模型 \(I_g^false=A_g S(x,y)\) 和真实三维目标积分模型一致，为网络分类结果提供了物理解释。

### 5.5 退化鲁棒性

在 per-gate 归一化 true/false 数据上，本文进一步测试了随机 gate dropout 和噪声退化。结果如下：

| 设置 | mean best val acc | std |
|---|---:|---:|
| Clean full stack | 1.0000 | 0.0000 |
| Random gate dropout | 0.9074 | 0.0642 |
| Gaussian noise + background scatter + Poisson noise | 0.9815 | 0.0321 |

结果表明，中等强度噪声、背景散射和 Poisson 光子噪声下模型仍保持较高准确率，而随机丢失一个 gate 会导致更明显下降。这说明当前判别机制对 gate 序列完整性更敏感，后续系统设计应保证多门采集稳定性。

## 6. 讨论

### 6.1 激光选通的意义

如果仅从普通图像分类角度看，深度图或单帧图像似乎也能用于目标识别。然而 hard projection 消融结果表明，当二维假目标的单帧外观被构造得接近真实目标时，单 gate 输入明显弱于完整 gate stack。经验响应版本的三个单门结果接近随机；rectangular-overlap 版本中 gate_0 和 gate_2 为随机水平，gate_1 虽然保留较强峰值线索，但仍低于完整三门序列。进一步地，曝光匹配后 gate_1 单门从 0.8704 降至 0.7222，而完整三门仍为 1.0000，说明完整 gate stack 的优势不能简单归因于某一 gate 的全局亮度差异。因此，本文的关键结论不是“网络能识别亮度不同的假目标”，而是“跨 gate 响应模式提供了比单帧图像更稳定、更充分的判别信息”。这正是激光距离选通成像相对于普通二维成像的价值所在。

### 6.2 关于假目标亮度

早期 flat-echo gain10 数据中，假目标存在偏亮问题，尤其 gate_0 的平均灰度明显高于真实三维目标。这类数据可作为 easy setting 验证流程是否跑通，但不适合作为主要论文证据。本文将 hard projection 和 rectangular-overlap 作为主设置，其中 rectangular-overlap 假目标没有像素饱和，且三个 gate 的前景平均亮度均低于 true3d。该设置下 gate_0 和 gate_2 单门为随机水平，说明分类结果并非简单由“假目标更亮”造成；但 gate_1 单门仍有 0.8704。曝光匹配控制实验将 flat_false 与 true3d 的三个 gate 全图均值对齐后，gate_1 单门下降到 0.7222，但未降到随机水平，提示峰值 gate 的局部面积、轮廓清晰度或阈值前景结构仍可能作为单帧线索。后续还需要通过局部曝光匹配、阈值无关指标和复杂背景进一步控制这些因素。

### 6.3 小样本军事数据限制

当前军事数据仅包含 44 个高质量筛选模型，覆盖坦克、战斗机和攻击直升机三类。该规模足以支持方法可行性验证和消融分析，但不足以支撑大范围军事目标泛化结论。后续应继续扩充高质量模型数量、增加视角变化、姿态变化、材质反射率变化和背景复杂度，并尝试引入真实或半实物激光选通数据。

### 6.4 与多模光神经网络的关系

本文当前使用电子神经网络作为验证基线，其输入为多 gate 图像序列。后续可将 gate stack 编码到光学输入平面，将中间特征提取替换为多模光纤散斑传播或其他光学计算前端，再通过电子读出层完成目标识别。由于本文已经给出了仿真数据、假目标建模、网络基线和物理诊断指标，因此可作为后续多模光神经网络高速识别系统的任务入口和对照基线。

## 7. 结论

本文围绕军事三维目标识别与二维平面假目标判别，构建了基于 Blender 的激光距离选通 gate stack 仿真流程，并在人工筛选的 44 个军事三维模型上完成了多组验证。针对平面假目标建模，本文将其表示为同一二维轮廓在不同 gate 中的强度缩放序列，并提出 hard projection 假目标以削弱单帧外观捷径；进一步地，本文用矩形激光脉冲与矩形接收门的重叠长度生成 rectangular-overlap 响应，使假目标强度变化更贴近距离选通物理过程。实验结果显示，在经验 hard projection 设置下，完整三门 gate stack 的 true3d/flat_false 判别平均最佳验证准确率为 1.0000，而三个单 gate 输入仅为 0.5370、0.6296 和 0.5370；在 rectangular-overlap 设置下，完整三门同样为 1.0000，gate_0 和 gate_2 单门为 0.5000，gate_1 单门为 0.8704；在进一步的 gate 均值曝光匹配设置下，完整三门仍为 1.0000，而三个单门为 0.5000、0.7222 和 0.5000。gate stack 物理诊断进一步显示，平面假目标跨 gate 高相关，而真实三维目标跨 gate 低相关、低 IoU。上述结果表明，激光距离选通序列能够提供普通单帧图像不具备的物理可解释判别信息。未来工作将进一步扩展军事模型规模，补充 gate 参数消融和真实退化建模，并探索与多模光神经网络高速识别系统的融合。

## 数据与代码可复现实验入口

主要结果文件：

```text
experiments/military_selected44_results_overview_2026-07-06.csv
dataset_new/military_true_false_selected44_brightness_summary_2026-07-06.csv
dataset_new/military_true_false_selected44_hard_rect_overlap_mean_classgate_matched_brightness_2026-07-06.csv
dataset_new/military_true_false_selected44_hard_projection_gate_diagnostics_by_class_2026-07-06.csv
dataset_new/military_true_false_selected44_hard_rect_overlap_gate_diagnostics_by_class_2026-07-06.csv
dataset_new/military_true_false_selected44_hard_rect_overlap_mean_classgate_matched_gate_diagnostics_by_class_2026-07-06.csv
dataset_new/military_true_false_selected44_gain10_per_gate_norm_gate_diagnostics_by_class_2026-07-06.csv
```

主要图表：

```text
artifacts/figures/military_selected44_2026-07-06/true3d_vs_hard_projection_gate_stack.png
artifacts/figures/military_selected44_2026-07-06/hard_projection_full_stack_vs_single_gate.png
artifacts/figures/military_selected44_2026-07-06/hard_rect_overlap_full_stack_vs_single_gate.png
artifacts/figures/military_selected44_2026-07-06/hard_rect_overlap_exposure_matched_full_stack_vs_single_gate.png
artifacts/figures/military_selected44_2026-07-06/hard_rect_overlap_gate1_residual_controls.png
artifacts/figures/military_selected44_2026-07-06/gate_stack_physical_diagnostics.png
artifacts/figures/military_selected44_2026-07-06/per_gate_norm_robustness.png
```

主要脚本：

```text
dataset_new/build_selected_subset.py
dataset_new/build_true_false_dataset.py
dataset_new/build_hard_flat_projection_dataset.py
dataset_new/diagnose_gate_stack.py
run_military_transfer_experiments.py
scripts/plot_military_selected44_results.py
scripts/build_military_selected44_ppt.py
```

## 待补参考文献位置

1. 激光距离选通成像原理与矩形脉冲/门函数重叠响应：待补文献。
2. 距离选通图像在目标识别中的应用：待补文献。
3. 二维诱饵、平面假目标或欺骗目标建模：待补文献。
4. 小样本迁移学习在目标识别中的应用：待补文献。
5. 光神经网络或多模光纤散斑识别系统：待补文献。
