# 基于激光选通仿真的三维军事目标与平面假目标判别方法研究

> Draft date: 2026-07-07  
> Status: 框架稿。当前结果来自 44 个筛选后的军事 3D 模型、Blender 选通仿真、v8 纯矩形重叠响应、per-gate max-normalized 控制与 mixed clean/noisy augmentation。文献引用、英文润色、多视角扩展实验仍需补充。

## 摘要

激光选通成像通过控制发射脉冲与接收门的时间重叠，在不同距离窗口内获取目标回波图像，为三维目标识别和假目标判别提供了区别于普通二维图像的深度相关观测。然而，在仿真数据驱动的目标识别任务中，网络可能利用单帧亮度、黑帧或渲染伪影等非物理捷径完成分类，从而削弱选通成像机制本身的研究意义。针对这一问题，本文构建了一套基于 Blender 的激光选通成像仿真与反捷径验证流程，用于研究真实三维军事目标与平面二维假目标在多 gate 图像序列中的差异。

方法上，本文将三维目标按照可见深度范围生成多 gate 回波图像，并将平面假目标建模为在相机深度方向被压缩到单一深度面的目标反射。对于矩形激光脉冲和矩形接收门，本文采用二者时间重叠长度作为 gate 响应权重，构造具有明确物理含义的回波强度变化。进一步地，为避免模型依赖单 gate 亮度差异，本文引入 per-gate 最大值归一化、single-gate ablation 以及 clean/noisy mixed augmentation 作为控制实验。实验在 44 个筛选后的军事三维模型上进行，比较完整三 gate 序列与单 gate 输入在干净、轻噪声和强噪声条件下的判别性能。

实验结果表明，原始 v8 仿真数据仍存在较强的单 gate 强度捷径，其中 gate2 的 p99 标量阈值分类准确率达到 0.9886。经过 per-gate 最大值归一化后，最强单 gate 标量捷径下降至 0.7955，且主导特征由强度统计转为边缘/形状统计。在 mixed augmentation 三 seed 实验中，完整三 gate 序列在独立 clean、light-noise 和 strong-noise 评估中的平均准确率分别为 0.9697、0.9545 和 0.7424，均高于对应单 gate 平均结果。进一步的 full-stack 融合方式对比显示，attention_residual 在三 seed 验证中达到 1.0000 的平均最佳验证准确率，但独立噪声评估中不同融合方式存在 clean、light-noise 和 strong-noise 条件下的鲁棒性权衡。结果说明，多 gate 选通序列在受控仿真条件下能够提供优于单 gate 输入的判别信息，但当前结论仍局限于小样本、单视角、仿真数据条件，后续需要多视角、多目标类别和更大规模军事模型验证。

关键词：激光选通成像；三维目标识别；二维假目标；Blender 仿真；多 gate 融合；反捷径验证

## 1 引言

军事目标识别通常面临目标姿态变化、背景干扰、遮挡和主动欺骗等问题。常规二维图像识别主要依赖纹理、轮廓和局部结构特征，当平面假目标或诱饵在外观上与真实目标接近时，单幅二维图像可能难以提供可靠判据。激光选通成像利用发射脉冲与接收门的时间控制，在不同距离窗口内记录目标回波，因此不仅包含二维成像信息，还包含与目标深度结构相关的距离选择信息。这一特点使其适合用于三维目标识别和二维假目标判别。

现有深度学习方法能够从图像数据中学习高维判别特征，但在物理成像仿真数据上直接训练网络存在一个关键风险：网络可能学习到数据生成过程中的捷径，而不是学习预期的物理差异。例如，当平面假目标只在某一个 gate 中出现，而其他 gate 接近全黑时，网络可以通过黑帧或峰值亮度完成分类；当真实三维目标与假目标的整体曝光不同，网络也可能依赖最大灰度、p99 灰度或前景均值等简单统计量。这类捷径会导致高准确率结果缺乏物理解释，也不利于后续向真实实验或光电融合系统迁移。

本文的研究目标不是单纯追求小样本分类准确率，而是建立一个可解释、可诊断、可扩展的激光选通仿真与判别验证框架。核心问题包括：

1. 如何在 Blender 中构造具有明确物理含义的真实三维目标与平面假目标 gate stack；
2. 如何检查数据中是否存在单 gate 亮度捷径或黑帧捷径；
3. 多 gate 输入相对于单 gate 输入是否在受控噪声条件下提供稳定收益；
4. 哪种轻量级 gate 融合网络更适合作为后续多模光神经网络研究的电子基线。

本文贡献可以概括为以下四点：

1. 构建了基于 Blender 的激光选通军事目标仿真流程，支持真实三维目标和相机深度压平的平面假目标生成；
2. 在矩形激光脉冲和矩形接收门假设下，采用二者重叠长度建模 gate 响应，形成 v5-v8 的参数演化和物理诊断链；
3. 引入 single-gate scalar shortcut diagnostics、per-gate max-normalization 和 mixed clean/noisy augmentation，用于区分真实多 gate 判别贡献与数据捷径；
4. 在筛选后的 44 个军事模型上完成三 seed 消融，验证完整 gate stack 在 clean、light-noise 和 strong-noise 条件下相对单 gate 输入的平均性能优势。

## 2 激光选通回波建模与假目标构造

### 2.1 矩形脉冲与矩形接收门响应

设激光发射脉冲为矩形函数，持续时间为 \(T_l\)，接收门也为矩形函数，持续时间为 \(T_g\)。目标距离对应的回波到达时间为 \(\tau\)，第 \(i\) 个接收门中心为 \(c_i\)。在忽略复杂散射和系统响应展宽的简化条件下，某一距离处回波被第 \(i\) 个接收门接收的相对权重可用两个矩形窗的重叠长度表示：

\[
W_i(\tau)=\frac{\left|\mathrm{rect}_{T_l}(t-\tau)\cap \mathrm{rect}_{T_g}(t-c_i)\right|}{\min(T_l,T_g)}.
\]

当两个矩形窗等宽或宽度接近时，响应随相对位移呈三角形；当一个窗口明显宽于另一个窗口时，响应可能呈梯形。该模型说明，平面假目标并不应被渲染成只出现局部切片的目标，而应表现为整目标轮廓在不同 gate 中按照重叠响应发生整体强度变化。

### 2.2 真实三维目标 gate stack

真实三维目标具有非零深度范围。对于给定视角，目标不同部件位于不同相机深度，因此不同 gate 会选择性地接收不同深度区域的回波。本文在 Blender 中根据可见深度范围自动设置 gate 覆盖区域，并生成三张灰度 gate 图像：

\[
X = [I_0, I_1, I_2],
\]

其中 \(I_i\) 表示第 \(i\) 个接收门对应的选通图像。

### 2.3 平面假目标构造

平面假目标被建模为深度方向上被压缩到单一相机深度面的目标反射。与早期 PNG 后处理方式不同，本文在 Blender 几何阶段直接将模型顶点投影到同一相机深度，保证元数据中可验证：

```text
flat_geometry_mode = flatten-camera-depth
flat_geometry_depth_max - flat_geometry_depth_min = 0
```

平面假目标的 gate 序列保留整目标轮廓，并根据目标所在深度与各接收门的矩形重叠响应改变整体强度。这里的强度变化不是人为设定的线性衰减，而是矩形激光脉冲与矩形接收门的重叠长度在不同 gate 中的采样结果；当接收门宽度大于脉冲宽度时，该响应可以表现为带平台的梯形函数。为避免某一 gate 固定承载所有假目标，本文采用 round-robin 策略将平面假目标分布到 gate0、gate1 和 gate2 附近。

## 3 方法

### 3.1 数据生成版本演化

本文在实验推进过程中形成了 v5-v8 的仿真参数演化。

| version | main setting | observed issue |
|---|---|---|
| v5 | reflectance randomized, residual floor `FlatMinResponse=0.35` | 假目标较稳定，但残余亮度缺乏严格物理依据 |
| v6 | pure rectangular response, `FlatMinResponse=0` | 非命中 gate 过黑，产生黑帧捷径 |
| v7 | larger gate overlap, pure rectangular response | 黑帧减少，但真实三维 gate stack 过度平滑 |
| v8 | reduced overlap, pure rectangular response, per-gate maxnorm control | 物理性更好，但仍需处理单 gate 强度和形状捷径 |

v8 的主要参数为：

```text
ReceiverGateWidth = 1.5
LaserPulseWidth = 0.45
AutoGateMargin = 0.12
FlatMinResponse = 0.0
FlatEchoGain = 2.0
ReflectanceMode = hash-log-uniform
ReflectanceMin = 0.6
ReflectanceMax = 2.8
FlatGeometryMode = flatten-camera-depth
FlatTargetGateIndexMode = round-robin
```

### 3.2 反捷径控制

为判断网络是否利用预期的多 gate 信息，本文设计三类控制。

第一，gate-stack diagnostics。统计同一样本不同 gate 之间的归一化相关性、mask IoU 和归一化绝对差异。平面假目标应在轮廓上更一致，真实三维目标应体现深度切片差异。

第二，single-gate scalar shortcut diagnostics。对每个 gate 的简单标量特征进行阈值分类，包括 mean、foreground mean、p95、p99、max value、edge density 等。如果某一标量即可接近网络分类性能，则说明存在非预期捷径。

第三，per-gate max-normalization。将每张 gate 图像按自身最大值归一到统一目标上限，以削弱绝对亮度差异。该控制不作为真实物理成像结果，而作为验证网络是否依赖亮度捷径的实验条件。

### 3.3 网络结构

每个样本输入为三张单通道 gate 图像：

\[
X \in \mathbb{R}^{S\times 1\times H\times W}, \quad S=3.
\]

网络首先用共享 CNN 编码器提取每个 gate 的特征：

\[
f_i = E(I_i).
\]

随后比较三种融合方式：

1. mean：对 gate 特征均值融合；
2. attention：学习 gate-level 权重并加权求和；
3. attention_residual：在 attention 融合基础上加入拼接特征的残差投影。

分类头为轻量 MLP，输出 true3d 与 flat_false 两类 logits。该 attention 是 gate-level discriminative weighting，不是 Transformer QKV self-attention。

### 3.4 Mixed clean/noisy augmentation

为提高模型对成像噪声和背景散射的稳健性，本文引入确定性 mixed augmentation。每个样本以概率 \(p=0.5\) 施加退化，否则保持干净：

```text
gaussian_noise_std = 0.02
poisson_peak = 80
background_scatter = 0.02
degradation_probability = 0.5
```

独立评估阶段使用 clean、light-noise 和 strong-noise 三种条件：

| condition | gaussian | poisson peak | background scatter |
|---|---:|---:|---:|
| clean | 0.00 | 0 | 0.00 |
| light noise | 0.02 | 80 | 0.02 |
| strong noise | 0.05 | 30 | 0.05 |

## 4 实验设置

实验使用人工筛选后的 44 个军事三维模型。每个模型生成真实三维 gate stack 与对应平面假目标 gate stack，形成二分类数据集：

| class | samples | gate images |
|---|---:|---:|
| true3d | 44 | 132 |
| flat_false | 44 | 132 |

训练与验证采用按 sample_id 分组的划分方式，避免同源样本泄漏。主要指标为最佳验证准确率和独立评估准确率。每个主要实验使用 seeds `42/332/2026` 三次重复。

## 5 结果

### 5.1 v8 原始数据仍存在单 gate 捷径

v8 原始 gate-stack 诊断如下。

| class | corr | mask IoU | absdiff | max/mean ratio |
|---|---:|---:|---:|---:|
| flat_false | 0.5848 | 0.5702 | 0.1177 | 103.8464 |
| true3d | 0.6736 | 0.6670 | 0.0934 | 9.3942 |

原始 v8 数据的 strongest scalar shortcut 显示 gate2 的 p99 特征可达到 0.9886 的阈值分类准确率，说明直接使用原始图像会留下明显强度捷径。

| gate | feature | threshold acc |
|---:|---|---:|
| 0 | max_value | 0.8636 |
| 1 | p95 | 0.8864 |
| 2 | p99 | 0.9886 |

### 5.2 Per-gate max-normalization 抑制亮度捷径

经过 per-gate 最大值归一化后，最强单 gate 标量捷径降低至 0.7955，且主导特征由强度统计转为 edge density。

| gate | feature | threshold acc |
|---:|---|---:|
| 0 | edge_density | 0.7955 |
| 1 | p99 | 0.7386 |
| 2 | edge_density | 0.7841 |

这说明归一化控制有效削弱了简单亮度判据，但 CNN 仍可能利用单 gate 形状差异，因此需要继续进行 full stack 与 single gate 网络消融。

### 5.3 Mixed augmentation 下 full gate stack 优于单 gate

在 per-gate max-normalized 控制集上，mixed augmentation 三 seed 训练结果如下。

| input | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| Full 3-gate stack | 0.9697 | 0.0525 | 0.9091 | 1.0000 |
| Gate 0 only | 0.9091 | 0.0909 | 0.8182 | 1.0000 |
| Gate 1 only | 0.7273 | 0.0909 | 0.6364 | 0.8182 |
| Gate 2 only | 0.8939 | 0.0525 | 0.8636 | 0.9545 |

独立评估结果如下。

| evaluation condition | input | mean acc | std | min | max |
|---|---|---:|---:|---:|---:|
| clean | Full 3-gate stack | 0.9697 | 0.0525 | 0.9091 | 1.0000 |
| clean | Gate 0 only | 0.9091 | 0.0909 | 0.8182 | 1.0000 |
| clean | Gate 1 only | 0.7121 | 0.0263 | 0.6818 | 0.7273 |
| clean | Gate 2 only | 0.8636 | 0.0455 | 0.8182 | 0.9091 |
| light noise | Full 3-gate stack | 0.9545 | 0.0455 | 0.9091 | 1.0000 |
| light noise | Gate 0 only | 0.8636 | 0.1202 | 0.7273 | 0.9545 |
| light noise | Gate 1 only | 0.5151 | 0.0694 | 0.4545 | 0.5909 |
| light noise | Gate 2 only | 0.7727 | 0.0909 | 0.6818 | 0.8636 |
| strong noise | Full 3-gate stack | 0.7424 | 0.1389 | 0.5909 | 0.8636 |
| strong noise | Gate 0 only | 0.6970 | 0.1144 | 0.5909 | 0.8182 |
| strong noise | Gate 1 only | 0.5152 | 0.0263 | 0.5000 | 0.5455 |
| strong noise | Gate 2 only | 0.5152 | 0.0263 | 0.5000 | 0.5455 |

结果表明，完整三 gate 输入在 clean、light-noise 和 strong-noise 三种条件下均取得最高平均准确率。特别是在 light-noise 条件下，full stack 平均准确率为 0.9545，高于最强单 gate 的 0.8636。

### 5.4 融合方式对比

full-stack 融合方式对比如下。

| fusion mode | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| attention | 0.9697 | 0.0525 | 0.9091 | 1.0000 |
| mean | 0.9545 | 0.0788 | 0.8636 | 1.0000 |
| attention_residual | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

独立评估结果如下。

| evaluation condition | fusion mode | mean acc | std | min | max |
|---|---|---:|---:|---:|---:|
| clean | attention | 0.9697 | 0.0525 | 0.9091 | 1.0000 |
| clean | mean | 0.9545 | 0.0788 | 0.8636 | 1.0000 |
| clean | attention_residual | 0.9848 | 0.0263 | 0.9545 | 1.0000 |
| light noise | attention | 0.9545 | 0.0455 | 0.9091 | 1.0000 |
| light noise | mean | 0.9697 | 0.0525 | 0.9091 | 1.0000 |
| light noise | attention_residual | 0.9545 | 0.0455 | 0.9091 | 1.0000 |
| strong noise | attention | 0.7424 | 0.1389 | 0.5909 | 0.8636 |
| strong noise | mean | 0.6515 | 0.0694 | 0.5909 | 0.7273 |
| strong noise | attention_residual | 0.6970 | 0.1721 | 0.5000 | 0.8182 |

结果表明，attention_residual 是当前验证准确率和 clean 独立评估表现最好的融合候选，mean 在 light-noise 条件下略优，而 attention 在 strong-noise 条件下更稳健。因此本文不应将某一种融合方式表述为全面最优，而应将融合方式对比作为 full gate stack 电子基线设计的一部分。

## 6 讨论

当前结果支持一个较为谨慎但清晰的结论：在受控 Blender 激光选通仿真中，完整 gate stack 相比任一单 gate 输入提供了更稳定的真实三维目标与平面假目标判别信息。然而，该结论成立的前提是必须显式处理仿真捷径。若直接使用原始 v8 图像，网络可能利用 gate2 的 p99 强度特征完成分类；若不进行 single-gate 消融，也无法说明多 gate 的必要性。

本文目前仍存在三个主要限制。

第一，数据规模较小。当前仅使用 44 个筛选后的军事模型，适合作为 controlled simulation validation，不足以支持部署级结论。

第二，视角单一。当前主要使用 top view，网络可能学习固定视角下的形状差异。后续需要多视角渲染，检验 gate-stack 机制是否在姿态变化下仍有效。

第三，仿真仍较简化。当前使用矩形脉冲和矩形门函数，尚未充分考虑真实系统中的脉冲展宽、探测器响应、复杂背景散射、大气衰减和材料 BRDF 差异。

## 7 结论

本文构建了一个面向激光选通成像的军事三维目标与平面假目标判别验证框架。通过 Blender 几何压平构造平面假目标，并采用矩形脉冲与矩形接收门重叠响应建模 gate 强度变化，本文形成了具有物理解释的 true3d/flat_false gate stack 数据。实验表明，原始仿真数据存在明显单 gate 强度捷径；per-gate max-normalization 能够有效削弱该捷径；在 mixed clean/noisy augmentation 和三 seed 独立评估下，完整三 gate 序列在 clean、light-noise 和 strong-noise 条件下均取得高于单 gate 的平均性能。融合方式对比进一步说明，attention_residual、mean 与 attention 在不同噪声条件下各有优势，后续工作将扩展多视角渲染、增加军事模型规模，并在更大数据集上重新评估融合头的稳定性。

## 8 图表计划

建议论文初稿至少包含以下图表：

| figure/table | content |
|---|---|
| Fig. 1 | 激光选通 true3d 与 flat_false 生成流程图，已生成 `writing/figures/fig1_gated_imaging_framework.png` |
| Fig. 2 | 矩形激光脉冲与矩形接收门重叠响应示意图，已生成 `writing/figures/fig2_rectangular_overlap_response.png` |
| Fig. 3 | true3d 与 flat_false 的三 gate 示例图，已生成 `writing/figures/fig3_true3d_flatfalse_gate_examples.png` |
| Fig. 4 | raw v8 与 per-gate maxnorm 的 scalar shortcut 对比，已生成 `writing/figures/fig4_scalar_shortcut_control.png` |
| Fig. 5 | full stack 与 single gate 在 clean/noisy 条件下的三 seed bar chart，已生成 `writing/figures/fig5_full_stack_vs_single_gate_robustness.png` |
| Fig. 6 | full-stack fusion mode 在 clean/noisy 条件下的三 seed bar chart，已生成 `writing/figures/fig6_full_stack_fusion_robustness.png` |
| Table 1 | v5-v8 参数演化与问题 |
| Table 2 | gate-stack diagnostics |
| Table 3 | mixed augmentation 三 seed 网络消融 |
| Table 4 | fusion mode 验证与独立评估对比 |

## 9 需要补充的内容

1. 文献引用：需要加入激光选通成像、3D 目标识别、false target / decoy detection、depth prior / gate stack 相关文献。
2. 多视角实验：建议每个模型增加至少 4-8 个视角。
3. 多视角融合实验：需要在 4-8 个视角下重复 full stack、single gate 与 fusion mode 对比。
4. 图像示例：需要从 v8 raw 与 per-gate maxnorm 数据集中挑选 2-3 个代表性模型。
5. 英文题目与摘要：若目标是 SCI，最终需要英文主稿或至少英文摘要。
