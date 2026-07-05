# 基于自适应距离门选通仿真的多层表面回波目标识别方法

Date: 2026-06-10

Status: 新版写作草稿。旧版 `writing/paper_draft.md`、`writing/paper_draft_attention_residual.md` 和实验段落文件保留不动。

## 摘要

激光距离选通成像通过控制激光脉冲与接收门之间的时间关系，选择性接收特定距离范围内的目标回波，从而在二维强度图像中引入距离分辨能力。与普通二维成像相比，多幅不同距离门下的选通图像能够提供近、中、远表面回波的互补信息，因而具有用于三维目标识别的潜力。然而，直接使用固定距离门对三维模型进行切片存在两个问题：不同模型的尺度和朝向并不一致，固定门位置容易只覆盖目标的一小部分；同时，如果将门外表面简单设为透明，虽然可以显示更多层内结构，但会偏离真实不透明目标的激光回波机理。

本文构建了一种基于物理启发式距离选通成像的三维目标识别框架。首先，利用 Blender 对三维模型进行统一尺度归一化和视角设定，并根据当前相机视角下模型的可见深度范围自动布置多个距离门。每个距离门的响应由激光脉冲函数和接收门函数的矩形窗卷积近似表示，并进一步结合表面法向、距离衰减和大气衰减项生成选通强度图像。其次，针对每个模型输出多幅 gate 图像、归一化深度图以及包含距离范围和门控参数的元数据，使数据生成过程具有可复现的物理参数记录。最后，采用共享切片编码器和多切片融合网络对多门控图像进行分类，并比较 mean、attention、concat 和 attention-residual 等融合策略。

现有 ModelNet10 预实验表明，多门控切片输入显著优于单切片输入，说明不同距离门确实提供了互补判别信息；attention-residual 融合在保留切片级权重解释能力的同时，取得了接近 concat 的分类性能。新版实验将进一步基于经过人工复核的军事三维目标数据集，验证自适应距离门选通仿真在坦克、战斗机、武装直升机、装甲车辆和军用发射车等目标识别任务中的有效性。

Keywords: 激光距离选通成像；自适应距离门；三维目标识别；物理启发式仿真；多切片融合；距离-能量相关

## 1. 引言

三维目标识别通常依赖点云、深度图、多视角图像或体素表示等数据形式。点云能够直接描述目标表面几何，但采样稀疏、传感器成本较高，且在远距离或恶劣天气下容易受到噪声影响。普通二维图像具有高空间分辨率和丰富纹理信息，但缺乏显式距离分辨能力。激光距离选通成像位于二者之间：它利用主动照明和时间门控机制，使相机只接收特定距离范围内的回波，从而在保持图像式空间分辨率的同时获得一定的距离选择能力。

距离选通三维成像文献通常并不是通过“看穿”目标内部来获得三维信息，而是利用多幅不同门控延时图像中的距离-能量关系，反演每个像素对应的目标表面距离。也就是说，真实激光选通成像观测的是不透明目标的可见外表面回波，而不是 CT 式内部切片。多幅选通图像所提供的三维信息主要来自同一视角下不同深度表面的回波强度变化、轮廓变化和遮挡关系。

本文关注的问题是：在三维目标识别任务中，基于距离选通机理生成的多幅表面回波图像是否能够提供有效判别信息？如果不同距离门的图像能够覆盖目标的近、中、远表面，神经网络是否可以利用这些互补信息提升分类性能？此外，在仿真数据生成中，如何避免固定 gate 设置与模型尺度、朝向不匹配，从而使生成数据更接近真实距离选通成像流程？

为回答这些问题，本文提出一种自适应距离门选通仿真流程。不同于早期版本中手动设定固定 gate centers 的做法，新流程先在模型归一化和相机视角确定后估计模型在相机坐标系下的深度范围，再将多个距离门自动布置到该范围内。这样，每个样本的 gate 图像都对应其自身的有效深度范围，而不是使用对所有模型都相同的固定距离参数。

本文贡献可概括为：

1. 提出一种面向三维目标识别的物理启发式距离选通仿真流程，根据模型可见深度范围自动设置 gate centers。
2. 将激光脉冲和接收门函数的卷积近似、表面法向项、距离衰减和大气衰减整合到 Blender 材质节点中，生成多门控表面回波图像。
3. 为每个样本额外导出归一化深度图和 metadata，记录深度范围、gate centers、门宽、脉冲宽度和渲染参数，为后续深度估计或物理参数消融提供依据。
4. 采用共享切片编码器和多切片融合网络进行目标识别，并以 ModelNet10 预实验验证多门控输入和 attention-residual 融合的有效性。
5. 针对 Objaverse 军事目标数据的标签噪声，设计严格关键词筛选、负例过滤和候选复核机制，为后续军事目标数据集构建提供更可靠流程。

## 2. 距离选通成像仿真原理

### 2.1 表面回波模型

在本文仿真中，每个三维模型首先被导入 Blender，并进行尺度归一化。给定相机视角后，模型表面点在相机坐标系下的深度定义为

```latex
R = |Z_{\mathrm{camera}}|.
```

对某一距离门 \(g\)，其中心距离为 \(R_g\)，接收门宽为 \(w_g\)，激光脉冲宽度为 \(w_p\)。本文将激光脉冲和接收门函数近似为矩形窗，其卷积结果等价于两个矩形窗在距离域上的重叠长度。距离门响应写为

```latex
W_g(R) =
\frac{
\min\left[
\max\left(
\frac{w_g+w_p}{2} - |R-R_g|,
0
\right),
\min(w_g,w_p)
\right]
}{
\min(w_g,w_p)
}.
```

因此，某一表面点在当前 gate 下的回波强度近似为

```latex
I_g(R) =
\left[
W_g(R)
\cdot \max(0,\mathbf n \cdot \mathbf l)
\cdot \frac{1}{R^p+\epsilon}
\cdot \exp(-2\sigma R)
\right]^{1/\gamma},
```

其中 \(\mathbf n\) 为表面法向，\(\mathbf l\) 为照明方向，\(p\) 为距离衰减指数，\(\epsilon\) 为避免近距离奇异的稳定项，\(\sigma\) 为大气消光系数，\(\gamma\) 为亮度压缩因子。

该模型保留了距离选通成像中的几个关键因素：距离门响应、表面朝向、距离衰减和传播衰减。但它仍属于物理启发式仿真，而不是完整的瞬态光传输或真实激光雷达物理引擎。

### 2.2 自适应 gate 布置

早期版本使用固定 gate centers，例如 \([7.4, 8.0, 8.6]\)。这种设置在单个样本上可以调参得到较好可视效果，但当模型类别、尺度和朝向变化较大时，固定 gate 容易出现两类问题：多个 gate 图像过于相似，或者只有一个 gate 命中目标主体。

新版流程采用 `--auto-gate-fit visible-bounds`。在模型归一化后，脚本计算模型包围盒八个顶点在相机坐标系下的深度范围：

```latex
R_{\min} = \min_j |Z_{\mathrm{camera},j}|,
\quad
R_{\max} = \max_j |Z_{\mathrm{camera},j}|.
```

给定 gate 数量 \(N\) 和边缘比例 \(m\)，自动 gate centers 定义为

```latex
R_{g,i}
=
R_{\min} + m(R_{\max}-R_{\min})
+
i
\frac{(1-2m)(R_{\max}-R_{\min})}{N-1},
\quad i=0,\ldots,N-1.
```

这种方式使每个样本的 gate 覆盖自身的近、中、远表面范围。以坦克样本 `01_Main_Battle_Tank_010_aa0919f0.glb` 为例，自动估计的深度范围为 \(6.7035\) 到 \(9.2965\)，对应的三个 gate centers 为

```text
6.9110, 8.0000, 9.0890
```

这比固定参数更接近真实系统中“先对准目标距离范围，再布置接收门”的成像逻辑。

### 2.3 遮挡、透明模式与物理合理性

本文实现了两种 gate 可见性模式：

```text
--gate-visibility emission
--gate-visibility transparent
```

`emission` 模式中，门外表面回波强度接近 0，但表面仍然保持不透明并遮挡后方结构。这更符合真实不透明目标的激光成像，因为坦克、飞机、车辆等目标不会被激光看穿。

`transparent` 模式中，门外表面被设为透明，使后方处于当前 gate 范围内的表面能够显示出来。这种模式可以产生更丰富的层内结构，但更接近几何 slab 可视化，而不是严格真实激光回波。本文将 `emission` 作为主要物理仿真模式，将 `transparent` 作为可视化和诊断工具。

### 2.4 深度图与元数据导出

为使仿真过程更接近距离-能量相关三维成像思想，新版脚本为每个样本额外导出归一化深度图：

```text
*_depth.png
```

该深度图将相机坐标深度按当前样本的 \(R_{\min}\) 和 \(R_{\max}\) 归一化，用于记录目标在当前视角下的深度分布。脚本同时输出 metadata：

```text
*_metadata.json
```

其中包含源模型路径、gate centers、门宽、脉冲宽度、深度范围、目标尺度、gate visibility 和 depth definition 等信息。该设计使后续实验能够追踪每个样本的物理参数，并为可能的深度辅助学习提供监督信号。

## 3. 数据集构建

### 3.1 ModelNet10 预实验数据

前期实验基于 ModelNet10 中的五个通用三维类别构建：

| 类别 | 含义 |
|---|---|
| chair | 椅子 |
| desk | 桌子 |
| sofa | 沙发 |
| bed | 床 |
| toilet | 马桶 |

在五类正常三维目标之外，额外构造 `image2d` / flat-target 异常类别。早期版本将该类别定义为一个含有目标信息的切片和两个全黑切片，用于测试网络是否能够区分正常多 gate 输入与缺帧式二维退化输入。新版实验不再将该设置作为主要二维假目标定义，而是采用 flat-echo 模式：同一个完整二维目标轮廓在多个 gate 下均有回波响应，但由于目标深度厚度很小，只有靠近平面深度的 gate 响应最强，其余 gate 保留可见的弱响应。该设置更接近真实二维平面假目标或平面反射片在距离选通系统中的表现。

这组数据的作用是验证网络结构和多切片融合策略，而不是作为最终军事目标实验结果。后续若继续保留二维假目标类，应优先使用 flat-echo 版本；早期 single-active 版本可作为缺帧/退化输入对照，而不应被解释为真实物理二维目标。

### 3.2 Objaverse 军事目标数据

新版实验计划使用 Objaverse 下载军事目标模型，包括：

| 类别 | 目标 |
|---|---|
| 01_Main_Battle_Tank | 主战坦克 |
| 02_Fighter_Jet | 战斗机 |
| 03_Attack_Helicopter | 武装直升机 |
| 04_Armored_Vehicle | 装甲车辆 |
| 05_Military_Truck_SAM | 军用卡车 / 防空导弹发射车 |

初始关键词下载存在两个问题：类别数量不均衡，以及部分模型并非对应物品。例如 `tank` 可能匹配水箱、储罐或玩具坦克，`truck` 也可能匹配普通民用卡车。为此，新版下载脚本采用强关键词、弱关键词上下文约束和类别负例过滤机制。筛选过程会输出 `candidate_review.csv`，用于人工复核候选模型名称、标签和匹配原因。

因此，后续正式实验应基于 `Military_3D_Dataset_curated` 或其他人工复核后的目录，而不应直接使用最初的粗筛数据。

## 4. 网络结构

### 4.1 多 gate 输入表示

每个样本由 \(S\) 张 gate 图像组成，输入张量为

```latex
\mathbf X \in \mathbb R^{B \times S \times C \times H \times W},
```

其中 \(S=3\)，\(C=1\)，\(H=W=224\)。每张 gate 图像首先通过共享 CNN 编码器得到切片级特征：

```latex
\mathbf f_i = E(\mathbf x_i),
\quad i=1,\ldots,S.
```

共享编码器保证不同距离门图像被映射到同一特征空间，同时减少参数量。

### 4.2 融合策略

本文比较四种多切片融合策略：

| 方法 | 融合方式 | 特点 |
|---|---|---|
| mean | 等权平均 | 最简单，参数最少 |
| attention | 学习 gate 权重并加权求和 | 可输出切片贡献权重 |
| concat | 直接拼接所有 gate 特征 | 信息保留最多，但解释性较弱 |
| attention-residual | attention 加权 + 拼接残差分支 | 在解释性和信息保留之间折中 |

attention 融合定义为

```latex
\mathbf f_{\mathrm{att}}
=
\sum_{i=1}^{S}\alpha_i \mathbf f_i,
\quad
\alpha_i =
\frac{\exp(a(\mathbf f_i))}
{\sum_j \exp(a(\mathbf f_j))}.
```

attention-residual 进一步加入拼接残差分支：

```latex
\mathbf f_{\mathrm{res}}
=
\mathrm{MLP}([\mathbf f_1,\ldots,\mathbf f_S]),
```

```latex
\mathbf f
=
\mathbf f_{\mathrm{att}} + \mathbf f_{\mathrm{res}}.
```

在当前阶段，网络主体不需要因为自动 gate 或 depth 输出而立即修改。自动 gate 仍然输出三张 gate 图像，现有 `MultiSliceObjectDataset` 和 `SliceAttentionClassifier` 可以直接读取。`*_depth.png` 暂时作为分析和潜在辅助监督，不直接输入分类网络，以避免变成 “gated images + oracle depth” 的不公平设置。

## 5. 实验设计

### 5.1 已完成预实验

ModelNet10 六分类预实验使用三个随机种子：

```text
42, 332, 2026
```

主要结果如下。

#### 多切片输入与单切片输入

| 输入方式 | 有效切片 | Seed 42 | Seed 332 | Seed 2026 | 平均准确率 | 标准差 |
|---|---|---:|---:|---:|---:|---:|
| Multi-slice attention | gate_0 + gate_1 + gate_2 | 94.17% | 90.83% | 93.33% | 92.78% | 1.73% |
| Single-gate | gate_0 | 80.00% | 75.83% | 85.00% | 80.28% | 4.59% |
| Single-gate | gate_1 | 82.50% | 81.67% | 82.50% | 82.22% | 0.48% |
| Single-gate | gate_2 | 84.17% | 81.67% | 86.67% | 84.17% | 2.50% |
| Single-gate-black | gate_0 | 82.50% | 75.00% | 86.67% | 81.39% | 5.91% |
| Single-gate-black | gate_1 | 78.33% | 77.50% | 78.33% | 78.06% | 0.48% |
| Single-gate-black | gate_2 | 85.00% | 81.67% | 83.33% | 83.33% | 1.67% |

该结果说明，完整三 gate 输入显著优于任意单 gate 输入，多门控图像确实提供了互补判别信息。

#### 融合策略消融

| 融合方式 | Seed 42 | Seed 332 | Seed 2026 | 平均准确率 | 标准差 |
|---|---:|---:|---:|---:|---:|
| mean | 92.50% | 89.17% | 94.17% | 91.94% | 2.55% |
| attention | 94.17% | 90.83% | 93.33% | 92.78% | 1.73% |
| attention-residual | 95.83% | 93.33% | 95.00% | 94.72% | 1.27% |
| concat | 96.67% | 93.33% | 95.83% | 95.28% | 1.73% |

concat 取得最高平均准确率，说明完整保留三个 gate 特征对分类有利。attention-residual 略低于 concat，但优于原始 attention，并保留切片级权重输出，因此更适合作为兼顾性能和解释性的主模型。

### 5.2 gate spacing 预实验

在五分类 ModelNet10 物理参数消融中，使用 attention-residual 模型比较 small、default 和 large 三种 gate spacing。结果显示 large spacing 平均准确率最高：

| Gate spacing | Seed 42 | Seed 332 | Seed 2026 | Mean | Std |
|---|---:|---:|---:|---:|---:|
| small | 93.00% | 91.00% | 92.00% | 92.00% | 1.00% |
| default | 95.00% | 90.00% | 93.00% | 92.67% | 2.52% |
| large | 97.00% | 93.00% | 94.00% | 94.67% | 2.08% |

该结果支持本文新版改进方向：gate 之间需要具有足够区分度。自动 gate fitting 的目的正是让每个样本的 gate 覆盖自身深度范围，并避免固定门参数导致图像过近或命中不足。

### 5.3 待完成新版实验

新版实验应包含以下内容：

1. Curated Objaverse 军事目标五分类实验。
2. 固定 gate 与 auto-gate 的对比。
3. `emission` 与 `transparent` 的对比，其中 `emission` 作为主物理模式，`transparent` 作为诊断模式。
4. attention、attention-residual、concat 的融合策略对比。
5. depth 图作为辅助监督的探索性实验，但不作为主分类结果。

## 6. 讨论

### 6.1 为什么自动 gate 更合理

真实距离选通成像系统需要根据目标距离范围设置接收门延时。如果 gate 位置与目标距离不匹配，即使目标存在，也可能只有少量回波被接收。固定 gate 参数在单一数据集或单一尺度下可以工作，但不适合尺度和朝向差异较大的 Objaverse 模型。自动 gate fitting 将 gate centers 绑定到每个模型的当前深度范围，因此更接近真实系统中“根据目标距离窗口进行成像”的流程。

### 6.2 为什么不直接使用透明切片作为主结果

透明模式可以让后方结构显示出来，视觉上更像“切到内部”，也可能提供更多分类信息。但真实激光照射不透明目标时，前表面会遮挡后表面。真实距离选通成像提供的是表面距离层信息，而不是内部体切片。因此，本文主实验应使用 `emission` 模式，以保持物理合理性；`transparent` 模式可以用于解释、调参或构建非物理的几何对照实验。

### 6.3 当前仿真的边界

本文仿真仍然有明显简化：

1. 使用相机坐标 \(Z\) 作为距离近似，而不是严格欧氏往返距离。
2. 激光脉冲和接收门被近似为矩形窗。
3. 未建模真实 BRDF、目标材质反射率差异和偏振效应。
4. 未加入传感器响应、读出噪声、光子噪声、背景光和散射介质。
5. 自动 gate 当前基于包围盒深度范围，而非逐像素可见表面深度分布。

因此，本文应将方法表述为“基于距离选通成像机理的物理启发式仿真”，而不是完整真实激光物理引擎。

### 6.4 网络是否需要修改

当前不需要立即修改网络主体。新版数据仍由多张 gate 图像组成，现有共享 CNN 编码器和多切片融合结构可以直接使用。短期内应优先验证数据生成改进是否带来更稳定的分类结果。只有在完成 auto-gate 主实验后，才建议考虑引入 depth 辅助分支：

```latex
\mathcal L =
\mathcal L_{\mathrm{cls}}
+
\lambda \mathcal L_{\mathrm{depth}}.
```

该分支可以让网络学习从 gated images 到 depth map 的距离-能量关系，但需要谨慎区分“真实可获得输入”和“训练时辅助监督”。

## 7. 结论

本文提出了一种自适应距离门选通仿真和多门控图像融合识别框架。与固定 gate 的早期版本相比，新流程根据每个模型在当前视角下的深度范围自动布置 gate centers，并输出 gate 图像、深度图和完整 metadata，使数据生成过程更符合距离选通成像的距离-能量相关思想。现有 ModelNet10 预实验表明，多 gate 输入显著优于单 gate 输入，attention-residual 在保持切片级解释能力的同时取得接近 concat 的分类性能。后续工作将重点转向经过人工复核的军事三维目标数据集，系统比较固定 gate 与 auto-gate、不同融合策略以及物理模式与诊断模式的差异。

## 8. 待补实验与图表

| 项目 | 状态 | 建议文件或输出 |
|---|---|---|
| Curated Objaverse 军事目标数据集 | 待完成 | `dataset_new/Military_3D_Dataset_curated` |
| auto-gate 批量切片 | 单样本已验证 | `dataset_new/dataset_gated_autogate` |
| 单样本 auto-gate 示例 | 已完成 | `dataset_new/slice_test_autogate_v4` |
| depth map 导出 | 已完成初版 | `*_depth.png` |
| metadata 导出 | 已完成 | `*_metadata.json` |
| 固定 gate vs auto-gate 实验 | 待完成 | 训练结果表 |
| military 5-class 分类结果 | 待完成 | `experiments/military_autogate_*` |
| 融合策略消融 | ModelNet 已有，military 待补 | mean / attention / attention-residual / concat |
| 方法流程图 | 需更新 | 加入 auto-gate 和 depth metadata |

## 9. 推荐命令

单样本验证：

```powershell
& "E:\新建文件夹 (2)\blender-launcher.exe" --background --python origindataset\gated_blender_physical.py -- --single-model "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset_new\Military_3D_Dataset\01_Main_Battle_Tank\01_Main_Battle_Tank_010_aa0919f0.glb" --output-root "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset_new\slice_test_autogate_v4" --target-size 6.0 --camera-view top --auto-gate-fit visible-bounds --auto-gate-margin 0.08 --receiver-gate-width 0.9 --laser-pulse-width 0.45 --gate-visibility emission --export-depth --render-device cpu
```

批量切片：

```powershell
& "E:\新建文件夹 (2)\blender-launcher.exe" --background --python origindataset\gated_blender_physical.py -- --input-root "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset_new\Military_3D_Dataset_curated" --output-root "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset_new\dataset_gated_autogate" --target-size 6.0 --camera-view top --auto-gate-fit visible-bounds --auto-gate-margin 0.08 --receiver-gate-width 0.9 --laser-pulse-width 0.45 --gate-visibility emission --export-depth --render-device cpu
```

训练建议：

```powershell
python train.py --dataset-root dataset_new\dataset_gated_autogate --classes 01_Main_Battle_Tank 02_Fighter_Jet 03_Attack_Helicopter 04_Armored_Vehicle 05_Military_Truck_SAM --expected-num-slices 3 --fusion-mode attention_residual
```
