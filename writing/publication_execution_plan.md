# 距离选通切片注意力网络论文执行计划

## 1. 总目标

把当前项目从“可以运行的 gated slice 分类 baseline”推进到“可以支撑论文初稿和投稿”的完整实验体系。

论文核心问题：

> 基于距离选通成像得到的多深度切片，是否包含可用于三维物体识别的结构信息？  
> gate center、gate width 等光学选通参数如何影响切片信息量和分类性能？  
> 注意力机制能否自动学习不同深度切片的重要性，并提供可解释分析？
> 当只有一个二维切片含有有效信息、其它切片为黑色时，模型性能与真正多切片输入相比下降多少？

项目主线：

```text
光学选通参数
    -> Blender 物理启发 gated slice 数据生成
    -> CNN / attention 网络识别
    -> 分类性能、鲁棒性、注意力权重分析
    -> 形成论文方法、实验和讨论
```

## 2. 暂定论文题目

英文题目候选：

```text
Gated-Viewing Slice Attention Network for 3D Object Classification from Depth-Selective Optical Slices
```

中文理解：

```text
基于距离选通切片注意力网络的三维物体分类方法
```

## 3. 预期贡献点

1. 构建一个基于 Blender 的物理启发 gated slice 数据生成流程。  
   不是普通几何切片，而是引入 gate response、距离衰减、表面法向和激光照明等因素。

2. 提出轻量级 Gated-Viewing Slice Attention Network。  
   共享 CNN 提取每个 gate 的切片特征，注意力模块学习不同深度切片的重要性。

3. 系统分析 gate 参数对识别性能和注意力分布的影响。  
   重点突出光学学生的优势：研究成像参数如何影响下游识别，而不是只堆深度学习模型。

4. 给出可解释分析。  
   通过 attention weights、confusion matrix、gate sparsity 和样例切片解释不同类别依赖哪些深度区域。

5. 增加二维图片识别对照。  
   使用同一批 gated slice 数据构造 2D baseline：只输入单个 gate 的二维图片，或保留多切片张量形状但仅一个 gate 有信息、其它 gate 全黑。该实验用于回答“单张二维图像是否已经足够”和“多深度切片是否带来额外识别信息”。

6. 增加第六类 `image2d` 异常/退化二维样本。  
   在 `dataset/image2d` 中生成与其它类别数量平衡的样本。每个样本仍包含 `gate_0/gate_1/gate_2` 三张图，但只有一个 gate 有二维图像信息，其余 gate 全黑。该类别用于测试注意力网络是否能把退化二维输入识别为独立类别，而不是误判成五个三维物体类。

## 4. 当前阶段

已经完成：

- 数据读取：`dataset.py`
- 训练入口：`train.py`
- CNN + attention 模型：`model.py`
- 训练曲线平滑和训练稳定化
- 中文代码注释
- CNN + attention 原理图
- 初步论文草稿：`writing/paper_draft.md`

当前关键缺口：

- 还缺系统消融实验。
- 还缺多 seed 统计。
- 还缺自动实验管理和结果汇总。
- 还缺高质量论文图表。
- 还缺面向投稿标准的论文初稿重写。

## 5. 优先实验路线

### 5.1 Baseline 固定

固定当前默认设置：

```text
gate_centers = [6.8, 7.4, 8.0]
receiver_gate_width = 1.0
laser_pulse_width = 0.45
range_loss_power = 2.0
```

训练 seed：

```text
42, 123, 2025
```

输出：

- best validation accuracy mean/std
- training curves
- confusion matrix
- attention weights
- summary table

当前默认类别扩展为：

```text
chair, desk, sofa, bed, toilet, image2d
```

### 5.2 Gate center / gate width 消融

Gate center 方案：

```text
A: [6.8, 7.4, 8.0]   当前方案，重叠大，不容易空
B: [6.8, 8.0, 9.2]   间隔大，前/中/后差异明显
C: [6.6, 7.8, 9.0]   折中覆盖
```

Gate width 方案：

```text
0.6, 1.0, 1.4
```

组合：

```text
3 gate center 设置 × 3 gate width 设置 × 3 seeds = 27 次训练
```

输出：

- gate 参数准确率表
- gate 参数热力图
- 每组样例切片可视化
- gate sparsity / active fraction 统计

### 5.3 模型结构消融

至少比较：

```text
Single gate_0
Single gate_1
Single gate_2
Single gate_0 + black gates
Single gate_1 + black gates
Single gate_2 + black gates
Average fusion
Concat fusion
CNN + attention
```

实现方式：

```text
--input-mode multi              # 默认多切片输入
--input-mode single-gate        # 真正二维输入：[B, 1, C, H, W]
--input-mode single-gate-black  # 保持 3 个切片，但只有指定 gate 有信息，其它 gate 全黑
--single-gate-index 0/1/2       # 选择 gate_0 / gate_1 / gate_2
```

二维对照实验优先级：

```text
1. single-gate gate_0 / gate_1 / gate_2
2. single-gate-black gate_0 / gate_1 / gate_2
3. 与 multi + attention 的结果比较 mean/std
```

3090 可用后增加：

```text
ResNet18 + attention
ResNet34 + attention
```

输出：

- 模型消融表
- 不同模型训练曲线
- attention vs non-attention 对比

### 5.4 物理参数消融

比较：

```text
range_loss_power = 0, 2, 4
```

含义：

- `0`：不考虑距离损耗
- `2`：扩展漫反射目标近似
- `4`：更强距离衰减

输出：

- 物理参数对准确率的影响
- 物理参数对 attention weights 的影响
- 样例切片亮度差异

### 5.5 鲁棒性实验

建议扰动：

```text
Gaussian noise
blur
intensity scaling
gate center offset
```

重点实验：

```text
训练 gate_centers: [6.8, 7.4, 8.0]
测试 gate offset: ±0.2, ±0.4
```

输出：

- 鲁棒性曲线
- 噪声强度 vs 准确率
- gate 偏移 vs 准确率

## 6. 3090 使用策略

3090 主要用于：

1. 跑多 seed 和大量消融。
2. 尝试更大 batch size。
3. 尝试更高输入分辨率。
4. 跑 ResNet18 / ResNet34 baseline。

建议优先级：

```text
第一优先级：多 seed + gate 参数消融
第二优先级：模型消融
第三优先级：更高分辨率和更强 backbone
```

不要一开始盲目上 ViT。当前数据规模较小，ViT 容易过拟合，论文解释也不一定更强。

## 7. 论文图表清单

必须完成：

- Fig. 1 方法整体结构图
- Fig. 2 Blender gated slice 生成流程
- Fig. 3 不同 gate 参数下的样例切片
- Fig. 4 训练曲线
- Fig. 5 confusion matrix
- Fig. 6 gate 参数消融热力图
- Fig. 7 每类 attention 权重分布
- Table 1 baseline 多 seed 结果
- Table 2 gate center / gate width 消融
- Table 3 二维单切片与多切片输入对照
- Table 4 模型结构消融
- Table 4 物理参数消融
- Table 5 鲁棒性实验

## 8. 论文初稿结构

```text
1. Introduction
   - 三维物体识别背景
   - 光学距离选通成像的优势
   - 当前问题：多深度切片的信息如何用于识别尚不清楚
   - 本文贡献

2. Related Work
   - 3D object recognition
   - range-gated imaging
   - optical/computational imaging with learning
   - attention-based multi-view fusion

3. Method
   - Blender gated imaging forward model
   - gated slice dataset construction
   - CNN-attention classifier
   - training details

4. Experiments
   - dataset setup
   - baseline performance
   - 2D single-slice baseline
   - gate parameter ablation
   - model ablation
   - robustness analysis
   - attention interpretation

5. Discussion
   - gate 间距和门宽如何影响信息量
   - 哪些类别依赖近/中/远切片
   - 仿真数据和简化物理模型的局限
   - 后续真实光学实验方向

6. Conclusion
```

## 9. 五天推进计划

### Day 1：实验管理脚本

目标：

- 建立批量实验 runner。
- 每次实验自动保存参数、模型、曲线、summary、attention csv。
- 自动汇总成一个总表。

完成标准：

```text
可以一条命令跑多个 seed，并生成 experiments/results.csv
```

### Day 2：Baseline 多 seed

目标：

- 当前 gate 参数跑 3 个 seed。
- 生成 baseline 结果表和平均值/标准差。

完成标准：

```text
Table 1 可以放进论文草稿
```

### Day 3：Gate 参数消融

目标：

- 批量生成不同 gate center / gate width 数据。
- 跑 9 组 gate 设置的训练。

完成标准：

```text
得到 gate 参数准确率表和热力图
```

### Day 4：模型消融

目标：

- 实现 single gate、single gate + black gates、average fusion、concat fusion。
- 与 attention 方法比较。

完成标准：

```text
得到二维/多切片对照表和模型消融表，证明多深度信息与 attention 的必要性
```

### Day 5：论文初稿整理

目标：

- 更新方法部分。
- 插入已完成图表。
- 写实验结果和讨论。

完成标准：

```text
形成一版可以给导师看的 paper draft
```

## 10. 最低可投稿版本标准

至少包含：

- baseline 多 seed 统计
- gate 参数消融
- 2D single-slice / black-slice 对照实验
- attention vs non-attention 消融
- 每类 attention 权重分析
- confusion matrix
- 样例 gated slice 可视化
- 方法原理图
- 清楚说明当前是物理启发仿真，不夸大为完整真实物理系统

## 11. 推荐投稿方向

较现实方向：

```text
Applied Optics
Optical Engineering
IEEE Photonics Journal
```

如果补充充分消融、更严谨物理模型，或者后续加入真实光学实验：

```text
Optics Express
```

当前策略：

先按 Applied Optics / Optical Engineering 的完整度要求推进，不急着追求过高目标。把实验体系做扎实，比单次最高准确率更重要。

## 12. 下一步立即任务

下一次工作从这里开始：

```text
实现 experiment runner：
1. 支持多 seed 训练；
2. 支持不同 artifact 输出目录；
3. 自动读取 summary.json；
4. 汇总 best_val_acc、参数和路径到 CSV；
5. 为后续 gate 参数消融留接口。
```

执行原则：

```text
先把实验管理搭好，再跑大量实验。
否则 3090 跑得越多，结果越容易乱。
```
