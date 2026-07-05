# 发表推进计划：距离选通多 gate 切片识别

日期：2026-06-02

依据：`writing/zotero_gated_imaging_insights_2026-06-02.md`

## 一句话论文主张

在简化但具有距离选通物理机制的 Blender 仿真数据上，本文验证多深度 gate 切片比单切片输入提供更强的三维目标判别信息；attention-residual 融合在接近 concat 高精度经验基线的同时，保留了 gate-level discriminative contribution 分析能力。

边界条件：当前结论应限定为仿真距离选通切片识别，不应写成真实硬件系统已经被完整验证。

## 术语锁定

| 建议术语 | 使用方式 |
|---|---|
| 距离选通多深度切片 | 描述 `gate_0/gate_1/gate_2`，避免写成普通 RGB-like 通道 |
| gate-level discriminative contribution | attention 权重的论文解释，不写成视觉显著性 |
| attention-residual fusion | 当前主模型 |
| concat fusion | 高精度经验基线，不写成理论上界 |
| physical-parameter ablation | 下一步主实验，不优先换复杂网络 |
| degradation robustness | 下一步主实验，用于接近真实 gated imaging 条件 |

## 结果章节证据阶梯

1. 系统/数据验证：展示 Blender 距离选通切片生成流程和 `image2d` 异常类设计。
2. 主结果：多 gate 输入优于单 gate 和 black-slice 控制。
3. 融合基线：mean、attention、attention-residual、concat 的 seed-matched 对比。
4. gate 贡献分析：按类别统计 attention 权重，并明确 attention 是判别贡献，不是视觉显著性。
5. 物理参数消融：门宽、脉宽、gate 间距、距离损耗、大气衰减如何影响准确率和 gate 贡献。
6. 退化鲁棒性：噪声、背景散射、远 gate 衰减、gate dropout 下，attention-residual 是否更稳。
7. 失败模式：报告最易混淆类别、退化下性能崩溃点、attention 转移现象。

## 已落实到代码的下一步接口

### 1. 渲染脚本的物理参数接口

脚本：`origindataset/gated_blender_physical.py`

新增参数：

```powershell
--gate-spacing
--gate-center-middle
--receiver-gate-width
--laser-pulse-width
--range-loss-power
--atmospheric-extinction
```

推荐先生成三组小规模物理参数数据集，每组每类少量模型做 smoke test，确认图像差异合理后再扩展：

```powershell
python origindataset\gated_blender_physical.py --output-root dataset_gate_spacing_small --gate-spacing 0.45 --gate-center-middle 7.4 --models-per-class 20
python origindataset\gated_blender_physical.py --output-root dataset_gate_spacing_default --gate-spacing 0.60 --gate-center-middle 7.4 --models-per-class 20
python origindataset\gated_blender_physical.py --output-root dataset_gate_spacing_large --gate-spacing 0.90 --gate-center-middle 7.4 --models-per-class 20
```

门宽消融：

```powershell
python origindataset\gated_blender_physical.py --output-root dataset_gate_width_narrow --receiver-gate-width 0.70 --laser-pulse-width 0.35 --models-per-class 20
python origindataset\gated_blender_physical.py --output-root dataset_gate_width_default --receiver-gate-width 1.00 --laser-pulse-width 0.45 --models-per-class 20
python origindataset\gated_blender_physical.py --output-root dataset_gate_width_wide --receiver-gate-width 1.40 --laser-pulse-width 0.65 --models-per-class 20
```

论文问题：gate 参数变化是否改变识别性能和 class-wise mean attention by gate。

### 2. 训练脚本的退化鲁棒性接口

脚本：`train.py`

新增参数：

```powershell
--gaussian-noise-std
--poisson-peak
--background-scatter
--background-sigma
--gate-attenuation-index
--gate-attenuation-factor
--gate-dropout-mode none|fixed|random
--gate-dropout-index
```

这些参数会写入 `summary.json` 和 `run_experiments.py` 结果 CSV；验证集的 `val_attention_weights.csv` 也会记录每个样本的退化条件和 dropout gate，便于后续做 attention 转移分析。

## 优先实验包

### A. 必做：干净数据 formal baseline

已有结果可保留：

| 实验 | 角色 |
|---|---|
| multi vs single-gate | 证明多 gate 互补性 |
| mean / attention / attention-residual / concat | 证明主模型与高精度基线关系 |
| attention by class/gate | 支撑 gate-level 判别贡献 |

不要重复大量跑，除非换了数据生成参数或修正了数据集。

### B. 必做：退化鲁棒性 formal set

先用 seed 42 smoke test，确认训练能跑通；再用 `42 332 2026` 正式跑。

Gaussian noise：

```powershell
python run_experiments.py --experiment-name robust_gaussian_005_attention_residual --fusion-mode attention_residual --seeds 42 332 2026 --gaussian-noise-std 0.05 --epochs 30 --batch-size 16
python run_experiments.py --experiment-name robust_gaussian_010_attention_residual --fusion-mode attention_residual --seeds 42 332 2026 --gaussian-noise-std 0.10 --epochs 30 --batch-size 16
```

Photon/Poisson noise：

```powershell
python run_experiments.py --experiment-name robust_poisson_30_attention_residual --fusion-mode attention_residual --seeds 42 332 2026 --poisson-peak 30 --epochs 30 --batch-size 16
python run_experiments.py --experiment-name robust_poisson_10_attention_residual --fusion-mode attention_residual --seeds 42 332 2026 --poisson-peak 10 --epochs 30 --batch-size 16
```

Background scatter：

```powershell
python run_experiments.py --experiment-name robust_scatter_010_attention_residual --fusion-mode attention_residual --seeds 42 332 2026 --background-scatter 0.10 --background-sigma 24 --epochs 30 --batch-size 16
```

远 gate 衰减：

```powershell
python run_experiments.py --experiment-name robust_far_gate_half_attention_residual --fusion-mode attention_residual --seeds 42 332 2026 --gate-attenuation-index 2 --gate-attenuation-factor 0.50 --epochs 30 --batch-size 16
```

随机 gate dropout：

```powershell
python run_experiments.py --experiment-name robust_random_dropout_attention_residual --fusion-mode attention_residual --seeds 42 332 2026 --gate-dropout-mode random --epochs 30 --batch-size 16
```

最小对比组：对每个退化强度，至少补跑 `concat` 和 `mean`：

```powershell
python run_experiments.py --experiment-name robust_gaussian_005_concat --fusion-mode concat --seeds 42 332 2026 --gaussian-noise-std 0.05 --epochs 30 --batch-size 16
python run_experiments.py --experiment-name robust_gaussian_005_mean --fusion-mode mean --seeds 42 332 2026 --gaussian-noise-std 0.05 --epochs 30 --batch-size 16
```

论文问题：attention-residual 是否在可解释性存在的前提下保持接近 concat 的鲁棒性；退化下 attention 是否转移到信噪比更高的 gate。

### C. 必做：物理参数消融 formal set

物理参数消融只使用五个真实 3D 类：

```text
chair, desk, sofa, bed, toilet
```

`image2d` 保留给六类异常/二维虚假目标识别实验，不混入 gate spacing 和 gate width 主消融。项目中已提供薄封装脚本：

```powershell
python run_physical_5class_experiments.py -- --experiment-name phys_gate_spacing_small_smoke --dataset-root dataset_gate_spacing_small --fusion-mode attention_residual --seeds 42 --epochs 2 --batch-size 16
```

该脚本等价于调用 `run_experiments.py --classes chair desk sofa bed toilet ...`。

优先级：

1. `gate_spacing`: small/default/large
2. `receiver_gate_width + laser_pulse_width`: narrow/default/wide
3. `range_loss_power`: 1.5/2.0/2.5
4. `atmospheric_extinction`: 0.00/0.03/0.08

每个参数组的数据集训练：

```powershell
python run_physical_5class_experiments.py -- --experiment-name phys_gate_spacing_small_attention_residual --dataset-root dataset_gate_spacing_small --fusion-mode attention_residual --seeds 42 332 2026 --epochs 30 --batch-size 16
python run_physical_5class_experiments.py -- --experiment-name phys_gate_spacing_default_attention_residual --dataset-root dataset_gate_spacing_default --fusion-mode attention_residual --seeds 42 332 2026 --epochs 30 --batch-size 16
python run_physical_5class_experiments.py -- --experiment-name phys_gate_spacing_large_attention_residual --dataset-root dataset_gate_spacing_large --fusion-mode attention_residual --seeds 42 332 2026 --epochs 30 --batch-size 16
```

论文问题：较窄门宽是否提高深度选择性但降低信号覆盖；较宽门宽是否提高稳定性但削弱 gate 间互补性。

## 图表清单

| 图表 | 内容 | 当前状态 |
|---|---|---|
| Figure 1 | 距离选通切片生成与数据集设计 | 已有 Nature 风格图 |
| Figure 2 | attention-residual 网络结构 | 已有 Nature/NN-SVG 风格图 |
| Figure 3 | 多 gate vs single-gate/black-slice 控制 | 已有结果图 |
| Figure 4 | fusion ablation | 已有结果图 |
| Figure 5 | class-wise gate contribution heatmap | 已有 seed 42 图，建议补 aggregate |
| Figure 6 | degradation robustness curves | 待跑 |
| Figure 7 | physical-parameter ablation | 待跑 |
| Table 1 | seed-matched accuracy summary | 已有基础结果，待合并鲁棒性 |
| Table 2 | physical parameter settings | 待补 |
| Table 3 | degradation settings and robustness gap | 待跑 |

## 稿件写作顺序

1. 先写 Results，不先扩写 Introduction。
2. Results 按“多 gate 价值 -> 融合消融 -> attention 贡献 -> 物理参数 -> 鲁棒性”组织。
3. Methods 明确输入为 `[B, S, C, H, W]`，`S=3` 是深度选择性 gated observations，不是普通 RGB 通道。
4. Discussion 主动承认仿真边界：尚未覆盖真实探测器响应、散射介质、偏振、材质反射率等。
5. Abstract 最后写，只报告已经完成的结果。

## 下一次最推荐执行

先在有 PyTorch 和 Blender 的实验室机器上执行两个 smoke test：

```powershell
python run_experiments.py --experiment-name robust_smoke_gaussian_attention_residual --fusion-mode attention_residual --seeds 42 --gaussian-noise-std 0.05 --epochs 2 --batch-size 16
python origindataset\gated_blender_physical.py --output-root dataset_gate_spacing_smoke --gate-spacing 0.45 --gate-center-middle 7.4 --models-per-class 5
```

如果这两步通过，就进入正式三种子鲁棒性实验和三组 gate spacing 消融。
