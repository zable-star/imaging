# 项目系统安排：军事三维目标与二维假目标判别

日期：2026-07-05

## 1. 总目标

当前阶段目标：

```text
完成稳定的网络训练，明确多 gate 距离选通图像是否能完成目标识别和二维假目标判别。
```

最终项目目标：

```text
构建一个从物理仿真、数据集构建、神经网络训练、假目标判别到多模光神经网络融合的完整研究链条。
```

## 2. 项目主线

建议采用三层递进叙事：

| 层次 | 问题 | 对应能力 |
|---|---|---|
| 数据层 | 如何获得有物理意义的多 gate 图像？ | Blender 仿真、gate 参数控制、数据筛查 |
| 算法层 | 如何利用多 gate 信息识别目标和假目标？ | CNN 编码、gate 融合、attention_residual、消融实验 |
| 系统层 | 如何迁移到军事目标和光神经网络？ | 小样本迁移、光学编码接口、光电联合识别 |

这比“我训练了一个分类网络”更完整，也更容易体现研究能力。

## 3. 当前最该完成的训练

### 3.1 六分类假目标判别

目的：

```text
证明二维假目标/异常 gate stack 与真实三维目标 gate stack 可被区分。
```

数据：

```text
chair, desk, sofa, bed, toilet, image2d
```

模型：

```text
attention_residual
```

必须保留的结果：

- `summary.json`
- `best_confusion_matrix.png`
- `training_curves.png`
- `val_attention_weights.csv`
- `attention_mean_by_class.png`

重点分析：

- `image2d` 是否能被稳定识别。
- `image2d` 的注意力是否集中在某些 gate。
- 黑 gate 或弱 gate 是否被模型当作异常证据。

### 3.2 五分类物理参数消融

目的：

```text
证明识别性能受距离选通物理参数影响，而不是普通图像分类。
```

先做：

- gate spacing: small / default / large
- num gates: 1 / 3 / 5

再做：

- receiver gate width: narrow / default / wide
- laser pulse width: narrow / default / wide
- range attenuation / background scatter / noise

重点结论应围绕：

```text
更合理的 gate 设置能提供更强的互补深度信息，attention_residual 能更稳定地融合多 gate 特征。
```

### 3.3 军事目标小样本迁移

目的：

```text
把项目从家具基线推进到军事三维目标应用。
```

建议第一版不要追求类别多。先做：

```text
tank / aircraft / helicopter / military_truck / missile_vehicle
```

如果数据质量不足，就压缩成 3 类：

```text
tank / aircraft / military_vehicle
```

训练策略：

1. 在可控数据上训练 `attention_residual`。
2. 保存 encoder 权重。
3. 替换分类头，微调军事类别。
4. 对比从零训练。
5. 报告小样本条件下迁移是否更稳。

## 4. 创新点提升路线

### 创新点 1：物理可解释 gate stack

当前问题：

```text
如果只说 Blender 渲染图片，创新性不够。
```

提升方式：

- 明确 gate 图像来自选通响应函数，而不是任意切片。
- 记录 gate center、gate width、pulse width、range loss。
- 在论文中用参数消融证明这些物理参数会改变识别结果。

### 创新点 2：二维假目标的物理建模

当前应避免：

```text
二维假目标只有一小块物体出现，导致假目标太容易、也不够物理。
```

建议建模：

```text
整个目标轮廓在多个 gate 中一致，但强度随 gate 响应变化。
```

这能把问题从“图像缺失检测”提升为：

```text
真实三维深度分布与平面二维诱饵响应模式的判别。
```

### 创新点 3：gate-level 判别贡献

当前 attention 不应被写成视觉显著性。建议固定表述：

```text
gate-level discriminative contribution
门控切片级判别贡献
```

论文中可以分析：

- 哪个 gate 对某类目标贡献更大。
- 参数变化后 attention 是否变得更均衡。
- 噪声/衰减下 attention 是否转向更可靠的 gate。

### 创新点 4：小样本军事迁移

军事模型少是问题，也可以转成研究价值：

```text
在军事目标样本有限、标签噪声较高的条件下，利用可控物理仿真预训练提升识别稳定性。
```

需要做的对比：

| 方法 | 目的 |
|---|---|
| 从零训练 | 基线 |
| 冻结 encoder 只训分类头 | 验证特征迁移 |
| 半冻结微调 | 实际推荐 |
| 全量微调 | 上限参考 |

### 创新点 5：多模光神经网络接口

当前电子网络可以作为以后光神经网络的对照组。

后续融合方式：

| 当前模块 | 光神经网络对应模块 |
|---|---|
| gate stack 图像 | 光学输入编码 |
| SliceEncoder | 光学传播/散斑特征提取 |
| fusion module | 电子或光电融合层 |
| classifier | 读出层 |

这样后续参与“多模光神经网络高速目标识别”项目时，可以说明自己已有：

- 光学成像仿真输入
- 目标识别训练基线
- 假目标判别任务
- gate/depth 物理参数分析
- 可迁移到光学前端的算法接口

## 5. 推荐时间安排

### 第 1 阶段：稳定训练结果

产出：

- 六分类 `attention_residual` 三随机种子结果
- 五分类 gate spacing 结果整理
- README 与实验说明稳定

验收标准：

```text
能清楚回答：网络训练了什么、输入是什么、输出是什么、二维假目标如何模拟、为什么 laser gated 有意义。
```

### 第 2 阶段：补强物理仿真

产出：

- num gates = 1 / 3 / 5 对比
- gate width / pulse width 对比
- 典型 gate stack 可视化

验收标准：

```text
能证明多 gate 不是普通多图像输入，而是与距离选通参数直接相关。
```

### 第 3 阶段：军事目标小样本实验

产出：

- 3 到 5 类高质量军事模型子集
- 真三维目标 gate stack
- 平面假目标 gate stack
- 从零训练 vs 迁移学习对比

验收标准：

```text
能展示该方法对军事目标有初步应用可行性，而不是只停留在家具类别。
```

### 第 4 阶段：论文创新点固化

产出：

- 方法流程图
- 网络结构图
- 仿真参数表
- 消融实验表
- 混淆矩阵
- gate attention 分析图

论文主问题：

```text
距离选通多 gate 观测如何提升目标识别与二维假目标判别？
```

### 第 5 阶段：多模光神经网络融合预研

产出：

- gate stack 到光学输入编码方案
- 电子 CNN baseline 与光学前端方案对照
- 速度/复杂度/可实现性讨论

验收标准：

```text
能把当前项目自然接到高速光神经网络目标识别，而不是另起炉灶。
```

## 6. 近期可执行清单

| 状态 | 事项 | 说明 |
|---|---|---|
| 已完成 | 固定当前 README 和路线图 | README 已改为军事三维目标、二维假目标和多模光神经网络衔接的项目首页 |
| 已完成 | 整理六分类 `attention_residual` 已有结果 | 三随机种子平均准确率约 0.9472 |
| 已完成 | 整理五分类 large gate spacing 已有结果 | 三随机种子平均准确率约 0.9467 |
| 已完成 | 设计迁移学习脚本入口 | `train.py` 支持预训练权重加载与冻结 encoder/attention/residual |
| 已完成 | 新增军事迁移包装脚本 | `run_military_transfer_experiments.py` 可快速启动军事小样本实验 |
| 已完成 | 用 `pytorch1` 跑通迁移 smoke | 六分类 1 epoch 冻结 encoder smoke 准确率 0.9583；二分类 smoke 准确率 0.9500 |
| 已完成 | 增加数据集训练就绪检查脚本 | `dataset_new/check_gate_dataset_ready.py` 可统计每类 raw models、gate pngs 和 valid samples |
| 已完成 | 增加精选军事模型复制脚本 | `dataset_new/build_selected_subset.py` 支持 `--expected-count 44`，防止误用 500 个候选 |
| 已确认 | 军事原始数据暂不能直接训练 | `dataset_new/Military_3D_Dataset` 是候选池；本轮只使用 `_review_sheets/thumbnail_review.csv` 中 keep=1 的 44 个模型 |
| 已完成 | 检查并修正平面假目标生成逻辑 | `flat-echo` 模式输出整目标轮廓，并通过 `--flat-echo-gain` 控制可见亮度 |
| 已完成 | 将 44 个精选模型落成机器可读清单 | 输出 `dataset_new/Military_3D_Selected44/selected_manifest.csv` |
| 已完成 | 渲染 44 个精选军事模型为真三维 gate stack | 输出 `dataset_new/Military_3D_Gated_Selected44`，共 132 张 gate PNG |
| 已完成 | 渲染 44 个精选军事模型的平面假目标 gate stack | 输出 `dataset_new/Military_3D_FlatEcho_Selected44_gain10`，共 132 张 gate PNG |
| 已完成 | 构建军事 true3d / flat_false 二分类数据集 | 输出 `dataset_new/Military_TrueFalse_Selected44_gain10`，共 88 个样本 |
| 已完成 | 跑军事迁移学习 20 epoch 三随机种子 | 冻结 encoder 和全量微调均达到 0.75 mean best val acc，std = 0 |
| 已完成 | 跑军事从零训练 20 epoch 三随机种子 | mean best val acc = 0.7083，std = 0.1443；最高 0.875 但波动更大 |
| 已完成 | 跑军事 true3d / flat_false 二分类迁移与从零训练 | 使用 `--split-group-by-sample-id` 防止同源模型泄漏；迁移与从零训练三 seed 均为 1.0 |
| 已完成 | 跑 true/false 单 gate 快速消融 | seed=42：gate_0 = 1.0，gate_1 = 0.8333，gate_2 = 0.7778 |
| 已完成 | 生成 per-gate 最大值归一化 true/false 数据集 | 输出 `dataset_new/Military_TrueFalse_Selected44_gain10_per_gate_norm`，用于削弱绝对亮度捷径 |
| 已完成 | 跑 per-gate 归一化 full stack 二分类 | 三随机种子 20 epoch：mean best val acc = 1.0，std = 0 |
| 已完成 | 跑 per-gate 归一化单 gate 三随机种子消融 | gate_0 = 0.9630，gate_1 = 0.9630，gate_2 = 0.8889；full stack 仍最稳 |
| 已完成 | 增加 gate stack 物理诊断脚本 | `dataset_new/diagnose_gate_stack.py` 输出 gate 间相关性、掩膜 IoU、归一化差分 |
| 已完成 | 验证真假目标 gate stack 的物理差异 | per-gate norm 下 flat_false corr = 0.9995 / IoU = 0.9880；true3d corr = 0.3244 / IoU = 0.3174 |
| 已完成 | 跑随机 gate dropout 鲁棒性测试 | 每个样本随机丢一门，mean best val acc = 0.9074，说明 gate 完整性影响稳定性 |
| 已完成 | 跑噪声/背景/Poisson 退化测试 | gaussian=0.05、background=0.04、poisson_peak=35，mean best val acc = 0.9815 |
| 已完成 | 构建 hard projection 平面假目标 | 从真实 3D gate stack 最大投影生成二维平面诱饵，输出 `dataset_new/Military_TrueFalse_Selected44_hard_projection` |
| 已完成 | 跑 hard projection full stack 与单 gate 消融 | full stack = 1.0；only gate_0 = 0.5370，only gate_1 = 0.6296，only gate_2 = 0.5370 |
| 已完成 | 构建 rectangular-overlap hard false target | 用矩形激光脉冲与矩形接收门的重叠长度生成假目标 gate response，输出 `dataset_new/Military_TrueFalse_Selected44_hard_rect_overlap` |
| 已完成 | 跑 rectangular-overlap full stack 与单 gate 消融 | full stack = 1.0；only gate_0 = 0.5000，only gate_1 = 0.8704，only gate_2 = 0.5000；说明峰值 gate 仍有残余单帧线索 |
| 已完成 | 构建 gate 均值曝光匹配控制集 | 输出 `dataset_new/Military_TrueFalse_Selected44_hard_rect_overlap_mean_classgate_matched`，flat_false 三个 gate 全图均值基本对齐 true3d |
| 已完成 | 跑曝光匹配 full stack 与单 gate 消融 | full stack = 1.0；only gate_0 = 0.5000，only gate_1 = 0.7222，only gate_2 = 0.5000；说明 gate_1 亮度捷径被削弱但仍有结构线索 |
| 已完成 | 生成论文证据矩阵 | `writing/paper_evidence_matrix_military_gated_false_target_2026-07-06.md` 梳理主张、证据强度和不能过度声称的点 |
| 已完成 | 形成中文论文初稿 | `writing/paper_draft_military_gated_false_target_2026-07-06.md` 已包含摘要、引言、方法、实验、讨论和结论 |
| 下一步 | 开始 num gates = 1 / 3 / 5 数据生成与训练 | 用来证明 gate 数增加时 attention_residual 是否更有价值 |
| 下一步 | 提高二维假目标难度 | 在 rectangular-overlap 基础上加入姿态偏移、曝光匹配、随机反射率和复杂背景，形成 easy / medium / hard 三档假目标 |
| 下一步 | 补充鲁棒性测试 | 在 true/false 数据上加入 gaussian noise、poisson noise、background scatter、gate attenuation/dropout |
| 下一步 | 剔除或复核低对比军事样本 | 重点检查 `01_Main_Battle_Tank_009_6e50bf75` 是否影响三分类稳定性 |
| 下一步 | 写一页“项目如何衔接多模光神经网络”的说明图 | 用于 PPT 和后续项目衔接 |

## 7. PPT 中可以讲的能力点

- 我不是只做分类，而是把目标识别放在距离选通成像链路中建模。
- 我能用 Blender 构建可控物理仿真数据，并分析 gate 参数影响。
- 我能设计二维假目标，使其符合平面目标在选通系统中的响应特点。
- 我能实现多 gate 神经网络并做融合消融。
- 我能用 attention 权重解释 gate-level 判别贡献。
- 我能处理军事 3D 数据标签噪声和小样本问题。
- 我能把电子网络基线扩展到未来光神经网络高速识别系统。

## 8. 2026-07-07 阶段更新：当前主线已经收敛到 v8 mixed-augmentation 框架

当前最适合作为论文主线的版本不是早期 `v4-v8` 版本号本身，而是以下物理/实验组合：

```text
balanced-overlap rectangular gated response
per-gate max-normalized anti-shortcut control
mixed clean/noisy augmentation
full-stack versus single-gate ablation
fusion-mode robustness comparison
```

已经完成的关键证据：

| 证据 | 结论 |
|---|---|
| raw v8 shortcut diagnostics | 原始仿真仍存在单 gate 强度捷径，不能直接宣称网络学到了深度一致性 |
| per-gate max-normalization | 明显削弱亮度捷径，把问题推进到形状/深度切片差异 |
| full stack vs single gate, three seeds | 完整三 gate stack 在 clean、light-noise、strong-noise 三种独立条件下平均准确率最高 |
| fusion-mode comparison | attention_residual 验证/clean 最好，mean 轻噪声略优，attention 强噪声更稳 |
| tests | 当前代码测试 `62 passed` |

当前 SCI 安全表述：

```text
本文提出的是一个物理可解释的激光选通仿真与反捷径验证框架。当前结果支持完整 gate stack 相比单 gate 在受控仿真中更稳定，但不支持部署级军事识别或某个融合头全面最优的结论。
```

下一阶段优先级：

1. 生成多视角数据，每个模型建议先做 4 个视角，若 3090 时间允许再扩到 8 个视角。
2. 在多视角数据上重复 full stack、single gate 和 fusion-mode 三组对比。
3. 为论文补 Fig. 1 方法流程图、Fig. 2 矩形重叠响应图、Fig. 3 true3d/flat_false gate 示例图。
4. 把 `writing/paper_draft_v8_mixaug_framework_2026-07-07.md` 转成英文主稿骨架，并补文献引用。
5. 3090 主要用于多视角训练和更大 batch/更多 epoch；当前本机足够做小规模代码验证、图表整理和论文写作。

## 9. 2026-07-07 阶段更新：多视角验证管线已经落地

已新增多视角数据生成与合并入口：

```text
scripts/run_v8_multiview_dataset.ps1
dataset_new/build_multiview_true_false_dataset.py
```

渲染包装脚本已支持：

```text
-ModelRotationDeg "0,0,<angle>"
```

本机已完成 smoke 验证：

```text
ViewRotationsZ = 0
ModelsPerClass = 1
true3d samples = 3
flat_false samples = 3
normalized gate images = 18
readiness = True
```

输出位置：

```text
dataset_new/_smoke_Military_TrueFalse_v8_mv_per_gate_maxnorm
dataset_new/_smoke_Military_TrueFalse_v8_mv_per_gate_maxnorm_readiness.csv
writing/multiview_v8_validation_plan_2026-07-07.md
```

这一步的意义：

```text
多视角验证已经从“后续建议”变成“可执行管线”。当前 smoke 只证明渲染、合并、归一化链路可用，不作为论文性能结果。真正的论文增强证据需要跑完整 44 模型 × 4 视角数据集，并重复 full stack / single gate / fusion mode 对比。
```

## 10. 2026-07-07 阶段更新：文献定位与论文叙事已经补齐第一版

已新增文献证据矩阵：

```text
writing/literature_evidence_matrix_gated_false_target_2026-07-07.md
```

已更新英文 SCI 工作稿：

```text
writing/sci_manuscript_v8_gated_false_target_draft_2026-07-07.md
```

本阶段加入的核心文献角色：

| 文献/方向 | 在论文中的作用 |
|---|---|
| 3D laser gated range-intensity correlation imaging | 支撑矩形脉冲/门函数、三角/梯形响应、噪声与系统物理限制 |
| Transformer-based 3D range-gated imaging with multiple depth priors | 支撑 gate stack 不是普通图像通道，而是带有光学深度先验的输入 |
| SAMFusion | 支撑 gated camera 在复杂环境和多模态感知中的应用价值 |
| Diffractive decoder 3D projection | 作为未来光神经网络/光电融合方向，不作为当前分类结果的直接证据 |
| Shortcut learning / domain randomization | 支撑为什么必须做亮度捷径、黑帧捷径、单 gate 消融和多视角验证 |

当前论文故事线应稳定为：

```text
激光选通成像提供深度相关的 gate-stack 物理观测；
但仿真数据容易产生亮度、黑帧、渲染偏差等捷径；
因此本文的核心贡献是物理可解释仿真 + 反捷径验证；
轻量网络只是验证 gate stack 在受控条件下确实提供了比单 gate 更稳定的信息。
```

下一步论文增强顺序：

1. 跑完整 44 模型 × 4 视角 v8 数据集。
2. 重复 full stack / single gate / fusion mode 对比。
3. 补军事假目标、诱饵识别或目标欺骗检测方向的文献。
4. 从 Zotero 导出最终参考文献，替换当前手工占位格式。

## 11. 2026-07-07 late update: four-view v8 validation finished

The previous "run the full 44-model x 4-view dataset" task is now complete.

Completed outputs:

```text
dataset_new/Military_TF_v8_mv4_norm
experiments/v8_mv4_norm_mixaug_attention_eval_summary_3seed.csv
experiments/v8_mv4_norm_mixaug_attention_eval_aggregate_3seed.csv
writing/v8_mv4_multiview_robustness_report_2026-07-07.md
writing/figures/fig7_mv4_full_stack_vs_single_gate_robustness.png
```

Key four-view result:

| condition | full stack | gate0 | gate1 | gate2 |
|---|---:|---:|---:|---:|
| clean | 0.9848 | 0.9697 | 0.9545 | 0.8939 |
| light noise | 0.9811 | 0.9432 | 0.9053 | 0.8750 |
| strong noise | 0.9205 | 0.7462 | 0.5265 | 0.6098 |

Updated project position:

```text
The main technical evidence is now:
1. physics-interpretable v8 gated simulation,
2. anti-shortcut scalar diagnostics and per-gate normalization,
3. full-stack versus single-gate ablation,
4. four-view model-level grouped validation.

The network remains a lightweight baseline. The stronger contribution is the simulation and validation protocol.
```

Four-view fusion comparison is also complete:

| condition | attention | mean | attention_residual |
|---|---:|---:|---:|
| clean | 0.9848 | 0.9848 | 0.9848 |
| light noise | 0.9811 | 0.9848 | 0.9394 |
| strong noise | 0.9205 | 0.7197 | 0.7727 |

This confirms that the network fusion head should not be presented as the main innovation. `attention_residual` is not consistently best after four-view validation. The stronger contribution is the simulation and validation protocol.

Updated next steps:

1. Add nuisance-aware training or simulator-level nuisance generation for reflectance, background, and occlusion shifts.
2. Add final references on military decoys, false targets, shortcut learning, and simulation-to-real transfer.
3. Prepare a compact SCI experiment table and PPT result slide from Fig. 5, Fig. 7, Fig. 8, and the Fig. 9 failure boundary.
4. Convert the Markdown working draft into a complete submission-style article using the research writing workflow.

## 12. 2026-07-08 update: hard-nuisance failure boundary

Structured nuisance stress tests were added after the four-view validation.

New hard-nuisance factors:

```text
low-frequency reflectance texture
weak background scatter
partial rectangular occlusion
paired true/false nuisance key
per-sample maximum preservation
```

Main result:

| condition | full stack | gate0 | gate1 | gate2 |
|---|---:|---:|---:|---:|
| hard_nuisance_v2 | 0.5000 | 0.4697 | 0.5000 | 0.4545 |
| hard_nuisance_v3_mild | 0.5000 | 0.4848 | 0.5000 | 0.4545 |

Interpretation:

```text
This is a limitation result, not a positive robustness result.
The current clean/noisy-trained models collapse under structured reflectance/background/occlusion shifts.
This strengthens the paper's honesty and gives the next innovation target:
nuisance-aware gated simulation and training.
```

New files:

```text
dataset_new/build_hard_nuisance_dataset.py
scripts/run_v8_mv4_hard_nuisance_eval.ps1
scripts/make_v8_hard_nuisance_boundary_figure.py
writing/v8_mv4_hard_nuisance_boundary_report_2026-07-08.md
writing/figures/fig9_hard_nuisance_failure_boundary.png
writing/daily_progress_2026-07-08.md
```
