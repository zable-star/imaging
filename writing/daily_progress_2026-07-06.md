# 2026-07-06 夜间推进记录：军事真假目标归一化诊断与 gate stack 证据

## 1. 今日继续推进目标

本轮延续当前总目标：

```text
基于人工 keep=1 的 44 个军事模型，继续推进真三维目标识别、二维平面假目标判别、选通堆栈物理诊断和项目文档更新。
```

本轮重点不再只是“跑出 1.0 准确率”，而是回答一个更关键的问题：

```text
网络区分 true3d / flat_false 时，到底是不是在利用激光选通 gate stack 的物理差异？
```

## 2. 已完成的新实验和工具

### 2.1 per-gate 归一化数据集

原始 true/false 二分类数据：

```text
dataset_new\Military_TrueFalse_Selected44_gain10
```

为了削弱绝对亮度捷径，生成逐 gate 最大值归一化版本：

```text
dataset_new\Military_TrueFalse_Selected44_gain10_per_gate_norm
```

命令：

```powershell
E:\ana\envs\pytorch1\python.exe dataset_new\normalize_gate_dataset.py --input-root dataset_new\Military_TrueFalse_Selected44_gain10 --output-root dataset_new\Military_TrueFalse_Selected44_gain10_per_gate_norm --mode per-gate-max --target-max 180 --min-source-max 2 --overwrite
```

训练就绪检查通过：

```text
dataset_new\military_true_false_selected44_gain10_per_gate_norm_readiness_2026-07-06.csv
```

质量审计：

```text
dataset_new\military_true_false_selected44_gain10_per_gate_norm_quality_2026-07-06.csv
```

结果：

| 指标 | 数值 |
|---|---:|
| images | 264 |
| samples | 88 |
| low contrast images | 7 |

### 2.2 新增 gate stack 物理诊断脚本

新增：

```text
dataset_new\diagnose_gate_stack.py
```

该脚本对每个样本三张 gate 计算：

| 指标 | 含义 |
|---|---|
| `mean_pair_corr_raw` | 原始灰度下 gate 间平均相关性 |
| `mean_pair_corr_maxnorm` | 每张图最大值归一化后 gate 间平均相关性 |
| `mean_pair_mask_iou` | gate 前景掩膜平均 IoU |
| `mean_pair_absdiff_maxnorm` | 最大值归一化后 gate 间平均绝对差 |
| `active_fraction_std` | 三个 gate 前景面积比例的标准差 |
| `max_mean_ratio` | 三个 gate 平均亮度的最大/最小比 |

诊断命令：

```powershell
E:\ana\envs\pytorch1\python.exe dataset_new\diagnose_gate_stack.py --root dataset_new\Military_TrueFalse_Selected44_gain10 --sample-csv-out dataset_new\military_true_false_selected44_gain10_gate_diagnostics_samples_2026-07-06.csv --class-csv-out dataset_new\military_true_false_selected44_gain10_gate_diagnostics_by_class_2026-07-06.csv

E:\ana\envs\pytorch1\python.exe dataset_new\diagnose_gate_stack.py --root dataset_new\Military_TrueFalse_Selected44_gain10_per_gate_norm --sample-csv-out dataset_new\military_true_false_selected44_gain10_per_gate_norm_gate_diagnostics_samples_2026-07-06.csv --class-csv-out dataset_new\military_true_false_selected44_gain10_per_gate_norm_gate_diagnostics_by_class_2026-07-06.csv
```

## 3. gate stack 物理诊断结果

### 3.1 原始 gain10 数据

| 类别 | samples | corr maxnorm | mask IoU | absdiff maxnorm |
|---|---:|---:|---:|---:|
| flat_false | 44 | 0.9995 | 0.9787 | 0.0054 |
| true3d | 44 | 0.3246 | 0.3065 | 0.1210 |

### 3.2 per-gate 归一化数据

| 类别 | samples | corr maxnorm | mask IoU | absdiff maxnorm |
|---|---:|---:|---:|---:|
| flat_false | 44 | 0.9995 | 0.9880 | 0.0055 |
| true3d | 44 | 0.3244 | 0.3174 | 0.1211 |

### 3.3 解释

平面假目标的回波可写成：

```text
I_g^false(x, y) = A_g S(x, y)
```

其中 `S(x, y)` 是同一整目标轮廓，`A_g` 是第 `g` 个选通门的整体响应强度。因此不同 gate 之间应该高度相似，只是整体亮度不同。

真实三维目标为：

```text
I_g^true(x, y) = integral rho(x, y, z) T(z) H_g(z) dz
```

不同 gate 对不同深度位置的结构响应不同，因此三张图之间会出现明显结构变化。

本轮诊断结果正好符合这个物理预期：

```text
flat_false: 三门结构几乎完全相同，corr 接近 1，IoU 接近 1。
true3d: 三门结构明显不同，corr 和 IoU 显著降低。
```

这比单独报告二分类 1.0 更有价值，因为它说明网络可利用的差异来自选通成像物理，而不是任意构造的标签差异。

## 4. per-gate 归一化训练结果

### 4.1 full gate stack

数据：

```text
dataset_new\Military_TrueFalse_Selected44_gain10_per_gate_norm
```

设置：

```text
input_mode = multi
fusion_mode = attention_residual
split_group_by_sample_id = true
epochs = 20
seeds = 42 / 332 / 2026
```

结果：

| 实验 | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| military_truefalse_gain10_per_gate_norm_scratch_20ep | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

### 4.2 单 gate 三随机种子消融

设置：

```text
input_mode = single-gate
epochs = 10
seeds = 42 / 332 / 2026
split_group_by_sample_id = true
```

结果：

| 输入 | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| only gate_0 | 0.9630 | 0.0321 | 0.9444 | 1.0000 |
| only gate_1 | 0.9630 | 0.0321 | 0.9444 | 1.0000 |
| only gate_2 | 0.8889 | 0.0556 | 0.8333 | 0.9444 |
| full gate stack | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

解释：

```text
单张 gate 已经能捕捉部分真假差异，说明当前 flat false 和 true3d 在单帧外观上仍存在可学习线索。
但 full gate stack 在三随机种子下最稳定，且 gate stack 诊断显示 true3d/flat_false 的核心区别正是门间结构一致性。
```

因此 PPT/论文里建议写成：

```text
多 gate 不是唯一信息来源，但它提供了稳定且物理可解释的判别证据；
尤其是 flat false 的跨 gate 高相似性和 true3d 的跨 gate 结构变化，构成二维平面诱饵判别的核心依据。
```

## 5. hard projection 假目标：压低单帧捷径

### 5.1 设计动机

per-gate 归一化后，单 gate 仍有较高准确率，说明 flat false 与 true3d 在单帧外观上还存在可学习差异。

为进一步证明“激光选通序列”的必要性，新增更难的假目标生成方式：

```text
从真实 3D gate stack 取最大投影，作为二维平面假目标轮廓 S(x, y)；
再用随机 flat depth 的 gate response 生成三张假目标 gate 图。
```

新增脚本：

```text
dataset_new\build_hard_flat_projection_dataset.py
```

生成命令：

```powershell
E:\ana\envs\pytorch1\python.exe dataset_new\build_hard_flat_projection_dataset.py --true-root dataset_new\Military_3D_Gated_Selected44 --output-root dataset_new\Military_3D_HardFlatProjection_Selected44 --expected-num-slices 3 --projection-mode max --response-sigma 0.65 --min-response 0.08 --reflectance-min 0.85 --reflectance-max 1.15 --seed 2026 --overwrite

E:\ana\envs\pytorch1\python.exe dataset_new\build_true_false_dataset.py --true-root dataset_new\Military_3D_Gated_Selected44 --false-root dataset_new\Military_3D_HardFlatProjection_Selected44 --output-root dataset_new\Military_TrueFalse_Selected44_hard_projection --expected-num-slices 3 --overwrite
```

### 5.2 hard projection 物理诊断

| 类别 | corr maxnorm | mask IoU | absdiff maxnorm |
|---|---:|---:|---:|
| flat_false | 0.9768 | 0.9301 | 0.0062 |
| true3d | 0.3246 | 0.3065 | 0.1210 |

解释：

```text
hard projection false target 的轮廓来自真实 3D gate stack，
因此单帧外观更接近真实目标；
但它仍是平面投影，所以跨 gate 高相似，而 true3d 仍是低相似。
```

### 5.3 hard projection 训练结果

| 输入 | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| full gate stack | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| only gate_0 | 0.5370 | 0.0321 | 0.5000 | 0.5556 |
| only gate_1 | 0.6296 | 0.0849 | 0.5556 | 0.7222 |
| only gate_2 | 0.5370 | 0.0642 | 0.5000 | 0.6111 |

这是本轮最强的选通意义证据：

```text
单独看任一 gate 时，真假目标几乎不可稳定区分；
输入完整三门 gate stack 时，网络可以稳定达到 1.0；
说明判别信息主要来自跨 gate 的结构一致性/变化模式，而不是普通单帧图像分类。
```

这组结果最适合用于 PPT 中回答：

```text
如果可以直接做图像分类，为什么还需要激光选通？
```

推荐回答：

```text
在 hard projection 设定下，单张图已经被刻意做得接近真实目标；
真正可区分的是三维目标随 gate 改变的深度结构响应，
和平面假目标跨 gate 重复同一投影轮廓的序列模式。
```

## 6. attention 结果

per-gate 归一化 full stack 的 `per_class_attention_summary.csv` 显示：

```text
flat_false: gate attention 接近均匀分布。
true3d: attention 更偏向 gate_0 / gate_2，不同 seed 中 gate_1 相对较低。
```

可解释为：

```text
flat_false 三张图高度相似，网络无需明显偏向某一门；
true3d 的不同 gate 包含不同结构响应，attention 会偏向更具判别性的深度窗口。
```

注意：这里的 attention 仍应称为：

```text
gate-level discriminative contribution
门控切片级判别贡献
```

不要把它写成像 Transformer 那样的空间注意力或视觉显著性。

## 7. 初步鲁棒性测试

在 per-gate 归一化 true/false 数据上追加两组 10 epoch、三随机种子实验。

### 6.1 随机 gate dropout

设置：

```text
gate_dropout_mode = random
每个样本随机置零一个 gate
```

结果：

| 实验 | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| military_truefalse_gain10_per_gate_norm_gate_dropout_random_scratch_10ep | 0.9074 | 0.0642 | 0.8333 | 0.9444 |

解释：

```text
随机丢失一个 gate 后准确率明显低于完整 full stack 的 1.0，
说明三门完整性确实有助于稳定判别。
```

### 6.2 噪声、背景散射和 Poisson 光子噪声

设置：

```text
gaussian_noise_std = 0.05
background_scatter = 0.04
poisson_peak = 35
```

结果：

| 实验 | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| military_truefalse_gain10_per_gate_norm_noise_bg_poisson_scratch_10ep | 0.9815 | 0.0321 | 0.9444 | 1.0000 |

解释：

```text
当前 true3d / flat_false 的主要差异是跨 gate 结构模式，
因此在中等噪声、背景和光子噪声下仍较稳定。
```

这组结果可以写成：

```text
方法对强度噪声具有一定鲁棒性，但对 gate 缺失更敏感。
这进一步支持 gate stack 完整序列对判别的重要性。
```

## 8. 本轮更新的文件

| 文件 | 作用 |
|---|---|
| `dataset_new/diagnose_gate_stack.py` | 新增 gate stack 物理诊断脚本 |
| `dataset_new/build_hard_flat_projection_dataset.py` | 从真实 3D gate stack 最大投影生成更难平面假目标 |
| `scripts/run_truefalse_single_gate_ablation.ps1` | 支持多 seeds 和 `ExperimentTag` |
| `scripts/plot_military_selected44_results.py` | 生成军事 44 模型结果图 |
| `scripts/build_military_selected44_ppt.py` | 生成可编辑 PPTX、讲稿和 true3d / hard projection gate stack 对照图 |
| `experiments/military_selected44_results_overview_2026-07-06.csv` | 最新军事 44 模型结果总表 |
| `README.md` | 增加当前军事 44 模型结果快照 |
| `writing/project_roadmap_2026-07-05.md` | 增加 per-gate 归一化和 gate stack 诊断结果 |
| `writing/daily_progress_2026-07-06.md` | 本文档 |

## 9. PPT 可用图

新增脚本：

```powershell
E:\ana\envs\pytorch1\python.exe scripts\plot_military_selected44_results.py
```

输出目录：

```text
artifacts\figures\military_selected44_2026-07-06
```

| 图 | 建议讲法 |
|---|---|
| `military_3class_transfer_vs_scratch.png` | 44 个军事样本下，迁移学习精度不一定显著更高，但稳定性优于从零训练 |
| `hard_projection_full_stack_vs_single_gate.png` | hard projection 设定下，单门接近随机，完整三门稳定 1.0，是“为什么需要选通序列”的主图 |
| `per_gate_norm_robustness.png` | 噪声/背景/Poisson 下仍较稳，随机丢门下降，说明 gate 完整性重要 |
| `gate_stack_physical_diagnostics.png` | 平面假目标跨 gate 高相关、高 IoU；真实 3D 目标跨 gate 低相关、低 IoU |
| `true3d_vs_hard_projection_gate_stack.png` | 展示 true3d gate 间结构变化与 hard projection 假目标整轮廓重复 |

PPT 中推荐把 `hard_projection_full_stack_vs_single_gate.png` 放在“激光选通意义”这一页，把 `gate_stack_physical_diagnostics.png` 放在前一页解释物理原因。

### 9.1 可编辑 PPTX 初稿

新增输出：

```text
presentation_outputs\military_selected44_gated_report_2026-07-06.pptx
presentation_outputs\military_selected44_gated_report_speaker_notes_2026-07-06.md
```

PPTX 结构：

| 页码 | 主题 |
|---:|---|
| 1 | 研究问题与总览 |
| 2 | 数据治理与仿真流程 |
| 3 | 网络结构与 gate-level 融合 |
| 4 | 军事三分类小样本迁移 |
| 5 | hard projection 核心证据 |
| 6 | gate stack 物理诊断 |
| 7 | 鲁棒性与 gate 缺失 |
| 8 | 阶段结论与下一步 |

检查结果：

```text
slide_count = 8
关键页包含图片：第 3/4/5/6/7 页
讲稿 Markdown 已单独输出
```

## 10. 对论文故事的影响

当前可以形成三层实验故事：

1. 可控家具基线证明：attention_residual 可以处理多 gate 目标识别和二维异常判别。
2. 精选军事 44 模型证明：在小样本军事目标上，迁移训练比从零训练更稳定。
3. true3d / flat_false 证明：平面假目标与真实三维目标在 gate stack 上存在物理可解释差异。

尤其第三点现在有了定量指标支撑：

```text
flat_false 跨 gate 高相关、高 IoU、低差分；
true3d 跨 gate 低相关、低 IoU、高差分。
```

这能把研究从“我训练了一个分类器”提升为：

```text
我构建了符合激光距离选通物理规律的真假目标仿真数据，
并用神经网络和统计诊断共同验证了二维平面诱饵与三维目标的 gate stack 差异。
```

hard projection 结果还能进一步强化为：

```text
当二维假目标单帧外观来自真实三维目标最大投影时，单 gate 几乎不能稳定判别；
完整 gate stack 仍可稳定判别，直接证明多选通门序列的必要性。
```

## 11. 下一步优先实验

建议继续按以下顺序推进：

1. **hard false target 生成**：随机 flat target depth，不固定对齐 gate_0。
2. **随机反射率扰动**：让 flat false 不只是同一轮廓的固定缩放。
3. **噪声与背景鲁棒性**：加入 gaussian noise、poisson noise、background scatter。
4. **gate dropout / attenuation**：测试丢失某个 gate 时 full stack 模型是否仍稳。
5. **低质量军事样本复核**：优先检查 `01_Main_Battle_Tank_009_6e50bf75`。

建议在 PPT 中把这些写成“下一阶段工作”，不是当前不足，而是将任务从 easy setting 推进到 hard setting。

## 12. rectangular-overlap 补充结果

在 hard projection 假目标基础上，已补充更物理的矩形脉冲-门重叠响应版本。该版本将平面假目标写成：

```text
I_g^false(x,y)=A_g S_hard(x,y)
```

其中 `S_hard` 来自真实 3D gate stack 的最大投影，`A_g` 由矩形激光脉冲与矩形接收门的重叠长度决定。该设置避免把假目标人为固定为线性衰减，也避免早期 `gain10` 版本中 gate_0 偏亮的问题。

新增数据与结果：

| 项目 | 结果 |
|---|---|
| 数据集 | `dataset_new/Military_TrueFalse_Selected44_hard_rect_overlap` |
| full 3-gate stack | 1.0000 |
| only gate_0 | 0.5000 |
| only gate_1 | 0.8704 |
| only gate_2 | 0.5000 |
| flat_false corr / IoU / absdiff | 0.9731 / 0.4299 / 0.0172 |
| true3d corr / IoU / absdiff | 0.3246 / 0.3065 / 0.1210 |

解释口径：

```text
rectangular-overlap 版本证明，在更符合矩形脉冲-门函数重叠响应的假目标设置下，
完整 gate stack 仍显著优于 gate_0 和 gate_2 单门输入。
gate_1 单门仍有较高准确率，说明峰值 gate 中存在残余亮度、面积或清晰度线索，
因此论文中应写成“完整 gate stack 提供更稳定、更充分的判别信息”，
不要写成“任意单门完全无效”。
```

同步更新文件：

```text
experiments/military_selected44_results_overview_2026-07-06.csv
scripts/summarize_truefalse_brightness.py
dataset_new/military_true_false_selected44_brightness_summary_2026-07-06.csv
scripts/plot_military_selected44_results.py
README.md
writing/project_roadmap_2026-07-05.md
writing/paper_evidence_matrix_military_gated_false_target_2026-07-06.md
writing/paper_draft_military_gated_false_target_2026-07-06.md
```

## 13. 曝光匹配控制实验

为进一步排除 `rectangular-overlap` 中 gate_1 单门偏高是否来自全局亮度差异，新增 gate 均值曝光匹配控制集：

```text
dataset_new/Military_TrueFalse_Selected44_hard_rect_overlap_mean_classgate_matched
```

构建方式：

```text
true3d 不变；
flat_false 每个 gate 使用一个类别级缩放因子；
使 flat_false gate_0/1/2 的全图均值分别匹配 true3d 对应 gate 的全图均值。
```

亮度审计结果：

| 类别 | gate_0 mean_all | gate_1 mean_all | gate_2 mean_all |
|---|---:|---:|---:|
| true3d | 0.0139 | 0.0500 | 0.0221 |
| flat_false matched | 0.0139 | 0.0501 | 0.0222 |

训练结果：

| 输入 | mean best val acc | std |
|---|---:|---:|
| Full 3-gate stack | 1.0000 | 0.0000 |
| Gate 0 only | 0.5000 | 0.0000 |
| Gate 1 only | 0.7222 | 0.1667 |
| Gate 2 only | 0.5000 | 0.0000 |

解释口径：

```text
曝光匹配后 gate_1 单门从 0.8704 降到 0.7222，
说明原 gate_1 的单帧优势有一部分来自全局曝光/亮度差异；
但 gate_1 仍高于随机，说明局部前景面积、轮廓清晰度或结构线索仍存在。
完整三门 gate stack 仍保持 1.0，是目前最强的“多 gate 序列必要性”证据。
```

新增图：

```text
artifacts/figures/military_selected44_2026-07-06/hard_rect_overlap_exposure_matched_full_stack_vs_single_gate.png
```

## 14. gate_1 残余单帧线索诊断

在 gate 均值曝光匹配后，`gate_1 only` 仍为 0.7222。为判断残余线索是否仍来自亮度，新增单 gate 标量特征诊断脚本：

```text
dataset_new/diagnose_single_gate_features.py
```

诊断结果显示，在 mean-all matched 数据中，gate_1 的可分特征主要与高灰度/前景亮度相关：

| gate_1 特征 | 简单阈值最佳准确率 | true3d 均值 | flat_false 均值 |
|---|---:|---:|---:|
| max_value | 0.7500 | 0.2913 | 0.2802 |
| p99 | 0.7386 | 0.2826 | 0.2661 |
| foreground_mean | 0.7159 | 0.2365 | 0.2276 |

随后进一步构建两组控制数据：

```text
dataset_new/Military_TrueFalse_Selected44_hard_rect_overlap_foreground_classgate_matched
dataset_new/Military_TrueFalse_Selected44_hard_rect_overlap_p99_classgate_matched
```

训练结果：

| 控制设置 | gate_1 only mean best val acc |
|---|---:|
| rectangular-overlap raw | 0.8704 |
| mean-all matched | 0.7222 |
| foreground-mean matched | 0.7222 |
| p99 matched | 0.7222 |

解释口径：

```text
gate_1 原始偏高的一部分来自全局曝光差异；
但进一步匹配前景均值和 p99 后，gate_1 单门没有继续下降，
说明残余单帧线索更可能来自局部形态、边缘、前景分布或平面最大投影与真实切片之间的结构差异。
这强化了 full gate stack 的必要性，也提醒论文中要避免声称“单门完全不可用”。
```

新增图：

```text
artifacts/figures/military_selected44_2026-07-06/hard_rect_overlap_gate1_residual_controls.png
```

## 15. 3090 论文级训练矩阵准备

用户说明后续如果需要训练网络，可以使用实验室 24GB RTX 3090。基于这一硬件条件，本轮将“继续跑实验”的目标从单个短训练改为论文级训练矩阵：

```text
多随机种子 + 更长 epoch + 主实验 + 单门消融 + 融合方式对比 + 鲁棒性 + 曝光匹配控制
```

已更新训练入口：

| 文件 | 更新 |
|---|---|
| `train.py` | 新增 `--num-workers`、`--use-amp`、`--cudnn-benchmark`，支持 3090 上更快训练 |
| `run_experiments.py` | 将 worker、AMP、cuDNN benchmark 写入命令和结果 CSV，保证实验可追溯 |
| `tests/test_run_experiments.py` | 补充参数转发测试 |

新增脚本：

```text
scripts/run_3090_paper_experiments.ps1
scripts/collect_paper_experiment_report.py
```

推荐先 dry run：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_3090_paper_experiments.ps1 -DryRun
```

3090 全套运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_3090_paper_experiments.ps1
```

如果时间有限，优先运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_3090_paper_experiments.ps1 -Stages core,ablation
```

默认训练设置：

```text
Seeds: 42 / 332 / 2026 / 730 / 1009
Batch size: 32
DataLoader workers: 4
AMP: enabled on CUDA
Main epochs: 80
Ablation epochs: 40
Fusion epochs: 60
Robustness epochs: 40
```

训练结束后自动汇总：

```text
experiments\paper3090_combined_results.csv
writing\paper3090_training_report_2026-07-06.md
```

新增规划文档：

```text
writing/3090_training_plan_2026-07-06.md
```

论文口径更新：

```text
当前 3 seed 结果作为 pilot evidence；
3090 的 5 seed 全矩阵结果将作为最终论文主结果。
如果两者冲突，以 3090 多 seed 结果为准。
```

## 16. 本机优先训练环境打通

用户说明当前电脑更方便使用 Codex，因此训练策略调整为：

```text
本机 RTX 3050 Ti 先跑轻/中量实验；
实在吃显存、耗时过长、或需要最终定稿 5 seed 长 epoch 时，再交给实验室 24GB RTX 3090。
```

本机硬件检查：

```text
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
显存: 4096 MiB
CUDA: available in E:\ana\envs\pytorch1
torch: 2.4.1
```

已完成环境修复：

```powershell
E:\ana\Scripts\conda.exe install -p E:\ana\envs\pytorch1 pytest -y
```

结果：

```text
pytest 7.4.4 已安装到 pytorch1
E:\ana\envs\pytorch1\python.exe -m pytest tests
43 passed
```

新增本机训练脚本：

```text
scripts/run_local_paper_experiments.ps1
```

默认设置：

```text
Seeds: 42 / 332 / 2026
Batch size: 8
DataLoader workers: 0
AMP: enabled on CUDA
Main epochs: 20
Ablation epochs: 10
```

推荐命令：

```powershell
# 快速检查命令
powershell -ExecutionPolicy Bypass -File scripts\run_local_paper_experiments.ps1 -DryRun -Stages smoke

# 快速链路测试
powershell -ExecutionPolicy Bypass -File scripts\run_local_paper_experiments.ps1 -Stages smoke

# 本机日常推进
powershell -ExecutionPolicy Bypass -File scripts\run_local_paper_experiments.ps1
```

本机 smoke 已完成：

| 实验 | epochs | seeds | mean best val acc | 说明 |
|---|---:|---|---:|---|
| `localgpu_smoke_truefalse_rect_matched_2ep` | 2 | 42/332/2026 | 0.5000 | 只验证训练链路，不作为论文结果 |

生成汇总：

```text
experiments\localgpu_combined_results.csv
writing\localgpu_training_report_2026-07-06.md
```

同时更新 `train.py`：

```text
AMP 调用从 torch.cuda.amp.* 切换到 torch.amp.* 兼容封装；
消除了 PyTorch 2.4.1 下的 FutureWarning；
保留旧版 PyTorch fallback。
```

## 17. 本机 5 epoch 主实验与脚本类别名修正

运行本机中量主实验：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local_paper_experiments.ps1 -Stages main -MainEpochs 5
```

前两组 true/false 训练成功；第三组三类军事 scratch 第一次失败，原因是新脚本中使用了简化类别名：

```text
tank / aircraft / helicopter
```

而当前真实目录名是：

```text
01_Main_Battle_Tank
02_Fighter_Jet
03_Attack_Helicopter
```

已修正：

```text
scripts/run_local_paper_experiments.ps1
scripts/run_3090_paper_experiments.ps1
```

并删除失败的 `localgpu_military3class_scratch_5ep` 记录后重新运行，补齐三类军事 5 epoch 结果。

本机 5 epoch 结果：

| 实验 | epochs | seeds | mean best val acc | std | 解释 |
|---|---:|---|---:|---:|---|
| `localgpu_truefalse_rect_matched_full_5ep` | 5 | 42/332/2026 | 0.5000 | 0.0000 | 5 epoch 未收敛，只作短训趋势 |
| `localgpu_truefalse_hard_projection_full_5ep` | 5 | 42/332/2026 | 0.5000 | 0.0000 | 5 epoch 未收敛，只作短训趋势 |
| `localgpu_military3class_scratch_5ep` | 5 | 42/332/2026 | 0.2917 | 0.0722 | 三类军事识别需要更长训练或迁移 |

重要解释：

```text
这些 5 epoch 结果不与此前 20 epoch 的结果矛盾；
它们说明当前小样本任务对训练轮数敏感，本机短训主要用于链路验证和趋势判断。
论文主结果仍应使用 20 epoch 以上的完整训练，最终优先使用 3090 的 5 seed 长训结果。
```

当前本机汇总文件：

```text
experiments\localgpu_combined_results.csv
writing\localgpu_training_report_2026-07-06.md
```

## 18. 本机 20 epoch 复现核心 true/false 结果

为确认 5 epoch 不收敛不是模型或数据问题，继续在本机 RTX 3050 Ti 上运行两组最核心的 full gate stack 真假目标实验：

```powershell
E:\ana\envs\pytorch1\python.exe run_military_transfer_experiments.py --classes true3d flat_false -- --experiment-name localgpu_truefalse_rect_matched_full_20ep --experiment-root experiments\localgpu_truefalse_rect_matched_full_20ep --dataset-root dataset_new\Military_TrueFalse_Selected44_hard_rect_overlap_mean_classgate_matched --fusion-mode attention_residual --input-mode multi --single-gate-index 0 --split-group-by-sample-id --seeds 42 332 2026 --epochs 20 --batch-size 8 --num-workers 0 --use-amp --skip-existing --keep-going --results-csv experiments\localgpu_truefalse_rect_matched_full_20ep\results.csv --aggregate-csv experiments\localgpu_truefalse_rect_matched_full_20ep\aggregate_results.csv

E:\ana\envs\pytorch1\python.exe run_military_transfer_experiments.py --classes true3d flat_false -- --experiment-name localgpu_truefalse_hard_projection_full_20ep --experiment-root experiments\localgpu_truefalse_hard_projection_full_20ep --dataset-root dataset_new\Military_TrueFalse_Selected44_hard_projection --fusion-mode attention_residual --input-mode multi --single-gate-index 0 --split-group-by-sample-id --seeds 42 332 2026 --epochs 20 --batch-size 8 --num-workers 0 --use-amp --skip-existing --keep-going --results-csv experiments\localgpu_truefalse_hard_projection_full_20ep\results.csv --aggregate-csv experiments\localgpu_truefalse_hard_projection_full_20ep\aggregate_results.csv
```

结果：

| 实验 | epochs | seeds | mean best val acc | std | 结论 |
|---|---:|---|---:|---:|---|
| `localgpu_truefalse_rect_matched_full_20ep` | 20 | 42/332/2026 | 1.0000 | 0.0000 | 曝光匹配后 full stack 仍可稳定判别 |
| `localgpu_truefalse_hard_projection_full_20ep` | 20 | 42/332/2026 | 1.0000 | 0.0000 | hard projection full stack 复现此前 pilot 结果 |

收敛拐点：

| 实验 | seed 42 | seed 332 | seed 2026 |
|---|---|---|---|
| rectangular-overlap mean matched | epoch 11 达到 0.9，epoch 12 达到 1.0 | epoch 12 达到 1.0 | epoch 11 达到 1.0 |
| hard projection | epoch 11 达到 1.0 | epoch 11 达到 1.0 | epoch 11 达到 1.0 |

解释口径：

```text
5 epoch 结果停留在 0.5，不能说明 gate stack 无效；
20 epoch 后两组核心 true/false 实验均稳定达到 1.0，
说明当前任务在本机设置下存在约 10-12 epoch 的收敛拐点。
论文中应使用 20 epoch 及以上结果作为主证据；
短训结果只用于说明训练轮数敏感性和本机调试流程。
```

当前本机汇总已刷新：

```text
experiments\localgpu_combined_results.csv
writing\localgpu_training_report_2026-07-06.md
```

## 19. 本机 rectangular-overlap 曝光匹配数据的 single-gate 20 epoch 消融

为形成公平闭环，将本机 full stack 20 epoch 与同一数据、同一 epoch、同一 seeds 的单门输入进行对比。

运行设置：

```text
dataset = dataset_new\Military_TrueFalse_Selected44_hard_rect_overlap_mean_classgate_matched
classes = true3d / flat_false
input_mode = single-gate
single_gate_index = 0 / 1 / 2
fusion_mode = attention_residual
epochs = 20
batch_size = 8
seeds = 42 / 332 / 2026
split_group_by_sample_id = true
AMP = true
```

训练命令按 gate 循环执行：

```powershell
foreach ($Gate in 0,1,2) {
  $ExperimentName = "localgpu_truefalse_rect_matched_single_gate${Gate}_20ep"
  E:\ana\envs\pytorch1\python.exe run_military_transfer_experiments.py --classes true3d flat_false -- --experiment-name $ExperimentName --experiment-root "experiments\$ExperimentName" --dataset-root dataset_new\Military_TrueFalse_Selected44_hard_rect_overlap_mean_classgate_matched --fusion-mode attention_residual --input-mode single-gate --single-gate-index $Gate --split-group-by-sample-id --seeds 42 332 2026 --epochs 20 --batch-size 8 --num-workers 0 --use-amp --skip-existing --keep-going --results-csv "experiments\$ExperimentName\results.csv" --aggregate-csv "experiments\$ExperimentName\aggregate_results.csv"
}
```

本机 20 epoch best-val 结果：

| 输入 | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| Full 3-gate stack | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Gate 0 only | 0.9630 | 0.0321 | 0.9444 | 1.0000 |
| Gate 1 only | 0.9259 | 0.0849 | 0.8333 | 1.0000 |
| Gate 2 only | 0.8704 | 0.0642 | 0.8333 | 0.9444 |

新增训练稳定性汇总脚本：

```text
scripts/summarize_training_stability.py
```

已生成：

```text
experiments\localgpu_rect_matched_20ep_training_stability.csv
```

稳定性观察：

```text
Full stack 三个 seed 的最后 5 个 epoch 均值均为 1.0；
single gate 的 best accuracy 可以很高，但 final accuracy 和最后 5 个 epoch 均值存在更明显波动。
Gate 0 / gate 1 / gate 2 均可在长训练中捕获残余单帧线索。
```

重要修正：

```text
此前 10 epoch / 5 epoch 单门结果较低，不能直接说明单帧完全不可用；
20 epoch 后，单门已经能够利用当前仿真中的局部形态、边缘或投影差异。
因此论文叙事应改为：
1. full gate stack 是最稳定、物理解释最明确的证据；
2. 单门长训练结果偏高说明当前仿真假目标仍有残余单帧线索；
3. 下一步需要局部直方图匹配、轮廓扰动、复杂背景和更严格的 hard projection 单门复现实验。
```

论文中不能再写：

```text
单门完全不能区分真假目标。
```

应写：

```text
在当前仿真设置下，单门图像在充分训练后仍保留部分可学习线索；
但 full gate stack 在多随机种子下保持最高且最稳定表现，并且与跨 gate 物理诊断一致。
```

## 20. 本机 hard projection 数据的 single-gate 20 epoch 复现

为补齐 hard projection 数据在同一训练轮数下的单门对照，继续运行三组 20 epoch、三随机种子单门消融。该结果非常重要，因为它修正了此前短训结果中“单门接近随机”的过强表述。

运行设置：

```text
dataset = dataset_new\Military_TrueFalse_Selected44_hard_projection
classes = true3d / flat_false
input_mode = single-gate
single_gate_index = 0 / 1 / 2
fusion_mode = attention_residual
epochs = 20
batch_size = 8
seeds = 42 / 332 / 2026
split_group_by_sample_id = true
AMP = true
```

本机 20 epoch best-val 结果：

| 输入 | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| Full 3-gate stack | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Gate 0 only | 0.8889 | 0.0000 | 0.8889 | 0.8889 |
| Gate 1 only | 0.6852 | 0.1156 | 0.5556 | 0.7778 |
| Gate 2 only | 0.8704 | 0.0321 | 0.8333 | 0.8889 |

稳定性文件：

```text
experiments\localgpu_hard_projection_20ep_training_stability.csv
```

稳定性观察：

```text
Full stack 三个 seed 均在 epoch 11 达到 1.0，并且最后 5 个 epoch 均值为 1.0。
hard projection 的 gate_1 单门仍明显弱于 full stack；
但 gate_0 和 gate_2 在 20 epoch 后升高到约 0.87-0.89，说明最大投影假目标仍残留可学习的单帧形态或边缘线索。
```

对论文叙事的修正：

```text
hard projection 的短训结果可以说明单门在早期或训练不足时较弱，
但 20 epoch 复现表明单门并非完全不可用。
因此论文中应把 hard projection 解释为“削弱但没有彻底消除单帧捷径”，
把 full gate stack 的优势写成“更稳定、收敛更充分、并与跨 gate 物理诊断一致”。
```

这也给下一阶段实验提出更明确要求：

```text
如果要把创新点继续抬高，下一步应做更强的单帧控制：
1. 局部直方图匹配，让 true3d 与 flat_false 的单 gate 局部亮度分布更接近；
2. 轮廓/边缘扰动，让最大投影假目标不固定保留过强边缘；
3. 复杂背景和散射噪声，让网络不能只依赖干净背景下的边缘差异；
4. 多视角扩充，让结果不只依赖 44 个模型的固定视角。
```

## 21. hard projection 前景直方图匹配控制：一个有价值的负结果

为继续排查单门长训练偏高的来源，新增前景直方图匹配控制脚本：

```text
dataset_new\match_false_target_histogram.py
```

该脚本保持 `true3d` 不变，只对 `flat_false` 的前景像素灰度做 gate-wise quantile histogram matching，使每个 gate 的 flat false 前景灰度分布向 true3d 前景灰度分布靠齐。当前先采用 `class-gate` 模式，即每个 gate 使用该 gate 下所有 true3d 前景像素的总体分布作为匹配目标。

生成命令：

```powershell
E:\ana\envs\pytorch1\python.exe dataset_new\match_false_target_histogram.py --input-root dataset_new\Military_TrueFalse_Selected44_hard_projection --output-root dataset_new\Military_TrueFalse_Selected44_hard_projection_hist_classgate_matched --match-scope class-gate --foreground-threshold 0.031372549 --num-quantiles 256 --overwrite
```

输出：

```text
dataset_new\Military_TrueFalse_Selected44_hard_projection_hist_classgate_matched
dataset_new\Military_TrueFalse_Selected44_hard_projection_hist_classgate_matched\histogram_match_manifest.csv
```

训练就绪检查：

| class | gate pngs | valid samples | ready |
|---|---:|---:|---|
| true3d | 132 | 44 | True |
| flat_false | 132 | 44 | True |

跨 gate 诊断：

| class | corr maxnorm | mask IoU | absdiff maxnorm |
|---|---:|---:|---:|
| flat_false | 0.9166 | 0.9301 | 0.0493 |
| true3d | 0.3246 | 0.3065 | 0.1210 |

单门标量特征诊断显示，单帧线索没有被完全压下去：

| 数据 | 最强单门特征 | 简单阈值准确率 | true3d mean | flat_false mean | 解释 |
|---|---|---:|---:|---:|---|
| 原始 hard projection | gate_2 edge_density | 0.8636 | 0.2787 | 0.0764 | 边缘/纹理差异明显 |
| 原始 hard projection | gate_0 foreground_ratio | 0.8182 | 0.0708 | 0.1853 | flat false 是整目标投影，面积偏大 |
| histogram matched | gate_1 max_value | 0.9318 | 0.2913 | 0.3988 | 直方图匹配没有消除高亮极值差异 |
| histogram matched | gate_0 foreground_ratio | 0.8182 | 0.0708 | 0.1861 | 面积差异仍然存在 |
| histogram matched | gate_1 edge_density | 0.7614 | 0.1845 | 0.2892 | 边缘/局部形态仍可分 |

结论：

```text
仅做前景灰度直方图匹配不足以消除单门线索。
单门长训练偏高不是单纯由全局亮度或前景均值造成，
更可能来自 flat_false 整目标投影带来的前景面积、边缘密度、轮廓清晰度和局部形态差异。
```

论文使用口径：

```text
这组结果应写成诊断性控制实验或局限分析：
我们尝试通过 gate-wise 前景直方图匹配削弱亮度分布捷径，
但单门可分性仍由面积与边缘结构维持。
因此后续需要引入复杂背景、边缘扰动、局部块级匹配或多视角数据，
而不是只做亮度缩放。
```

对创新点的影响：

```text
这个负结果反而能让论文故事更扎实：
1. 我们不是只报告 1.0 准确率，而是在主动排查网络可能学到的捷径；
2. 结果证明当前仿真任务中的 single-gate residual cue 具有具体来源；
3. 下一步方法改进可以自然落到“更强的物理仿真与更严格的控制数据集”。
```

新增测试：

```text
tests\test_match_false_target_histogram.py
```

测试结果：

```text
4 passed
```

## 22. histogram matched 控制集的网络训练结果

在完成前景直方图匹配控制集后，继续用同一套本机训练协议跑 full stack 与三组单门 20 epoch 训练，检验该控制是否真的压低了单门捷径。

训练设置：

```text
dataset = dataset_new\Military_TrueFalse_Selected44_hard_projection_hist_classgate_matched
classes = true3d / flat_false
epochs = 20
batch_size = 8
seeds = 42 / 332 / 2026
split_group_by_sample_id = true
AMP = true
```

运行结果：

| 输入 | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| Full 3-gate stack | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Gate 0 only | 0.9444 | 0.0000 | 0.9444 | 0.9444 |
| Gate 1 only | 0.8889 | 0.0556 | 0.8333 | 0.9444 |
| Gate 2 only | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

稳定性文件：

```text
experiments\localgpu_hard_projection_hist_20ep_training_stability.csv
```

本机总表已刷新：

```text
experiments\localgpu_combined_results.csv
writing\localgpu_training_report_2026-07-06.md
```

关键解释：

```text
histogram matched 控制集没有削弱单门，反而使 gate_0 / gate_1 / gate_2 的单门结果整体偏高。
这说明单门残余线索不是简单的前景灰度分布问题；
网络仍然可以利用整目标最大投影带来的面积、边缘、轮廓和局部形态差异。
```

论文使用口径应进一步收紧：

```text
前景直方图匹配实验不能作为“单帧捷径已被排除”的正证据。
它应写成诊断性负结果：亮度直方图控制不足以消除单门可分性，
因此下一阶段必须从几何形态和成像背景层面继续增强假目标控制。
```

已同步更新实验脚本：

```text
scripts\run_local_paper_experiments.ps1
scripts\run_3090_paper_experiments.ps1
```

本地复现命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local_paper_experiments.ps1 -Stages hist -MainEpochs 20
```

3090 复现命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_3090_paper_experiments.ps1 -Stages hist -AblationEpochs 40
```

## 23. 面积与极值控制：进一步确认单门捷径来自后处理伪迹

在 histogram matched 控制集上，单门仍然很高，说明仅匹配灰度直方图不能压低面积、边缘和形态线索。因此继续构建更强的几何控制。

### 23.1 新增面积匹配脚本

新增脚本：

```text
dataset_new\match_false_target_geometry.py
```

用途：

```text
copy true3d unchanged；
对 flat_false 按 gate 匹配 true3d 的平均前景面积；
用棋盘距离优先保留离背景边界更远的核心像素，削弱整目标投影面积过大的捷径。
```

生成命令：

```powershell
E:\ana\envs\pytorch1\python.exe dataset_new\match_false_target_geometry.py --input-root dataset_new\Military_TrueFalse_Selected44_hard_projection_hist_classgate_matched --output-root dataset_new\Military_TrueFalse_Selected44_hard_projection_hist_area_classgate_matched --match-scope class-gate-mean --foreground-threshold 0.031372549 --overwrite
```

面积匹配后的诊断：

| 数据 | gate | 特征 | true3d mean | flat_false mean | 简单阈值准确率 |
|---|---:|---|---:|---:|---:|
| hist + area | 0 | foreground_ratio | 0.0708 | 0.0613 | 0.7386 |
| hist + area | 1 | foreground_ratio | 0.2028 | 0.1687 | 0.7500 |
| hist + area | 2 | foreground_ratio | 0.1098 | 0.0919 | 0.7273 |
| hist + area | 1 | max_value | 0.2913 | 0.3987 | 0.9318 |

解释：

```text
面积控制有效压低了 foreground_ratio 和 bbox_area_ratio 捷径，
但 gate_1 的极亮像素仍然是强单门线索。
```

### 23.2 max-value 缩放与 clipmax 控制

先尝试用 `max_value` 作为曝光匹配统计量，扩展：

```text
dataset_new\match_false_target_exposure.py --match-stat max_value
```

但直接缩放 false 图像会把整体前景亮度压低，制造新的 p99 / foreground_mean 捷径。因此继续新增只裁剪极亮像素的脚本：

```text
dataset_new\clip_false_target_intensity.py
```

clipmax 数据集：

```text
dataset_new\Military_TrueFalse_Selected44_hard_projection_hist_area_clipmax_classgate_matched
```

生成命令：

```powershell
E:\ana\envs\pytorch1\python.exe dataset_new\clip_false_target_intensity.py --input-root dataset_new\Military_TrueFalse_Selected44_hard_projection_hist_area_classgate_matched --output-root dataset_new\Military_TrueFalse_Selected44_hard_projection_hist_area_clipmax_classgate_matched --clip-stat max_value --clip-scope class-gate-mean --foreground-threshold 0.031372549 --overwrite
```

clipmax 后的单门特征诊断：

| gate | 最强特征 | 简单阈值准确率 | true3d mean | flat_false mean |
|---:|---|---:|---:|---:|
| 0 | max_value | 0.8182 | 0.2786 | 0.2521 |
| 1 | max_value | 0.8182 | 0.2913 | 0.2836 |
| 2 | max_value | 0.8182 | 0.2280 | 0.2052 |
| 1 | foreground_ratio | 0.7500 | 0.2028 | 0.1687 |

### 23.3 clipmax 控制集网络训练

训练设置：

```text
dataset = dataset_new\Military_TrueFalse_Selected44_hard_projection_hist_area_clipmax_classgate_matched
epochs = 20
batch_size = 8
seeds = 42 / 332 / 2026
```

训练结果：

| 输入 | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| Full 3-gate stack | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Gate 0 only | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Gate 1 only | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Gate 2 only | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

稳定性文件：

```text
experiments\localgpu_hard_projection_hist_area_clipmax_20ep_training_stability.csv
```

本机总表已刷新：

```text
experiments\localgpu_combined_results.csv
writing\localgpu_training_report_2026-07-06.md
```

关键结论：

```text
像素级后处理控制没有形成更强正证据，反而暴露出新的图像伪迹：
面积裁剪、核心像素保留和强度裁剪会生成稳定的单帧模式，网络很容易学习。
因此这组结果不能用于证明 gate stack 优势；
它应写成局限与方法改进依据：二维假目标控制应尽量回到 Blender/物理生成阶段，
而不是在 PNG 上做过多后处理。
```

下一步更有价值的方向：

```text
1. 在 Blender 中渲染真实平面假目标板，而不是 PNG 后处理；
2. 给 true3d 和 flat_false 同时加入复杂背景、地面、散射和探测噪声；
3. 对 false target 加入姿态偏移、轻微轮廓扰动、反射率纹理；
4. 做多视角扩充，减少模型固定角度造成的形态捷径。
```

已同步脚本 stage：

```text
scripts\run_local_paper_experiments.ps1: -Stages geom
scripts\run_3090_paper_experiments.ps1: -Stages geom
```
