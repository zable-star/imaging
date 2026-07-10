# 距离选通多切片目标识别基线

本项目面向 **军事三维目标识别与二维假目标判别**。当前阶段先用 Blender 距离选通仿真生成多 gate 图像序列，再用轻量神经网络验证两件事：

1. 同一目标在不同选通门下的多张切片是否能提供互补判别信息。
2. 网络是否能区分真实三维目标的多 gate 响应和二维平面假目标/异常输入。

最终目标是把该基线发展为可服务论文和后续 **多模光神经网络高速目标识别** 项目的前置平台：先在电子神经网络中证明数据、物理仿真和判别逻辑成立，再逐步迁移到光学编码、光纤传输和光电联合识别框架。

## 当前研究定位

项目目前分为两条证据线。

| 方向 | 数据设置 | 目的 |
|---|---|---|
| 三维目标识别 | `chair / desk / sofa / bed / toilet` 或后续军事三维类别 | 验证多 gate 距离选通图像是否支持目标类别识别 |
| 二维假目标判别 | 在真实三维类别外加入 `image2d` 类 | 验证网络是否能识别二维平面/异常 gate 序列 |

现阶段 ModelNet10 家具类别主要作为可控基线；军事三维目标是最终应用方向。由于军事 3D 模型数量少、标签噪声大，建议采用“先稳定基线，再小样本迁移到军事目标”的路线，而不是直接在低质量军事数据上强行训练。

## 方法概览

每个样本由同一目标的多张距离选通切片组成：

```text
sample = [gate_0.png, gate_1.png, gate_2.png, ...]
```

这些 gate 不是普通 RGB 通道，而是不同深度响应窗口下的灰度观测。网络输入形状为：

```text
[B, S, C, H, W]
```

其中 `B` 是 batch size，`S` 是 gate 数量，`C=1`，默认图像尺寸为 `224 x 224`。

当前模型结构：

```text
每张 gate 图像
    -> 共享 CNN 编码器 SliceEncoder
    -> gate-level 特征 f_i
    -> mean / attention / concat / attention_residual 融合
    -> MLP 分类器
    -> 类别 logits
```

当前 attention 是轻量 MLP 打分：

```text
alpha_i = softmax(MLP(f_i))
f_att = sum_i alpha_i f_i
```

它不是 Transformer/QKV attention。这里的 attention 权重应解释为 **gate-level discriminative contribution（门控切片级判别贡献）**，不要解释成视觉显著性。

## 核心创新点

项目建议围绕以下创新点展开，而不是单纯追求换大网络：

1. **距离选通物理仿真驱动的数据构建**  
   用 Blender 模拟不同 gate spacing、门宽、脉宽、距离衰减和背景退化，使数据具备明确物理含义。

2. **多 gate 判别贡献分析**  
   比较 `mean / attention / concat / attention_residual`，说明多深度选通观测如何影响分类，以及网络在不同 gate 上的判别依赖。

3. **二维假目标与真实三维目标的序列差异建模**  
   二维假目标不是简单的一张假图，而是“整目标轮廓在多个 gate 中强度随选通响应变化”的异常序列。该设计更接近激光选通下平面诱饵的物理表现。

4. **面向小样本军事目标的迁移训练路线**  
   先在 ModelNet10/可控仿真上预训练 gate 融合网络，再用少量筛查后的军事 3D 模型微调，重点体现数据治理、物理建模和迁移学习能力。

5. **向多模光神经网络的可融合接口**  
   当前 CNN 融合模型可作为电子基线；后续可把 gate stack 编码到光学输入，将中间传播替换为多模光纤散斑/光学层，再比较电子网络和光电联合网络的速度、精度与鲁棒性。

## 数据集设计

### 1. 可控基线数据

默认使用 ModelNet10 中的五个类别：

| 类别 | 标签 |
|---|---:|
| chair | 0 |
| desk | 1 |
| sofa | 2 |
| bed | 3 |
| toilet | 4 |
| image2d | 5 |

其中 `image2d` 是二维异常/假目标类，用于检验网络是否能识别非真实三维 gate 序列。

### 2. 二维假目标建模原则

对平面二维假目标，更合理的 gate 序列不是“只有一小部分物体出现”，而是：

```text
整个目标轮廓在各 gate 中都保持一致，
但强度根据平面目标深度与接收门响应逐渐变弱，甚至消失。
```

也就是说，二维假目标的关键异常不是形状残缺，而是缺少真实三维目标应有的深度分层响应。

### 3. 军事三维目标数据

军事数据建议采用筛查后的小样本策略：

| 阶段 | 目标 |
|---|---|
| 粗筛 | 去除标签错误、空模型、非目标类别模型 |
| 精筛 | 每类保留外形清楚、尺度正常、方向可统一的模型 |
| 渲染 | 使用相同 gate 参数生成真三维目标序列 |
| 假目标 | 由整目标轮廓生成平面回波序列 |
| 训练 | 用 ModelNet10 预训练权重迁移到军事类别 |

军事类别数量不宜一开始过多。建议先做 3 到 5 类，例如坦克、飞机、直升机、军车、导弹车。每类优先保证质量，再扩大数量。

## 训练任务安排

### 任务 A：六分类二维假目标判别

目标：

```text
chair / desk / sofa / bed / toilet / image2d
```

用途：

- 证明网络能区分真实三维 gate stack 和二维异常 gate stack。
- 输出混淆矩阵、attention 权重和 per-class accuracy。
- 作为 PPT 和论文中“假目标判别能力”的第一组证据。

推荐命令：

```powershell
python run_experiments.py --experiment-name six_class_attention_residual_seedmatched --dataset-root dataset --fusion-mode attention_residual --seeds 42 332 2026 --epochs 30 --batch-size 16
```

### 任务 B：五分类物理参数消融

目标：

```text
chair / desk / sofa / bed / toilet
```

用途：

- 分析 gate spacing、门宽、脉宽等物理参数对识别效果的影响。
- 避免 `image2d` 黑 gate 模式干扰物理参数结论。

推荐命令：

```powershell
python run_physical_5class_experiments.py -- --experiment-name phys_gate_spacing_large_attention_residual --dataset-root dataset_gate_spacing_large --fusion-mode attention_residual --seeds 42 332 2026 --epochs 30 --batch-size 16
```

### 任务 C：融合方式对比

至少比较：

| 融合方式 | 作用 |
|---|---|
| `mean` | 无学习权重的平均融合基线 |
| `attention` | 可解释 gate 权重 |
| `attention_residual` | 当前主模型，兼顾 gate 权重和特征保留 |
| `concat` | 高信息量经验基线，但解释性较弱 |

论文中不建议把 `concat` 写成理论上限，只写成已测试方法中的高精度经验基线。

### 任务 D：军事小样本迁移

建议顺序：

1. 用可控数据训练 `attention_residual` 基线。
2. 冻结或半冻结 `SliceEncoder`，替换分类头。
3. 用少量高质量军事三维模型微调。
4. 与从零训练对比。
5. 报告准确率、混淆矩阵、类别间混淆原因和 gate-level attention。

从零训练示例：

```powershell
python run_military_transfer_experiments.py --classes tank aircraft military_vehicle -- --experiment-name military_scratch_attention_residual --dataset-root dataset_new\Military_3D_Dataset --fusion-mode attention_residual --seeds 42 332 2026 --epochs 30 --batch-size 8
```

冻结 encoder 迁移示例：

```powershell
python run_military_transfer_experiments.py --classes tank aircraft military_vehicle -- --experiment-name military_transfer_frozen_encoder --dataset-root dataset_new\Military_3D_Dataset --fusion-mode attention_residual --pretrained-model-path experiments\six_class_attention_residual_seedmatched\six_class_attention_residual_seed42\slice_attention_model.pth --freeze-encoder --seeds 42 332 2026 --epochs 30 --batch-size 8
```

半冻结微调示例：

```powershell
python run_military_transfer_experiments.py --classes tank aircraft military_vehicle -- --experiment-name military_transfer_finetune --dataset-root dataset_new\Military_3D_Dataset --fusion-mode attention_residual --pretrained-model-path experiments\six_class_attention_residual_seedmatched\six_class_attention_residual_seed42\slice_attention_model.pth --seeds 42 332 2026 --epochs 30 --batch-size 8 --lr 0.0001
```

`train.py` 已支持这些迁移参数：

| 参数 | 作用 |
|---|---|
| `--pretrained-model-path` | 加载已有模型权重 |
| `--pretrained-include-classifier` | 类别数相同且希望复用分类头时开启；军事迁移通常不开 |
| `--freeze-encoder` | 冻结共享 CNN 编码器，只训练融合层和分类头 |
| `--freeze-attention` | 冻结 gate attention 打分器 |
| `--freeze-residual` | 冻结 `attention_residual` 中的 concat residual 投影 |

军事数据必须先使用人工筛选后的 44 个模型，不要直接使用 500 个候选模型。推荐流程：

```powershell
# 1. 使用人工缩略图审查表，只复制 keep=1 的 44 个模型
E:\ana\envs\pytorch1\python.exe dataset_new\build_selected_subset.py --review-csv dataset_new\Military_3D_Dataset\_review_sheets\thumbnail_review.csv --output-root dataset_new\Military_3D_Selected44 --expected-count 44

# 2. 将 44 个有效 glb 渲染成 gate stack
powershell -ExecutionPolicy Bypass -File scripts\render_selected_military_gates.ps1

# 3. 检查渲染结果是否可训练
E:\ana\envs\pytorch1\python.exe dataset_new\check_gate_dataset_ready.py --root dataset_new\Military_3D_Gated_Selected44 --expected-num-slices 3

# 4. 生成同一批模型的二维平面假目标 gate stack
powershell -ExecutionPolicy Bypass -File scripts\render_selected_military_gates.ps1 -OutputRoot dataset_new\Military_3D_FlatEcho_Selected44_gain10 -TargetMode flat-echo -FlatTargetGateIndex 0 -FlatMinResponse 0.18 -FlatEchoGain 10

# 5. 合并成 true3d / flat_false 二分类数据集
E:\ana\envs\pytorch1\python.exe dataset_new\build_true_false_dataset.py --true-root dataset_new\Military_3D_Gated_Selected44 --false-root dataset_new\Military_3D_FlatEcho_Selected44_gain10 --output-root dataset_new\Military_TrueFalse_Selected44_gain10 --expected-num-slices 3 --overwrite
```

### 任务 E：3090 论文级训练矩阵

当前本机 RTX 3050 Ti Laptop GPU 可以承担日常短/中量训练；实验室 24GB RTX 3090 用于最终长 epoch、多随机种子全矩阵验证。建议采用“本机先推进，本机跑不动或需要最终定稿结果时再上 3090”的策略。

本机优先运行：

```powershell
# 先检查命令
powershell -ExecutionPolicy Bypass -File scripts\run_local_paper_experiments.ps1 -DryRun

# 默认跑 main + ablation：主实验和单门消融
powershell -ExecutionPolicy Bypass -File scripts\run_local_paper_experiments.ps1

# 只做快速 smoke
powershell -ExecutionPolicy Bypass -File scripts\run_local_paper_experiments.ps1 -Stages smoke
```

本机默认设置：

```text
seeds = 42 / 332 / 2026
batch_size = 8
num_workers = 0
AMP = true
main epochs = 20
ablation epochs = 10
```

本机结果会汇总到：

```text
experiments\localgpu_combined_results.csv
writing\localgpu_training_report_2026-07-06.md
```

如果使用实验室 24GB RTX 3090，建议不要只把单个实验 epoch 加长，而是系统补齐论文需要的验证矩阵：

| 实验组 | 目的 |
|---|---|
| `core` | 三类军事目标识别、迁移学习和 true/false 主实验 |
| `ablation` | full gate stack 与 gate_0/gate_1/gate_2 单门输入对比 |
| `fusion` | `mean / attention / attention_residual / concat` 融合方式对比 |
| `robustness` | 噪声、背景散射、Poisson 光子噪声、gate dropout、单 gate 衰减 |
| `controls` | foreground-mean 和 p99 曝光匹配控制，继续压低亮度捷径 |

推荐先做 dry run 检查命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_3090_paper_experiments.ps1 -DryRun
```

确认无误后在 3090 上运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_3090_paper_experiments.ps1
```

如果时间有限，先跑核心和单门消融：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_3090_paper_experiments.ps1 -Stages core,ablation
```

该脚本默认使用：

```text
Python = E:\ana\envs\pytorch1\python.exe
seeds = 42 / 332 / 2026 / 730 / 1009
batch_size = 32
num_workers = 4
AMP = true
```

训练完成后会自动汇总：

```text
experiments\paper3090_combined_results.csv
writing\paper3090_training_report_2026-07-06.md
```

写论文时，3090 结果应该优先替换当前 3 随机种子的初步结果；如果 5 随机种子下结论仍成立，核心说服力会明显强于目前的快速试跑。

## 主要文件

| 文件/目录 | 作用 |
|---|---|
| `dataset.py` | 按“一个样本对应多张 gate 图像”的方式读取数据 |
| `model.py` | 定义 `mean / attention / concat / attention_residual` 四种融合模型 |
| `train.py` | 单次训练入口，保存模型、曲线、混淆矩阵和 attention CSV |
| `run_experiments.py` | 多随机种子、多设置实验管理 |
| `run_physical_5class_experiments.py` | 五分类物理消融包装脚本 |
| `run_military_transfer_experiments.py` | 军事小样本迁移实验包装脚本 |
| `make_image2d_class.py` | 构建二维假目标/异常类 |
| `dataset_new/generate.py` | 从 Objaverse 关键词筛选军事 3D 模型候选 |
| `dataset_new/build_selected_subset.py` | 按人工筛选清单复制军事有效模型，支持 `--expected-count 44` 防误用 |
| `dataset_new/build_true_false_dataset.py` | 将真实三维与平面假目标 gate stack 合并成 `true3d / flat_false` 二分类训练集 |
| `dataset_new/check_gate_dataset_ready.py` | 检查数据目录是否已经具备可训练的 `*_gate_*.png` 样本 |
| `dataset_new/audit_gate_image_quality.py` | 审计 gate 图像亮度、前景像素和低对比样本 |
| `dataset_new/normalize_gate_dataset.py` | 生成强度归一化 gate 数据集，削弱绝对亮度捷径 |
| `dataset_new/diagnose_gate_stack.py` | 量化 gate 间相关性、掩膜 IoU 和归一化差分 |
| `dataset_new/review_dataset.py` | 辅助人工筛查军事模型 |
| `origindataset/gated_blender_physical.py` | Blender 距离选通渲染脚本 |
| `scripts/run_truck_gate_renders.ps1` | 单个军事车辆真假目标渲染示例 |
| `scripts/render_selected_military_gates.ps1` | 将 44 个精选军事模型批量渲染为 gate stack |
| `scripts/run_truefalse_single_gate_ablation.ps1` | 对军事 true/false 数据做 gate_0/1/2 单 gate 快速消融 |
| `scripts/run_local_paper_experiments.ps1` | 在本机 RTX 3050 Ti 上运行轻/中量论文推进实验 |
| `scripts/run_3090_paper_experiments.ps1` | 在 24GB 3090 上运行论文级多种子训练矩阵 |
| `scripts/collect_paper_experiment_report.py` | 汇总 `paper3090_*` aggregate 结果为 CSV 和 Markdown 报告 |
| `scripts/summarize_gate_experiment.py` | 从 `val_attention_weights.csv` 汇总 per-class accuracy 与平均 gate attention |
| `scripts/plot_military_selected44_results.py` | 生成军事 44 模型 PPT/论文用结果图 |
| `scripts/build_military_selected44_ppt.py` | 生成军事 44 模型汇报 PPTX、讲稿和 gate stack 对照图 |
| `plot_experiment_results.py` | 汇总实验结果并绘图 |
| `writing/` | 论文草稿、交接文档、实验解释与项目路线 |
| `writing/project_roadmap_2026-07-05.md` | 当前训练、创新点提升、军事目标迁移和光神经网络融合路线图 |
| `writing/daily_progress_2026-07-05.md` | 今日推进记录：成像推导、网络推导、迁移学习接口和已有结果 |
| `writing/paper_evidence_matrix_military_gated_false_target_2026-07-06.md` | 论文主张、证据、风险表述和结果表矩阵 |
| `writing/paper_draft_military_gated_false_target_2026-07-06.md` | 基于当前结果形成的中文论文初稿 |
| `writing/ppt_military_selected44_storyline_2026-07-06.md` | 军事 44 模型汇报页级大纲 |
| `presentation_html/` | 文献分享 PPT HTML 与讲稿材料 |

## 输出结果

每次训练通常输出：

| 文件 | 说明 |
|---|---|
| `summary.json` | 实验配置与最佳验证结果 |
| `training_history.csv` | 每轮训练损失和验证精度 |
| `training_curves.png` | 训练曲线 |
| `best_confusion_matrix.png` | 最佳模型混淆矩阵 |
| `val_attention_weights.csv` | 每个验证样本的预测、真实标签、gate 权重和类别概率 |

多次实验会汇总到：

```text
experiments/results.csv
experiments/aggregate_results.csv
```

## 当前军事 44 模型结果快照

人工筛选后的军事小样本集来自：

```text
dataset_new\Military_3D_Dataset\_review_sheets\thumbnail_review.csv
```

当前 keep=1 共 44 个模型：

| 类别 | 数量 |
|---|---:|
| 01_Main_Battle_Tank | 12 |
| 02_Fighter_Jet | 20 |
| 03_Attack_Helicopter | 12 |

已生成并检查通过的数据集：

| 数据集 | 样本数 | 用途 |
|---|---:|---|
| `dataset_new\Military_3D_Gated_Selected44` | 44 | 真实三维军事目标三分类 |
| `dataset_new\Military_3D_FlatEcho_Selected44_gain10` | 44 | 同源模型的平面假目标 gate stack |
| `dataset_new\Military_TrueFalse_Selected44_gain10` | 88 | `true3d / flat_false` 二分类 |
| `dataset_new\Military_TrueFalse_Selected44_gain10_per_gate_norm` | 88 | 逐 gate 最大值归一化二分类，削弱亮度捷径 |
| `dataset_new\Military_TrueFalse_Selected44_hard_projection` | 88 | 由真实 3D gate stack 最大投影生成的更难平面假目标 |
| `dataset_new\Military_TrueFalse_Selected44_hard_rect_overlap` | 88 | hard projection + 矩形激光脉冲/矩形接收门重叠响应，更接近物理选通响应 |
| `dataset_new\Military_TrueFalse_Selected44_hard_rect_overlap_mean_classgate_matched` | 88 | rectangular-overlap 的 gate 均值曝光匹配控制集 |

关键结果汇总：

| 实验 | mean best val acc | 说明 |
|---|---:|---|
| 军事三分类，迁移冻结 encoder | 0.7500 | 3 seed 稳定，std = 0 |
| 军事三分类，从零训练 | 0.7083 | 最高 0.875，但 std = 0.1443 |
| true3d / flat_false，原始 gain10 | 1.0000 | 二分类链路跑通 |
| true3d / flat_false，per-gate 归一化 full stack | 1.0000 | 去掉绝对亮度后仍可稳定区分 |
| per-gate 归一化 only gate_0 | 0.9630 | 单门仍有信息，但低于 full stack |
| per-gate 归一化 only gate_1 | 0.9630 | 单门仍有信息，但低于 full stack |
| per-gate 归一化 only gate_2 | 0.8889 | 第三门单独最弱 |
| per-gate 归一化随机丢一门 | 0.9074 | gate stack 完整性会影响稳定性 |
| per-gate 归一化 + 噪声/背景/Poisson | 0.9815 | 对常见成像退化仍较稳 |
| hard projection full stack | 1.0000 | 更难假目标下三门序列仍稳定可分 |
| hard projection only gate_0 | 0.5370 | 单张图接近随机 |
| hard projection only gate_1 | 0.6296 | 单张图明显弱于 full stack |
| hard projection only gate_2 | 0.5370 | 单张图接近随机 |
| rectangular-overlap full stack | 1.0000 | 矩形脉冲-门重叠响应下三门序列仍稳定可分 |
| rectangular-overlap only gate_0 | 0.5000 | 单张图为随机水平 |
| rectangular-overlap only gate_1 | 0.8704 | 峰值 gate 仍保留部分单帧线索，但低于 full stack |
| rectangular-overlap only gate_2 | 0.5000 | 单张图为随机水平 |
| exposure-matched rectangular-overlap full stack | 1.0000 | gate 均值曝光匹配后三门序列仍稳定可分 |
| exposure-matched rectangular-overlap only gate_0 | 0.5000 | 单张图为随机水平 |
| exposure-matched rectangular-overlap only gate_1 | 0.7222 | 曝光匹配后明显低于未匹配 gate_1 |
| exposure-matched rectangular-overlap only gate_2 | 0.5000 | 单张图为随机水平 |
| foreground-matched rectangular-overlap only gate_1 | 0.7222 | 前景均值匹配后 gate_1 残余仍存在 |
| p99-matched rectangular-overlap only gate_1 | 0.7222 | 高分位亮度匹配后 gate_1 残余仍存在 |

gate stack 物理诊断结果：

| 数据设置 | 类别 | mean pair corr maxnorm | mean pair mask IoU | mean pair absdiff maxnorm |
|---|---|---:|---:|---:|
| per-gate norm | flat_false | 0.9995 | 0.9880 | 0.0055 |
| per-gate norm | true3d | 0.3244 | 0.3174 | 0.1211 |
| hard projection | flat_false | 0.9768 | 0.9301 | 0.0062 |
| hard projection | true3d | 0.3246 | 0.3065 | 0.1210 |
| rectangular-overlap | flat_false | 0.9731 | 0.4299 | 0.0172 |
| rectangular-overlap | true3d | 0.3246 | 0.3065 | 0.1210 |
| rectangular-overlap exposure matched | flat_false | 0.9698 | 0.4274 | 0.0210 |
| rectangular-overlap exposure matched | true3d | 0.3246 | 0.3065 | 0.1210 |

解释：

```text
平面假目标的三张 gate 几乎是同一整目标轮廓的强度缩放；
真实三维目标在不同 gate 中呈现明显结构变化。
这为“激光选通 gate stack 能区分二维平面诱饵和三维目标”提供了直接的物理证据。
```

hard projection 版本进一步把单帧捷径压低：

```text
二维假目标轮廓直接来自真实三维 gate stack 的最大投影；
单独输入任一 gate 时准确率接近随机，
但输入完整三门序列时仍达到 1.0。
```

rectangular-overlap 版本进一步把假目标强度系数改成矩形激光脉冲和矩形接收门的重叠长度：

```text
gate_0 和 gate_2 单独输入为 0.5；
gate_1 单独输入仍有 0.8704，说明峰值 gate 有残余单帧线索；
完整三门 gate stack 仍为 1.0，是当前更物理的主结果。
```

进一步做 gate 均值曝光匹配后：

```text
flat_false 与 true3d 的 gate0/1/2 全图均值基本对齐；
gate_1 单独输入从 0.8704 降到 0.7222；
完整三门 gate stack 仍为 1.0。
```

继续对 gate_1 做 foreground mean 和 p99 高分位匹配后，gate_1 单门仍为 0.7222。这说明残余单帧线索不只是全图均值、前景均值或高分位亮度，而更可能来自局部形态、边缘或前景结构差异。

这更适合放在 PPT 或论文中解释“为什么需要激光选通序列，而不是只做普通单张图分类”。

详细总表：

```text
experiments\military_selected44_results_overview_2026-07-06.csv
dataset_new\military_true_false_selected44_brightness_summary_2026-07-06.csv
dataset_new\military_true_false_selected44_hard_rect_overlap_gate_diagnostics_by_class_2026-07-06.csv
dataset_new\military_true_false_selected44_hard_rect_overlap_mean_classgate_matched_brightness_2026-07-06.csv
dataset_new\military_true_false_selected44_hard_rect_overlap_mean_classgate_matched_gate_diagnostics_by_class_2026-07-06.csv
```

PPT 可用图已生成到：

```text
artifacts\figures\military_selected44_2026-07-06
```

| 图 | 用途 |
|---|---|
| `military_3class_transfer_vs_scratch.png` | 展示军事三分类迁移学习比从零训练更稳定 |
| `hard_projection_full_stack_vs_single_gate.png` | 展示 hard projection 下 full stack 明显优于单 gate |
| `hard_rect_overlap_full_stack_vs_single_gate.png` | 展示矩形脉冲-门重叠响应下 full stack 优于单 gate |
| `hard_rect_overlap_exposure_matched_full_stack_vs_single_gate.png` | 展示曝光匹配后 full stack 仍优于单 gate |
| `hard_rect_overlap_gate1_residual_controls.png` | 展示 gate_1 残余单帧线索的均值/前景/p99 匹配控制 |
| `per_gate_norm_robustness.png` | 展示随机丢门、噪声背景退化下的鲁棒性 |
| `gate_stack_physical_diagnostics.png` | 展示 flat_false 与 true3d 的跨 gate 相关性/IoU 差异 |

可编辑 PPTX 初稿和讲稿：

```text
presentation_outputs\military_selected44_gated_report_2026-07-06.pptx
presentation_outputs\military_selected44_gated_report_speaker_notes_2026-07-06.md
```

## 当前优先级

| 优先级 | 工作 | 产出 |
|---|---|---|
| P0 | 继续补强 true/false 假目标难度 | 已完成 rectangular-overlap；下一步加入姿态偏移、曝光匹配和复杂背景 |
| P0 | 跑通五分类 num gates = 1 / 3 / 5 消融 | 物理仿真有效性证据 |
| P1 | 剔除低对比军事样本后重跑三分类 | 更稳的军事目标应用证明 |
| P1 | 做迁移学习更多冻结策略对比 | 小样本能力证明 |
| P2 | 加入噪声、衰减、背景散射、gate dropout | 鲁棒性证明 |
| P2 | 设计 gate/depth prior 辅助任务 | 创新点升级 |
| P3 | 接入多模光纤散斑或光学编码模块 | 面向高速光神经网络融合 |

## 环境与运行

在本机如果默认 Python 没有 PyTorch，请切换到安装了 PyTorch 的环境后运行训练。实验室机器可参考交接文档中的解释器路径。

运行测试：

```powershell
python -m pytest tests -v
```

快速查看实验脚本参数：

```powershell
python run_experiments.py --help
python run_physical_5class_experiments.py --help
python run_military_transfer_experiments.py --help
```

## 论文叙事建议

建议把论文主线写成：

```text
距离选通成像能够在不同门延迟下获得目标的多深度响应图像。本文构建多 gate 仿真数据，并研究这些深度选择性观测对目标识别和二维假目标判别的贡献。通过共享 CNN 编码和 gate-level 融合网络，模型能够在保持判别贡献可解释性的同时完成多类目标识别；进一步通过物理参数消融、假目标建模和小样本军事目标迁移，验证该方法向高速多模光神经网络识别系统融合的可行性。
```

这条叙事能同时体现：

- 物理成像理解能力
- 数据集构建与筛查能力
- 神经网络训练与消融实验能力
- 小样本迁移学习能力
- 面向光电融合系统的工程扩展能力
