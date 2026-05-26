# 线程交接文档：gated slice attention 论文项目

交接日期：2026-05-26  
工作目录：`E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline`

## 1. 项目总目标

把当前项目从一个可以运行的 gated slice 分类 baseline，推进到可以支撑论文初稿和后续投稿的完整实验体系。

论文主线：

```text
光学距离选通参数
    -> Blender 物理启发 gated slice 数据生成
    -> CNN / attention 网络识别
    -> 分类性能、鲁棒性、注意力权重分析
    -> 形成论文方法、实验和讨论
```

核心科学问题：

1. 距离选通成像得到的多深度切片是否包含三维物体识别信息？
2. gate center、gate width 等光学参数如何影响切片信息量和分类性能？
3. 注意力机制能否自动学习不同深度切片的重要性，并提供可解释分析？

详细执行计划已保存：

```text
writing/publication_execution_plan.md
```

## 2. 当前数据和默认实验设置

数据集位置：

```text
dataset/
```

当前数据规模：

```text
5 类：chair, desk, sofa, bed, toilet
每类 100 个样本
总计 500 个样本
每个样本 3 张 gated slice：gate_0 / gate_1 / gate_2
```

当前 Blender gate 默认参数在：

```text
origindataset/gated_blender_physical.py
```

关键参数：

```python
gate_centers = [6.8, 7.4, 8.0]
receiver_gate_width = 1.0
laser_pulse_width = 0.45
range_loss_power = 2.0
```

说明：

- 三个 gate center 当前是等间距，间距为 `0.6`。
- 有效响应范围约为 `(receiver_gate_width + laser_pulse_width) / 2 = 0.725`。
- 因此相邻 gate 有明显重叠，不容易切不到特征，但切片差异可能偏弱。

## 3. 已完成的代码改动

### 3.1 `model.py`

新增模型文件：

```text
model.py
```

包含：

- `SliceEncoder`
- `SliceAttentionClassifier`

模型逻辑：

```text
输入 x: [B, S, C, H, W]
    -> reshape 为 [B*S, C, H, W]
    -> 共享 CNN 编码每张切片
    -> 得到 [B, S, 128] 切片特征
    -> attention 为每张切片生成权重
    -> 加权融合为物体级特征
    -> classifier 输出类别 logits
```

代码中已经加了中文注释。

### 3.2 `train.py`

已做训练稳定化：

- 默认学习率从 `1e-3` 改为 `3e-4`
- 增加 `--min-lr`
- 增加余弦退火 `CosineAnnealingLR`
- 增加 `label_smoothing`
- 增加梯度裁剪 `grad_clip`
- 固定 cuDNN 随机性
- 曲线图同时画 raw 和 EMA 平滑曲线
- 保存 `training_history.csv`

重要新改动：

```text
已把 random_split 改成 stratified_split
```

原因：

原来验证集是普通随机划分，导致混淆矩阵每类数量不一致。现在按类别分层划分。

当前默认：

```python
VAL_RATIO = 0.2
```

所以验证集为：

```text
总验证集 100 个
每类 20 个
```

已验证输出：

```text
val_len 100
{'chair': 20, 'desk': 20, 'sofa': 20, 'bed': 20, 'toilet': 20}
```

如果用户想每类 25 个，需要运行：

```powershell
python train.py --val-ratio 0.25
```

### 3.3 测试

新增：

```text
tests/test_model.py
tests/test_train_utils.py
```

覆盖内容：

- 模型输出形状
- attention 权重归一化
- EMA 函数
- stratified split 每类数量均衡
- 非法 `val_ratio` 报错

注意：

- 默认 Python 环境没有 `torch`，所以部分测试会跳过。
- `pytorch1` conda 环境有 `torch`，但之前发现缺少 `pytest` 模块。

已成功运行过：

```powershell
python -m py_compile train.py model.py
E:\ana\Scripts\conda.exe run -n pytorch1 python -c "from train import build_class_dirs, DEFAULT_CLASSES, DATASET_ROOT, VAL_RATIO, SEED, stratified_split; from dataset import MultiSliceObjectDataset; ds=MultiSliceObjectDataset(build_class_dirs(DATASET_ROOT, DEFAULT_CLASSES), expected_num_slices=3); train_set,val_set=stratified_split(ds, VAL_RATIO, SEED); counts={c:0 for c in ds.class_names}; [counts.__setitem__(ds.samples[idx].class_name, counts[ds.samples[idx].class_name]+1) for idx in val_set.indices]; print('val_len', len(val_set)); print(counts)"
```

## 4. 已生成的图和文档

论文执行计划：

```text
writing/publication_execution_plan.md
```

CNN + attention 原理图：

```text
artifacts/cnn_attention_principle.svg
artifacts/cnn_attention_principle_nature.svg
```

推荐使用 Nature 风格版本：

```text
artifacts/cnn_attention_principle_nature.svg
```

当前论文草稿：

```text
writing/paper_draft.md
```

另有：

```text
writing/full.md
```

注意：`writing/` 当前是未跟踪目录，里面是用户/本线程正在整理的论文材料，不要删除。

## 5. 当前工作区状态提示

已知未提交改动包括：

```text
M train.py
?? model.py
?? tests/test_model.py
?? tests/test_train_utils.py
?? writing/
```

还新增过图文件：

```text
artifacts/cnn_attention_principle.svg
artifacts/cnn_attention_principle_nature.svg
```

下一个线程接手时应先运行：

```powershell
git status --short
```

不要随意回滚这些改动。

## 6. 下一步最高优先级

下一步不要先换大模型，应该先做实验管理脚本。

目标：

```text
实现 experiment runner：
1. 支持多 seed 训练；
2. 支持自动创建 artifact 输出目录；
3. 自动读取每次训练的 summary.json；
4. 汇总 best_val_acc、参数和输出路径到 CSV；
5. 后续可扩展 gate 参数消融。
```

建议新增文件：

```text
run_experiments.py
```

建议输出目录结构：

```text
experiments/
  baseline_seed42/
    summary.json
    training_curves.png
    training_history.csv
    best_confusion_matrix.png
    val_attention_weights.csv
  baseline_seed123/
  baseline_seed2025/
  results.csv
```

第一版 runner 只需要支持 baseline 多 seed：

```text
seed = 42, 123, 2025
```

完成后再扩展到 gate 参数消融。

## 7. 推荐后续实验顺序

### Step 1：baseline 多 seed

固定当前参数：

```text
gate_centers = [6.8, 7.4, 8.0]
receiver_gate_width = 1.0
range_loss_power = 2.0
val_ratio = 0.2
```

跑：

```text
seed = 42, 123, 2025
```

输出：

```text
Table 1: baseline accuracy mean/std
Fig: training curves
Fig: confusion matrix
Fig: attention weights by class
```

### Step 2：gate center / gate width 消融

gate center：

```text
A: [6.8, 7.4, 8.0]
B: [6.8, 8.0, 9.2]
C: [6.6, 7.8, 9.0]
```

gate width：

```text
0.6, 1.0, 1.4
```

组合：

```text
3 × 3 × 3 seeds = 27 次训练
```

### Step 3：模型消融

至少比较：

```text
Single gate_0
Single gate_1
Single gate_2
Average fusion
Concat fusion
CNN + attention
```

3090 可用后可加：

```text
ResNet18 + attention
ResNet34 + attention
```

### Step 4：物理参数和鲁棒性实验

物理参数：

```text
range_loss_power = 0, 2, 4
```

鲁棒性：

```text
Gaussian noise
blur
intensity scaling
gate center offset
```

## 8. 3090 使用策略

3090 应优先用于：

1. 多 seed 和大量消融；
2. 更大 batch size；
3. 更高输入分辨率；
4. ResNet18 / ResNet34 baseline。

不要一开始盲目上 ViT。当前数据量只有 500 个样本，ViT 容易过拟合，论文解释也不一定更强。

## 9. 期刊目标

当前最现实目标：

```text
Applied Optics
Optical Engineering
IEEE Photonics Journal
```

如果后续补充充分消融、更严谨物理模型，或者加入真实光学实验：

```text
Optics Express
```

当前阶段不要以 Nature 系列为目标。先把工程和实验体系做扎实。

## 10. 给下一个线程的第一句话建议

可以直接从这里开始：

```text
请读取 writing/thread_handoff_2026-05-26.md 和 writing/publication_execution_plan.md。
接下来先实现 run_experiments.py，用于 baseline 多 seed 实验，并把结果汇总到 experiments/results.csv。
不要回滚现有 train.py、model.py 和 tests 的改动。
```
