# 切片注意力分类基线

基于 ModelNet10 的五个物体类别和一个二维异常类别，验证 **同一物体的多张选通切片是否能够支持分类，以及网络更依赖哪些切片**。

这版模型暂时去掉了多模光纤传播模块，先专注于"切片本身的信息是否有效"。

## 核心思路

- 一个普通物体样本 = 同一物体的3张正交切片（gate_0 / gate_1 / gate_2）
- 一个 `image2d` 异常样本 = 只有一个 gate 含有二维图像信息，其它 gate 为全黑图
- 每张切片都是单通道灰度图
- 使用共享 CNN（`SliceEncoder`）分别提取每张切片的128维特征
- 使用注意力模块学习每张切片的重要性权重
- 对切片特征加权融合后送入分类器，输出5类概率

## 数据集

使用 ModelNet10 中的5个类别，每类100个 gated slice 样本，并额外生成一个 `image2d` 二维异常类别：

| 类别 | 标签 |
|------|------|
| chair | 0 |
| desk | 1 |
| sofa | 2 |
| bed | 3 |
| toilet | 4 |
| image2d | 5 |

## 输入形式

模型输入张量形状为 `[num_slices, 1, H, W]`，默认 `num_slices=3, H=W=224`。

一个样本不被视为普通的3通道图像，而是 **同一物体的3张顺序切片**。

## 文件说明

| 文件 | 作用 |
|------|------|
| `dataset.py` | 按"一个物体对应多张切片"的方式读取数据，自动按 gate_0 / gate_1 / gate_2 排序 |
| `model.py` | 模型结构：`SliceEncoder`（CNN编码器）+ `SliceAttentionClassifier`（注意力融合+分类） |
| `train.py` | 训练入口：构建数据集、划分训练/验证集、训练、保存结果 |
| `analyze_attention.py` | 分析验证集注意力权重分布，生成每个样本的注意力柱状图和各类别/各gate的均值图 |
| `analyze_gate_sparsity.py` | 统计各gate切片图像的空白比例和平均强度 |
| `convert_off_to_obj_dataset.py` | 将 ModelNet10 的 OFF 文件转换为 OBJ 文件 |

## 数据准备流程

### 1. OFF → OBJ 转换

```powershell
python convert_off_to_obj_dataset.py
```

输入：`origindataset\ModelNet10\ModelNet10\<class>\<split>\*.off`

输出：`obj_dataset\<class>\<split>\*.obj`

### 2. Blender 渲染选通切片

```powershell
blender --background --python origindataset\gated_blender_physical.py -- --input-root obj_dataset --output-root dataset
```

输出：`dataset\<class>\*_gate_0.png / *_gate_1.png / *_gate_2.png`

### 2.5 生成二维异常类别

```powershell
python make_image2d_class.py --overwrite
```

输出：`dataset\image2d\*_gate_0.png / *_gate_1.png / *_gate_2.png`

每个 `image2d` 样本只有一个 gate 从原有物体切片复制而来，其余 gate 是全黑图；`dataset\image2d\manifest.csv` 会记录来源。

### 3. 训练

```powershell
python train.py
```

可选参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-root` | `dataset/` | 切片数据集路径 |
| `--classes` | chair desk sofa bed toilet image2d | 类别列表 |
| `--epochs` | 30 | 训练轮数 |
| `--batch-size` | 8 | 批大小 |
| `--lr` | 1e-3 | 学习率 |
| `--val-ratio` | 0.2 | 验证集比例 |
| `--seed` | 42 | 随机种子 |

## 输出结果

训练结果保存在 `artifacts/` 目录下：

| 文件 | 说明 |
|------|------|
| `training_curves.png` | 训练损失与验证精度曲线 |
| `best_confusion_matrix.png` | 最佳验证结果对应的混淆矩阵 |
| `val_attention_weights.csv` | 验证集每个样本的预测/真实标签、注意力权重、类别概率 |
| `attention_per_sample.png` | 每个验证样本的注意力权重堆叠柱状图 |
| `attention_mean_by_gate.png` | 各gate的全局和按类别平均注意力权重 |
| `summary.json` | 训练结果摘要 |

### val_attention_weights.csv 字段说明

| 字段 | 含义 |
|------|------|
| `pred` / `gt` | 预测/真实类别标签（0-4） |
| `attn_gate_0` / `attn_gate_1` / `attn_gate_2` | 三个切片的注意力权重（softmax归一化，和为1） |
| `prob_class_0` ~ `prob_class_4` | 模型对5个类别的预测概率（和为1） |

## 运行分析

```powershell
# 注意力权重分析
python analyze_attention.py

# 切片稀疏度统计
python analyze_gate_sparsity.py --dataset-root dataset
```

## 运行测试

```powershell
python -m pytest tests/ -v
```

## 这版基线的意义

这版基线先回答一个更基础的问题：

**同一物体的多张选通切片是否已经具有足够的分类信息？**

如果这一步成立，后续可以进一步研究：

- 如何把这些切片映射到光学输入
- 如何在多模光纤中进行物理传播
- 如何利用 speckle 完成后续识别

换句话说，这个基线既是一个更简单的分类模型，也是后续光学神经网络设计的参照组。
