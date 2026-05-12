# 切片注意力分类基线

这个文件夹保存的是一个简化版目标分类基线，用来验证：

**同一物体的多张单色切片，是否能够支持分类，以及网络更依赖哪些切片。**

这版模型暂时去掉了多模光纤传播模块，先专注于“切片本身的信息是否有效”。

## 核心思路

- 一个样本 = 同一物体的全部切片
- 每张切片都是单通道灰度图
- 使用共享 CNN 分别提取每张切片的特征
- 使用注意力模块学习每张切片的重要性
- 对切片特征加权融合后进行分类

## 输入形式

以一个物体包含 3 张切片为例：

- `gate_0`
- `gate_1`
- `gate_2`

模型输入张量形状为：

`[num_slices, 1, H, W]`

当前这版默认设置为：

- `num_slices = 3`
- `H = W = 224`

也就是说，一个样本不再被视为普通的 3 通道图像，而是被视为：

**“同一物体的 3 张顺序切片”**

## 文件说明

- [dataset.py](/E:/wjz/test1/dataset/dataset_obj/slice_attention_baseline/dataset.py)  
  负责按“一个物体对应多张切片”的方式读取数据，并自动按 `gate_0 / gate_1 / gate_2` 排序。

- [model.py](/E:/wjz/test1/dataset/dataset_obj/slice_attention_baseline/model.py)  
  定义模型结构，包括：
  - 切片编码器 `SliceEncoder`
  - 注意力分类器 `SliceAttentionClassifier`

- [train.py](/E:/wjz/test1/dataset/dataset_obj/slice_attention_baseline/train.py)  
  训练入口，负责：
  - 构建数据集
  - 划分训练集/验证集
  - 训练模型
  - 保存结果文件

## 当前使用的数据集路径

- `E:\wjz\test1\dataset\dataset_obj\chair_gated_physical`
- `E:\wjz\test1\dataset\dataset_obj\desk_gated_physical`

这两个目录中的图片文件名格式为：

- `chair_0001_gate_0.png`
- `chair_0001_gate_1.png`
- `chair_0001_gate_2.png`

程序会自动把同一物体的不同 gate 图像组合成一个样本。

## 输出结果

训练结果保存在：

- `E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\artifacts`

主要输出文件包括：

- `training_curves.png`：训练损失与验证精度曲线
- `best_confusion_matrix.png`：最佳验证结果对应的混淆矩阵
- `val_attention_weights.csv`：验证集中每个样本的注意力权重
- `summary.json`：训练结果摘要

其中 `val_attention_weights.csv` 最值得关注，因为它可以帮助分析：

- 网络更关注哪一张切片
- 哪些 gate 对分类贡献更大

## 运行方式

```powershell
python E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\train.py
```

## Five-class ModelNet10 pipeline

This workspace now includes a five-class data preparation flow based on
`origindataset\ModelNet10\ModelNet10`.

Selected classes:

- `chair`
- `desk`
- `sofa`
- `bed`
- `toilet`

Default size is moderate: `80` train models and `20` test models per class,
for `500` OBJ files total.

1. Convert OFF files to OBJ:

```powershell
python E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\convert_off_to_obj_dataset.py
```

Output:

```text
E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\obj_dataset\<class>\<split>\*.obj
```

2. Render physically gated slices in Blender:

```powershell
blender --background --python E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\origindataset\gated_blender_physical.py -- --input-root E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\obj_dataset --output-root E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset
```

Output:

```text
E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset\<class>\*_gate_0.png
E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset\<class>\*_gate_1.png
E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset\<class>\*_gate_2.png
```

3. Train the five-class classifier:

```powershell
python E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\train.py
```

## 这版基线的意义

这版基线的作用是先回答一个更基础的问题：

**不考虑多模光纤传播时，同一物体的多张选通切片是否已经具有足够的分类信息？**

如果这一步成立，那么后续就可以进一步研究：

- 如何把这些切片映射到光学输入
- 如何在多模光纤中进行物理传播
- 如何利用 speckle 完成后续识别

换句话说，这个基线既是一个更简单的分类模型，也是后续光学神经网络设计的参照组。
