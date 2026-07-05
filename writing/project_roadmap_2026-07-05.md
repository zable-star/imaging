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

1. 固定当前 README 和路线图。
2. 跑六分类 `attention_residual` 正式训练。
3. 检查 `image2d` 的假目标生成是否是整目标轮廓强度衰减。
4. 整理 gate spacing 已有结果为一张论文表和一张图。
5. 开始 num gates = 1 / 3 / 5 数据生成与训练。
6. 对军事模型做人工筛查，只保留高质量小集合。
7. 设计迁移学习脚本：加载预训练 encoder，替换分类头。
8. 写一页“项目如何衔接多模光神经网络”的说明图。

## 7. PPT 中可以讲的能力点

- 我不是只做分类，而是把目标识别放在距离选通成像链路中建模。
- 我能用 Blender 构建可控物理仿真数据，并分析 gate 参数影响。
- 我能设计二维假目标，使其符合平面目标在选通系统中的响应特点。
- 我能实现多 gate 神经网络并做融合消融。
- 我能用 attention 权重解释 gate-level 判别贡献。
- 我能处理军事 3D 数据标签噪声和小样本问题。
- 我能把电子网络基线扩展到未来光神经网络高速识别系统。

