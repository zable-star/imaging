# 精选军事 44 模型汇报大纲：激光选通与二维假目标判别

## 第 1 页：研究问题

标题：

```text
基于激光距离选通序列的军事三维目标识别与二维假目标判别
```

要讲：

```text
普通图像分类只看二维投影，容易把真实三维目标和平面诱饵混淆。
激光距离选通能够获得同一目标在不同深度门下的响应序列，
因此不仅看“长什么样”，还看“随距离门如何变化”。
```

建议图：

```text
gate stack 示意图或真实/假目标 gate_0, gate_1, gate_2 样例
```

## 第 2 页：数据治理与仿真流程

要讲：

```text
原始军事 3D 数据标签噪声较大，因此没有直接使用全量数据，
而是通过 thumbnail_review.csv 人工筛选 keep=1 的模型。
```

关键数字：

| 类别 | 数量 |
|---|---:|
| Main Battle Tank | 12 |
| Fighter Jet | 20 |
| Attack Helicopter | 12 |
| 合计 | 44 |

流程：

```text
人工筛选 44 个模型
-> Blender 距离选通渲染 true3d gate stack
-> 构造 flat false / hard projection 假目标
-> readiness 和 quality audit
-> 网络训练与消融
```

## 第 3 页：网络结构

要讲：

```text
每个样本输入三张 gate 图像。
每张图经过共享 CNN 编码器得到 gate-level feature，
再通过 attention_residual 融合，最后用分类头输出类别。
```

注意表述：

```text
这里的 attention 是 gate-level discriminative contribution，
不是 Transformer 的 QKV attention，也不是空间显著性图。
```

## 第 4 页：军事三分类小样本迁移

建议图：

```text
artifacts\figures\military_selected44_2026-07-06\military_3class_transfer_vs_scratch.png
```

要讲：

```text
军事样本只有 44 个，直接从零训练波动较大。
迁移学习平均精度不一定显著更高，但三随机种子下更稳定。
这说明可控仿真预训练可作为军事小样本识别的稳定起点。
```

关键结果：

| 方法 | mean best val acc | std |
|---|---:|---:|
| transfer frozen | 0.7500 | 0.0000 |
| transfer finetune | 0.7500 | 0.0000 |
| scratch | 0.7083 | 0.1443 |

## 第 5 页：为什么需要激光选通

建议图：

```text
artifacts\figures\military_selected44_2026-07-06\hard_projection_full_stack_vs_single_gate.png
```

要讲：

```text
为了避免模型只靠单张图外观，我们构造 hard projection 假目标：
把真实 3D gate stack 的最大投影作为二维平面假目标轮廓。
这样单张图和真实目标更接近。
```

核心结论：

| 输入 | mean best val acc |
|---|---:|
| full 3-gate stack | 1.0000 |
| only gate_0 | 0.5370 |
| only gate_1 | 0.6296 |
| only gate_2 | 0.5370 |

讲稿句：

```text
可以看到，单独看任何一个 gate 时，准确率接近随机；
但输入完整三门序列后，模型稳定达到 100%。
这说明真正有效的信息不是单帧二维外观，而是跨 gate 的深度响应变化。
```

## 第 6 页：物理诊断证据

建议图：

```text
artifacts\figures\military_selected44_2026-07-06\gate_stack_physical_diagnostics.png
```

要讲：

```text
平面假目标的不同 gate 本质上是同一个二维轮廓的强度缩放，
所以跨 gate 相关性和前景 IoU 很高。
真实三维目标在不同深度门中响应不同部件，
所以跨 gate 相关性和 IoU 明显降低。
```

关键数字：

| 数据 | flat_false corr / IoU | true3d corr / IoU |
|---|---:|---:|
| per-gate norm | 0.9995 / 0.9880 | 0.3244 / 0.3174 |
| hard projection | 0.9768 / 0.9301 | 0.3246 / 0.3065 |

## 第 7 页：鲁棒性与局限

建议图：

```text
artifacts\figures\military_selected44_2026-07-06\per_gate_norm_robustness.png
```

要讲：

```text
噪声、背景散射和 Poisson 光子噪声下仍保持较高准确率；
随机丢失一个 gate 后准确率下降，说明 gate stack 完整性对稳定判别很重要。
```

关键结果：

| 设置 | mean best val acc |
|---|---:|
| clean full stack | 1.0000 |
| random gate dropout | 0.9074 |
| noise + background + Poisson | 0.9815 |

## 第 8 页：当前结论与下一步

当前结论：

```text
1. Blender 距离选通仿真可以构造可训练的军事 gate stack 数据。
2. 小样本军事三分类中，迁移学习比从零训练更稳定。
3. hard projection 结果证明，二维假目标判别的关键不是单帧外观，而是跨 gate 响应模式。
4. gate stack 物理诊断为这一点提供了可解释指标。
```

下一步：

```text
1. 增加更多高质量军事模型，扩大每类样本数。
2. 做 num gates = 1 / 3 / 5 和 gate width / pulse width 消融。
3. 引入更真实的噪声、背景、材质反射率和目标姿态变化。
4. 将电子网络基线迁移到多模光神经网络高速识别框架。
```

一句话收束：

```text
这项工作不是简单把三张图输入 CNN，
而是利用激光距离选通提供的深度响应序列，
把真实三维目标和平面二维诱饵在物理响应上的差异转化为可学习、可解释的判别特征。
```
