# 论文证据矩阵：军事 gate stack 与二维假目标判别

日期：2026-07-06

## 1. 当前论文主张与证据对应

| 主张 | 当前证据 | 证据强度 | 可写入论文方式 | 仍需补强 |
|---|---|---|---|---|
| Blender 距离选通仿真可以生成可训练的军事目标 gate stack | `Military_3D_Gated_Selected44`：44 个样本、132 张 gate PNG；readiness 通过 | 中 | 写成“构建并验证了一个小规模仿真数据集” | 补充更多模型、视角和真实系统参数 |
| 平面假目标在多 gate 中应表现为同一整目标轮廓的强度缩放 | 公式 `I_g^false(x,y)=A_g S(x,y)`；flat/hard false 的跨 gate corr 高；rectangular-overlap 版本 corr=0.9731 | 强 | 写入方法建模与物理解释 | 需要文献引用支撑激光选通响应模型；rectangular-overlap 的弱响应 gate 会降低前景 IoU |
| 真实三维目标在不同 gate 中出现结构变化 | true3d gate stack 诊断：hard projection 数据中 corr=0.3246，IoU=0.3065 | 强 | 写入结果与讨论 | 增加更多真实目标姿态可增强普适性 |
| hard projection 能削弱但不能彻底消除单帧外观捷径 | 早期 hard projection 单门结果较低：gate_0=0.5370，gate_1=0.6296，gate_2=0.5370；但本机 20 epoch 复现中 gate_0=0.8889，gate_1=0.6852，gate_2=0.8704 | 中 | 作为核心消融实验和局限讨论：hard projection 降低了单帧捷径难度，但最大投影仍保留边缘/形态残余线索 | 需要更强单帧控制，如局部直方图匹配、轮廓扰动、复杂背景 |
| 完整 gate stack 对二维假目标判别是关键 | hard projection full stack=1.0000；rectangular-overlap full stack=1.0000，均高于单门；本机 `localgpu_*_full_20ep` 对 hard projection 与 exposure-matched rectangular-overlap 均复现 mean=1.0000/std=0 | 强 | 作为主结果 | 需要避免写成真实世界已验证；5 epoch 短训不收敛，论文主结果必须使用充分训练轮数 |
| rectangular-overlap 假目标不是因过亮而容易分类 | 亮度审计：rectangular-overlap flat_false 三门前景均值 0.1031/0.1463/0.0891，低于 true3d 的 0.1689/0.2365/0.1525，饱和比例均为 0 | 强 | 写入讨论中排除亮度捷径 | gate_1 单门仍偏高，需继续做曝光匹配和复杂背景 |
| 曝光匹配后 full stack 仍保持优势 | class-gate mean matched 后 full stack=1.0000，gate_0=0.5000，gate_1=0.7222，gate_2=0.5000；本机 20 epoch 三种子复现 full stack=1.0000，首次达到 0.9/1.0 约在 epoch 11-12 | 强 | 作为新增控制实验，说明 full stack 优势不只是全局曝光差异 | gate_1 未完全随机，仍需控制局部面积/轮廓清晰度 |
| gate_1 残余单帧线索不是单纯均值/高分位亮度 | gate_1 foreground-mean matched=0.7222，p99 matched=0.7222；与 mean matched 相同 | 中 | 写入讨论：残余来自分布形状或局部结构线索 | 需要后续局部直方图匹配或更复杂背景继续验证 |
| 前景直方图匹配不足以消除单门残余线索 | 新增 hard projection histogram matched 控制集；flat_false corr=0.9166，IoU=0.9301；单门特征仍可分：gate_1 max_value 阈值准确率 0.9318，gate_0 foreground_ratio 0.8182，gate_1 edge_density 0.7614；网络单门 20 epoch 仍很高：gate_0=0.9444，gate_1=0.8889，gate_2=1.0000 | 强 | 写成诊断性负结果：亮度分布不是唯一捷径，面积、边缘和形态仍需控制 | 需要轮廓扰动、复杂背景、局部块级匹配或多视角扩充 |
| 像素级面积/极值后处理控制会产生新的单门伪迹 | hist+area+clipmax 控制集：单门特征阈值最高约 0.8182，但网络 full/gate0/gate1/gate2 20 epoch 全部 1.0000 | 强 | 写成方法局限：PNG 后处理式控制可能生成稳定伪迹，不能作为排除单帧捷径的正证据 | 下一步应在 Blender/物理渲染阶段构造平面假目标，并加入复杂背景/多视角 |
| 单门长训练仍能捕获残余单帧线索 | 本机 20 epoch exposure-matched rectangular-overlap：gate_0=0.9630，gate_1=0.9259，gate_2=0.8704；本机 20 epoch hard projection：gate_0=0.8889，gate_1=0.6852，gate_2=0.8704；full stack=1.0000 且最后 5 epoch 更稳定 | 中 | 写入讨论和局限：当前仿真假目标仍保留局部形态/边缘/投影差异 | 不应声称“单门完全不可用”；下一步需做更强单帧控制 |
| 小样本军事三分类中迁移学习更稳定 | transfer frozen/finetune mean=0.7500 std=0；scratch mean=0.7083 std=0.1443 | 中 | 写成“小样本条件下稳定性更好” | 数据量太小，不宜写成显著提升 |
| 方法对中等噪声/背景/Poisson 退化较稳 | per-gate norm noise+bg+Poisson mean=0.9815 | 中 | 写成仿真退化下仍保持较高准确率 | 需要更多噪声强度曲线 |
| gate 缺失会影响稳定性 | random gate dropout mean=0.9074，低于 clean full stack=1.0000 | 中 | 写成 gate 序列完整性重要 | 可补固定丢 gate_0/1/2 分析 |

## 2. 推荐论文创新点表述

### 创新点 1：物理约束的二维平面假目标 gate stack 建模

可写：

```text
不同于将二维假目标简化为单帧异常图像，本文根据距离选通响应将平面假目标建模为同一二维轮廓在多个 gate 中的强度缩放序列。
```

证据：

```text
per-gate norm flat_false: corr=0.9995, IoU=0.9880
hard projection flat_false: corr=0.9768, IoU=0.9301
rectangular-overlap flat_false: corr=0.9731, IoU=0.4299
```

注：rectangular-overlap 的 corr 仍高，说明轮廓结构保持一致；IoU 下降主要来自部分 gate 响应接近背景阈值，前景掩膜被压缩或消失。这是物理弱回波造成的阈值效应，不应解释为平面轮廓发生结构变化。

### 创新点 2：hard projection false target 与矩形脉冲-门重叠响应

可写：

```text
为削弱单帧外观捷径，本文进一步提出 hard projection 假目标：利用真实三维 gate stack 的最大投影作为平面诱饵轮廓，再按平面目标 gate response 生成多门序列。对于矩形激光脉冲和矩形接收门，本文进一步用二者在回波到达时间上的重叠长度生成 piecewise-linear gate response，使假目标强度变化更接近距离选通物理过程。
```

证据：

```text
hard projection full stack=1.0000
hard projection only gate_0=0.5370
hard projection only gate_1=0.6296
hard projection only gate_2=0.5370

rectangular-overlap full stack=1.0000
rectangular-overlap only gate_0=0.5000
rectangular-overlap only gate_1=0.8704
rectangular-overlap only gate_2=0.5000

rectangular-overlap exposure matched full stack=1.0000
rectangular-overlap exposure matched only gate_0=0.5000
rectangular-overlap exposure matched only gate_1=0.7222
rectangular-overlap exposure matched only gate_2=0.5000
rectangular-overlap foreground matched only gate_1=0.7222
rectangular-overlap p99 matched only gate_1=0.7222
```

### 创新点 3：gate stack 物理诊断指标

可写：

```text
本文使用跨 gate 相关性、前景掩膜 IoU 和归一化差分作为 gate stack 的物理诊断指标，以区分平面投影序列与真实三维深度响应序列。
```

证据：

```text
hard projection flat_false: corr=0.9768, IoU=0.9301, absdiff=0.0062
hard projection true3d: corr=0.3246, IoU=0.3065, absdiff=0.1210
rectangular-overlap flat_false: corr=0.9731, IoU=0.4299, absdiff=0.0172
rectangular-overlap true3d: corr=0.3246, IoU=0.3065, absdiff=0.1210
rectangular-overlap exposure matched flat_false: corr=0.9698, IoU=0.4274, absdiff=0.0210
rectangular-overlap exposure matched true3d: corr=0.3246, IoU=0.3065, absdiff=0.1210
```

### 创新点 4：军事小样本迁移验证

可写：

```text
在人工筛选的 44 个军事三维模型上，本文比较了从零训练与预训练迁移，初步验证了小样本军事目标识别中的稳定性收益。
```

证据：

```text
transfer frozen mean=0.7500, std=0
transfer finetune mean=0.7500, std=0
scratch mean=0.7083, std=0.1443
```

## 3. 不能过度声称的点

| 风险表述 | 问题 | 建议替代表述 |
|---|---|---|
| 本方法已经可用于真实战场目标识别 | 目前是 Blender 仿真和小样本 3D 模型 | 本文在仿真条件下验证了方法可行性 |
| 迁移学习显著提升准确率 | 当前迁移 mean=0.75，scratch mean=0.7083，但样本少 | 迁移学习降低了三随机种子下的波动 |
| 二维假目标完全等价于真实诱饵 | 当前是平面投影近似 | 本文构造了符合平面目标 gate response 的仿真假目标 |
| 网络学到了真实深度 | 网络输入是 gate 图像，不是显式深度图 | 网络利用了多 gate 响应模式 |
| 100% 准确率证明方法完全可靠 | 样本规模小，且为仿真数据；rectangular-overlap gate_1 单门仍有 0.8704，曝光匹配后仍有 0.7222 | hard projection、rectangular-overlap 与曝光匹配消融显示 gate stack 对当前仿真任务具有关键判别作用 |
| 5 epoch 结果为 0.5 说明方法无效 | 本机短训显示 true/false full stack 在 5 epoch 未收敛，但 20 epoch 在 epoch 11-12 后稳定达到 1.0 | 当前任务对训练轮数敏感，论文主结果采用充分训练后的多种子结果 |

## 4. 论文结果表建议

### 表 1：军事三分类迁移结果

| 方法 | mean best val acc | std | seeds |
|---|---:|---:|---|
| Transfer frozen encoder | 0.7500 | 0.0000 | 42/332/2026 |
| Transfer finetune | 0.7500 | 0.0000 | 42/332/2026 |
| Scratch | 0.7083 | 0.1443 | 42/332/2026 |

### 表 2：hard projection 消融

| 响应模型 | 输入 | mean best val acc | std |
|---|---|---:|---:|
| Gaussian/empirical response | Full 3-gate stack | 1.0000 | 0.0000 |
| Gaussian/empirical response | Gate 0 only | 0.5370 | 0.0321 |
| Gaussian/empirical response | Gate 1 only | 0.6296 | 0.0849 |
| Gaussian/empirical response | Gate 2 only | 0.5370 | 0.0642 |
| Rectangular pulse-gate overlap | Full 3-gate stack | 1.0000 | 0.0000 |
| Rectangular pulse-gate overlap | Gate 0 only | 0.5000 | 0.0000 |
| Rectangular pulse-gate overlap | Gate 1 only | 0.8704 | 0.0849 |
| Rectangular pulse-gate overlap | Gate 2 only | 0.5000 | 0.0000 |
| Rectangular overlap + exposure matched | Full 3-gate stack | 1.0000 | 0.0000 |
| Rectangular overlap + exposure matched | Gate 0 only, 10 epochs | 0.5000 | 0.0000 |
| Rectangular overlap + exposure matched | Gate 1 only, 10 epochs | 0.7222 | 0.1667 |
| Rectangular overlap + exposure matched | Gate 2 only, 10 epochs | 0.5000 | 0.0000 |
| Local GPU reproducibility: hard projection | Full 3-gate stack, 20 epochs | 1.0000 | 0.0000 |
| Local GPU residual single-frame cue: hard projection | Gate 0 only, 20 epochs | 0.8889 | 0.0000 |
| Local GPU residual single-frame cue: hard projection | Gate 1 only, 20 epochs | 0.6852 | 0.1156 |
| Local GPU residual single-frame cue: hard projection | Gate 2 only, 20 epochs | 0.8704 | 0.0321 |
| Local GPU diagnostic control: hard projection histogram matched | Full 3-gate stack, 20 epochs | 1.0000 | 0.0000 |
| Local GPU residual single-frame cue: hard projection histogram matched | Gate 0 only, 20 epochs | 0.9444 | 0.0000 |
| Local GPU residual single-frame cue: hard projection histogram matched | Gate 1 only, 20 epochs | 0.8889 | 0.0556 |
| Local GPU residual single-frame cue: hard projection histogram matched | Gate 2 only, 20 epochs | 1.0000 | 0.0000 |
| Local GPU diagnostic control: hist + area + clipmax | Full 3-gate stack, 20 epochs | 1.0000 | 0.0000 |
| Local GPU artifact check: hist + area + clipmax | Gate 0 only, 20 epochs | 1.0000 | 0.0000 |
| Local GPU artifact check: hist + area + clipmax | Gate 1 only, 20 epochs | 1.0000 | 0.0000 |
| Local GPU artifact check: hist + area + clipmax | Gate 2 only, 20 epochs | 1.0000 | 0.0000 |
| Local GPU reproducibility: rectangular overlap + exposure matched | Full 3-gate stack, 20 epochs | 1.0000 | 0.0000 |
| Local GPU residual single-frame cue: rectangular overlap + exposure matched | Gate 0 only, 20 epochs | 0.9630 | 0.0321 |
| Local GPU residual single-frame cue: rectangular overlap + exposure matched | Gate 1 only, 20 epochs | 0.9259 | 0.0849 |
| Local GPU residual single-frame cue: rectangular overlap + exposure matched | Gate 2 only, 20 epochs | 0.8704 | 0.0642 |

注：本机 5 epoch 对应 true/false full stack 结果仍为 0.5000，说明短训不能作为论文主结果；20 epoch 后两组 full stack 均在第 11-12 个 epoch 附近达到稳定高精度。另一方面，20 epoch 单门结果明显升高，说明当前仿真中仍有残余单帧线索；论文应把 full stack 写成“更稳定且物理可解释”，而不是写成“单帧完全不可用”。hard projection 的 20 epoch 单门复现尤其提示：最大投影能削弱外观捷径，但仍可能保留边缘、轮廓和局部形态线索。histogram matched 控制集的单门结果进一步说明，单纯匹配前景灰度直方图并不能消除这些线索，甚至可能强化 gate_2 的单门可分性。hist+area+clipmax 控制集则进一步提醒：在 PNG 上做面积裁剪和极值裁剪会产生新的稳定单帧伪迹，因此后续更应转向 Blender 平面目标物理渲染、复杂背景和多视角扩充。

### 表 3：gate stack 诊断

| 数据 | 类别 | corr | mask IoU | absdiff |
|---|---|---:|---:|---:|
| per-gate norm | flat_false | 0.9995 | 0.9880 | 0.0055 |
| per-gate norm | true3d | 0.3244 | 0.3174 | 0.1211 |
| hard projection | flat_false | 0.9768 | 0.9301 | 0.0062 |
| hard projection | true3d | 0.3246 | 0.3065 | 0.1210 |
| rectangular-overlap | flat_false | 0.9731 | 0.4299 | 0.0172 |
| rectangular-overlap | true3d | 0.3246 | 0.3065 | 0.1210 |
| rectangular-overlap exposure matched | flat_false | 0.9698 | 0.4274 | 0.0210 |
| rectangular-overlap exposure matched | true3d | 0.3246 | 0.3065 | 0.1210 |
| hard projection histogram matched | flat_false | 0.9166 | 0.9301 | 0.0493 |
| hard projection histogram matched | true3d | 0.3246 | 0.3065 | 0.1210 |

## 5. 推荐论文图

| 图号 | 文件 | 作用 |
|---|---|---|
| Fig. 1 | `true3d_vs_hard_projection_gate_stack.png` | 展示 true3d 与 hard projection gate stack 差异 |
| Fig. 2 | `military_3class_transfer_vs_scratch.png` | 小样本迁移学习验证 |
| Fig. 3 | `hard_projection_full_stack_vs_single_gate.png` | 核心消融：full stack vs single gate |
| Fig. 4 | `hard_rect_overlap_full_stack_vs_single_gate.png` | 更物理的矩形脉冲-门重叠响应消融 |
| Fig. 5 | `hard_rect_overlap_exposure_matched_full_stack_vs_single_gate.png` | 曝光匹配后 full stack 仍优于 single gate |
| Fig. 6 | `hard_rect_overlap_gate1_residual_controls.png` | gate_1 残余单帧线索控制实验 |
| Fig. 7 | `gate_stack_physical_diagnostics.png` | 物理诊断指标 |
| Fig. 8 | `per_gate_norm_robustness.png` | 鲁棒性验证 |

## 6. 论文短板与下一步验证

1. 样本规模仍小：需要继续扩充高质量军事模型。
2. Blender 参数需要更细：应补 gate width、pulse width、gate spacing、num gates 消融。
3. 假目标模型需要更多形态：hard projection 和 rectangular-overlap 是重要一步，但还可以加入姿态偏移、反射率扰动和复杂背景。
4. 单门残余线索需要更强控制：前景直方图匹配仍保留 foreground ratio、edge density 和 max value 等可分特征；像素级面积/极值后处理又会产生新的伪迹。后续应优先在 Blender 中构造平面板假目标、复杂背景和多视角数据，而不是继续依赖 PNG 后处理。
5. 需要文献支撑：激光选通响应、二维诱饵/平面目标建模、多 gate/depth cue 识别等位置需要正式引用。
6. 需要外部验证：目前没有真实激光选通数据，只能写成仿真可行性研究。
