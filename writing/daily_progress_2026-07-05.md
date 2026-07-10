# 2026-07-05 今日推进记录：训练、推导与军事迁移路线

## 1. 今日目标

今天围绕当前主线推进：

```text
完成网络训练能力建设，最终服务军事三维目标识别与二维假目标判别。
```

具体推进点：

1. 给训练脚本增加小样本迁移学习能力。
2. 给批量实验脚本增加预训练权重与冻结策略参数。
3. 新增军事小样本迁移实验包装脚本。
4. 整理现有实验结果。
5. 补充物理成像、二维假目标、网络融合、迁移学习的推导说明。

## 2. 当前已有实验结果

### 2.1 六分类假目标判别

实验目录：

```text
experiments/six_class_attention_residual_seedmatched
```

类别：

```text
chair / desk / sofa / bed / toilet / image2d
```

融合方式：

```text
attention_residual
```

三随机种子结果：

| Seed | Best validation accuracy |
|---:|---:|
| 42 | 0.9583 |
| 332 | 0.9333 |
| 2026 | 0.9500 |

汇总：

| 指标 | 数值 |
|---|---:|
| Mean accuracy | 0.9472 |
| Std | 0.0127 |
| Min | 0.9333 |
| Max | 0.9583 |

当前意义：

```text
在可控 ModelNet10 + image2d 设置下，attention_residual 已经能够较稳定地区分真实三维 gate stack 和二维异常/假目标 gate stack。
```

### 2.2 五分类 gate spacing 物理参数消融

实验目录：

```text
experiments/phys_gate_spacing_large_attention_residual_seed*
```

类别：

```text
chair / desk / sofa / bed / toilet
```

设置：

```text
large gate spacing
attention_residual
```

三随机种子结果：

| Seed | Best validation accuracy |
|---:|---:|
| 42 | 0.9700 |
| 332 | 0.9300 |
| 2026 | 0.9400 |

汇总：

| 指标 | 数值 |
|---|---:|
| Mean accuracy | 0.9467 |
| Std | 0.0208 |
| Min | 0.9300 |
| Max | 0.9700 |

当前意义：

```text
五分类物理消融结果说明：在不引入 image2d 的情况下，多 gate 距离选通图像本身可以支持较高准确率的目标识别。
```

## 3. 距离选通成像的基本推导

### 3.1 单个 gate 的接收信号

对激光距离选通成像，可以把第 `g` 个 gate 的图像写成：

```text
I_g(x, y) = ∫ ρ(x, y, z) · T(z) · H_g(z) dz + B_g(x, y) + N_g(x, y)
```

其中：

| 符号 | 含义 |
|---|---|
| `I_g(x, y)` | 第 `g` 个选通门得到的灰度图 |
| `ρ(x, y, z)` | 目标在空间位置上的反射率/散射强度 |
| `T(z)` | 距离传播衰减和介质透过率 |
| `H_g(z)` | 第 `g` 个 gate 对深度 `z` 的响应函数 |
| `B_g(x, y)` | 背景散射、杂散光或系统偏置 |
| `N_g(x, y)` | 探测噪声、光子噪声、读出噪声等 |

`H_g(z)` 来自激光脉冲与接收门的时间重叠。若目标深度为 `z`，回波到达时间近似为：

```text
t_z = 2z / c
```

第 `g` 个接收门中心时间为 `τ_g`，对应中心距离：

```text
z_g = cτ_g / 2
```

当 `z` 接近 `z_g` 时，回波与接收门重叠大，`H_g(z)` 高；当 `z` 远离 `z_g` 时，`H_g(z)` 低甚至为 0。

所以多 gate 图像不是普通多通道图像，而是：

```text
同一目标在多个深度响应窗口下的观测序列。
```

### 3.2 三维真实目标的 gate stack

真实三维目标有深度厚度。不同部件位于不同 `z`，因此：

```text
I_g(x, y) = ∫ ρ(x, y, z) · T(z) · H_g(z) dz
```

会随 `g` 改变局部结构强度。

直观理解：

```text
gate_0 可能更强调目标近端结构，
gate_1 可能响应主体中部，
gate_2 可能响应远端或后部结构。
```

因此三维目标的判别信息来自两类特征：

1. 单张 gate 内的二维轮廓、局部纹理和投影结构。
2. 多张 gate 之间的深度响应变化模式。

这也是当前网络需要输入 `[gate_0, gate_1, gate_2]` 序列，而不是只输入一张图的原因。

### 3.3 二维平面假目标的 gate stack

对于平面二维假目标，可以近似认为目标所有有效反射点处在同一深度 `z_0`。若其二维轮廓为 `S(x, y)`，则：

```text
ρ_false(x, y, z) = S(x, y) · δ(z - z_0)
```

代入 gate 成像模型：

```text
I_g^false(x, y)
= ∫ S(x, y) · δ(z - z_0) · T(z) · H_g(z) dz
= S(x, y) · T(z_0) · H_g(z_0)
```

令：

```text
A_g = T(z_0) · H_g(z_0)
```

则：

```text
I_g^false(x, y) = A_g · S(x, y)
```

这说明平面二维假目标的关键规律是：

```text
不同 gate 中看到的是同一个完整轮廓 S(x, y)，但整体强度 A_g 随 gate 响应变化。
```

因此，二维假目标不应该被模拟成“只有一小块目标出现”。更合理的仿真是：

```text
gate_0: 整个目标轮廓，强度 A_0
gate_1: 整个目标轮廓，强度 A_1
gate_2: 整个目标轮廓，强度 A_2
```

其中某些 `A_g` 可以很小甚至接近 0。

真实三维目标和二维平面假目标的差别不是“有没有图”，而是：

```text
真实三维目标：不同 gate 中局部结构会随深度分层变化。
二维平面假目标：不同 gate 中主要是同一轮廓做整体强度缩放。
```

这就是二维假目标判别的物理基础。

## 4. 网络融合模型推导

### 4.1 共享编码器

每个 gate 图像先经过同一个 CNN 编码器：

```text
f_i = E(I_i),    i = 0, 1, ..., S-1
```

其中：

| 符号 | 含义 |
|---|---|
| `I_i` | 第 `i` 张 gate 图像 |
| `E(·)` | 共享 SliceEncoder |
| `f_i` | 第 `i` 张 gate 的特征向量 |
| `S` | gate 数量 |

共享编码器的好处：

```text
不同 gate 使用同一套图像特征提取规则，减少参数量，也避免每个 gate 学出互不一致的低层特征。
```

### 4.2 mean 融合

最简单的融合为平均：

```text
f_mean = (1/S) · Σ_i f_i
```

优点：

```text
稳定、参数少、不易过拟合。
```

缺点：

```text
无法区分某些 gate 是否更有效，也无法表达 gate 质量差异。
```

### 4.3 attention 融合

当前 attention 是 gate-level MLP attention：

```text
s_i = MLP_att(f_i)
α_i = exp(s_i) / Σ_j exp(s_j)
f_att = Σ_i α_i f_i
```

其中：

```text
Σ_i α_i = 1
```

解释：

```text
α_i 表示第 i 个 gate 对最终分类的判别贡献。
```

注意：

```text
这里的 attention 不是 Transformer，也不是 Q/K/V 自注意力。
```

它更适合当前项目，因为当前每个样本只有 3 到 5 个 gate，token 数很少，用轻量 MLP attention 更不容易过拟合，也更容易解释。

### 4.4 concat 融合

concat 直接保留全部 gate 特征：

```text
f_concat = [f_0, f_1, ..., f_{S-1}]
```

优点：

```text
信息保留最多，经常能得到较高准确率。
```

缺点：

```text
参数量随 gate 数增加，且没有天然 gate 权重解释。
```

因此论文里不应把 concat 写成理论上限，只能写成：

```text
已测试融合方式中的高信息量经验基线。
```

### 4.5 attention_residual 融合

当前主模型：

```text
f_att = Σ_i α_i f_i
f_res = MLP_res([f_0, f_1, ..., f_{S-1}])
f = f_att + f_res
```

解释：

```text
attention 分支提供 gate-level 判别贡献；
residual 分支补偿 attention 加权和可能丢失的特征组合信息。
```

这正好对应当前项目的平衡点：

1. 比 mean 更能表达 gate 差异。
2. 比纯 attention 更能保留多 gate 组合信息。
3. 比 concat 更容易解释 gate-level contribution。

## 5. 分类损失与 softmax 说明

网络最后输出 logits：

```text
o = C(f)
```

其中 `o_k` 是第 `k` 类的未归一化分数。

概率由 softmax 得到：

```text
p_k = exp(o_k) / Σ_j exp(o_j)
```

训练时使用交叉熵：

```text
L_cls = -log p_y
```

在 PyTorch 中，`CrossEntropyLoss` 直接接收 logits，不需要在模型里手动加 softmax。也就是说：

```text
模型最后一层输出 logits；
训练损失内部计算 log-softmax；
评估和保存概率时再显式 softmax。
```

这回答了之前的问题：

```text
最后全连接层后面训练时不是显式 softmax，但分类概率解释时是 softmax。
```

## 6. 小样本军事迁移学习推导

### 6.1 为什么需要迁移

军事 3D 数据的问题：

1. 样本少。
2. 标签噪声大。
3. 很多模型不是对应类别。
4. 有些模型渲染后为空或目标太小。

如果直接从零训练，模型容易学到偶然特征：

```text
θ* = argmin_θ L_military(θ)
```

当军事样本很少时，`L_military` 对真实分布估计不稳定，泛化风险较高。

### 6.2 预训练 + 微调

先在可控数据上训练：

```text
θ_source = argmin_θ L_source(θ)
```

其中 source 可以是 ModelNet10 gate stack 或更大规模可控仿真数据。

再将 encoder 权重迁移到军事数据：

```text
θ_E ← θ_E^source
θ_head 随机初始化
```

目标变为：

```text
θ_target = argmin_θ L_military(θ; θ_E 初始化为 θ_E^source)
```

### 6.3 三种训练策略

| 策略 | 数学形式 | 适用情况 |
|---|---|---|
| 从零训练 | 全部参数随机初始化 | 军事样本较多时作为基线 |
| 冻结 encoder | 固定 `θ_E`，只训练融合层和分类头 | 军事样本很少时更稳 |
| 半冻结微调 | 所有参数可训练，但学习率更低 | 样本数量中等，想适应军事外观 |

冻结 encoder 时：

```text
θ_E = constant
θ_fusion, θ_head = argmin L_military
```

半冻结微调时：

```text
θ_all = argmin L_military
```

但通常使用更低学习率，例如：

```text
lr = 1e-4 或 5e-5
```

### 6.4 今天代码中已加入的迁移能力

`train.py` 新增：

| 参数 | 作用 |
|---|---|
| `--pretrained-model-path` | 加载已有模型权重 |
| `--pretrained-include-classifier` | 是否加载分类头；军事迁移通常不加载 |
| `--freeze-encoder` | 冻结共享 CNN 编码器 |
| `--freeze-attention` | 冻结 attention 打分器 |
| `--freeze-residual` | 冻结 residual projection |

`summary.json` 会记录：

- 预训练路径
- 加载了多少权重
- 跳过了多少权重
- 是否冻结 encoder / attention / residual
- 总参数量
- 可训练参数量

这对论文和汇报有用，因为可以明确说明：

```text
迁移学习不是口头方案，而是已经进入训练代码和实验记录。
```

## 7. 军事目标实验矩阵

建议第一轮只做 3 类：

```text
tank / aircraft / military_vehicle
```

原因：

```text
样本质量优先于类别数量。类别过多但标签不准，会让训练结果没有解释价值。
```

第一轮实验矩阵：

| 实验 | 初始化 | 冻结 | 目的 |
|---|---|---|---|
| military_scratch | 随机 | 不冻结 | 从零训练基线 |
| military_transfer_frozen_encoder | ModelNet gate 预训练 | 冻结 encoder | 检验通用 gate 特征迁移 |
| military_transfer_finetune | ModelNet gate 预训练 | 不冻结，低学习率 | 检验军事域适应能力 |

建议指标：

- mean accuracy
- per-class accuracy
- confusion matrix
- attention by class/gate
- 每类样本数
- 标签筛查通过率

如果迁移优于从零训练，可以形成论文中的一个实用结论：

```text
在军事三维模型样本有限的条件下，可控距离选通仿真预训练能够提升目标识别稳定性。
```

## 8. 多模光神经网络融合推导

当前电子基线为：

```text
I_0, I_1, ..., I_{S-1}
    -> CNN encoder
    -> gate fusion
    -> classifier
```

后续多模光神经网络可以把中间特征提取替换为光学传播：

```text
I_stack
    -> optical encoding
    -> multimode fiber / scattering propagation
    -> camera readout speckle
    -> electronic lightweight classifier
```

如果把光学传播记作 `Φ_opt`，则：

```text
u = Enc_opt(I_stack)
s = |Φ_opt(u)|^2
o = C_elec(s)
```

其中：

| 符号 | 含义 |
|---|---|
| `Enc_opt` | gate stack 到光场输入的编码 |
| `Φ_opt` | 多模光纤或散射介质传播 |
| `s` | 相机采集到的 speckle 强度 |
| `C_elec` | 电子读出分类器 |

当前项目的价值在于：

```text
先证明 gate stack 本身具有目标识别和假目标判别信息，
再讨论这些信息能否通过光学前端高速编码和读取。
```

因此它不是和多模光神经网络割裂的项目，而是后者的输入建模、数据生成和电子基线。

## 9. 今天本机已运行的检查

### 9.1 语法检查

```powershell
python -m py_compile train.py run_experiments.py run_military_transfer_experiments.py
```

结果：

```text
通过。
```

### 9.2 军事迁移 dry-run

运行：

```powershell
python run_military_transfer_experiments.py --classes tank aircraft military_vehicle -- --experiment-name military_transfer_smoke --dataset-root dataset_new\Military_3D_Dataset --pretrained-model-path experiments\six_class_attention_residual_seedmatched\six_class_attention_residual_seed42\slice_attention_model.pth --freeze-encoder --seeds 42 --epochs 2 --batch-size 8 --dry-run
```

结果：

```text
命令成功展开到 train.py，并正确传入 --pretrained-model-path 与 --freeze-encoder。
```

### 9.3 通用实验 dry-run

运行：

```powershell
python run_experiments.py --experiment-name transfer_command_check --pretrained-model-path experiments\six_class_attention_residual_seedmatched\six_class_attention_residual_seed42\slice_attention_model.pth --freeze-encoder --freeze-attention --seeds 42 --epochs 1 --dry-run
```

结果：

```text
命令成功展开到 train.py，并正确传入迁移学习相关参数。
```

### 9.4 pytorch1 迁移 smoke：六分类数据

运行环境：

```text
E:\ana\envs\pytorch1\python.exe
```

运行：

```powershell
E:\ana\envs\pytorch1\python.exe run_experiments.py --experiment-name pytorch1_transfer_smoke --experiment-root experiments\pytorch1_transfer_smoke --dataset-root dataset --fusion-mode attention_residual --pretrained-model-path experiments\six_class_attention_residual_seedmatched\six_class_attention_residual_seed42\slice_attention_model.pth --freeze-encoder --seeds 42 --epochs 1 --batch-size 16 --results-csv experiments\pytorch1_transfer_smoke\results.csv --aggregate-csv experiments\pytorch1_transfer_smoke\aggregate_results.csv
```

结果：

| 指标 | 数值 |
|---|---:|
| Loaded pretrained keys | 28 |
| Skipped classifier keys | 6 |
| Total parameters | 176,679 |
| Trainable parameters | 67,271 |
| Epochs | 1 |
| Validation accuracy | 0.9583 |

解释：

```text
迁移接口已在真实 PyTorch + CUDA 环境中跑通。默认跳过分类头，只加载 encoder、attention 和 residual projection；冻结 encoder 后仍能训练融合层和分类头。
```

### 9.5 pytorch1 迁移 smoke：二分类子集

运行：

```powershell
E:\ana\envs\pytorch1\python.exe train.py --dataset-root dataset --classes chair desk --artifact-dir experiments\pytorch1_transfer_smoke_2class --model-path experiments\pytorch1_transfer_smoke_2class\slice_attention_model.pth --fusion-mode attention_residual --pretrained-model-path experiments\six_class_attention_residual_seedmatched\six_class_attention_residual_seed42\slice_attention_model.pth --freeze-encoder --epochs 1 --batch-size 16 --seed 42
```

结果：

| 指标 | 数值 |
|---|---:|
| Samples | 200 |
| Loaded pretrained keys | 28 |
| Skipped classifier keys | 6 |
| Total parameters | 176,419 |
| Trainable parameters | 67,011 |
| Epochs | 1 |
| Validation accuracy | 0.9500 |

解释：

```text
不同类别数下，预训练特征加载和分类头重建逻辑正常。这正是军事小样本迁移需要的能力。
```

### 9.6 数据集训练就绪检查

新增脚本：

```text
dataset_new/check_gate_dataset_ready.py
```

军事原始目录检查：

```powershell
E:\ana\envs\pytorch1\python.exe dataset_new\check_gate_dataset_ready.py --root dataset_new\Military_3D_Dataset --expected-num-slices 3 --csv-out dataset_new\military_gate_readiness_2026-07-05.csv
```

结果：

| Class | Raw models | Valid gated samples | Ready |
|---|---:|---:|---|
| 01_Main_Battle_Tank | 100 | 0 | False |
| 02_Fighter_Jet | 100 | 0 | False |
| 03_Attack_Helicopter | 100 | 0 | False |
| 04_Armored_Vehicle | 100 | 0 | False |
| 05_Military_Truck_SAM | 100 | 0 | False |

可控六分类目录检查：

| Class | Valid gated samples | Ready |
|---|---:|---|
| chair | 100 | True |
| desk | 100 | True |
| sofa | 100 | True |
| bed | 100 | True |
| toilet | 100 | True |
| image2d | 100 | True |

large gate spacing 五分类目录检查：

| Class | Valid gated samples | Ready |
|---|---:|---|
| chair | 100 | True |
| desk | 100 | True |
| sofa | 100 | True |
| bed | 100 | True |
| toilet | 100 | True |

关键结论：

```text
军事数据目前不是网络不能训练，而是还没有渲染成训练脚本需要的 gate stack。下一步必须先将筛查后的 .glb 军事模型渲染为 *_gate_0/1/2.png，再启动军事迁移训练。
```

### 9.7 人工筛选 44 个军事模型的安全入口

用户明确说明：

```text
军事目录不要直接用，很多都是废的；已人工挑选 44 个。
```

因此后续不能使用：

```text
dataset_new\Military_3D_Dataset
```

作为训练输入。这个目录只是候选池。

已新增脚本：

```text
dataset_new/build_selected_subset.py
scripts/render_selected_military_gates.ps1
```

作用：

```text
从人工 review CSV 中复制 keep=1 的模型到独立目录，并用 --expected-count 44 防止误把 500 个候选全复制。
```

本机验证命令：

```powershell
E:\ana\envs\pytorch1\python.exe dataset_new\build_selected_subset.py --review-csv dataset_new\Military_3D_Dataset\manual_review.csv --output-root dataset_new\Military_3D_Selected44 --expected-count 44 --dry-run
```

当前结果：

```text
RuntimeError: Expected 44 selected rows, found 500.
```

解释：

```text
当前 manual_review.csv 中 keep=1 仍然是 500 行，不是用户筛选出的 44 行。因此 44 个有效模型还没有写入这个 CSV，或者存在另一个尚未定位到的清单/目录。
```

下一步需要：

1. 将 `manual_review.csv` 中废模型的 `keep` 改为 `0`，只保留 44 行 `keep=1`；或
2. 提供一个只包含 44 个模型路径的 selected CSV；或
3. 把 44 个 `.glb` 放入单独目录。

只有完成这一步，才进入：

```text
Military_3D_Selected44 -> Blender gate rendering -> Military_3D_Gated_Selected44 -> transfer training
```

渲染 dry-run 已验证：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\render_selected_military_gates.ps1 -DryRun
```

输出会打印：

```text
InputRoot: dataset_new\Military_3D_Selected44
OutputRoot: dataset_new\Military_3D_Gated_Selected44
target-mode: physical-3d
camera-view: top
auto-gate-fit: visible-bounds
render-device: cpu
```

真正渲染前需要设置：

```powershell
$env:BLENDER_LAUNCHER="你的blender-launcher.exe路径"
```

或：

```powershell
$env:BLENDER_EXE="你的blender.exe路径"
```

## 10. 下一步最值得跑的本机/实验室任务

### 10.1 如果本机有 PyTorch 环境

本机已经确认可用：

```text
E:\ana\envs\pytorch1\python.exe
torch 2.4.1
CUDA available
```

后续军事数据渲染完成后，先跑一个小 smoke：

```powershell
E:\ana\envs\pytorch1\python.exe run_military_transfer_experiments.py --classes 01_Main_Battle_Tank 02_Fighter_Jet 04_Armored_Vehicle -- --experiment-name military_transfer_smoke --dataset-root dataset_new\Military_3D_Gated --fusion-mode attention_residual --pretrained-model-path experiments\six_class_attention_residual_seedmatched\six_class_attention_residual_seed42\slice_attention_model.pth --freeze-encoder --seeds 42 --epochs 2 --batch-size 4
```

目的不是追求准确率，而是验证：

1. 军事数据目录是否符合训练读取规则。
2. 类别名是否匹配。
3. 预训练权重是否能加载。
4. 冻结参数是否生效。
5. 输出文件是否完整。

### 10.2 军事数据下一步

当前 `dataset_new\Military_3D_Dataset` 是原始 `.glb` 候选池，不可直接训练。建议先生成 44 个精选模型目录：

```text
dataset_new\Military_3D_Selected44
```

然后再生成新的训练目录：

```text
dataset_new\Military_3D_Gated_Selected44
```

目标结构：

```text
dataset_new\Military_3D_Gated_Selected44\01_Main_Battle_Tank\xxx_gate_0.png
dataset_new\Military_3D_Gated_Selected44\01_Main_Battle_Tank\xxx_gate_1.png
dataset_new\Military_3D_Gated_Selected44\01_Main_Battle_Tank\xxx_gate_2.png
...
```

渲染后先运行：

```powershell
E:\ana\envs\pytorch1\python.exe dataset_new\check_gate_dataset_ready.py --root dataset_new\Military_3D_Gated_Selected44 --expected-num-slices 3
```

只有当每类 `valid_samples > 0` 后，才进入迁移训练。

## 11. 当前判断

今天推进后，项目叙事可以更明确地写成：

```text
本项目不是单纯训练一个 CNN 分类器，而是围绕距离选通成像构建多 gate 物理仿真数据，分析真实三维目标与二维平面假目标在 gate stack 上的响应差异，并通过 attention_residual 融合网络完成目标识别、假目标判别和 gate-level 判别贡献分析。进一步地，项目已加入小样本迁移学习接口，为军事三维目标识别和后续多模光神经网络高速识别系统提供电子基线与数据入口。
```

## 12. 夜间继续推进结果：精选军事 44 模型、真假目标数据与训练

### 12.1 人工筛选清单已经接入

本轮不再直接使用 `dataset_new\Military_3D_Dataset` 候选池，而是使用用户人工筛选后的：

```text
dataset_new\Military_3D_Dataset\_review_sheets\thumbnail_review.csv
```

统计结果：

| 类别 | keep=1 数量 |
|---|---:|
| 01_Main_Battle_Tank | 12 |
| 02_Fighter_Jet | 20 |
| 03_Attack_Helicopter | 12 |
| 合计 | 44 |

已经生成精选模型目录：

```text
dataset_new\Military_3D_Selected44
```

并写出清单：

```text
dataset_new\Military_3D_Selected44\selected_manifest.csv
dataset_new\Military_3D_Selected44\selected_manifest.json
```

### 12.2 真三维与平面假目标 gate stack 已生成

真实三维目标输出：

```text
dataset_new\Military_3D_Gated_Selected44
```

平面二维假目标输出：

```text
dataset_new\Military_3D_FlatEcho_Selected44_gain10
```

其中平面假目标使用：

```text
--target-mode flat-echo
--flat-target-gate-index 0
--flat-min-response 0.18
--flat-echo-gain 10
```

这样假目标不再是“只有局部出现”，而是同一整目标轮廓在不同 gate 中以不同强度出现。

训练就绪检查：

| 数据集 | 类别/样本 | gate PNG | valid samples | 结论 |
|---|---:|---:|---:|---|
| Military_3D_Gated_Selected44 | 12 / 20 / 12 | 132 | 44 | ready |
| Military_3D_FlatEcho_Selected44_gain10 | 12 / 20 / 12 | 132 | 44 | ready |
| Military_TrueFalse_Selected44_gain10 | true3d 44 / flat_false 44 | 264 | 88 | ready |

质量审计结果：

| 数据集 | low contrast gate 图像 |
|---|---:|
| true3d | 5 / 132 |
| flat_false gain10 | 3 / 132 |

主要可疑样本集中在：

```text
01_Main_Battle_Tank_009_6e50bf75
```

建议后续人工复核该模型是否需要剔除或重渲染。

### 12.3 军事三分类：迁移学习与从零训练

任务：

```text
01_Main_Battle_Tank / 02_Fighter_Jet / 03_Attack_Helicopter
```

数据：

```text
dataset_new\Military_3D_Gated_Selected44
```

三随机种子 20 epoch 结果：

| 实验 | seeds | mean best val acc | std | 解释 |
|---|---|---:|---:|---|
| transfer frozen encoder | 42 / 332 / 2026 | 0.7500 | 0.0000 | 稳定，但上限受小样本限制 |
| transfer finetune | 42 / 332 / 2026 | 0.7500 | 0.0000 | 与冻结 encoder 相同 |
| scratch | 42 / 332 / 2026 | 0.7083 | 0.1443 | 最高可到 0.875，但波动明显 |

当前更严谨的结论不是“迁移一定更高”，而是：

```text
在 44 个军事样本的小数据条件下，预训练迁移使结果更稳定；从零训练可能偶然更高，但随机种子敏感性更强。
```

### 12.4 军事 true3d / flat_false 二分类

二分类数据集：

```text
dataset_new\Military_TrueFalse_Selected44_gain10
```

训练时加入：

```text
--split-group-by-sample-id
```

目的：同一源模型生成的 true3d 和 flat_false 必须同时进入 train 或 val，避免同源模型跨集合造成评估泄漏。

结果：

| 实验 | seeds | mean best val acc | std |
|---|---|---:|---:|
| transfer frozen encoder | 42 / 332 / 2026 | 1.0000 | 0.0000 |
| scratch | 42 / 332 / 2026 | 1.0000 | 0.0000 |

解释：

```text
当前 flat-echo 假目标和真实 3D gate stack 的差异已经足够明显，网络即使从零训练也能稳定区分。
```

这说明“真假目标判别链路”已经跑通，但也暴露出下一步问题：

```text
当前假目标难度偏低，后续应加入强度归一化、随机反射率、噪声、背景散射、不同 flat depth 和更弱的 gate 差异。
```

### 12.5 true/false 单 gate 快速消融

设置：

```text
seed = 42
epochs = 10
split_group_by_sample_id = true
```

结果：

| 输入 | best val acc |
|---|---:|
| only gate_0 | 1.0000 |
| only gate_1 | 0.8333 |
| only gate_2 | 0.7778 |
| full gate stack | 1.0000 |

解释：

```text
gate_0 单独已经很强，说明当前 flat-echo 数据中最强 gate 存在明显亮度/形态线索；gate_1 和 gate_2 单独下降，说明多 gate 仍有互补性，但下一步必须避免模型只依赖 gate_0 的强度差。
```

### 12.6 本轮新增或更新的关键工具

| 文件 | 作用 |
|---|---|
| `dataset_new/build_true_false_dataset.py` | 合并 true3d 与 flat_false 二分类数据 |
| `dataset_new/audit_gate_image_quality.py` | 审计 gate 图像亮度和低对比样本 |
| `scripts/summarize_gate_experiment.py` | 汇总 per-class accuracy 和 gate attention |
| `scripts/run_truefalse_single_gate_ablation.ps1` | 一键跑 true/false 单 gate 快速消融 |
| `scripts/render_selected_military_gates.ps1` | 新增 flat-echo 参数传递，包括 `FlatEchoGain` |

### 12.7 下一步最值得做

1. 对 true/false 数据加入强度归一化或随机反射率，削弱 gate_0 亮度捷径。
2. 做 flat target depth 的随机化，让假目标不总是对齐 gate_0。
3. 对 true/false 二分类加入噪声、背景散射、gate attenuation/dropout 鲁棒性测试。
4. 对军事三分类尝试合并低质量样本剔除版本，比较是否提升稳定性。
5. 把三分类“小样本迁移更稳定”和二分类“当前假目标可分但偏容易”写成 PPT 的两条实验结论。
