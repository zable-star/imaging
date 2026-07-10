# 本机 GPU 论文训练结果汇总

本文件由 `scripts/collect_paper_experiment_report.py` 自动汇总 `localgpu_*` 实验生成。
主要用途是把多种子训练结果快速落到论文证据表中。

注意：本机 GPU 结果用于日常推进、链路验证和趋势判断；短 epoch 结果不能替代 20-80 epoch 的论文主结果。

## 三类军事目标识别

| experiment | runs | mean acc | std | min | max | seeds |
|---|---:|---:|---:|---:|---:|---|
| localgpu_military3class_scratch_5ep | 3 | 0.2917 | 0.0722 | 0.2500 | 0.3750 | 42 332 2026 |

## 链路测试

| experiment | runs | mean acc | std | min | max | seeds |
|---|---:|---:|---:|---:|---:|---|
| localgpu_smoke_truefalse_rect_matched_2ep | 3 | 0.5000 | 0.0000 | 0.5000 | 0.5000 | 42 332 2026 |

## 真假目标主实验

| experiment | runs | mean acc | std | min | max | seeds |
|---|---:|---:|---:|---:|---:|---|
| localgpu_truefalse_hard_projection_full_20ep | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 42 332 2026 |
| localgpu_truefalse_hard_projection_full_5ep | 3 | 0.5000 | 0.0000 | 0.5000 | 0.5000 | 42 332 2026 |
| localgpu_truefalse_hard_projection_hist_area_clipmax_full_20ep | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 42 332 2026 |
| localgpu_truefalse_hard_projection_hist_full_20ep | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 42 332 2026 |
| localgpu_truefalse_rect_matched_full_20ep | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 42 332 2026 |
| localgpu_truefalse_rect_matched_full_5ep | 3 | 0.5000 | 0.0000 | 0.5000 | 0.5000 | 42 332 2026 |

## 单门/残余线索消融

| experiment | runs | mean acc | std | min | max | seeds |
|---|---:|---:|---:|---:|---:|---|
| localgpu_truefalse_hard_projection_hist_area_clipmax_single_gate0_20ep | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 42 332 2026 |
| localgpu_truefalse_hard_projection_hist_area_clipmax_single_gate1_20ep | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 42 332 2026 |
| localgpu_truefalse_hard_projection_hist_area_clipmax_single_gate2_20ep | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 42 332 2026 |
| localgpu_truefalse_hard_projection_hist_single_gate0_20ep | 3 | 0.9444 | 0.0000 | 0.9444 | 0.9444 | 42 332 2026 |
| localgpu_truefalse_hard_projection_hist_single_gate1_20ep | 3 | 0.8889 | 0.0556 | 0.8333 | 0.9444 | 42 332 2026 |
| localgpu_truefalse_hard_projection_hist_single_gate2_20ep | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 42 332 2026 |
| localgpu_truefalse_hard_projection_single_gate0_20ep | 3 | 0.8889 | 0.0000 | 0.8889 | 0.8889 | 42 332 2026 |
| localgpu_truefalse_hard_projection_single_gate1_20ep | 3 | 0.6852 | 0.1156 | 0.5556 | 0.7778 | 42 332 2026 |
| localgpu_truefalse_hard_projection_single_gate2_20ep | 3 | 0.8704 | 0.0321 | 0.8333 | 0.8889 | 42 332 2026 |
| localgpu_truefalse_rect_matched_single_gate0_20ep | 3 | 0.9630 | 0.0321 | 0.9444 | 1.0000 | 42 332 2026 |
| localgpu_truefalse_rect_matched_single_gate1_20ep | 3 | 0.9259 | 0.0849 | 0.8333 | 1.0000 | 42 332 2026 |
| localgpu_truefalse_rect_matched_single_gate2_20ep | 3 | 0.8704 | 0.0642 | 0.8333 | 0.9444 | 42 332 2026 |

## 写论文时的使用口径

- 三类军事目标识别用于说明小样本军事目标迁移是否比从零训练更稳定。
- 真假目标主实验和单门消融用于回答激光选通是否提供了单帧图像之外的判别信息。
- 如果单门长训练结果较高，应解释为当前仿真仍存在残余单帧形态线索；不要写成单门完全不可用。
- 曝光匹配和鲁棒性实验用于排除亮度捷径，并说明方法在仿真退化下仍可工作。
- 所有结果仍属于 Blender 仿真和 44 个精选军事模型条件下的验证，论文中不要写成真实外场系统已经验证。
