# SCI 下一步验证计划：激光选通三维目标与二维假目标判别

## 当前论文主线

当前最稳妥的论文主张是：

```text
在受控 Blender 激光选通仿真中，完整三 gate 序列比单 gate 图像提供更稳定的三维目标/平面假目标判别证据。
但这个结论必须建立在反捷径诊断、per-gate 归一化、同源模型分组验证和结构化扰动域随机化之上。
```

不要把论文写成“提出一个新网络”。当前网络只是轻量电子基线，用来验证 gate stack 是否含有有效判别信息。

## 已经可以支撑的证据

| 证据链 | 当前结果 | 论文作用 |
|---|---:|---|
| 原始 v8 捷径诊断 | gate2 p99 阈值分类 0.9886 | 证明不能直接相信高准确率 |
| per-gate 最大值归一化 | 最强 scalar shortcut 降到 0.7955 | 削弱亮度捷径 |
| 单视角 full stack vs single gate | full 在 clean/light/strong 下均值最高 | 初步证明多 gate 有价值 |
| 四视角分组验证 | full: 0.9848 / 0.9811 / 0.9205 | 降低固定视角泄漏 |
| 融合方式对比 | attention 在 strong noise 最稳 | 网络不是核心创新 |
| hard-nuisance 边界 | normal mixaug 在 hard v2/v3 均为 0.5000 | 暴露结构化扰动失效 |
| domain mix + strong noise | clean 0.9394, light 0.9242, strong 0.9091, hard mild 0.9394, hard strong 0.7727 | 当前最强创新升级结果 |

核心图：

```text
Fig. 7  四视角 full stack vs single gate
Fig. 9  hard-nuisance failure boundary
Fig. 10 domain-randomized training strategy comparison
```

## 还缺的论文级验证

### 1. domain-mix 后的 single-gate 消融

当前 Fig. 10 只比较训练策略，没有证明 domain-randomized training 下 full stack 仍优于单 gate。

需要补：

```text
domain mix + strong noise full stack
domain mix + strong noise gate0
domain mix + strong noise gate1
domain mix + strong noise gate2
```

建议先在本机跑 smoke：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_v8_domainmix_single_gate_ablation.ps1 -Seeds 42 -Epochs 8
```

已完成本机 smoke。注意：这是 seed42、8 epoch 的预实验，不能作为正式论文结果。

| condition | full stack seed42 20ep | gate0 smoke | gate1 smoke | gate2 smoke |
|---|---:|---:|---:|---:|
| clean | 0.9545 | 0.8636 | 0.5455 | 0.8182 |
| light noise | 0.9545 | 0.8636 | 0.5455 | 0.7727 |
| strong noise | 0.9091 | 0.8068 | 0.5000 | 0.7727 |
| hard mild | 1.0000 | 0.8182 | 0.5000 | 0.6818 |
| hard strong | 0.7727 | 0.6364 | 0.5000 | 0.5455 |

结论：

```text
smoke 支持继续做正式 domain-mix full stack vs single gate 消融。
但由于 single-gate 只跑了 8 epoch，不能直接写进论文主结果。
```

如果 smoke 有价值，再去 3090 跑完整三种子：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_v8_domainmix_single_gate_ablation.ps1 -Seeds 42,332,2026 -Epochs 20
```

### 2. held-out nuisance 泛化

当前 domain-mix 训练使用 normal + hard_v3_mild，测试包含 hard_v2，因此已经有一个更强扰动测试。但还不够系统。

建议补至少两类 held-out：

```text
hard_v4_reflectance_only
hard_v5_background_occlusion_only
```

目的不是追求所有都高，而是判断模型到底依赖哪类结构化因素。

### 3. 更多视角或 elevation

当前四视角是 yaw = 0/90/180/270。论文可以先用，但更强版本应补：

```text
yaw = 0/45/90/135/180/225/270/315
或者增加一个 elevation 角度
```

这项渲染成本较高，优先级低于 domain-mix single-gate 消融。

### 4. 文献和引用

投稿前必须补齐：

```text
激光选通/GRICI 物理模型
gated camera 或 range-gated imaging 感知应用
depth prior / transformer 3DRGI
shortcut learning
domain randomization / simulation-to-real
军事伪装、诱饵、假目标相关文献
```

当前文稿中的引用是工作占位引用，投稿前要从 Zotero 或 publisher 页面核对格式。

## 当前最推荐的推进顺序

1. 本机跑 `domain-mix single-gate smoke`，确认消融方向是否明显。
2. 如果 smoke 显示 full stack 仍有优势，把完整 single-gate 三种子矩阵放到 3090 跑。
3. 把 Fig. 10 扩展成 Fig. 10A/B：A 为策略对比，B 为 domain-mix full vs single gate。
4. 完成英文 SCI 正文二轮整理：删掉过程性描述，保留 Methods/Results/Discussion。
5. 最后整理 Zotero 引用和投稿格式。

## 当前能写的结论

可以写：

```text
Explicit domain-randomized simulation improves the robustness boundary under structured nuisance shifts.
```

可以写：

```text
The complete gate stack provides more stable evidence than individual gates in the controlled and grouped validation protocol.
```

不能写：

```text
The method is robust to real battlefield decoys.
```

不能写：

```text
The attention network is the main innovation.
```

## 当前关键文件

```text
writing/sci_manuscript_v8_gated_false_target_draft_2026-07-07.md
writing/sci_claims_evidence_matrix_2026-07-07.md
writing/v8_mv4_domain_randomization_strategy_report_2026-07-08.md
writing/v8_mv4_domainmix_single_gate_smoke_report_2026-07-08.md
writing/figures/fig10_domain_randomization_strategy_comparison.png
scripts/run_v8_domainmix_single_gate_ablation.ps1
scripts/run_v8_domainmix_strongaug_eval.ps1
experiments/v8_mv4_strategy_comparison_aggregate_2026-07-08.csv
experiments/v8_mv4_domainmix_single_gate_smoke_eval_aggregate_seed42_8ep.csv
```
