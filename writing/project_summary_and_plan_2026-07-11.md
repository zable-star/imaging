# 项目汇总与投稿规划
日期：2026-07-11  
仓库：`E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline`  
远程：`origin` → `git@github.com:zable-star/imaging.git`（`master`）  
目标期刊：Optics & Laser Technology（OLT）

---

## 1. 一句话项目定位

本项目不是“新网络架构”论文，而是：

> **物理可解释的激光距离选通成像仿真 + 反捷径验证 + 域随机化鲁棒边界分析**  
> 任务：真三维目标 vs 平面假目标判别

主主张（已冻结）：
1. 三门控堆栈比单门更稳
2. 原始仿真存在亮度捷径，必须先诊断/控制
3. 结构化干扰下 clean/noisy 训练会塌，域混合训练能抬高鲁棒边界
4. 结论限于受控仿真，不宣称外场部署可用

---

## 2. 当前完成度总览

| 模块 | 状态 | 说明 |
|---|---|---|
| 主线实验 | 完成并冻结 | domainmix + strongaug + attention + full 3-gate |
| 关键结果表/图 | 完成 | Table 1–9，Fig.1–11 |
| 研究初稿 | 可交导师 | 章节完整、主张克制 |
| Blocking 清理 | 基本完成 | 图注去路径、声明默认模板、DOCX 重建 |
| 投稿终稿 | 未完成 | 缺作者信息、DOI 终核、可选公开代码链接 |
| 真实数据验证 | 未做 | 纯仿真证据链 |
| 代码/结果入库 | 已 push | 必要代码 + 汇总 CSV + 投稿包 |

### 当前判断
- **组会 / 导师审阅：可以现在交**
- **期刊 Submit：还差作者/合规信息与终核，不差主实验**

---

## 3. 关键文件地图

### 3.1 投稿核心（优先看这些）
| 文件 | 作用 |
|---|---|
| `writing/sci_manuscript_OLT_target_2026-07-10.md` | 主手稿 Markdown |
| `writing/sci_manuscript_OLT_target_2026-07-10.docx` | 带图 Word 稿 |
| `writing/figures/fig1_*.png` … `fig11_*.png` | 论文图 |
| `writing/OLT_cover_letter_2026-07-10.md` | 封面信草稿 |
| `writing/OLT_highlights_2026-07-10.txt` | Elsevier Highlights |
| `writing/OLT_presubmission_checklist_2026-07-10.md` | 投前检查清单 |
| `writing/OLT_submission_plan_2026-07-10.md` | OLT 投稿计划 |
| `writing/current_mainline_best_results_2026-07-10.md` | 主线结果冻结清单 |
| `writing/manuscript_result_tables_paste_final_2026-07-10.md` | 表格粘贴终稿素材 |
| `writing/project_summary_and_plan_2026-07-11.md` | 本汇总与规划文档 |

### 3.2 训练 / 评估代码
| 文件 | 作用 |
|---|---|
| `train.py` | 主训练入口（含 noise/domainmix 等） |
| `dataset.py` | 数据读取与样本组织 |
| `model.py` | 轻量 CNN + fusion baseline |
| `run_experiments.py` | 实验编排 |
| `run_military_transfer_experiments.py` | 军用/迁移相关实验入口 |
| `scripts/evaluate_*.py` | 评估与网格评测 |
| `scripts/make_v8_*.py` | 论文图生成 |
| `scripts/run_v8_*.ps1` | 主线实验批处理脚本 |
| `tests/*.py` | 数据构建与训练工具测试 |

### 3.3 数据构建与诊断
| 路径 | 作用 |
|---|---|
| `dataset_new/*.py` | 真/假目标数据集构建、归一化、捷径诊断、hard nuisance 构造 |
| `dataset_new/*_gate_stack_*.csv` | 样本/类别元数据 |
| `dataset_new/*_readiness.csv` / `*_quality.csv` | 数据集就绪与质量检查 |
| `dataset_new/*_separability.csv` | 单门标量可分性诊断 |

注意：大图数据集目录（如 `Military_TF_v8_mv4_norm/`、hard nuisance 图像目录）本地存在，通常不入库。

### 3.4 主线结果 CSV（写论文时优先引用）
| 文件 | 对应结论 |
|---|---|
| `experiments/v8_mv4_strategy_comparison_aggregate_2026-07-08.csv` | 训练策略对比（Table 8 / Fig.10） |
| `experiments/v8_mv4_domainmix_full_vs_single_gate_eval_aggregate_3seed.csv` | domainmix 全栈 vs 单门（Table 9 / Fig.11） |
| `experiments/v8_mv4_norm_mixaug_attention_eval_aggregate_3seed.csv` | clean 上限（高 clean，硬干扰差） |
| `experiments/v8_mv4_hard_nuisance_*_eval_aggregate_3seed.csv` | 硬干扰失败边界 |
| 其他 `experiments/*aggregate*.csv` | 消融与历史对照 |

### 3.5 过程/备份（一般不投、不发）
- `writing/*.bak_*`
- `writing/_*.py`、`writing/_*.txt`（临时脚本与日志）
- `_archive_unused_*`
- `handoff_*`、`presentation_outputs/`
- `experiments/*/` 单次 run 目录（`.gitignore` 已忽略）

---

## 4. 主线实验冻结表

### 4.1 主线配置
| 项 | 取值 |
|---|---|
| 数据家族 | Military TF v8，四视角 mv4，per-gate max-normalized |
| 任务 | binary：true3d vs flat_false |
| 输入 | 三门控灰度堆栈 |
| 网络 | shared CNN + attention fusion（轻量 baseline） |
| 训练 | domain mixture + strong-noise augmentation |
| seeds | 42 / 332 / 2026 |
| 验证 | grouped validation（同模型多视角同 split） |

### 4.2 主线数字（domainmix full stack）
| 条件 | 准确率 |
|---|---:|
| clean | 0.9394 |
| light noise | 0.9242 |
| strong noise | 0.9091 |
| mild nuisance | 0.9394 |
| strong nuisance | 0.7727 |

对照提醒：
- clean/noisy 训练可到 clean ≈ 0.9848，但 hard domain ≈ 0.50
- 因此论文主线强调 **鲁棒边界**，不强调最高 clean 精度

### 4.3 反捷径关键证据
- raw 第三门 P99 阈值准确率 ≈ 0.9886
- per-gate max-norm 后最强标量捷径降至 ≈ 0.7955
- 单门消融：domainmix 后全栈仍全面优于 Gate 0/1/2

---

## 5. 手稿结构与状态

`sci_manuscript_OLT_target_2026-07-10.md` 当前结构：
1. Highlights / Abstract / Keywords
2. Introduction
3. Related Work（约 17 篇，4 小节）
4. Physical Model and Simulation
5. Dataset and Anti-Shortcut Protocol
6. Neural Baseline and Training Setup
7. Results（6.1–6.7）
8. Discussion / Limitations / Conclusion
9. Declaration / Funding / Data Availability / References

### 已做 Blocking 清理
- [x] 去掉图注内部 `Source file: writing/figures/...`
- [x] 默认利益冲突声明
- [x] 默认无外部基金声明
- [x] 默认数据/代码 “upon reasonable request”
- [x] DOCX 重嵌 Fig.1–11

### 仍开放项
- [ ] 作者姓名 / 排序 / 单位 / 通讯作者
- [ ] 若有基金，替换默认 funding 句
- [ ] 若公开 GitHub，替换 data/code 句
- [ ] 参考文献 DOI/页码终核（尤其会议文 [12][13]）

---

## 6. Git 与发布边界

### 建议入库
- 代码、脚本、测试
- 汇总 CSV、论文图、手稿与投稿附件
- 数据元数据 CSV（samples/classes/readiness）

### 不建议入库
- 渲染图像大数据集
- 单次训练 run 目录与 checkpoint
- 备份文件、临时清理脚本日志

### 已知远程
- `origin`: GitHub `zable-star/imaging`
- `private`: `imaging-classification`
- `gitee`: 备用镜像

投稿前确认：公开仓库是否与“数据/代码可用性”声明一致。

---

## 7. 距离投稿的差距（按优先级）

### P0：不填就不能投
1. 作者信息页
2. 通讯作者邮箱
3. 基金信息确认（有/无）
4. 数据/代码公开策略确认
5. 文献 DOI 终核

### P1：强烈建议（提高命中率）
1. Abstract / Introduction / Discussion 英文终润
2. Related Work 对比小表（Gated2Depth / Gated2Gated / Liu OLT / 本文）
3. Methods 补：44 模型筛选标准
4. Methods 补：Blender 版本、分辨率、相机/门控关键参数
5. Limitations 再强调：无真实门控外场数据

### P2：审稿后大概率被要，可不阻塞首投
1. 更大俯仰/视角变化
2. 更真实材料/大气/探测器噪声
3. 少量真实或半真实门控样例
4. 与传统 RIP/相关法的定量对比

---

## 8. 两周执行计划（建议）

### Day 0–1：合规填空
- 收齐作者、单位、通讯邮箱、基金
- 定数据公开方式（request / GitHub）
- 更新 MD + DOCX 声明区

### Day 1–2：终核与格式
- CrossRef/出版社页核对 References
- 全文扫描内部标签（v8/mv4/mixaug/gate0 等）
- 确认表题图注编号连续

### Day 2–4：科学包装增强
- 写对比小表
- 补模型筛选 + 渲染设置段落
- 英文润色摘要/引言/讨论

### Day 4–5：导师审
- 只先发：摘要 + 结论 + 封面信 + Highlights
- 收集“主叙事是否同意”的反馈

### Day 5–7：打包提交
- 最终 Word/PDF
- Highlights 单独文件
- Cover letter
- 图源文件（按 OLT 要求）
- 建议审稿人 3–6 名（可选）

---

## 9. 投稿叙事（对外统一口径）

### 可以说
- 我们做了门控成像物理仿真与反捷径验证框架
- 证明完整三门控堆栈在受控仿真中比单门更稳
- 证明不诊断捷径、不纳入结构化干扰，会得到误导性高精度

### 不要说
- 提出了全新 SOTA 网络
- 已在真实战场/外场验证
- 可直接部署军事识别系统

### 封面信主句
> This manuscript presents a physics-interpretable gated-imaging simulation and anti-shortcut validation framework for true 3D versus planar false-target discrimination, not a new neural architecture.

---

## 10. 推荐下一步（按顺序）

1. **你提供**：作者信息 + 基金 + 代码是否公开
2. **我执行**：填入手稿声明与作者区，重建 DOCX
3. **你/导师**：确认主叙事
4. **共同**：DOI 终核 + 英文终润
5. **提交**：OLT online submission

### 成功标准（可投稿）
- [ ] 作者信息齐全
- [ ] 三份声明与事实一致
- [ ] 图注无内部路径
- [ ] 主线数字与 CSV 一致
- [ ] 封面信/Highlights/Word 齐套
- [ ] 导师确认“可投”

### 成功标准（更稳）
- [ ] 对比小表 + 渲染细节 + 英文终润完成
- [ ] 至少一轮外部阅读无重大叙事漏洞

---

## 11. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 审稿人批“纯仿真” | major/reject 风险 | 强化 limitation + 仿真验证价值；后续补真实样例 |
| 审稿人以为在吹网络 | 贡献被误解 | 全文固定“framework / baseline”表述 |
| 第三方模型版权 | 数据公开受阻 | 保持 “assets not redistributed” |
| clean 精度高于 robust 主线 | 叙事混乱 | 明确：主线看鲁棒，不看最高 clean |

---

## 12. 快速入口（每天开工先看）

1. `writing/project_summary_and_plan_2026-07-11.md`（本文）
2. `writing/current_mainline_best_results_2026-07-10.md`
3. `writing/sci_manuscript_OLT_target_2026-07-10.md`
4. `writing/OLT_presubmission_checklist_2026-07-10.md`
5. `experiments/v8_mv4_domainmix_full_vs_single_gate_eval_aggregate_3seed.csv`

---

## 13. 当前结论

**项目已从“能跑实验”推进到“可内部评审的完整研究初稿 + 投稿准备包”。**  
主科学证据链已闭环；距离 OLT 正式投稿，主要剩：

> **作者/基金/公开策略 + 文献终核 +（建议）一版英文与对比包装。**

不建议现在回头大改网络或重跑主线，除非导师明确要求换主张。