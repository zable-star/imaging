# OLT 投前完成度检查清单（2026-07-10）

目标期刊：Optics & Laser Technology  
主手稿：`writing/sci_manuscript_OLT_target_2026-07-10.md`  
Word 稿：`writing/sci_manuscript_OLT_target_2026-07-10.docx`  
封面信：`writing/OLT_cover_letter_2026-07-10.md`

## 0. 一句话状态

- **研究初稿**：已完成（可交导师/组会）
- **投稿终稿**：未完成（还差作者信息、声明、文献终核、语言终润）
- **主线主张已冻结**：物理可解释门控仿真 + 反捷径验证 + 域随机化鲁棒边界
- **不要改主线数字**，除非重新跑完整 3-seed 正式实验

---

## 1. 已完成（可打勾）

### 科学内容
- [x] 完整章节：Abstract → Introduction → Related Work → Method → Dataset/Protocol → Experiments → Discussion → Limitations → Conclusion
- [x] 主线结果齐全：Table 1–9、Fig.1–11
- [x] 主张边界克制：仿真验证框架，不宣称部署可用，不主打网络创新
- [x] 反捷径证据：raw P99 shortcut、per-gate max-norm、single-gate ablation
- [x] 鲁棒性证据：hard-nuisance failure boundary + domain-mix improvement
- [x] 相关工作扩写 + 参考文献 17 条
- [x] Highlights 已写
- [x] DOCX 已插图

### 主线结果（冻结）
- 主训练：domain mix + strong noise + attention + full 3-gate stack
- 主结果：clean 0.9394 / light 0.9242 / strong 0.9091 / hard mild 0.9394 / hard strong 0.7727
- 对照：clean mixaug 可到 0.9848，但 hard domain 掉到约 0.50

---

## 2. 投稿前必须完成（Blocking）

这些不做，不建议点 Submit。

### A. 作者与合规信息
- [ ] 作者姓名、排序、单位中英文
- [ ] 通讯作者邮箱与地址
- [ ] Declaration of Competing Interest 最终句
- [ ] Funding 最终句（有基金写编号；无基金用标准无基金句）
- [ ] Acknowledgements（如有）
- [ ] Data/Code Availability 最终句（见下方推荐模板）

### B. 文献与格式
- [ ] 新参考文献 DOI / 会议页码 CrossRef 或出版社页终核
- [ ] 文内引用编号与 References 一一对应（当前 1–17 已全覆盖）
- [ ] 图注去掉内部路径备注（如 `Source file: writing/figures/...`）或改为补充材料说明
- [ ] 表题统一为 OLT/Elsevier 风格（Table 1. ...）
- [ ] 检查正文是否还残留内部标签：`v8`、`mv4`、`mixaug`、裸 `p99`、`gate0` 等

### C. 文件包
- [ ] 最终 Word 或 LaTeX（按系统要求）
- [ ] Highlights 单独 3–5 条（已有，可单独复制）
- [ ] 高分辨率图（建议 TIFF/PDF 或高清 PNG，按期刊要求）
- [ ] 封面信
- [ ] 建议审稿人 3–6 名（可选但推荐）
- [ ] 作者贡献声明（若期刊/学校要求 CRediT）

---

## 3. 强烈建议完成（会明显提高命中率）

- [ ] 英文全文再润色 1 遍（尤其 Abstract / Introduction / Discussion）
- [ ] 增加 1 个 Related Work / 方法对比小表（Gated2Depth / Gated2Gated / Liu OLT / 本文）
- [ ] Methods 补 1 段：44 个军用模型筛选标准
- [ ] Methods 补：Blender 版本、渲染分辨率、相机/光照关键设置
- [ ] Limitations 再明确一句：无真实门控外场数据，结论限于仿真域
- [ ] 若可能：补 elevation / 更大视角变化中的一小段验证
- [ ] 若可能：1 个真实或半真实门控样例做定性对比（即使不做完整训练）

---

## 4. 可不阻塞首投、但审稿后大概率被要

- 更多模型数量 / 类别多样性
- 更真实 BRDF、大气、探测器噪声、时间抖动
- 真实平面假目标（折叠、支撑结构、非均匀反射）
- 与传统门控测距/强度相关方法的定量对比
- 代码与数据公开仓库

---

## 5. 推荐可直接粘贴的声明模板

### 5.1 Competing Interest
```text
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.
```

### 5.2 Funding（无外部基金时）
```text
This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.
```

### 5.3 Data and Code Availability（稳健版，推荐）
```text
The simulation scripts, training code, and aggregated experimental results supporting this study are available from the corresponding author upon reasonable request. The selected third-party 3D model assets are subject to their original licenses and are not redistributed with this paper.
```

### 5.4 Data and Code Availability（若可公开仓库）
```text
The simulation and training code, together with aggregated result tables, will be made available at [GitHub URL] upon publication. Third-party 3D model assets are not redistributed due to license restrictions; model identifiers and preprocessing steps are described in the manuscript.
```

---

## 6. 投稿时主叙事（封面信与摘要一致）

**主贡献：**
1. 物理可解释的 Blender 门控成像仿真（真 3D vs 平面假目标）
2. 反捷径验证协议（标量捷径、归一化、分组验证、单门消融）
3. 结构化干扰失败边界 + 域随机化改善

**不要主打：**
- 新 attention 架构
- 军事部署可用
- 真实外场已验证

**一句话：**
> This is a physics-interpretable gated-imaging simulation and anti-shortcut validation study, not a new network architecture paper.

---

## 7. 建议时间表

| 阶段 | 内容 | 建议用时 |
|---|---|---|
| Day 0–1 | 作者信息/基金/声明填完；图注去内部路径 | 0.5–1 天 |
| Day 1–2 | 文献 DOI 终核 + 术语扫描 | 0.5 天 |
| Day 2–4 | 英文润色 + 对比小表 + 模型筛选段 | 1–2 天 |
| Day 4–5 | 导师审一版 | 视反馈 |
| Day 5–7 | 改完打包投 OLT | 1 天 |

---

## 8. 是否现在可投？

| 场景 | 建议 |
|---|---|
| 交导师看 | **可以，现在就交** |
| 直接投 OLT | **先完成第 2 节 Blocking 项** |
| 想提高一档命中率 | 再完成第 3 节中的润色 + 对比表 + 渲染细节 |

---

## 9. 当前文件索引

- 主稿 MD：`writing/sci_manuscript_OLT_target_2026-07-10.md`
- 主稿 DOCX：`writing/sci_manuscript_OLT_target_2026-07-10.docx`
- 主线结果：`writing/current_mainline_best_results_2026-07-10.md`
- 投稿计划：`writing/OLT_submission_plan_2026-07-10.md`
- 封面信：`writing/OLT_cover_letter_2026-07-10.md`
- 本检查清单：`writing/OLT_presubmission_checklist_2026-07-10.md`