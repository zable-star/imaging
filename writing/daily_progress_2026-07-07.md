# Daily progress: 2026-07-07

## 1. Main decision

The project has moved the 2D false-target construction back into Blender instead of relying on PNG post-processing.

Reason: previous PNG-level controls such as histogram matching, foreground-area clipping, and clipmax created stable image artifacts. Those controls were useful as negative diagnostics, but they should not be treated as positive evidence for a paper.

## 2. Blender-side simulation changes

Updated file:

```text
origindataset/gated_blender_physical.py
```

Key changes:

1. Added `--flat-geometry-mode`.
   - `material-only`: old behavior, keeps the 3D mesh but uses flat-depth brightness.
   - `flatten-camera-depth`: projects all mesh vertices onto a single camera-space depth plane.

2. Added `--flat-target-gate-index-mode`.
   - `fixed`: use one selected gate.
   - `round-robin`: distribute flat false targets across gate 0/1/2.
   - `hash`: deterministic hash-based gate placement.

3. Fixed GLB normalization.
   - The old origin/scale workflow could preserve parent/world transforms, producing impossible camera depths around 180.
   - The new workflow bakes imported world vertices, removes parent transform, centers the mesh, and scales it into the target-size volume.

4. Metadata now records:
   - `flat_geometry_mode`
   - `flat_target_gate_index_mode`
   - resolved `flat_target_gate_index`
   - `flat_geometry_depth_min`
   - `flat_geometry_depth_max`

This gives direct evidence that the flat false target is actually planar in camera depth.

## 3. Rendered datasets

True 3D dataset:

```text
dataset_new/Military_3D_Gated_Selected44_blender_norm_v2
```

Flat false-target datasets:

```text
dataset_new/Military_FlatFalse_Selected44_blender_flat_v2
dataset_new/Military_FlatFalse_Selected44_blender_flat_rr_v3
dataset_new/Military_FlatFalse_Selected44_blender_flat_rr_gain2_min035_v4
```

Current best controlled binary dataset:

```text
dataset_new/Military_TrueFalse_Selected44_blender_flat_rr_gain2_min035_v4
```

Composition:

| class | samples | gate images | status |
|---|---:|---:|---|
| true3d | 44 | 132 | ready |
| flat_false | 44 | 132 | ready |

Flat false metadata check:

| item | value |
|---|---:|
| metadata files | 44 |
| gate index distribution | `{0: 15, 1: 15, 2: 14}` |
| max flattened depth span | 0.0 |
| metadata mode errors | 0 |

## 4. Gate-stack physical diagnostics

Diagnostics file:

```text
dataset_new/military_true_false_selected44_blender_flat_rr_gain2_min035_v4_gate_stack_classes.csv
```

| class | samples | corr | mask IoU | absdiff |
|---|---:|---:|---:|---:|
| flat_false | 44 | 0.9993 | 0.9809 | 0.0063 |
| true3d | 44 | 0.2992 | 0.2864 | 0.1327 |

Interpretation:

The Blender-side false target now behaves like a planar echo: across gates the normalized shape is almost identical. The true 3D target behaves like depth slicing: different gates see different parts of the object.

This is strong physical-diagnostic evidence. It is more useful for the paper than the earlier PNG post-processing controls.

## 5. Single-gate shortcut diagnostics

Diagnostics file:

```text
dataset_new/military_true_false_selected44_blender_flat_rr_gain2_min035_v4_single_gate_feature_separability.csv
```

Strongest scalar shortcut per gate:

| gate | strongest feature | threshold acc | true3d mean | flat false mean |
|---:|---|---:|---:|---:|
| 0 | max_value | 0.9432 | 0.2813 | 0.1214 |
| 1 | max_value | 1.0000 | 0.2914 | 0.1189 |
| 2 | max_value | 0.9545 | 0.2334 | 0.1127 |

Interpretation:

Even after Blender-side geometry flattening and round-robin gate placement, single-gate brightness cues remain strong. This is a current limitation, not a positive result.

Paper wording should be:

```text
The full gate stack provides a physically interpretable representation of 3D slicing versus planar echo consistency. However, the current simulation still leaves residual single-frame intensity cues, so stronger background, reflectance, exposure, and detector controls are required before claiming robust gate-stack-only discrimination.
```

## 6. Network ablation, three local seeds

Training artifacts:

```text
experiments/localgpu_blender_flat_rr_gain2_min035_v4_full_20ep_seed*
experiments/localgpu_blender_flat_rr_gain2_min035_v4_gate0_20ep_seed*
experiments/localgpu_blender_flat_rr_gain2_min035_v4_gate1_20ep_seed*
experiments/localgpu_blender_flat_rr_gain2_min035_v4_gate2_20ep_seed*
```

Summary:

| input | runs | mean best val acc | std | min | max | seeds |
|---|---:|---:|---:|---:|---:|---|
| Full 3-gate stack | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 42/332/2026 |
| Gate 0 only | 3 | 0.8939 | 0.0263 | 0.8636 | 0.9091 | 42/332/2026 |
| Gate 1 only | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 42/332/2026 |
| Gate 2 only | 3 | 0.9697 | 0.0263 | 0.9545 | 1.0000 | 42/332/2026 |

Interpretation:

Full stack works consistently, but single-gate results are also high. Gate 1 reaches 1.0000 across all three seeds. The result currently supports two points:

1. The new Blender false-target construction is usable by the network.
2. The simulation is not yet strict enough to prove that the network relies only on multi-gate physical consistency.

## 7. Generated evidence report

Report files:

```text
writing/blender_flat_rr_gain2_min035_v4_report_2026-07-07.md
experiments/localgpu_blender_flat_rr_gain2_min035_v4_ablation_summary.csv
experiments/localgpu_blender_flat_rr_gain2_min035_v4_ablation_aggregate.csv
```

Report script:

```text
scripts/collect_blender_flat_v4_report.py
```

## 8. Degraded-scene sanity check

Degradation setting:

```text
gaussian_noise_std = 0.03
poisson_peak = 60
background_scatter = 0.02
seed = 42
epochs = 20
```

Artifacts:

```text
experiments/localgpu_blender_flat_rr_gain2_min035_v4_degraded_full_20ep_seed42
experiments/localgpu_blender_flat_rr_gain2_min035_v4_degraded_gate1_20ep_seed42
```

Result:

| input | best val acc |
|---|---:|
| Full 3-gate stack, degraded | 1.0000 |
| Gate 1 only, degraded | 1.0000 |

Interpretation:

This degradation check does not remove the strongest single-gate shortcut. Therefore the remaining shortcut is probably not just random sensor noise; it is tied to the simulated geometry/brightness distribution itself, especially max intensity and foreground statistics. The next improvement should be done in Blender/rendering physics, not only in training-time perturbation.

## 9. Next required steps

Priority 1: add stronger physics controls in Blender, not PNG post-processing.

Recommended controls:

1. Add weak background scatter/noise during training and/or rendering.
2. Randomize flat false reflectance so max intensity cannot separate classes.
3. Add detector noise and mild blur to both true and false.
4. Render multiple views to reduce fixed-view shape shortcuts.
5. Run multi-seed training on the current v4 dataset and the next v5 dataset.
6. Move the final long runs to the 24 GB RTX 3090 machine.

Priority 2: paper story.

Current safe story:

```text
We built a physics-inspired Blender pipeline for gated laser imaging and constructed a planar false-target control by flattening the target in camera depth. Gate-stack diagnostics show a clear difference between true 3D depth slicing and planar echo consistency. Network experiments validate that the generated data can support true/false discrimination, while single-gate ablations reveal remaining simulation shortcuts that motivate stronger physical controls.
```

## 10. v5 reflectance-randomized Blender dataset

New dataset roots:

```text
dataset_new/Military_3D_Gated_Selected44_blender_refl_v5
dataset_new/Military_FlatFalse_Selected44_blender_refl_rr_gain2_min035_v5
dataset_new/Military_TrueFalse_Selected44_blender_refl_rr_gain2_min035_v5
```

Key settings:

```text
FlatGeometryMode = flatten-camera-depth
FlatTargetGateIndexMode = round-robin
FlatMinResponse = 0.35
FlatEchoGain = 2.0
ReflectanceMode = hash-log-uniform
ReflectanceMin = 0.6
ReflectanceMax = 2.8
```

Metadata check:

| item | value |
|---|---:|
| flat false metadata files | 44 |
| target-gate distribution | 15 / 15 / 14 |
| max flattened depth span | 0 |
| reflectance range | 0.6067 to 2.7272 |

Gate-stack diagnostics:

| class | corr | mask IoU | absdiff |
|---|---:|---:|---:|
| flat_false | 0.9993 | 0.9826 | 0.0058 |
| true3d | 0.3011 | 0.2892 | 0.1330 |

Three-seed network ablation:

| input | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| Full 3-gate stack | 0.9545 | 0.0000 | 0.9545 | 0.9545 |
| Gate 0 only | 0.8788 | 0.0263 | 0.8636 | 0.9091 |
| Gate 1 only | 0.9394 | 0.0525 | 0.9091 | 1.0000 |
| Gate 2 only | 0.9545 | 0.0455 | 0.9091 | 1.0000 |

Interpretation:

v5 is more honest than v4 because reflectance variation is introduced in Blender and lowers the strongest single-gate shortcut, especially gate 1. However, gate 2 still matches the full stack in mean accuracy, so v5 cannot yet prove that the network relies mainly on multi-gate depth consistency.

Generated report:

```text
writing/blender_refl_rr_gain2_min035_v5_report_2026-07-07.md
experiments/localgpu_blender_refl_rr_gain2_min035_v5_ablation_summary.csv
experiments/localgpu_blender_refl_rr_gain2_min035_v5_ablation_aggregate.csv
scripts/collect_blender_refl_v5_report.py
scripts/run_blender_refl_v5_ablation.ps1
```

## 11. v6 pure rectangular-overlap false target check

Question tested:

```text
If both the laser pulse and receiver gate are rectangular, should the planar false target simply use the rectangular-overlap response instead of a residual brightness floor?
```

v6 changed only:

```text
FlatMinResponse = 0.0
```

New dataset roots:

```text
dataset_new/Military_FlatFalse_Selected44_blender_refl_rr_gain2_min0_v6
dataset_new/Military_TrueFalse_Selected44_blender_refl_rr_gain2_min0_v6
```

Gate-stack diagnostics:

| class | corr | mask IoU | absdiff | max/mean ratio |
|---|---:|---:|---:|---:|
| flat_false | 0.4193 | 0.3596 | 0.1686 | 150.7863 |
| true3d | 0.3011 | 0.2892 | 0.1330 | 34.7651 |

Single-gate scalar shortcut:

| gate | strongest feature | threshold acc |
|---:|---|---:|
| 1 | max_value | 0.9432 |
| 2 | edge_density | 0.9205 |
| 0 | edge_density | 0.8750 |

seed42 network check:

| input | best val acc |
|---|---:|
| Full 3-gate stack | 1.0000 |
| Gate 1 only | 0.9545 |

Interpretation:

Pure rectangular-overlap response is physically cleaner than a fixed residual floor, but with the current gate spacing it makes non-hit gates nearly black. This creates a new black-frame/peak-frame shortcut and weakens the intended "same full silhouette across gates" flat-target diagnostic. The next version should not simply set the floor to zero; it should jointly tune gate spacing, receiver width, pulse width, exposure normalization, background, and noise.

Detailed note:

```text
writing/flat_false_gate_response_v5_v6_note_2026-07-07.md
```

## 12. v7 overlapping-gate experiment

Purpose:

```text
Test whether a pure rectangular-overlap flat target can avoid the v6 black-frame shortcut by increasing natural gate overlap, without returning to the v5 residual brightness floor.
```

Settings:

```text
ReceiverGateWidth = 1.5
LaserPulseWidth = 0.45
AutoGateMargin = 0.16
FlatMinResponse = 0.0
FlatEchoGain = 2.0
ReflectanceMode = hash-log-uniform
```

Dataset roots:

```text
dataset_new/Military_3D_Gated_Selected44_blender_refl_overlap_w15_m016_v7
dataset_new/Military_FlatFalse_Selected44_blender_refl_overlap_w15_m016_v7
dataset_new/Military_TrueFalse_Selected44_blender_refl_overlap_w15_m016_v7
```

Metadata:

| item | value |
|---|---:|
| flat false metadata files | 44 |
| max flattened depth span | 0 |
| non-hit zero response fraction | 0.2955 |

Gate-stack diagnostics:

| class | corr | mask IoU | absdiff | max/mean ratio |
|---|---:|---:|---:|---:|
| flat_false | 0.6427 | 0.6158 | 0.1007 | 88.4508 |
| true3d | 0.7325 | 0.7273 | 0.0792 | 9.2153 |

Attention-fusion ablation, three seeds:

| input | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| Full 3-gate stack | 0.9091 | 0.0455 | 0.8636 | 0.9545 |
| Gate 0 only | 0.7576 | 0.0262 | 0.7273 | 0.7727 |
| Gate 1 only | 0.8939 | 0.0525 | 0.8636 | 0.9545 |
| Gate 2 only | 0.9242 | 0.0946 | 0.8182 | 1.0000 |

Fusion comparison on full stack:

| fusion | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| attention | 0.9091 | 0.0455 | 0.8636 | 0.9545 |
| mean | 0.9394 | 0.0262 | 0.9091 | 0.9545 |
| attention_residual | 0.9394 | 0.0262 | 0.9091 | 0.9545 |

Interpretation:

v7 reduces the v6 black-frame problem and makes full-stack mean/attention_residual fusion more stable. However, the true3d gate-stack correlation becomes high, meaning the gate overlap may over-smooth 3D depth slicing. Gate 2 remains competitive in some seeds. v7 is therefore a useful parameter-study result, not the final main result.

Generated reports:

```text
writing/blender_refl_overlap_w15_m016_v7_report_2026-07-07.md
writing/v7_overlap_gate_evidence_and_v8_plan_2026-07-07.md
experiments/localgpu_blender_refl_overlap_w15_m016_v7_ablation_aggregate.csv
experiments/localgpu_blender_refl_overlap_w15_m016_v7_mean_aggregate.csv
experiments/localgpu_blender_refl_overlap_w15_m016_v7_attention_residual_aggregate.csv
```

## 13. v8 pure-overlap anti-shortcut and robustness control

Purpose:

```text
Test a less over-smoothed pure rectangular-overlap setting, then add anti-shortcut normalization and mixed clean/noisy augmentation to separate physical evidence from single-frame shortcuts.
```

v8 settings:

```text
ReceiverGateWidth = 1.5
LaserPulseWidth = 0.45
AutoGateMargin = 0.12
FlatMinResponse = 0.0
FlatEchoGain = 2.0
ReflectanceMode = hash-log-uniform
FlatGeometryMode = flatten-camera-depth
FlatTargetGateIndexMode = round-robin
```

Dataset roots:

```text
dataset_new/Military_3D_Gated_Selected44_blender_refl_overlap_w15_m012_v8
dataset_new/Military_FlatFalse_Selected44_blender_refl_overlap_w15_m012_v8
dataset_new/Military_TrueFalse_Selected44_blender_refl_overlap_w15_m012_v8
dataset_new/Military_TrueFalse_Selected44_blender_refl_overlap_w15_m012_v8_per_gate_maxnorm
```

Raw v8 diagnostics:

| class | corr | mask IoU | absdiff | max/mean ratio |
|---|---:|---:|---:|---:|
| flat_false | 0.5848 | 0.5702 | 0.1177 | 103.8464 |
| true3d | 0.6736 | 0.6670 | 0.0934 | 9.3942 |

Compared with v7, true3d gate correlation is reduced, so v8 weakens over-smoothing. However, raw v8 still has a strong gate2 scalar shortcut: `p99` threshold accuracy reaches `0.9886`.

Per-gate max-normalized control:

```text
dataset_new/normalize_gate_dataset.py --mode per-gate-max --target-max 180 --min-source-max 2
```

After normalization, the strongest scalar single-gate shortcut drops to `0.7955`, and the dominant shortcut changes from intensity to edge/shape statistics. This is useful as an anti-brightness control, but single-gate CNNs can still exploit slice shape.

Seed42 normalized-control ablation:

| input | best val acc |
|---|---:|
| Full 3-gate stack | 1.0000 |
| Gate 0 only | 0.9545 |
| Gate 1 only | 0.9091 |
| Gate 2 only | 0.9545 |

New evaluation tooling:

```text
scripts/evaluate_gate_model.py
train.py --degradation-probability
```

`evaluate_gate_model.py` loads a saved `best_model.pth` and evaluates clean/noisy test conditions without retraining. `--degradation-probability` enables deterministic clean/noisy mixed augmentation. Default is `1.0`, so previous experiments are unchanged.

Test status after the code change:

```text
60 passed
```

Mixed augmentation setting:

```text
gaussian_noise_std = 0.02
poisson_peak = 80
background_scatter = 0.02
degradation_probability = 0.5
```

Seed42 independent evaluation:

| input | clean | light noise | strong noise |
|---|---:|---:|---:|
| Full 3-gate stack | 1.0000 | 1.0000 | 0.7727 |
| Gate 0 only | 0.9091 | 0.9091 | 0.8182 |
| Gate 1 only | 0.6818 | 0.4545 | 0.5455 |
| Gate 2 only | 0.9091 | 0.8636 | 0.5455 |

Interpretation:

The current best story is not "the problem is solved." It is:

```text
The simulation and network now include explicit anti-shortcut controls. Raw pure-overlap data still contains intensity shortcuts; per-gate normalization suppresses those shortcuts; mixed clean/noisy training gives the full gate stack the best clean and light-noise performance in seed42. Strong-noise performance remains inconclusive and needs multi-seed plus fusion-mode verification.
```

Generated report and summary:

```text
writing/blender_refl_overlap_w15_m012_v8_report_2026-07-07.md
writing/blender_refl_overlap_w15_m012_v8_per_gate_maxnorm_report_2026-07-07.md
writing/v8_mixaug_robustness_report_2026-07-07.md
experiments/v8_per_gate_maxnorm_eval_summary_seed42.csv
```

## Late update: three-seed independent robustness and fusion figures

The mixed clean/noisy setting was expanded from seed42 to three seeds (`42/332/2026`) for the normalized v8 dataset.

Full-stack versus single-gate independent evaluation:

| evaluation condition | full stack | gate0 | gate1 | gate2 |
|---|---:|---:|---:|---:|
| clean | 0.9697 | 0.9091 | 0.7121 | 0.8636 |
| light noise | 0.9545 | 0.8636 | 0.5151 | 0.7727 |
| strong noise | 0.7424 | 0.6970 | 0.5152 | 0.5152 |

Interpretation: the full 3-gate stack now has the highest mean accuracy under clean, light-noise, and strong-noise independent evaluations. This is stronger than the earlier seed42-only result and can be used as the main controlled validation result.

Full-stack fusion independent evaluation:

| evaluation condition | attention | mean | attention_residual |
|---|---:|---:|---:|
| clean | 0.9697 | 0.9545 | 0.9848 |
| light noise | 0.9545 | 0.9697 | 0.9545 |
| strong noise | 0.7424 | 0.6515 | 0.6970 |

Interpretation: `attention_residual` is best on validation and clean independent evaluation, `mean` is slightly better under light noise, and `attention` is best under strong noise. The paper should therefore claim a gate-stack advantage, not that one fusion head is universally optimal.

New result tables:

```text
experiments/v8_per_gate_maxnorm_full_fusion_mixaug_eval_summary_3seed.csv
experiments/v8_per_gate_maxnorm_full_fusion_mixaug_eval_aggregate_3seed.csv
```

New paper/PPT figures:

```text
writing/figures/fig5_full_stack_vs_single_gate_robustness.png
writing/figures/fig5_full_stack_vs_single_gate_robustness.pdf
writing/figures/fig6_full_stack_fusion_robustness.png
writing/figures/fig6_full_stack_fusion_robustness.pdf
writing/figure_captions_v8_mixaug_2026-07-07.md
```

Updated writing files:

```text
writing/v8_mixaug_robustness_report_2026-07-07.md
writing/paper_draft_v8_mixaug_framework_2026-07-07.md
```

Current test status:

```text
60 passed
```

## Manuscript figure and English draft update

Generated a reproducible figure script:

```text
scripts/make_v8_paper_figures.py
```

Generated manuscript figures:

```text
writing/figures/fig1_gated_imaging_framework.png
writing/figures/fig1_gated_imaging_framework.pdf
writing/figures/fig2_rectangular_overlap_response.png
writing/figures/fig2_rectangular_overlap_response.pdf
writing/figures/fig3_true3d_flatfalse_gate_examples.png
writing/figures/fig3_true3d_flatfalse_gate_examples.pdf
writing/figures/fig4_scalar_shortcut_control.png
writing/figures/fig4_scalar_shortcut_control.pdf
writing/figures/source_manifest_v8_paper_figures.csv
```

Important correction for the physical story:

```text
For a planar false target, the whole-object silhouette should remain visible across gates when the pulse-gate response is nonzero. Its brightness should follow sampled rectangular pulse-gate overlap, not an arbitrary linear fade or a partial object slice.
```

Created an English SCI-style working manuscript:

```text
writing/sci_manuscript_v8_gated_false_target_draft_2026-07-07.md
```

This draft currently includes the abstract, introduction, physical model, anti-shortcut validation protocol, network baseline, experiments, results, discussion, limitations, conclusion, figure plan, evidence files, and citation tasks. References are intentionally left as tasks/placeholders to avoid invented citation details.

## Multi-view pipeline update

Added multi-view dataset tooling:

```text
scripts/run_v8_multiview_dataset.ps1
dataset_new/build_multiview_true_false_dataset.py
```

Updated the selected military renderer wrapper:

```text
scripts/render_selected_military_gates.ps1
```

New exposed argument:

```text
-ModelRotationDeg "0,0,<angle>"
```

Dry-run validation completed for the default four-view command path (`0/90/180/270` degrees).

Actual smoke run completed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_v8_multiview_dataset.ps1 `
  -ModelsPerClass 1 `
  -ViewRotationsZ 0 `
  -TrueOutputRoot .\dataset_new\_smoke_Military_3D_Gated_v8_mv `
  -FalseOutputRoot .\dataset_new\_smoke_Military_FlatFalse_v8_mv `
  -CombinedOutputRoot .\dataset_new\_smoke_Military_TrueFalse_v8_mv `
  -NormalizedOutputRoot .\dataset_new\_smoke_Military_TrueFalse_v8_mv_per_gate_maxnorm
```

Smoke result:

| class | gate pngs | valid samples | ready |
|---|---:|---:|---|
| true3d | 9 | 3 | True |
| flat_false | 9 | 3 | True |

New plan document:

```text
writing/multiview_v8_validation_plan_2026-07-07.md
```

Interpretation:

```text
Historical note at the time of this smoke run:
the multi-view pipeline was operational, but not yet a performance result.
This was superseded later the same day by the completed 44-model x 4-view evaluation and fusion ablation.
```

## Literature and manuscript positioning update

The four local Markdown-converted papers in:

```text
E:\wjz\非线性光学\文献
```

were mapped to the current gated false-target discrimination project.

New evidence matrix:

```text
writing/literature_evidence_matrix_gated_false_target_2026-07-07.md
```

Relevance ranking for the current SCI story:

| rank | paper | role in this project |
|---:|---|---|
| 1 | Transformer-based 3D range-gated imaging method with multiple depth priors | Closest related work for gate-stack depth priors, adaptive depth bins, and optical-prior-guided learning |
| 2 | Range precision prediction model for three-dimensional laser gated range-intensity correlation imaging | Core physical support for rectangular pulse/gate response, triangular/trapezoidal RIPs, and noise-aware gated imaging physics |
| 3 | SAMFusion | Application motivation for gated cameras in robust multimodal perception and future fusion |
| 4 | Snapshot 3D image projection using a diffractive decoder | Future optical-neural/diffractive computing direction, not direct evidence for the current classifier |

Updated the SCI-style manuscript:

```text
writing/sci_manuscript_v8_gated_false_target_draft_2026-07-07.md
```

Main manuscript changes:

```text
1. Added a Related Work section after Introduction.
2. Re-numbered the later sections so Physical Model is now Section 3.
3. Inserted working citation placeholders:
   [Song2026GRICI]
   [Liu2026DepthPriors]
   [Palladin2024SAMFusion]
   [Isil2026DiffractiveDecoder]
   [Geirhos2020Shortcut]
   [Tobin2017DomainRandomization]
4. Added a References section with verified DOI information where available.
5. Updated Evidence Files and Citation Tasks.
```

Current safest paper positioning:

```text
The main contribution is not a new neural architecture.
The contribution is a physics-interpretable gated-imaging simulation and anti-shortcut validation protocol for real-3D versus planar-false-target discrimination.
The lightweight network is evidence that the corrected gate stack carries useful discriminative information after shortcut controls.
```

Immediate next steps:

```text
1. Full multi-view v8 render on the 44 selected military models.
2. Repeat full-stack versus single-gate ablations on the multi-view normalized dataset.
3. Add shortcut-learning and synthetic-data simulation-to-real references.
4. Export final references from Zotero before any formal submission.
```

## Late update: four-view v8 validation completed

The full 44-model, four-view v8 dataset has now been generated and evaluated.

Dataset:

```text
dataset_new/Military_TF_v8_mv4_norm
```

Scale:

```text
true3d samples = 176
flat_false samples = 176
total samples = 352
total gate PNGs = 1056
views = 0 / 90 / 180 / 270 degrees
```

Important split fix:

```text
train.py now strips the view prefix from multi-view sample IDs before grouped splitting.
This prevents different views of the same source model from leaking across train/validation partitions.
```

Four-view independent evaluation, three seeds:

| evaluation condition | full stack | gate0 | gate1 | gate2 |
|---|---:|---:|---:|---:|
| clean | 0.9848 | 0.9697 | 0.9545 | 0.8939 |
| light noise | 0.9811 | 0.9432 | 0.9053 | 0.8750 |
| strong noise | 0.9205 | 0.7462 | 0.5265 | 0.6098 |

Interpretation:

```text
The full 3-gate stack remains the best mean performer after four-view yaw expansion.
The strong-noise gap is now much clearer than in the single-view result.
Single gates are still informative under clean/light noise, so the claim should remain:
gate-stack evidence is more stable, not that all single-frame shortcuts are eliminated.
```

New scripts and outputs:

```text
scripts/run_v8_mv4_eval_grid.ps1
scripts/summarize_eval_grid.py
scripts/make_v8_mv4_robustness_figure.py
experiments/v8_mv4_norm_mixaug_attention_eval_summary_3seed.csv
experiments/v8_mv4_norm_mixaug_attention_eval_aggregate_3seed.csv
writing/v8_mv4_multiview_robustness_report_2026-07-07.md
writing/figures/fig7_mv4_full_stack_vs_single_gate_robustness.png
writing/figures/fig7_mv4_full_stack_vs_single_gate_robustness.pdf
```

Manuscript update:

```text
writing/sci_manuscript_v8_gated_false_target_draft_2026-07-07.md
```

The draft now treats four-view validation as completed evidence, not a future task.

Verification:

```text
pytest tests/test_train_utils.py tests/test_run_experiments.py tests/test_build_multiview_true_false_dataset.py
22 passed
```

## Late update: four-view fusion ablation completed

The four-view dataset was used to compare full-stack fusion modes:

```text
attention
mean
attention_residual
```

Training outputs:

```text
experiments/v8_mv4_norm_mixaug_attention_full_20ep
experiments/v8_mv4_norm_mixaug_mean_full_20ep
experiments/v8_mv4_norm_mixaug_attention_residual_full_20ep
```

Independent evaluation, three seeds:

| condition | attention | mean | attention_residual |
|---|---:|---:|---:|
| clean | 0.9848 | 0.9848 | 0.9848 |
| light noise | 0.9811 | 0.9848 | 0.9394 |
| strong noise | 0.9205 | 0.7197 | 0.7727 |

Interpretation:

```text
The network fusion module is not the main innovation.
attention_residual is not consistently best after four-view validation.
Simple attention is substantially more robust under strong noise.
The paper should emphasize gated simulation + anti-shortcut validation + full-stack evidence.
```

New files:

```text
scripts/run_v8_mv4_fusion_experiments.ps1
scripts/evaluate_v8_mv4_fusion_grid.py
scripts/make_v8_mv4_fusion_figure.py
experiments/v8_mv4_norm_mixaug_full_fusion_eval_summary_3seed.csv
experiments/v8_mv4_norm_mixaug_full_fusion_eval_aggregate_3seed.csv
writing/v8_mv4_fusion_robustness_report_2026-07-07.md
writing/figures/fig8_mv4_full_stack_fusion_robustness.png
writing/figures/fig8_mv4_full_stack_fusion_robustness.pdf
```

Updated manuscript:

```text
writing/sci_manuscript_v8_gated_false_target_draft_2026-07-07.md
```
