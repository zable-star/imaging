# Blender false-target evidence report

This report records the v7 overlap-gate attention_residual fusion revision of the laser-gated military true/false-target dataset.
The purpose is to move the work from a visually plausible demo toward a paper-grade validation chain.

## Dataset construction

- True target root: `dataset_new/Military_3D_Gated_Selected44_blender_refl_overlap_w15_m016_v7`.
- Flat false root: `dataset_new/Military_FlatFalse_Selected44_blender_refl_overlap_w15_m016_v7`.
- Binary dataset: `dataset_new/Military_TrueFalse_Selected44_blender_refl_overlap_w15_m016_v7`.
- Flat false metadata files: 44.
- Flat target gate distribution: {0: 15, 1: 15, 2: 14}.
- Maximum flattened camera-depth span: 0.
- Reflectance range: 0.6067 to 2.7272; mean 1.4714.
- Metadata mode errors: 0.

Interpretation: the planar false target is generated in Blender with metadata-recorded geometry, gate response,
and reflectance settings. This is preferable to PNG-level histogram or clipping post-processing because the
sample metadata records the physical nuisance factors.

## Gate-stack diagnostics

| class | samples | corr | mask IoU | absdiff | max/mean ratio |
|---|---|---|---|---|---|
| flat_false | 44 | 0.6427 | 0.6158 | 0.1007 | 88.4508 |
| true3d | 44 | 0.7325 | 0.7273 | 0.0792 | 9.2153 |

Interpretation: this table should be read as a physics-control diagnostic, not only as an accuracy metric.
A useful setting should keep planar false-target responses interpretable while preserving enough true-3D
cross-gate diversity. If the true3d correlation becomes too high, the gates may be over-smoothed; if
flat_false correlation becomes too low, the false target may have black-frame shortcuts.

## Network ablation, three seeds

| input | runs | mean | std | min | max | seeds |
|---|---|---|---|---|---|---|
| Full 3-gate stack | 3 | 0.9394 | 0.0262 | 0.9091 | 0.9545 | 42/332/2026 |

Per-run details:

| input | best val acc | best epoch | seed |
|---|---|---|---|
| Full 3-gate stack | 0.9545 | 13 | 2026 |
| Full 3-gate stack | 0.9545 | 13 | 332 |
| Full 3-gate stack | 0.9091 | 10 | 42 |

## v7 attention comparison

| input | v7 attention mean | current mean | delta |
|---|---|---|---|
| Full 3-gate stack | 0.9091 | 0.9394 | +0.0303 |

## Single-gate shortcut diagnostics

| gate | strongest scalar feature | threshold acc | true3d mean | flat false mean | Cohen d |
|---|---|---|---|---|---|
| 0 | max_value | 0.8636 | 0.3423 | 0.1540 | -2.017 |
| 1 | max_value | 0.8864 | 0.3379 | 0.1821 | -2.266 |
| 2 | p99 | 0.9886 | 0.3117 | 0.1225 | -2.756 |

## Paper-use interpretation

- Stronger claim now supported: the simulator can generate metadata-proven planar false targets and 3D targets whose gate-stack statistics differ in the expected direction.
- More cautious claim: single-gate classifiers and scalar shortcuts must remain explicit controls before claiming robust gate-stack discrimination.
- Current limitation: the dataset has only 44 selected military models per class condition, so the result is a controlled simulation validation rather than deployment evidence.
- Next validation step: add background/clutter, range/exposure balancing, multi-view renders, and train a stricter model with single-gate and gate-dropout controls on the 3090.
