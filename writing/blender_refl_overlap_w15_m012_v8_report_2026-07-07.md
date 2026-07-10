# Blender false-target evidence report

This report records the v8 overlap gate w=1.5 margin=0.12 revision of the laser-gated military true/false-target dataset.
The purpose is to move the work from a visually plausible demo toward a paper-grade validation chain.

## Dataset construction

- True target root: `Military_3D_Gated_Selected44_blender_refl_overlap_w15_m012_v8`.
- Flat false root: `Military_FlatFalse_Selected44_blender_refl_overlap_w15_m012_v8`.
- Binary dataset: `Military_TrueFalse_Selected44_blender_refl_overlap_w15_m012_v8`.
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
| flat_false | 44 | 0.5848 | 0.5702 | 0.1177 | 103.8464 |
| true3d | 44 | 0.6736 | 0.6670 | 0.0934 | 9.3942 |

Interpretation: this table should be read as a physics-control diagnostic, not only as an accuracy metric.
A useful setting should keep planar false-target responses interpretable while preserving enough true-3D
cross-gate diversity. If the true3d correlation becomes too high, the gates may be over-smoothed; if
flat_false correlation becomes too low, the false target may have black-frame shortcuts.

## Network ablation, three seeds

| input | runs | mean | std | min | max | seeds |
|---|---|---|---|---|---|---|
| Full 3-gate stack | 1 | 0.9091 | 0.0000 | 0.9091 | 0.9091 | 42 |
| Gate 0 only | 1 | 0.6818 | 0.0000 | 0.6818 | 0.6818 | 42 |
| Gate 1 only | 1 | 0.8636 | 0.0000 | 0.8636 | 0.8636 | 42 |
| Gate 2 only | 1 | 0.9545 | 0.0000 | 0.9545 | 0.9545 | 42 |

Per-run details:

| input | best val acc | best epoch | seed |
|---|---|---|---|
| Full 3-gate stack | 0.9091 | 10 | 42 |
| Gate 0 only | 0.6818 | 7 | 42 |
| Gate 1 only | 0.8636 | 12 | 42 |
| Gate 2 only | 0.9545 | 12 | 42 |

## v7 overlap gate w=1.5 margin=0.16 comparison

| input | v7 overlap gate w=1.5 margin=0.16 mean | current mean | delta |
|---|---|---|---|
| Full 3-gate stack | 0.9091 | 0.9091 | +0.0000 |
| Gate 0 only | 0.7576 | 0.6818 | -0.0758 |
| Gate 1 only | 0.8939 | 0.8636 | -0.0303 |
| Gate 2 only | 0.9242 | 0.9545 | +0.0303 |

## Single-gate shortcut diagnostics

| gate | strongest scalar feature | threshold acc | true3d mean | flat false mean | Cohen d |
|---|---|---|---|---|---|
| 0 | max_value | 0.8636 | 0.3426 | 0.1440 | -2.041 |
| 1 | p95 | 0.8864 | 0.3177 | 0.1597 | -2.291 |
| 2 | p99 | 0.9886 | 0.3055 | 0.1103 | -2.886 |

## Paper-use interpretation

- Stronger claim now supported: the simulator can generate metadata-proven planar false targets and 3D targets whose gate-stack statistics differ in the expected direction.
- More cautious claim: single-gate classifiers and scalar shortcuts must remain explicit controls before claiming robust gate-stack discrimination.
- Current limitation: the dataset has only 44 selected military models per class condition, so the result is a controlled simulation validation rather than deployment evidence.
- Next validation step: add background/clutter, range/exposure balancing, multi-view renders, and train a stricter model with single-gate and gate-dropout controls on the 3090.
