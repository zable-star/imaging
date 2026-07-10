# Blender false-target evidence report

This report records the v8 overlap gate per-gate max-normalized anti-shortcut control revision of the laser-gated military true/false-target dataset.
The purpose is to move the work from a visually plausible demo toward a paper-grade validation chain.

## Dataset construction

- True target root: `Military_TrueFalse_Selected44_blender_refl_overlap_w15_m012_v8_per_gate_maxnorm/true3d`.
- Flat false root: `Military_TrueFalse_Selected44_blender_refl_overlap_w15_m012_v8_per_gate_maxnorm/flat_false`.
- Binary dataset: `Military_TrueFalse_Selected44_blender_refl_overlap_w15_m012_v8_per_gate_maxnorm`.
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
| flat_false | 44 | 0.5848 | 0.5363 | 0.1178 | 320.9733 |
| true3d | 44 | 0.6736 | 0.6712 | 0.0934 | 9.3275 |

Interpretation: this table should be read as a physics-control diagnostic, not only as an accuracy metric.
A useful setting should keep planar false-target responses interpretable while preserving enough true-3D
cross-gate diversity. If the true3d correlation becomes too high, the gates may be over-smoothed; if
flat_false correlation becomes too low, the false target may have black-frame shortcuts.

## Network ablation, three seeds

| input | runs | mean | std | min | max | seeds |
|---|---|---|---|---|---|---|
| Full 3-gate stack | 1 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 42 |
| Gate 0 only | 1 | 0.9545 | 0.0000 | 0.9545 | 0.9545 | 42 |
| Gate 1 only | 1 | 0.9091 | 0.0000 | 0.9091 | 0.9091 | 42 |
| Gate 2 only | 1 | 0.9545 | 0.0000 | 0.9545 | 0.9545 | 42 |

Per-run details:

| input | best val acc | best epoch | seed |
|---|---|---|---|
| Full 3-gate stack | 1.0000 | 14 | 42 |
| Gate 0 only | 0.9545 | 12 | 42 |
| Gate 1 only | 0.9091 | 16 | 42 |
| Gate 2 only | 0.9545 | 10 | 42 |

## raw v8 comparison

| input | raw v8 mean | current mean | delta |
|---|---|---|---|
| Full 3-gate stack | 0.9091 | 1.0000 | +0.0909 |
| Gate 0 only | 0.6818 | 0.9545 | +0.2727 |
| Gate 1 only | 0.8636 | 0.9091 | +0.0455 |
| Gate 2 only | 0.9545 | 0.9545 | +0.0000 |

## Single-gate shortcut diagnostics

| gate | strongest scalar feature | threshold acc | true3d mean | flat false mean | Cohen d |
|---|---|---|---|---|---|
| 0 | edge_density | 0.7955 | 0.3664 | 0.1357 | -1.398 |
| 1 | p99 | 0.7386 | 0.6874 | 0.6004 | -0.573 |
| 2 | edge_density | 0.7841 | 0.3499 | 0.1559 | -1.194 |

## Paper-use interpretation

- Stronger claim now supported: the simulator can generate metadata-proven planar false targets and 3D targets whose gate-stack statistics differ in the expected direction.
- More cautious claim: single-gate classifiers and scalar shortcuts must remain explicit controls before claiming robust gate-stack discrimination.
- Current limitation: the dataset has only 44 selected military models per class condition, so the result is a controlled simulation validation rather than deployment evidence.
- Next validation step: add background/clutter, range/exposure balancing, multi-view renders, and train a stricter model with single-gate and gate-dropout controls on the 3090.
