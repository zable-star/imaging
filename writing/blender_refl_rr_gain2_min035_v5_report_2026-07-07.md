# Blender reflectance-randomized false-target v5 evidence report

This report records the current v5 revision of the laser-gated military true/false-target dataset.
The purpose is to move the work from a visually plausible demo toward a paper-grade validation chain.

## Dataset construction

- True target root: `dataset_new/Military_3D_Gated_Selected44_blender_refl_v5`.
- Flat false root: `dataset_new/Military_FlatFalse_Selected44_blender_refl_rr_gain2_min035_v5`.
- Binary dataset: `dataset_new/Military_TrueFalse_Selected44_blender_refl_rr_gain2_min035_v5`.
- Flat false metadata files: 44.
- Flat target gate distribution: {0: 15, 1: 15, 2: 14}.
- Maximum flattened camera-depth span: 0.
- Reflectance range: 0.6067 to 2.7272; mean 1.4714.
- Metadata mode errors: 0.

Interpretation: v5 keeps the planar false-target physics from v4, but moves a brightness-control factor into
the Blender rendering stage through deterministic hash-log-uniform reflectance variation. This is preferable to
PNG-level histogram or clipping post-processing because the sample metadata records the physical nuisance factor.

## Gate-stack diagnostics

| class | samples | corr | mask IoU | absdiff | max/mean ratio |
|---|---|---|---|---|---|
| flat_false | 44 | 0.9993 | 0.9826 | 0.0058 | 2.0919 |
| true3d | 44 | 0.3011 | 0.2892 | 0.1330 | 34.7651 |

Interpretation: flat false samples remain almost identical across normalized gates, while true 3D samples
show low gate-pair correlation and larger cross-gate differences. This preserves the central physical contrast:
a planar echo has gate-consistent silhouette information, while a 3D target is sliced by depth.

## Network ablation, three seeds

| input | runs | mean | std | min | max | seeds |
|---|---|---|---|---|---|---|
| Full 3-gate stack | 3 | 0.9545 | 0.0000 | 0.9545 | 0.9545 | 42/332/2026 |
| Gate 0 only | 3 | 0.8788 | 0.0263 | 0.8636 | 0.9091 | 42/332/2026 |
| Gate 1 only | 3 | 0.9394 | 0.0525 | 0.9091 | 1.0000 | 42/332/2026 |
| Gate 2 only | 3 | 0.9545 | 0.0455 | 0.9091 | 1.0000 | 42/332/2026 |

Per-run details:

| input | best val acc | best epoch | seed |
|---|---|---|---|
| Full 3-gate stack | 0.9545 | 9 | 2026 |
| Full 3-gate stack | 0.9545 | 7 | 332 |
| Full 3-gate stack | 0.9545 | 8 | 42 |
| Gate 0 only | 0.9091 | 10 | 2026 |
| Gate 0 only | 0.8636 | 13 | 332 |
| Gate 0 only | 0.8636 | 13 | 42 |
| Gate 1 only | 0.9091 | 10 | 2026 |
| Gate 1 only | 1.0000 | 8 | 332 |
| Gate 1 only | 0.9091 | 12 | 42 |
| Gate 2 only | 1.0000 | 14 | 2026 |
| Gate 2 only | 0.9545 | 10 | 332 |
| Gate 2 only | 0.9091 | 10 | 42 |

## v4 to v5 comparison

| input | v4 mean | v5 mean | delta |
|---|---|---|---|
| Full 3-gate stack | 1.0000 | 0.9545 | -0.0455 |
| Gate 0 only | 0.8939 | 0.8788 | -0.0151 |
| Gate 1 only | 1.0000 | 0.9394 | -0.0606 |
| Gate 2 only | 0.9697 | 0.9545 | -0.0152 |

## Single-gate shortcut diagnostics

| gate | strongest scalar feature | threshold acc | true3d mean | flat false mean | Cohen d |
|---|---|---|---|---|---|
| 0 | foreground_ratio | 0.8864 | 0.0733 | 0.2476 | 1.508 |
| 1 | p99 | 0.9318 | 0.3186 | 0.1441 | -2.853 |
| 2 | max_value | 0.8636 | 0.2650 | 0.1430 | -2.006 |

## Paper-use interpretation

- Stronger claim now supported: the simulator can generate metadata-proven planar false targets and 3D targets whose gate-stack statistics differ in the expected direction.
- More cautious claim: reflectance randomization reduces some direct brightness shortcuts compared with v4, but single-gate classifiers are still too strong.
- Current limitation: the dataset has only 44 selected military models per class condition, so the result is a controlled simulation validation rather than deployment evidence.
- Next validation step: add background/clutter, range/exposure balancing, multi-view renders, and train a stricter model with single-gate and gate-dropout controls on the 3090.
