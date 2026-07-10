# Blender flat false-target v4 evidence report

This report records the current physics-side dataset revision and the first local network ablation.
It should be treated as evidence plus limitation, not as a final paper result.

## Dataset construction

- True target root: `dataset_new/Military_3D_Gated_Selected44_blender_norm_v2`.
- Flat false root: `dataset_new/Military_FlatFalse_Selected44_blender_flat_rr_gain2_min035_v4`.
- Binary dataset: `dataset_new/Military_TrueFalse_Selected44_blender_flat_rr_gain2_min035_v4`.
- Flat false metadata files: 44.
- Flat target gate distribution: {0: 15, 1: 15, 2: 14}.
- Maximum flattened camera-depth span: 0.
- Metadata mode errors: 0.

Interpretation: the false target is now generated in Blender as a camera-depth-flattened target,
with round-robin gate placement and stronger in-render brightness, instead of PNG-level post-processing.

## Gate-stack diagnostics

| class | samples | corr | mask IoU | absdiff |
|---|---|---|---|---|
| flat_false | 44 | 0.9993 | 0.9809 | 0.0063 |
| true3d | 44 | 0.2992 | 0.2864 | 0.1327 |

Interpretation: flat false samples have nearly identical normalized gate stacks, while true 3D samples
show much lower cross-gate correlation and larger normalized differences. This supports the physical
story of planar echo consistency versus 3D depth slicing.

## Network ablation, three seeds

| input | runs | mean | std | min | max | seeds |
|---|---|---|---|---|---|---|
| Full 3-gate stack | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 42/332/2026 |
| Gate 0 only | 3 | 0.8939 | 0.0263 | 0.8636 | 0.9091 | 42/332/2026 |
| Gate 1 only | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 42/332/2026 |
| Gate 2 only | 3 | 0.9697 | 0.0263 | 0.9545 | 1.0000 | 42/332/2026 |

Per-run details:

| input | best val acc | best epoch | seed |
|---|---|---|---|
| Full 3-gate stack | 1.0000 | 10 | 2026 |
| Full 3-gate stack | 1.0000 | 8 | 332 |
| Full 3-gate stack | 1.0000 | 8 | 42 |
| Gate 0 only | 0.9091 | 8 | 2026 |
| Gate 0 only | 0.8636 | 11 | 332 |
| Gate 0 only | 0.9091 | 18 | 42 |
| Gate 1 only | 1.0000 | 9 | 2026 |
| Gate 1 only | 1.0000 | 9 | 332 |
| Gate 1 only | 1.0000 | 11 | 42 |
| Gate 2 only | 1.0000 | 10 | 2026 |
| Gate 2 only | 0.9545 | 10 | 332 |
| Gate 2 only | 0.9545 | 9 | 42 |

Interpretation: the full gate stack reaches 1.0000 across all three seeds, but single-gate inputs remain high.
Gate 1 also reaches 1.0000 across all three seeds, so the result cannot be claimed as pure gate-stack superiority.

## Single-gate shortcut diagnostics

| gate | strongest scalar feature | threshold acc | true3d mean | flat false mean | Cohen d |
|---|---|---|---|---|---|
| 0 | max_value | 0.9432 | 0.2813 | 0.1214 | -2.630 |
| 1 | max_value | 1.0000 | 0.2914 | 0.1189 | -5.051 |
| 2 | max_value | 0.9545 | 0.2334 | 0.1127 | -2.630 |

Interpretation: max-value and foreground statistics still separate the two classes in single gates.
This is a limitation of the current simulation and should drive the next revision: add background, detector noise,
reflectance variation, distance/exposure balancing, and multi-view rendering before claiming strong generalization.

## Degraded-scene sanity check

Setting:

```text
gaussian_noise_std = 0.03
poisson_peak = 60
background_scatter = 0.02
seed = 42
epochs = 20
```

| input | best val acc |
|---|---:|
| Full 3-gate stack, degraded | 1.0000 |
| Gate 1 only, degraded | 1.0000 |

Interpretation: adding deterministic sensor/background degradation at this level does not remove the strongest
single-gate shortcut. The remaining shortcut is therefore likely tied to the rendering geometry and brightness
distribution, not merely to clean images.

## Paper-use wording

- Safe claim: Blender-side flattened false targets produce a physically interpretable control with high gate-stack consistency.
- Safe claim: full gate stacks are stable, but current single-gate shortcuts remain and must be treated as limitations.
- Do not claim: a single gate is useless, or the method is already validated on real battlefield data.
- Next required evidence: multi-seed training on v4/v5, background/noise robustness, and a 3090 run with higher epochs.
