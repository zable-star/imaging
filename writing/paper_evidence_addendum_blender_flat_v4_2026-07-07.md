# Evidence addendum: Blender-side planar false target v4

## Core contribution supported by current evidence

The current strongest contribution is not a final classifier result. It is a more physically controlled simulation pipeline:

```text
3D true target: physical gated rendering over object depth.
2D false target: same model silhouette flattened to one camera-space depth plane in Blender.
```

This is a meaningful improvement over PNG-level post-processing because the false target is now defined before rendering, with metadata proving its geometry.

## Current dataset

```text
dataset_new/Military_TrueFalse_Selected44_blender_flat_rr_gain2_min035_v4
```

Dataset facts:

| item | value |
|---|---:|
| selected military models | 44 |
| true3d samples | 44 |
| flat_false samples | 44 |
| gates per sample | 3 |
| flat false gate distribution | `{0: 15, 1: 15, 2: 14}` |
| max flat false depth span | 0.0 |

## Physical diagnostic evidence

| class | corr | mask IoU | absdiff |
|---|---:|---:|---:|
| flat_false | 0.9993 | 0.9809 | 0.0063 |
| true3d | 0.2992 | 0.2864 | 0.1327 |

Meaning:

The planar false target keeps nearly the same normalized shape across gates, while the real 3D target changes strongly across gates. This directly supports the proposed physical distinction:

```text
planar false target = echo consistency across gates
true 3D target = depth-dependent gated slices
```

## Network evidence

Three-seed local ablation:

| input | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| Full 3-gate stack | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Gate 0 only | 0.8939 | 0.0263 | 0.8636 | 0.9091 |
| Gate 1 only | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Gate 2 only | 0.9697 | 0.0263 | 0.9545 | 1.0000 |

Meaning:

The classifier can separate true3d and flat_false on the current Blender dataset, but the high single-gate results show that the current simulation still contains single-frame shortcuts.

## Negative evidence that must be reported honestly

Single-gate scalar shortcuts:

| gate | strongest scalar feature | threshold acc |
|---:|---|---:|
| 0 | max_value | 0.9432 |
| 1 | max_value | 1.0000 |
| 2 | max_value | 0.9545 |

Training-time degradation sanity check:

| input | best val acc |
|---|---:|
| Full stack with noise/background | 1.0000 |
| Gate 1 only with noise/background | 1.0000 |

Meaning:

The strongest shortcut is not removed by simple sensor/background degradation. It likely comes from the rendering geometry and brightness distribution. The next step should be a rendering-side v5 dataset, not more PNG post-processing.

## Safe claims for the paper

1. A Blender-side gated-imaging pipeline was constructed for selected military 3D targets.
2. A planar false-target control was implemented by flattening target geometry in camera depth before rendering.
3. Gate-stack diagnostics reveal a clear physical difference between 3D depth slicing and planar echo consistency.
4. Network experiments verify separability on the simulated dataset.
5. Single-gate ablations expose remaining simulation shortcuts, motivating stricter rendering controls.

## Claims to avoid

1. Do not claim single-gate images are useless.
2. Do not claim the classifier has learned pure depth physics.
3. Do not claim real-world battlefield validation.
4. Do not use perfect accuracy alone as the paper's main evidence.

## Next v5 controls

Priority controls:

1. Multi-view rendering for each model.
2. Reflectance randomization for both true and flat false targets.
3. Background/ground-plane clutter rendered in Blender.
4. Detector noise and mild blur applied consistently to both classes.
5. Exposure balancing at render time, not post-processing time.
6. Final multi-seed 3090 training after the v5 dataset is generated.

This addendum should be treated as the bridge between current implementation and the eventual manuscript.
