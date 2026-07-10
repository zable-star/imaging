# v7 overlapping-gate evidence and v8 plan

## 1. Why v7 was tested

v6 used the pure rectangular-overlap response with `FlatMinResponse=0`, but the original gate spacing was too wide. Many non-hit gates became nearly black, creating a black-frame shortcut.

v7 therefore keeps the pure rectangular-overlap response but increases gate overlap:

```text
ReceiverGateWidth = 1.5
LaserPulseWidth = 0.45
AutoGateMargin = 0.16
FlatMinResponse = 0.0
FlatEchoGain = 2.0
ReflectanceMode = hash-log-uniform
```

This is a more physical alternative to the v5 residual floor. The neighboring-gate brightness comes from the overlap of the rectangular laser pulse and rectangular receiver gate, not from a manually imposed minimum response.

## 2. Dataset evidence

Dataset roots:

```text
dataset_new/Military_3D_Gated_Selected44_blender_refl_overlap_w15_m016_v7
dataset_new/Military_FlatFalse_Selected44_blender_refl_overlap_w15_m016_v7
dataset_new/Military_TrueFalse_Selected44_blender_refl_overlap_w15_m016_v7
```

Metadata check:

| item | value |
|---|---:|
| true samples | 44 |
| flat false samples | 44 |
| gate images per class | 132 |
| flat false metadata files | 44 |
| flat target gate distribution | 15 / 15 / 14 |
| max flattened depth span | 0 |
| reflectance range | 0.6067 to 2.7272 |
| non-hit zero response fraction | 0.2955 |

The non-hit zero response fraction is much lower than the pure v6 setting with the original narrow gate layout, so v7 reduces the black-frame problem.

## 3. Gate-stack diagnostics

| class | corr | mask IoU | absdiff | max/mean ratio |
|---|---:|---:|---:|---:|
| flat_false | 0.6427 | 0.6158 | 0.1007 | 88.4508 |
| true3d | 0.7325 | 0.7273 | 0.0792 | 9.2153 |

Interpretation:

- Positive: v7 avoids the extremely sparse false-target stack seen in v6.
- Negative: true3d gate stacks become too correlated. The wider/closer gates smooth the 3D depth slicing, so the physical contrast between true 3D slicing and planar echo consistency is weaker than in v5.

This means v7 is not a clean final physics setting, but it is an important parameter study: it shows that gate overlap must be tuned, not simply maximized.

## 4. Three-seed network ablation

Attention fusion:

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

- v7 makes some single-gate runs harder, especially gate 0.
- Mean and attention_residual full-stack fusion are more stable than plain attention.
- However, gate 2 can still reach high accuracy in some seeds, including 1.0000 for seed 332. This remains a shortcut risk.

## 5. Current best paper statement

Safe wording:

```text
We introduce a Blender-based planar false-target simulator with metadata-controlled camera-depth flattening, deterministic reflectance variation, and rectangular gate-response modeling. Parameter studies show that the gate response cannot be tuned in isolation: a narrow non-overlapping gate creates black-frame shortcuts, while excessive overlap smooths the true 3D depth slicing. A moderate overlap setting improves the network-level full-stack behavior under mean/attention-residual fusion, but single-gate controls remain competitive, requiring stricter nuisance control before making a final robustness claim.
```

Do not claim yet:

```text
The final model already proves robust 3D-versus-2D false-target discrimination.
```

## 6. v8 plan

The next version should optimize the trade-off between v5 and v7.

Recommended v8 candidate:

```text
ReceiverGateWidth = 1.3 or 1.5
LaserPulseWidth = 0.45
AutoGateMargin = 0.08 or 0.12
FlatMinResponse = 0.0
ReflectanceMode = hash-log-uniform
```

Rationale:

- v5 preserved true3d gate diversity but used a nonphysical residual floor.
- v6 removed the floor but created black-frame shortcuts.
- v7 reduced black-frame shortcuts but over-smoothed true3d slices.
- v8 should use less overlap than v7 while keeping the pure rectangular response.

Validation requirements for v8:

1. Gate-stack diagnostics must show true3d correlation lower than v7.
2. Flat_false should avoid the v6 black-frame collapse.
3. Full-stack mean or attention_residual should outperform every single-gate input over three seeds.
4. Scalar single-gate threshold accuracy must be reported as a limitation if it remains high.
5. Final 3090 runs should include noise/background and gate-dropout robustness.
