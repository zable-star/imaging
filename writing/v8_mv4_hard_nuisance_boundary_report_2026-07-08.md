# v8 mv4 hard-nuisance boundary report

This report records an additional stress test beyond clean/noisy evaluation. The goal is not to claim improved performance. The goal is to identify whether the current clean/noisy-trained models generalize to more realistic nuisance changes.

## Dataset construction

Source dataset:

```text
dataset_new/Military_TF_v8_mv4_norm
```

Hard-nuisance variants:

```text
dataset_new/Military_TF_v8_mv4_hard_nuisance_v2
dataset_new/Military_TF_v8_mv4_hard_nuisance_v3_mild
```

Both variants apply deterministic nuisance transformations to both `true3d` and `flat_false` classes:

- Low-frequency spatial reflectance texture.
- Weak background scatter.
- Partial rectangular occlusion.
- Same nuisance key for paired true/false samples with the same source model ID.
- Per-sample maximum preservation to avoid introducing a new max-brightness shortcut.

The v3 variant uses milder parameters than v2.

## Dataset readiness and diagnostics

v2:

```text
true3d valid samples = 176
flat_false valid samples = 176
low_contrast_images = 0
mean max gray: true3d = 180, flat_false = 180
```

v3 mild:

```text
true3d valid samples = 176
flat_false valid samples = 176
low_contrast_images = 0
mean max gray: true3d = 180, flat_false = 180
```

Gate-stack diagnostics:

| dataset | class | corr maxnorm | mask IoU | absdiff maxnorm |
|---|---|---:|---:|---:|
| v2 | flat_false | 0.5962 | 0.6435 | 0.1645 |
| v2 | true3d | 0.6823 | 0.6555 | 0.0618 |
| v3 mild | flat_false | 0.5859 | 0.6545 | 0.1643 |
| v3 mild | true3d | 0.6753 | 0.6715 | 0.0779 |

## Independent evaluation

The saved four-view v8 attention models trained on `Military_TF_v8_mv4_norm` were evaluated directly on the hard-nuisance datasets. The validation split remains model-level grouped.

| condition | full stack | gate0 | gate1 | gate2 |
|---|---:|---:|---:|---:|
| hard_nuisance_v2 | 0.5000 | 0.4697 | 0.5000 | 0.4545 |
| hard_nuisance_v3_mild | 0.5000 | 0.4848 | 0.5000 | 0.4545 |

Figure:

```text
writing/figures/fig9_hard_nuisance_failure_boundary.png
writing/figures/fig9_hard_nuisance_failure_boundary.pdf
```

## Interpretation

This result is a failure-boundary result.

The previous clean/light/strong-noise experiments show that the full gate stack is more robust than single-gate inputs under additive noise and background scatter used during mixed augmentation. However, the hard-nuisance datasets introduce spatial reflectance changes, preserved-max background shifts, and partial occlusion. Under this distribution shift, the current clean/noisy-trained models collapse to approximately chance accuracy.

This should be written as a limitation and next-step motivation:

```text
The current framework validates the physical value of gate-stack observations under controlled v8 simulation and noise perturbations, but it does not yet solve domain shift caused by structured reflectance, background, and occlusion changes. Future work should incorporate these nuisance factors directly into the simulator and training protocol.
```

Do not write:

```text
The method is robust to realistic reflectance, background, and occlusion changes.
```

## Files

```text
dataset_new/build_hard_nuisance_dataset.py
scripts/run_v8_mv4_hard_nuisance_eval.ps1
scripts/make_v8_hard_nuisance_boundary_figure.py
dataset_new/Military_TF_v8_mv4_hard_nuisance_v2_readiness.csv
dataset_new/Military_TF_v8_mv4_hard_nuisance_v2_quality.csv
dataset_new/Military_TF_v8_mv4_hard_nuisance_v2_gate_stack_classes.csv
dataset_new/Military_TF_v8_mv4_hard_nuisance_v3_mild_readiness.csv
dataset_new/Military_TF_v8_mv4_hard_nuisance_v3_mild_quality.csv
dataset_new/Military_TF_v8_mv4_hard_nuisance_v3_mild_gate_stack_classes.csv
experiments/v8_mv4_hard_nuisance_v2_eval_aggregate_3seed.csv
experiments/v8_mv4_hard_nuisance_v3_mild_eval_aggregate_3seed.csv
```
