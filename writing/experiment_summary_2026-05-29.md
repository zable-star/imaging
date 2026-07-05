# Experiment Summary - 2026-05-29

Source directory:

```text
experiments/
```

Current formal seed set:

```text
42, 332, 2026
```

Metric:

```text
best validation accuracy
```

Note:

```text
experiments/baseline_seed123 and experiments/baseline_seed2025 are older runs from 2026-05-26.
The current aggregate CSV files use seeds 42, 332, and 2026.
```

## 1. Fusion Ablation

All rows use the six-class dataset:

```text
chair, desk, sofa, bed, toilet, image2d
```

Input mode:

```text
multi: [gate_0, gate_1, gate_2]
```

| Experiment | Fusion | Seed 42 | Seed 332 | Seed 2026 | Mean | Std | Min | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| six_class_attention | attention | 0.9417 | 0.9083 | 0.9333 | 0.9278 | 0.0173 | 0.9083 | 0.9417 |
| six_class_mean | mean | 0.9250 | 0.8917 | 0.9417 | 0.9194 | 0.0255 | 0.8917 | 0.9417 |
| six_class_concat | concat | 0.9667 | 0.9333 | 0.9583 | 0.9528 | 0.0173 | 0.9333 | 0.9667 |

Key observation:

```text
Concat fusion currently performs best: 95.28% mean accuracy.
Attention fusion is second: 92.78%.
Mean fusion is slightly lower: 91.94%.
```

Interpretation:

```text
The multi-slice representation is strong, but the current attention module is not the highest-accuracy fusion method.
Concat fusion has a larger classifier input and may preserve more slice-specific information.
Attention remains valuable for interpretability, especially for analyzing which gate contributes to decisions.
```

## 2. 2D Single-Gate Controls

These experiments test whether a single 2D slice is sufficient.

| Experiment | Input Mode | Gate | Seed 42 | Seed 332 | Seed 2026 | Mean | Std | Gap vs multi-attention |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| six_class_attention | multi | all | 0.9417 | 0.9083 | 0.9333 | 0.9278 | 0.0173 | 0.0000 |
| single_gate_g0 | single-gate | 0 | 0.8000 | 0.7583 | 0.8500 | 0.8028 | 0.0459 | -0.1250 |
| single_gate_g1 | single-gate | 1 | 0.8250 | 0.8167 | 0.8250 | 0.8222 | 0.0048 | -0.1056 |
| single_gate_g2 | single-gate | 2 | 0.8417 | 0.8167 | 0.8667 | 0.8417 | 0.0250 | -0.0861 |

Key observation:

```text
The best single-gate result is gate_2 at 84.17%.
The full multi-slice attention model reaches 92.78%.
The multi-slice model is 8.61 percentage points higher than the best single-gate baseline.
```

Interpretation:

```text
A single 2D slice contains useful classification information, but it does not match the full multi-depth input.
This supports the claim that multiple gated slices provide complementary information.
```

## 3. Black-Slice Controls

These experiments keep the three-slice tensor shape, but only one gate contains image information and the other two gates are black.

| Experiment | Input Mode | Informative Gate | Seed 42 | Seed 332 | Seed 2026 | Mean | Std | Gap vs multi-attention |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| single_gate_black_g0 | single-gate-black | 0 | 0.8250 | 0.7500 | 0.8667 | 0.8139 | 0.0591 | -0.1139 |
| single_gate_black_g1 | single-gate-black | 1 | 0.7833 | 0.7750 | 0.7833 | 0.7806 | 0.0048 | -0.1472 |
| single_gate_black_g2 | single-gate-black | 2 | 0.8500 | 0.8167 | 0.8333 | 0.8333 | 0.0167 | -0.0944 |

Key observation:

```text
Black-slice controls remain below the full multi-slice model.
The best black-slice control is gate_2 at 83.33%, still 9.44 percentage points below multi-slice attention.
```

Comparison with true single-gate inputs:

| Gate | Single-Gate Mean | Black-Slice Mean | Difference |
|---:|---:|---:|---:|
| 0 | 0.8028 | 0.8139 | +0.0111 |
| 1 | 0.8222 | 0.7806 | -0.0417 |
| 2 | 0.8417 | 0.8333 | -0.0083 |

Interpretation:

```text
Adding black placeholder slices does not consistently improve performance over true single-gate input.
The black-slice pattern can be discriminative for the image2d class, but it does not replace the information in real multi-depth slices.
```

## 4. Main Conclusions

1. Multi-slice input is clearly better than any single-slice 2D control.

```text
multi attention: 92.78%
best single-gate: 84.17%
best black-slice: 83.33%
```

2. Concat fusion is the strongest classifier in current results.

```text
concat: 95.28%
attention: 92.78%
mean: 91.94%
```

3. Attention is still useful for interpretation.

```text
Even if concat is more accurate, attention provides gate-level weights that help explain how the model uses depth-selective slices.
```

4. The 2D abnormal class is learnable.

```text
The six-class models reach high accuracy, showing that image2d samples are distinguishable from the five 3D object classes.
```

## 5. Suggested Paper Tables

Table 1:

```text
Six-class baseline: attention / mean / concat fusion ablation.
```

Table 2:

```text
Multi-slice vs single-gate 2D controls.
```

Table 3:

```text
Single-gate vs black-slice controls.
```

Suggested discussion point:

```text
The current attention model is interpretable but not the highest-accuracy fusion strategy. This motivates reporting both accuracy-oriented concat fusion and interpretable attention fusion, or improving the attention module in the next model iteration.
```

## 5.1 Next Network Variant

Implemented after the initial fusion ablation:

```text
--fusion-mode attention_residual
```

Motivation:

```text
Pure attention fusion compresses all slice features into one 128-dimensional vector.
Concat fusion preserves slice-specific information and obtains the highest accuracy, but is less interpretable.
The attention_residual mode keeps learned attention weights, then adds a projected concat residual back to the attention-fused feature.
```

Model form:

```text
attention_fused = sum_i alpha_i * f_i
residual = MLP([f_0, f_1, f_2])
fused = attention_fused + residual
```

Smoke test:

```text
1 epoch run succeeded.
This smoke result is not a paper result.
```

Recommended formal run:

```powershell
"C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_experiments.py --experiment-name six_class_attention_residual --fusion-mode attention_residual --experiment-root experiments\six_class_attention_residual
```

Expected role:

```text
If attention_residual approaches concat while retaining attention weights, it can become the improved main network.
If not, concat remains the accuracy baseline and attention remains the interpretable baseline.
```

Seed-matched formal result:

```text
seeds = 42, 332, 2026
attention_residual mean accuracy = 0.9472
std = 0.0127
min = 0.9333
max = 0.9583
```

Seed-level comparison:

| Method | Seed 42 | Seed 332 | Seed 2026 | Mean | Std |
|---|---:|---:|---:|---:|---:|
| attention | 0.9417 | 0.9083 | 0.9333 | 0.9278 | 0.0173 |
| attention_residual | 0.9583 | 0.9333 | 0.9500 | 0.9472 | 0.0127 |
| concat | 0.9667 | 0.9333 | 0.9583 | 0.9528 | 0.0173 |

Updated interpretation:

```text
attention_residual improves over the original attention model on every matched seed.
It slightly trails concat in mean accuracy, but keeps interpretable attention weights.
Therefore, attention_residual is a stronger interpretable main-network candidate, while concat remains a high-accuracy empirical baseline among the tested fusion methods.
```

## 6. Generated Figures

Generated by:

```powershell
python plot_experiment_results.py
```

Output directory:

```text
artifacts/figures/
```

Figures:

```text
artifacts/figures/fusion_ablation_accuracy.png
artifacts/figures/single_gate_controls_accuracy.png
artifacts/figures/black_slice_controls_accuracy.png
artifacts/figures/seed_stability_accuracy.png
```

Supporting CSV:

```text
artifacts/figures/experiment_plot_summary.csv
```

Recommended usage:

```text
fusion_ablation_accuracy.png:
  Use as the model/fusion ablation figure. It shows concat > attention > mean.

single_gate_controls_accuracy.png:
  Use as the main evidence that multi-slice input outperforms any single 2D gate.

black_slice_controls_accuracy.png:
  Use to compare true 2D single-gate input and black-slice placeholder input.

seed_stability_accuracy.png:
  Use in supplementary material or internal reports to show seed-level stability.
```
