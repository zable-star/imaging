# v8 anti-shortcut and robustness report

This report records the v8 follow-up experiments for the selected 44 military models.
The goal is to turn the false-target simulation from a visually plausible demo into a more paper-grade validation chain.

## 1. Physical setting

v8 keeps the pure rectangular-overlap response while reducing gate overlap relative to v7:

```text
ReceiverGateWidth = 1.5
LaserPulseWidth = 0.45
AutoGateMargin = 0.12
FlatMinResponse = 0.0
FlatEchoGain = 2.0
ReflectanceMode = hash-log-uniform
ReflectanceMin = 0.6
ReflectanceMax = 2.8
FlatGeometryMode = flatten-camera-depth
FlatTargetGateIndexMode = round-robin
```

Dataset roots:

```text
dataset_new/Military_3D_Gated_Selected44_blender_refl_overlap_w15_m012_v8
dataset_new/Military_FlatFalse_Selected44_blender_refl_overlap_w15_m012_v8
dataset_new/Military_TrueFalse_Selected44_blender_refl_overlap_w15_m012_v8
```

Generated manuscript figures for the simulation mechanism:

```text
writing/figures/fig1_gated_imaging_framework.png
writing/figures/fig2_rectangular_overlap_response.png
writing/figures/fig3_true3d_flatfalse_gate_examples.png
writing/figures/fig4_scalar_shortcut_control.png
writing/figures/source_manifest_v8_paper_figures.csv
```

Important physical interpretation: with rectangular laser pulse and rectangular receiver gate, the per-gate response is determined by temporal overlap length. It is triangular when the two windows have equal width and trapezoidal when one window is wider. Therefore, a planar false target should not be modeled as an arbitrary linear fade; it should preserve the whole-object silhouette and sample the overlap response across gates.

Readiness:

| class | samples | gate images | ready |
|---|---:|---:|---|
| true3d | 44 | 132 | True |
| flat_false | 44 | 132 | True |

## 2. v8 raw gate-stack diagnostics

| class | samples | corr | mask IoU | absdiff | max/mean ratio |
|---|---:|---:|---:|---:|---:|
| flat_false | 44 | 0.5848 | 0.5702 | 0.1177 | 103.8464 |
| true3d | 44 | 0.6736 | 0.6670 | 0.0934 | 9.3942 |

Compared with v7, v8 lowers true3d gate correlation from about 0.73 to 0.67, so the true 3D slices are less over-smoothed.
However, scalar single-gate shortcuts remain strong in the raw images.

Strongest raw single-gate scalar shortcuts:

| gate | strongest scalar feature | threshold acc | true3d mean | flat false mean | Cohen d |
|---:|---|---:|---:|---:|---:|
| 0 | max_value | 0.8636 | 0.3426 | 0.1440 | -2.041 |
| 1 | p95 | 0.8864 | 0.3177 | 0.1597 | -2.291 |
| 2 | p99 | 0.9886 | 0.3055 | 0.1103 | -2.886 |

Raw seed42 network ablation:

| input | best val acc |
|---|---:|
| Full 3-gate stack | 0.9091 |
| Gate 0 only | 0.6818 |
| Gate 1 only | 0.8636 |
| Gate 2 only | 0.9545 |

Interpretation: v8 is physically cleaner than a residual brightness floor, but raw images still allow a gate2 intensity shortcut.

## 3. Per-gate max-normalized anti-shortcut control

A control dataset was created with:

```text
dataset_new/normalize_gate_dataset.py
--mode per-gate-max
--target-max 180
--min-source-max 2
```

Output:

```text
dataset_new/Military_TrueFalse_Selected44_blender_refl_overlap_w15_m012_v8_per_gate_maxnorm
```

After per-gate max normalization, the strongest scalar single-gate shortcut drops from `0.9886` to `0.7955`, and the dominant feature changes from intensity (`max/p99`) to shape/edge density.

| gate | strongest scalar feature | threshold acc | true3d mean | flat false mean | Cohen d |
|---:|---|---:|---:|---:|---:|
| 0 | edge_density | 0.7955 | 0.3664 | 0.1357 | -1.398 |
| 1 | p99 | 0.7386 | 0.6874 | 0.6004 | -0.573 |
| 2 | edge_density | 0.7841 | 0.3499 | 0.1559 | -1.194 |

Seed42 network ablation on the normalized control:

| input | best val acc |
|---|---:|
| Full 3-gate stack | 1.0000 |
| Gate 0 only | 0.9545 |
| Gate 1 only | 0.9091 |
| Gate 2 only | 0.9545 |

Interpretation: normalization suppresses simple brightness shortcuts, but single-gate CNNs can still use slice-shape differences. This should be described as a useful control, not as final proof that only gate-stack consistency is used.

## 4. Training/evaluation separation for robustness

A new script was added:

```text
scripts/evaluate_gate_model.py
```

Purpose:

```text
Load a trained best_model.pth, reconstruct the same seed/validation split, and evaluate clean or degraded test conditions without retraining.
```

`train.py` was also extended with:

```text
--degradation-probability
```

This enables deterministic clean/noisy mixed augmentation. Default is `1.0`, so old experiments are unchanged.

The ablation runner was updated to expose the same degradation controls:

```text
scripts/run_blender_refl_v5_ablation.ps1
```

Example 3090-ready command:

```powershell
foreach ($seed in @(42, 332, 2026)) {
  powershell -ExecutionPolicy Bypass -File scripts\run_blender_refl_v5_ablation.ps1 `
    -DatasetRoot dataset_new\Military_TrueFalse_Selected44_blender_refl_overlap_w15_m012_v8_per_gate_maxnorm `
    -ExperimentTag blender_refl_overlap_w15_m012_v8_per_gate_maxnorm_mixaug_p05 `
    -Seeds $seed `
    -Epochs 20 `
    -BatchSize 8 `
    -FusionMode attention `
    -GaussianNoiseStd 0.02 `
    -PoissonPeak 80 `
    -BackgroundScatter 0.02 `
    -DegradationProbability 0.5 `
    -UseAmp `
    -CudnnBenchmark
}
```

Test status:

```text
60 passed
```

## 5. Robustness evidence

Evaluation summary:

```text
experiments/v8_per_gate_maxnorm_mixaug_eval_summary_3seed.csv
experiments/v8_per_gate_maxnorm_mixaug_eval_aggregate_3seed.csv
```

Conditions:

| condition | gaussian | poisson peak | background scatter |
|---|---:|---:|---:|
| clean | 0.00 | 0 | 0.00 |
| light noise | 0.02 | 80 | 0.02 |
| strong noise | 0.05 | 30 | 0.05 |

### Clean-trained full-stack model

| evaluation condition | full-stack acc |
|---|---:|
| clean | 1.0000 |
| light noise | 0.5000 |
| strong noise | 0.5000 |

The clean model is not robust to even mild test-time noise/domain shift.

### Fully noisy-trained full-stack model

| evaluation condition | full-stack acc |
|---|---:|
| clean | 0.6364 |
| light noise | 1.0000 |
| strong noise | 0.5455 |

The fully noisy model over-specializes to the noisy distribution and loses clean-domain performance.

### Mixed clean/noisy augmented models, seed42

The mixed setting used:

```text
gaussian_noise_std = 0.02
poisson_peak = 80
background_scatter = 0.02
degradation_probability = 0.5
```

| input | clean | light noise | strong noise |
|---|---:|---:|---:|
| Full 3-gate stack | 1.0000 | 1.0000 | 0.7727 |
| Gate 0 only | 0.9091 | 0.9091 | 0.8182 |
| Gate 1 only | 0.6818 | 0.4545 | 0.5455 |
| Gate 2 only | 0.9091 | 0.8636 | 0.5455 |

Interpretation:

- Mixed augmentation is the first setting here that keeps full-stack performance high on both clean and light-noise validation.
- Full-stack outperforms every single gate under clean and light-noise evaluation.
- Strong-noise evidence is not yet conclusive: gate0 is slightly higher than full-stack in this seed, while gate1/gate2 collapse. This must be checked with more seeds and possibly better fusion.

### Mixed clean/noisy augmented models, three seeds

Training aggregate:

```text
experiments/localgpu_blender_refl_overlap_w15_m012_v8_per_gate_maxnorm_mixaug_p05_g002_p80_b002_ablation_aggregate.csv
```

| input | mean best val acc | std | min | max | seeds |
|---|---:|---:|---:|---:|---|
| Full 3-gate stack | 0.9697 | 0.0525 | 0.9091 | 1.0000 | 42/332/2026 |
| Gate 0 only | 0.9091 | 0.0909 | 0.8182 | 1.0000 | 42/332/2026 |
| Gate 1 only | 0.7273 | 0.0909 | 0.6364 | 0.8182 | 42/332/2026 |
| Gate 2 only | 0.8939 | 0.0525 | 0.8636 | 0.9545 | 42/332/2026 |

Independent evaluation aggregate:

| evaluation condition | input | mean acc | std | min | max |
|---|---|---:|---:|---:|---:|
| clean | Full 3-gate stack | 0.9697 | 0.0525 | 0.9091 | 1.0000 |
| clean | Gate 0 only | 0.9091 | 0.0909 | 0.8182 | 1.0000 |
| clean | Gate 1 only | 0.7121 | 0.0263 | 0.6818 | 0.7273 |
| clean | Gate 2 only | 0.8636 | 0.0455 | 0.8182 | 0.9091 |
| light noise | Full 3-gate stack | 0.9545 | 0.0455 | 0.9091 | 1.0000 |
| light noise | Gate 0 only | 0.8636 | 0.1202 | 0.7273 | 0.9545 |
| light noise | Gate 1 only | 0.5151 | 0.0694 | 0.4545 | 0.5909 |
| light noise | Gate 2 only | 0.7727 | 0.0909 | 0.6818 | 0.8636 |
| strong noise | Full 3-gate stack | 0.7424 | 0.1389 | 0.5909 | 0.8636 |
| strong noise | Gate 0 only | 0.6970 | 0.1144 | 0.5909 | 0.8182 |
| strong noise | Gate 1 only | 0.5152 | 0.0263 | 0.5000 | 0.5455 |
| strong noise | Gate 2 only | 0.5152 | 0.0263 | 0.5000 | 0.5455 |

Generated figure:

```text
writing/figures/fig5_full_stack_vs_single_gate_robustness.png
writing/figures/fig5_full_stack_vs_single_gate_robustness.pdf
```

Interpretation:

- The full 3-gate stack has the highest mean accuracy in all three independent evaluation conditions.
- The strongest single gate is gate0, but it is lower than full-stack on average under clean, light-noise, and strong-noise evaluation.
- The three-seed result is now strong enough to use as a main controlled simulation result, while still requiring larger/multi-view datasets before making deployment claims.

### Full-stack fusion comparison

Fusion aggregate:

```text
experiments/localgpu_blender_refl_overlap_w15_m012_v8_per_gate_maxnorm_full_fusion_mixaug_p05_g002_p80_b002_aggregate.csv
experiments/v8_per_gate_maxnorm_full_fusion_mixaug_eval_summary_3seed.csv
experiments/v8_per_gate_maxnorm_full_fusion_mixaug_eval_aggregate_3seed.csv
```

| fusion mode | mean best val acc | std | min | max | seeds |
|---|---:|---:|---:|---:|---|
| attention | 0.9697 | 0.0525 | 0.9091 | 1.0000 | 42/332/2026 |
| mean | 0.9545 | 0.0788 | 0.8636 | 1.0000 | 42/332/2026 |
| attention_residual | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 42/332/2026 |

Independent evaluation aggregate:

| evaluation condition | fusion mode | mean acc | std | min | max |
|---|---|---:|---:|---:|---:|
| clean | attention | 0.9697 | 0.0525 | 0.9091 | 1.0000 |
| clean | mean | 0.9545 | 0.0788 | 0.8636 | 1.0000 |
| clean | attention_residual | 0.9848 | 0.0263 | 0.9545 | 1.0000 |
| light noise | attention | 0.9545 | 0.0455 | 0.9091 | 1.0000 |
| light noise | mean | 0.9697 | 0.0525 | 0.9091 | 1.0000 |
| light noise | attention_residual | 0.9545 | 0.0455 | 0.9091 | 1.0000 |
| strong noise | attention | 0.7424 | 0.1389 | 0.5909 | 0.8636 |
| strong noise | mean | 0.6515 | 0.0694 | 0.5909 | 0.7273 |
| strong noise | attention_residual | 0.6970 | 0.1721 | 0.5000 | 0.8182 |

Generated figure:

```text
writing/figures/fig6_full_stack_fusion_robustness.png
writing/figures/fig6_full_stack_fusion_robustness.pdf
```

Interpretation: fusion choice changes the robustness trade-off. `attention_residual` is the strongest validation candidate and has the best clean independent accuracy, `mean` is slightly better under light noise, and the simpler `attention` fusion is strongest under strong noise. For the current SCI story, the safest main statement is that the full gate stack is useful, while the final fusion head remains an engineering choice that should be rechecked after multi-view expansion.

## 6. Paper-use conclusion

Safe claims supported by current evidence:

1. The Blender pipeline can generate metadata-proven true 3D and planar false-target gated sequences.
2. Pure rectangular gate overlap is physically interpretable, but parameter choices can introduce black-frame or intensity shortcuts.
3. Per-gate normalization and single-gate ablations are necessary controls before claiming gate-stack discrimination.
4. Mixed clean/noisy augmentation improves the robustness story and provides a clearer reason for using the full gate stack under clean, light-noise, and strong-noise controlled evaluations.
5. Fusion-mode comparison shows a validation/robustness trade-off: `attention_residual` gives the best validation and clean-test result, while `attention` is more robust in the current strong-noise test.

Claims not yet safe:

1. Do not claim deployment-ready military target discrimination.
2. Do not claim that the current model always relies on multi-gate depth consistency.
3. Do not claim general strong-noise robustness outside this controlled simulation setting.

Next required experiments:

1. Add multi-view renders to reduce fixed top-view shape shortcuts.
2. Repeat the full-stack/single-gate/fusion comparison on the multi-view dataset.
3. Move the final larger multi-view and fusion runs to the 24 GB RTX 3090 machine.
