# Multi-view v8 validation plan

Draft date: 2026-07-07

## Purpose

The current main result is based on a selected 44-model, single-view, controlled Blender simulation. This is enough to support a framework-level claim, but not enough for a strong SCI claim about target-pose robustness. The next validation step is therefore a multi-view extension of the v8 rectangular-overlap dataset.

## Implemented tools

New scripts:

```text
scripts/run_v8_multiview_dataset.ps1
dataset_new/build_multiview_true_false_dataset.py
```

Updated script:

```text
scripts/render_selected_military_gates.ps1
```

The renderer now exposes:

```text
-ModelRotationDeg "0,0,<angle>"
```

This uses the existing Blender `--model-rotation-deg` argument to create top-view azimuth changes without rewriting the Blender camera model.

## Multi-view construction

Default view set:

```text
view_z000
view_z090
view_z180
view_z270
```

Physical setting:

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

Expected full dataset scale:

| item | count |
|---|---:|
| selected military models | 44 |
| views per model | 4 |
| binary classes | 2 |
| gate images per sample | 3 |
| true3d samples | 176 |
| flat_false samples | 176 |
| total samples | 352 |
| total gate PNGs | 1056 |

## Smoke validation already completed

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_v8_multiview_dataset.ps1 `
  -ModelsPerClass 1 `
  -ViewRotationsZ 0 `
  -TrueOutputRoot .\dataset_new\_smoke_Military_3D_Gated_v8_mv `
  -FalseOutputRoot .\dataset_new\_smoke_Military_FlatFalse_v8_mv `
  -CombinedOutputRoot .\dataset_new\_smoke_Military_TrueFalse_v8_mv `
  -NormalizedOutputRoot .\dataset_new\_smoke_Military_TrueFalse_v8_mv_per_gate_maxnorm
```

Smoke outputs:

```text
dataset_new/_smoke_Military_3D_Gated_v8_mv
dataset_new/_smoke_Military_FlatFalse_v8_mv
dataset_new/_smoke_Military_TrueFalse_v8_mv
dataset_new/_smoke_Military_TrueFalse_v8_mv_per_gate_maxnorm
dataset_new/_smoke_Military_TrueFalse_v8_mv_per_gate_maxnorm_readiness.csv
```

Readiness result:

| class | gate pngs | valid samples | ready |
|---|---:|---:|---|
| true3d | 9 | 3 | True |
| flat_false | 9 | 3 | True |

Interpretation: the multi-view data pipeline is operational. This smoke result is not a training result and should not be used as paper evidence for performance. It only proves that rendering, view-labeled merging, and per-gate normalization work end to end.

## Full dataset command

Recommended on the lab RTX 3090 machine:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_v8_multiview_dataset.ps1 `
  -ViewRotationsZ 0,90,180,270 `
  -ModelsPerClass 0 `
  -RenderDevice gpu
```

If running on the current local machine, use CPU or a small subset first:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_v8_multiview_dataset.ps1 `
  -ViewRotationsZ 0,90 `
  -ModelsPerClass 2 `
  -RenderDevice cpu
```

## Required experiments after full rendering

Use the normalized multi-view dataset:

```text
dataset_new/Military_TrueFalse_Selected44_blender_refl_overlap_w15_m012_v8_mv4_per_gate_maxnorm
```

Minimum experiment set:

1. Full 3-gate stack, attention fusion, seeds `42/332/2026`.
2. Single-gate ablations: gate0, gate1, gate2, seeds `42/332/2026`.
3. Fusion comparison: attention, mean, attention_residual, seeds `42/332/2026`.
4. Independent clean/light/strong-noise evaluation using `scripts/evaluate_gate_model.py`.

Recommended training command pattern:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_blender_refl_v5_ablation.ps1 `
  -DatasetRoot dataset_new\Military_TrueFalse_Selected44_blender_refl_overlap_w15_m012_v8_mv4_per_gate_maxnorm `
  -ExperimentTag blender_refl_overlap_w15_m012_v8_mv4_per_gate_maxnorm_mixaug_p05 `
  -Seeds 42,332,2026 `
  -Epochs 20 `
  -BatchSize 8 `
  -FusionMode attention `
  -GaussianNoiseStd 0.02 `
  -PoissonPeak 80 `
  -BackgroundScatter 0.02 `
  -DegradationProbability 0.5 `
  -UseAmp `
  -CudnnBenchmark
```

## Paper claim after multi-view validation

If the full stack remains better than single-gate inputs after multi-view expansion, the paper can strengthen the result from:

```text
single-view controlled simulation evidence
```

to:

```text
pose-augmented controlled simulation evidence
```

Still avoid claiming deployment-ready recognition unless real sensor data or a much broader military model dataset is added.

