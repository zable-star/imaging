# Thread Handoff: Physical Gate Ablation and Next Experiments

## Record Time

- Created on: 2026-06-16
- Timezone: Asia/Hong_Kong
- Local project directory: `E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline`
- Lab project directory: `D:\学生文件夹\王剑哲\slice_attention_baseline`
- Lab Python interpreter: `C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe`
- Lab Blender executable used: `D:\学生文件夹\王剑哲\blender\blender.exe`

## Resume Prompt

When opening a new Codex window, paste:

```text
项目目录是 E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline。
请先读取 writing\handoffs\index.md 和 writing\handoffs\handoff_2026-06-16_physical_ablation.md，
然后继续推进 slice attention baseline 项目的物理参数消融和论文整理。
```

If working on the lab machine, use:

```text
项目目录是 D:\学生文件夹\王剑哲\slice_attention_baseline。
Python 解释器是 C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe。
Blender 路径是 D:\学生文件夹\王剑哲\blender\blender.exe。
请根据 writing\handoffs\handoff_2026-06-16_physical_ablation.md 继续实验。
```

## Current Research Direction

The project now has two separate evidence lines:

1. Six-class abnormal-input study:
   - Classes: `chair`, `desk`, `sofa`, `bed`, `toilet`, `image2d`
   - Purpose: prove the network can distinguish normal multi-gate 3D objects from 2D-like false or degenerate targets.

2. Five-class physical-parameter ablation:
   - Classes: `chair`, `desk`, `sofa`, `bed`, `toilet`
   - Purpose: analyze how range-gated imaging parameters affect recognition accuracy and gate-level discriminative contribution.
   - `image2d` should not be included in the main physical ablation, because gate spacing/width changes can make normal 3D gate slices darker and confound the abnormal black-gate pattern.

Use the phrase `gate-level discriminative contribution` for attention weights. Do not describe the current attention module as Transformer/QKV attention.

## Important Code State

| File | Role |
|---|---|
| `model.py` | Defines `mean`, `attention`, `concat`, and `attention_residual` fusion. Current attention is MLP gate-scoring attention, not QKV attention. |
| `origindataset\gated_blender_physical.py` | Blender gated-slice renderer. It supports physical parameters and now includes `--render-device auto|cpu|gpu`. |
| `run_experiments.py` | General multi-seed runner. Defaults to six classes unless `--classes` is supplied. |
| `run_physical_5class_experiments.py` | Thin wrapper for five-class physical ablations; automatically passes `--classes chair desk sofa bed toilet`. |
| `scripts\plot_gate_spacing_results.py` | Python plotting script for the gate-spacing ablation figure. |
| `writing\physical_gate_spacing_results_2026-06-02.md` | Paper-ready summary of gate-spacing results. |
| `writing\publication_next_steps_from_zotero_2026-06-02.md` | Publication execution plan from Zotero/literature insights. |

## Model Clarification

### Current residual fusion

The `attention_residual` branch is enabled by:

```text
--fusion-mode attention_residual
```

In `model.py`, the model computes:

```text
f_att = sum_i alpha_i f_i
f_res = MLP([f_0, f_1, f_2])
f = f_att + f_res
```

This is a fusion-level residual, not a ResNet block.

### Current attention mechanism

The current attention module is:

```text
alpha_i = softmax(MLP(f_i))
f_att = sum_i alpha_i f_i
```

It does not use Q/K/V matrices. This is intentional because each sample currently has only a small number of gate slices, and lightweight MLP attention is easier to interpret and less prone to overfitting. If future experiments use 5/7/9 gates or patch-level gate tokens, QKV self-attention can be introduced as a later variant.

## Blender Renderer Notes

Several lab-side issues were fixed in the local script and should be copied to the lab machine when needed:

1. Removed accidental `matplotlib` and `torch` imports from the Blender script.
2. Fixed damaged encoding/line-break issues where `laser_pulse_width` and `atmospheric_extinction` were swallowed by comments.
3. Added `--render-device cpu` support to avoid Blender 5.1 CUDA kernel failures.
4. Fixed OBJ import logic so it tracks newly imported mesh objects instead of relying on `bpy.context.selected_objects`.

Recommended smoke render command on the lab machine:

```powershell
$Proj = "D:\学生文件夹\王剑哲\slice_attention_baseline"
$Blender = "D:\学生文件夹\王剑哲\blender\blender.exe"
Set-Location $Proj

& $Blender --background --python origindataset\gated_blender_physical.py -- --output-root dataset_gate_spacing_smoke --gate-spacing 0.45 --gate-center-middle 7.4 --models-per-class 5 --render-device cpu
```

The smoke render successfully produced:

```text
bed    15
chair  15
desk   15
sofa   15
toilet 15
```

Visual inspection of `chair\test_chair_0890_gate_0/1/2.png` showed non-black, centered targets with visible gate-dependent depth response.

## Completed Physical Result: Gate Spacing

Five-class `attention_residual` experiments were run for:

```text
small   spacing = 0.45
default spacing = 0.60
large   spacing = 0.90
```

Seeds:

```text
42, 332, 2026
```

Accuracy:

| Gate spacing | Seed 42 | Seed 332 | Seed 2026 | Mean | Std |
|---|---:|---:|---:|---:|---:|
| small | 0.9300 | 0.9100 | 0.9200 | 0.9200 | 0.0100 |
| default | 0.9500 | 0.9000 | 0.9300 | 0.9267 | 0.0252 |
| large | 0.9700 | 0.9300 | 0.9400 | 0.9467 | 0.0208 |

Interpretation:

```text
Large gate spacing currently performs best, suggesting that more separated gate response windows provide stronger complementary depth-selective observations under the current rendering geometry.
```

Overall attention contribution:

| Gate spacing | gate_0 | gate_1 | gate_2 |
|---|---:|---:|---:|
| small | 0.3140 | 0.3404 | 0.3456 |
| default | 0.3211 | 0.3402 | 0.3387 |
| large | 0.3299 | 0.3375 | 0.3326 |

Interpretation:

```text
Small spacing is more biased toward gate_2, while large spacing produces a more balanced gate contribution pattern.
```

Generated figure files:

```text
artifacts\figures\physical_gate_spacing_ablation.svg
artifacts\figures\physical_gate_spacing_ablation.pdf
artifacts\figures\physical_gate_spacing_ablation.png
artifacts\figures\physical_gate_spacing_ablation.tiff
artifacts\figures\physical_gate_spacing_ablation_source.csv
```

Plot script:

```text
scripts\plot_gate_spacing_results.py
```

## Recommended Next Experiment: Number of Gates

The next question is whether increasing the number of gated slices makes attention more useful.

Recommended design:

```text
Fix total depth coverage and vary only num_gates.
```

Use current large-spacing coverage as the reference:

```text
center = 7.4
range approximately [6.5, 8.3]
```

Suggested gate-center sets:

| num_gates | gate centers | Role |
|---:|---|---|
| 1 | `7.4` | single-depth baseline |
| 2 | `6.95,7.85` | two depth windows |
| 3 | `6.5,7.4,8.3` | current large-spacing baseline |
| 5 | `6.5,6.95,7.4,7.85,8.3` | denser depth sampling |
| 7 | optional | redundancy/saturation check |

If time is limited, run only:

```text
num_gates = 1, 3, 5
```

For each gate count, compare at least:

```text
mean
attention_residual
```

If time allows, add:

```text
concat
```

The key paper question is not simply whether more gates improve accuracy, but:

```text
Does attention_residual gain more over mean as the number of gates increases?
```

If yes, this supports the claim that learned gate selection becomes more valuable when more depth-selective observations are available.

## Suggested Lab Commands for num_gates Rendering

Use CPU rendering if Blender CUDA still fails:

```powershell
$Proj = "D:\学生文件夹\王剑哲\slice_attention_baseline"
$Blender = "D:\学生文件夹\王剑哲\blender\blender.exe"
Set-Location $Proj

& $Blender --background --python origindataset\gated_blender_physical.py -- --output-root dataset_num_gates_1 --num-gates 1 --gate-centers 7.4 --models-per-class 0 --render-device cpu

& $Blender --background --python origindataset\gated_blender_physical.py -- --output-root dataset_num_gates_3 --num-gates 3 --gate-centers 6.5,7.4,8.3 --models-per-class 0 --render-device cpu

& $Blender --background --python origindataset\gated_blender_physical.py -- --output-root dataset_num_gates_5 --num-gates 5 --gate-centers 6.5,6.95,7.4,7.85,8.3 --models-per-class 0 --render-device cpu
```

For small smoke tests, change `--models-per-class 0` to `--models-per-class 5` or `20`.

## Suggested Lab Commands for num_gates Training

Smoke example:

```powershell
& "C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_physical_5class_experiments.py -- --experiment-name phys_num_gates_3_attention_residual_smoke --dataset-root dataset_num_gates_3 --expected-num-slices 3 --fusion-mode attention_residual --seeds 42 --epochs 2 --batch-size 16
```

Formal examples:

```powershell
& "C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_physical_5class_experiments.py -- --experiment-name phys_num_gates_1_mean --dataset-root dataset_num_gates_1 --expected-num-slices 1 --fusion-mode mean --seeds 42 332 2026 --epochs 30 --batch-size 16

& "C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_physical_5class_experiments.py -- --experiment-name phys_num_gates_1_attention_residual --dataset-root dataset_num_gates_1 --expected-num-slices 1 --fusion-mode attention_residual --seeds 42 332 2026 --epochs 30 --batch-size 16

& "C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_physical_5class_experiments.py -- --experiment-name phys_num_gates_3_mean --dataset-root dataset_num_gates_3 --expected-num-slices 3 --fusion-mode mean --seeds 42 332 2026 --epochs 30 --batch-size 16

& "C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_physical_5class_experiments.py -- --experiment-name phys_num_gates_3_attention_residual --dataset-root dataset_num_gates_3 --expected-num-slices 3 --fusion-mode attention_residual --seeds 42 332 2026 --epochs 30 --batch-size 16

& "C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_physical_5class_experiments.py -- --experiment-name phys_num_gates_5_mean --dataset-root dataset_num_gates_5 --expected-num-slices 5 --fusion-mode mean --seeds 42 332 2026 --epochs 30 --batch-size 16

& "C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_physical_5class_experiments.py -- --experiment-name phys_num_gates_5_attention_residual --dataset-root dataset_num_gates_5 --expected-num-slices 5 --fusion-mode attention_residual --seeds 42 332 2026 --epochs 30 --batch-size 16
```

Important: always set `--expected-num-slices` to match the dataset gate count.

## Later Experiment: Gate Width

After `num_gates`, run gate-width ablation:

| Setting | receiver_gate_width | laser_pulse_width |
|---|---:|---:|
| narrow | 0.70 | 0.35 |
| default | 1.00 | 0.45 |
| wide | 1.40 | 0.65 |

This will separate the effect of gate response-window breadth from gate spacing.

## Files to Copy Back After Lab Runs

For analysis on the local machine, copy:

```text
experiments\phys_num_gates_*
experiments\results.csv
experiments\aggregate_results.csv
dataset_num_gates_* samples if visual inspection is needed
```

For each run, the most useful files are:

```text
summary.json
training_history.csv
val_attention_weights.csv
best_confusion_matrix.png
training_curves.png
```

## Caution

- The worktree contains many generated experiment directories and figures. Do not revert unrelated files.
- Keep six-class `image2d` experiments separate from five-class physical ablations.
- Do not describe the current attention module as QKV/Transformer attention unless a new QKV model is actually implemented.
- For PointNet comparisons, do not claim speed superiority without FPS/latency measurement. Use cautious language: the gated-slice method avoids explicit point-cloud reconstruction and is potentially compatible with faster imaging pipelines.
