# Current Results Snapshot - 2026-05-27

## 1. Six-class baseline

Dataset:

```text
chair, desk, sofa, bed, toilet, image2d
```

Current single-seed training result from `artifacts/summary.json`:

```text
num_samples = 600
val_ratio = 0.2
val_samples = 120
seed = 42
epochs = 30
fusion_mode = attention
best_val_acc = 0.9417
```

Current artifacts:

```text
artifacts/summary.json
artifacts/training_curves.png
artifacts/training_history.csv
artifacts/best_confusion_matrix.png
artifacts/val_attention_weights.csv
artifacts/attention_mean_by_class.csv
artifacts/attention_mean_by_class.png
artifacts/attention_image2d_active_vs_black.csv
```

## 2. Attention summary

Mean attention by validation class:

| class | gate_0 | gate_1 | gate_2 | max gate |
|---|---:|---:|---:|---|
| chair | 0.3527 | 0.3225 | 0.3248 | gate_0 |
| desk | 0.3129 | 0.3200 | 0.3671 | gate_2 |
| sofa | 0.3532 | 0.3224 | 0.3244 | gate_0 |
| bed | 0.3728 | 0.3183 | 0.3089 | gate_0 |
| toilet | 0.3320 | 0.3281 | 0.3399 | gate_2 |
| image2d | 0.3306 | 0.3452 | 0.3242 | gate_1 |

For the `image2d` class, the attention mechanism gives more average weight to black gates than to the informative gate:

```text
image2d informative gate mean attention = 0.2530
image2d black gates mean attention      = 0.3735
```

Interpretation:

```text
The classifier likely treats the two black slices as discriminative evidence for the image2d abnormal/degenerate class. This is useful for discussion: attention highlights discriminative evidence, not necessarily the human-defined visually informative slice.
```

## 3. New model ablation interface

`model.py`, `train.py`, and `run_experiments.py` now support:

```text
--fusion-mode attention
--fusion-mode mean
--fusion-mode concat
```

Smoke-tested with 1 epoch:

```text
mean fusion:   runs successfully
concat fusion: runs successfully
```

These smoke results are only functional checks and should not be used as paper results.

## 4. Recommended 3090 experiment commands

Run attention baseline with three seeds:

```powershell
"C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_experiments.py --experiment-name six_class_attention --fusion-mode attention
```

Run fusion ablations:

```powershell
"C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_experiments.py --experiment-name six_class_mean --fusion-mode mean
"C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_experiments.py --experiment-name six_class_concat --fusion-mode concat
```

Run 2D single-gate controls:

```powershell
"C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_experiments.py --experiment-name single_gate_g0 --input-mode single-gate --single-gate-index 0
"C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_experiments.py --experiment-name single_gate_g1 --input-mode single-gate --single-gate-index 1
"C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_experiments.py --experiment-name single_gate_g2 --input-mode single-gate --single-gate-index 2
```

Run black-slice controls:

```powershell
"C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_experiments.py --experiment-name single_gate_black_g0 --input-mode single-gate-black --single-gate-index 0
"C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_experiments.py --experiment-name single_gate_black_g1 --input-mode single-gate-black --single-gate-index 1
"C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_experiments.py --experiment-name single_gate_black_g2 --input-mode single-gate-black --single-gate-index 2
```

## 5. Next writing step

The next paper draft update should add:

```text
1. Six-class dataset construction, including image2d generation.
2. Baseline six-class result.
3. Attention finding: black slices can become discriminative evidence.
4. Planned fusion ablation: attention vs mean vs concat.
```
