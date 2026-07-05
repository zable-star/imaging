# Thread Handoff: Attention-Residual Experiments

## Record Time

- Created on: 2026-06-02
- Timezone: Asia/Hong_Kong
- Local project directory: `E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline`
- Lab project directory: `D:\学生文件夹\王剑哲\slice_attention_baseline`
- Lab Python interpreter: `C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe`

## Resume Prompt

When opening a new Codex window, paste:

```text
项目目录是 E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline。
请先读取 writing\handoffs\index.md 和 writing\handoffs\handoff_2026-06-02_attention_residual.md，
然后继续推进 slice attention baseline 项目。
```

If working on the lab machine, paste:

```text
项目目录是 D:\学生文件夹\王剑哲\slice_attention_baseline。
Python 解释器是 C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe。
请先读取 writing\handoffs\index.md 和 writing\handoffs\handoff_2026-06-02_attention_residual.md，
后续终端命令都使用上述实验室目录和 Python 解释器。
```

## Current Research Goal

The project studies target recognition from laser range-gated simulation slices. Each sample contains three gated slice images. The current paper direction is to show that multi-slice gated input is useful, that attention can provide interpretable slice weights, and that an improved attention-residual fusion gives better accuracy while preserving interpretability.

The dataset currently has six classes:

```text
chair
desk
sofa
bed
toilet
image2d
```

The `image2d` class is an abnormal/degenerate input class. Each `image2d` sample has one informative slice and two all-black slices. It was added to test whether the model can distinguish normal 3D gated-slice inputs from 2D-like abnormal inputs.

## Important Code Files

| File | Role |
|---|---|
| `model.py` | Defines the slice encoder and fusion modes: `mean`, `attention`, `concat`, `attention_residual` |
| `train.py` | Main training script; supports multi-slice and single-gate controls |
| `run_experiments.py` | Multi-seed experiment runner |
| `plot_experiment_results.py` | Generates experiment comparison figures in `artifacts\figures` |
| `analyze_attention_scores.py` | Summarizes validation attention weights and produces attention heatmaps |
| `make_image2d_class.py` | Builds the `image2d` abnormal class |
| `origindataset\gated_blender_physical.py` | Generates physically motivated gated slices; now supports gate parameter ablations |
| `writing\paper_experiment_section_attention_residual.md` | Latest paper-style experiment section |
| `writing\paper_draft_attention_residual.md` | New unified Chinese paper draft using the latest six-class attention-residual results |
| `writing\publication_next_steps_from_zotero_2026-06-02.md` | Publication-oriented next experiment plan based on Zotero gated-imaging insights |

## Current Model Design

The most important model variant is `attention_residual`.

Original attention fusion:

```text
f_att = sum_i alpha_i f_i
```

Attention-residual fusion:

```text
f_att = sum_i alpha_i f_i
f_res = MLP([f_0, f_1, f_2])
f = f_att + f_res
```

Interpretation:

- `attention` keeps explicit gate weights but compresses the three slice features into one weighted sum.
- `concat` keeps the most feature information but does not naturally provide interpretable gate weights.
- `attention_residual` is the current recommended main model because it improves attention while preserving gate-weight interpretability.

## Main Experiment Results

The formal fusion comparison uses matched seeds:

```text
42, 332, 2026
```

| Fusion mode | Seed 42 | Seed 332 | Seed 2026 | Mean accuracy | Std |
|---|---:|---:|---:|---:|---:|
| mean | 92.50% | 89.17% | 94.17% | 91.94% | 2.55% |
| attention | 94.17% | 90.83% | 93.33% | 92.78% | 1.73% |
| attention_residual | 95.83% | 93.33% | 95.00% | 94.72% | 1.27% |
| concat | 96.67% | 93.33% | 95.83% | 95.28% | 1.73% |

Current conclusion:

```text
concat has the highest empirical accuracy among the tested fusion modes.
attention_residual is the preferred main model for the paper because it is close to concat in accuracy and keeps attention interpretability.
```

Do not describe concat as a strict theoretical upper bound. It is only an empirical high-performance baseline among the tested fusion methods.

## Figure Files

Use these figures in the paper/result discussion:

| Figure | Path |
|---|---|
| Fusion ablation | `artifacts\figures\fusion_ablation_accuracy.png` |
| Method architecture | `artifacts\figures\method_architecture_attention_residual_nature.png` / `.svg` / `.pdf` / `.tiff` |
| NN-SVG-style network architecture | `artifacts\figures\slice_attention_residual_nnsvg_style.png` / `.svg` / `.pdf` / `.tiff`; config note at `artifacts\figures\slice_attention_residual_nnsvg_config.md` |
| Gated-slice rendering pipeline | `artifacts\figures\gated_slice_rendering_pipeline_nature.png` / `.svg` / `.pdf` / `.tiff` |
| Attention-residual confusion matrix | `experiments\six_class_attention_residual_seedmatched\six_class_attention_residual_seed42\best_confusion_matrix.png` |
| Mean attention by class | `experiments\six_class_attention_residual_seedmatched\six_class_attention_residual_seed42\attention_mean_by_class.png` |

Latest paper-style section:

```text
writing\paper_experiment_section_attention_residual.md
```

Latest unified draft:

```text
writing\paper_draft_attention_residual.md
```

## Confusion Matrix Result

For `attention_residual`, seed 42:

- Best validation accuracy: 95.83%
- Validation samples: 20 per class

Confusion matrix rows are ground truth and columns are predictions:

| GT \ Pred | chair | desk | sofa | bed | toilet | image2d |
|---|---:|---:|---:|---:|---:|---:|
| chair | 17 | 0 | 1 | 1 | 1 | 0 |
| desk | 0 | 19 | 0 | 1 | 0 | 0 |
| sofa | 0 | 0 | 20 | 0 | 0 | 0 |
| bed | 0 | 0 | 1 | 19 | 0 | 0 |
| toilet | 0 | 0 | 0 | 0 | 20 | 0 |
| image2d | 0 | 0 | 0 | 0 | 0 | 20 |

Key interpretation:

- `image2d` is classified perfectly in this validation split.
- Most remaining errors are from `chair`.
- This supports the claim that the model can distinguish normal 3D gated-slice sequences from 2D abnormal inputs.

## Attention Result

Mean attention by class for `attention_residual`, seed 42:

| Class | gate_0 | gate_1 | gate_2 | Highest gate |
|---|---:|---:|---:|---|
| chair | 0.3238 | 0.3383 | 0.3378 | gate_1 |
| desk | 0.3457 | 0.3366 | 0.3177 | gate_0 |
| sofa | 0.3335 | 0.3347 | 0.3318 | gate_1 |
| bed | 0.3184 | 0.3342 | 0.3474 | gate_2 |
| toilet | 0.3394 | 0.3306 | 0.3301 | gate_0 |
| image2d | 0.3297 | 0.3649 | 0.3054 | gate_1 |

For `image2d`:

| Group | Mean attention |
|---|---:|
| Informative slice | 0.1752 |
| Black slices | 0.4124 |

Important interpretation:

```text
Attention weight means discriminative contribution, not simple visual saliency.
For image2d, the black slices are themselves strong evidence of the abnormal class, so the model assigns high attention to black slices.
```

## Lab Commands

Use these command styles on the lab machine.

Train attention-residual:

```powershell
Set-Location "D:\学生文件夹\王剑哲\slice_attention_baseline"
& "C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" train.py --dataset-root dataset --artifact-dir artifacts --fusion-mode attention_residual --epochs 30 --batch-size 16 --seed 42
```

Run matched-seed attention-residual experiment:

```powershell
Set-Location "D:\学生文件夹\王剑哲\slice_attention_baseline"
& "C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" run_experiments.py --dataset-root dataset --output-root experiments --experiment-name six_class_attention_residual_seedmatched --fusion-mode attention_residual --seeds 42 332 2026 --epochs 30 --batch-size 16
```

Replot experiment figures:

```powershell
Set-Location "D:\学生文件夹\王剑哲\slice_attention_baseline"
& "C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" plot_experiment_results.py --experiments-root experiments --output-dir artifacts\figures
```

Analyze attention scores:

```powershell
Set-Location "D:\学生文件夹\王剑哲\slice_attention_baseline"
& "C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" analyze_attention_scores.py --artifact-dir experiments\six_class_attention_residual_seedmatched\six_class_attention_residual_seed42 --dataset-root dataset
```

## Suggested Next Steps

Progress continued in this window:

1. `writing\paper_experiment_section_attention_residual.md` was polished with a multi-slice vs single-gate control subsection, corrected concat wording, a stronger attention interpretation paragraph, and a physical simulation scope subsection.
2. `writing\experiment_summary_2026-05-29.md` was updated to describe concat as a high-accuracy empirical baseline rather than an upper baseline.
3. `writing\paper_draft_attention_residual.md` was created as a unified Chinese paper draft based on the latest six-class attention-residual results.
4. Nature-style method and rendering-pipeline figures were generated with `scripts\draw_nature_figures.py`. The rendering-pipeline figure uses real dataset thumbnails: normal gated slices from `dataset\chair\test_chair_0890_gate_*.png` and the `image2d` control from `dataset\image2d\image2d_0000_gate_*.png`.

Updated next steps:

1. Continue polishing `writing\paper_draft_attention_residual.md` into the main advisor-facing draft.
2. Use `writing\publication_next_steps_from_zotero_2026-06-02.md` as the current publication execution plan.
3. Run degradation robustness smoke tests with `attention_residual`, then formal three-seed runs for Gaussian noise, Poisson noise, background scatter, far-gate attenuation, and gate dropout.
4. Generate small physical-parameter ablation datasets with `origindataset\gated_blender_physical.py`, especially `gate_spacing` and `receiver_gate_width / laser_pulse_width`.
5. For each new run, compare `attention_residual` against at least `concat` and `mean`; analyze whether attention shifts toward higher-SNR gates under degradation.
6. Keep describing attention weights as `gate-level discriminative contribution`, not visual saliency.

## New Publication-Oriented Interfaces

`origindataset\gated_blender_physical.py` now accepts:

```text
--gate-spacing
--gate-center-middle
--receiver-gate-width
--laser-pulse-width
--range-loss-power
--atmospheric-extinction
```

`train.py` and `run_experiments.py` now accept:

```text
--gaussian-noise-std
--poisson-peak
--background-scatter
--background-sigma
--gate-attenuation-index
--gate-attenuation-factor
--gate-dropout-mode none|fixed|random
--gate-dropout-index
```

These degradation parameters are written to `summary.json` and runner CSV outputs. Validation attention CSV rows also include the degradation metadata so later analysis can group attention by robustness condition.

Recommended smoke commands:

```powershell
python run_experiments.py --experiment-name robust_smoke_gaussian_attention_residual --fusion-mode attention_residual --seeds 42 --gaussian-noise-std 0.05 --epochs 2 --batch-size 16
python origindataset\gated_blender_physical.py --output-root dataset_gate_spacing_smoke --gate-spacing 0.45 --gate-center-middle 7.4 --models-per-class 5
```

## Caution

The project may have untracked experiment outputs and edited code. Do not revert changes unless the user explicitly asks. Treat `experiments\six_class_attention_residual_seedmatched` as the current formal attention-residual result directory.
