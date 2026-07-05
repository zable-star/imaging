# Thread Handoff: Method Comparison Direction

## Record Time

- Created on: 2026-06-03
- Timezone: Asia/Hong_Kong
- Local project directory: `E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline`
- Lab project directory: `D:\学生文件夹\王剑哲\slice_attention_baseline`
- Lab Python interpreter: `C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe`

## Main Update

The paper direction has been adjusted.

Previous emphasis:

```text
attention mechanism as the central contribution
```

Current emphasis:

```text
comparison and discussion of multiple gated-slice fusion methods
```

The paper should compare methods by:

- accuracy
- stability across seeds
- structural complexity
- computation time / FPS / memory after future benchmarking
- confusion matrix behavior
- auxiliary slice-weight interpretation when attention-based methods are used

## Updated Main Document

The current experiment section has been rewritten:

```text
writing\paper_experiment_section_attention_residual.md
```

Despite the filename, the content is no longer only about attention-residual. It is now titled:

```text
多门控切片融合方法实验对比与结果分析
```

## Current Interpretation

Do not claim that attention is necessarily the best method.

Current result summary:

| Method | Mean accuracy | Std | Main interpretation |
|---|---:|---:|---|
| Mean | 91.94% | 2.55% | simplest low-cost baseline |
| Attention | 92.78% | 1.73% | adaptive weighting helps slightly, but weighted sum compresses information |
| Attention-residual | 94.72% | 1.27% | better accuracy and stability than attention, with auxiliary slice-weight analysis |
| Concat | 95.28% | 1.73% | highest accuracy among current methods, but larger classifier input and weaker slice-level interpretation |

Recommended writing:

```text
The paper compares multiple fusion strategies for gated-slice object recognition.
Concat achieves the highest accuracy in the current experiments.
Attention-residual is close to concat and provides auxiliary slice-level analysis.
Mean is the lowest-complexity baseline.
The final method choice should consider both accuracy and computation cost.
```

## Computation Time Status

Important:

```text
Current result files do not yet contain training time, inference time, FPS, or GPU memory.
```

Therefore, the paper should not yet make a hard claim about which method is fastest in real measured runtime.

Current safe statement:

```text
Mean has the lightest fusion operation.
Concat has simple fusion but a larger classifier input dimension.
Attention adds gate-weight computation.
Attention-residual adds both attention and a residual projection branch.
Because the shared CNN encoder dominates the computation, actual runtime must be measured on the same hardware.
```

## Suggested Next Experiment

Add or run a benchmarking script that records:

- parameter count
- average inference time per sample
- FPS
- training time per epoch
- GPU memory

Run the benchmark on the lab RTX 3090 using:

```powershell
Set-Location "D:\学生文件夹\王剑哲\slice_attention_baseline"
& "C:\Users\Administrator\anaconda3\envs\pytorch1\python.exe" benchmark_fusion_methods.py --dataset-root dataset --experiments-root experiments --output-dir artifacts\benchmarks
```

The script may not exist yet. If needed, create it next.

## Next Steps

1. Add a real benchmarking script for fusion methods.
2. Run it on RTX 3090.
3. Add a computation-time table to `writing\paper_experiment_section_attention_residual.md`.
4. Optionally rename the document later to something like `paper_experiment_section_method_comparison.md`.
