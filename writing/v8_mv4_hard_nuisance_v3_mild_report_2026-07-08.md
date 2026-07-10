# v8 mv4 multiview robustness report

> Note: this auto-generated report is superseded by `writing/v8_mv4_hard_nuisance_boundary_report_2026-07-08.md`.
> The hard-nuisance result should be interpreted as a failure-boundary result, not as positive robustness evidence.

This report evaluates saved v8 multiview attention models on the grouped validation split.
The split keeps all rendered views of the same source model in the same partition.

## Setup

- Dataset: `dataset_new/Military_TF_v8_mv4_norm`
- Views: 0, 90, 180, and 270 degrees around Z
- Classes: `true3d` and `flat_false`
- Seeds: 42, 332, 2026
- Evaluation conditions: clean, light noise, strong noise

## Aggregate accuracy

| input | condition | runs | mean acc | std | min | max | seeds |
|---|---|---:|---:|---:|---:|---:|---|
| full | hard_nuisance_v3_mild | 3 | 0.5000 | 0.0000 | 0.5000 | 0.5000 | 42 332 2026 |
| gate0 | hard_nuisance_v3_mild | 3 | 0.4848 | 0.0262 | 0.4545 | 0.5000 | 42 332 2026 |
| gate1 | hard_nuisance_v3_mild | 3 | 0.5000 | 0.0000 | 0.5000 | 0.5000 | 42 332 2026 |
| gate2 | hard_nuisance_v3_mild | 3 | 0.4545 | 0.0787 | 0.3636 | 0.5000 | 42 332 2026 |

## Interpretation

- `hard_nuisance_v3_mild`: best mean accuracy is `0.5000` with `full`.

The defensible claim is that the full gate stack gives the strongest or most stable validation performance under the current v8 multiview simulation. Single-gate inputs are still informative, so the result should be presented as added depth-gated evidence rather than complete removal of shape shortcuts.

## Files

- Per-run CSV: `experiments\v8_mv4_norm_mixaug_attention_eval_summary_3seed.csv`
- Aggregate CSV: `experiments\v8_mv4_norm_mixaug_attention_eval_aggregate_3seed.csv`
