# v8 mv4 multiview robustness report

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
| full | clean | 1 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 42 |
| full | light_noise_g002_p80_b002 | 1 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 42 |
| full | strong_noise_g005_p30_b005 | 1 | 0.2614 | 0.0000 | 0.2614 | 0.2614 | 42 |
| full | hard_nuisance_v2 | 1 | 0.7727 | 0.0000 | 0.7727 | 0.7727 | 42 |
| full | hard_nuisance_v3_mild | 1 | 0.8636 | 0.0000 | 0.8636 | 0.8636 | 42 |

## Interpretation

- `clean`: best mean accuracy is `1.0000` with `full`.
- `light_noise_g002_p80_b002`: best mean accuracy is `1.0000` with `full`.
- `strong_noise_g005_p30_b005`: best mean accuracy is `0.2614` with `full`.
- `hard_nuisance_v2`: best mean accuracy is `0.7727` with `full`.
- `hard_nuisance_v3_mild`: best mean accuracy is `0.8636` with `full`.

## Files

- Per-run CSV: `experiments\v8_mv4_norm_mixaug_attention_eval_summary_3seed.csv`
- Aggregate CSV: `experiments\v8_mv4_norm_mixaug_attention_eval_aggregate_3seed.csv`
