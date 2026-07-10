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
| full | clean | 3 | 0.9394 | 0.0262 | 0.9091 | 0.9545 | 42 332 2026 |
| full | light_noise_g002_p80_b002 | 3 | 0.9242 | 0.0262 | 0.9091 | 0.9545 | 42 332 2026 |
| full | strong_noise_g005_p30_b005 | 3 | 0.9091 | 0.0000 | 0.9091 | 0.9091 | 42 332 2026 |
| full | hard_nuisance_v2 | 3 | 0.7727 | 0.0000 | 0.7727 | 0.7727 | 42 332 2026 |
| full | hard_nuisance_v3_mild | 3 | 0.9394 | 0.0525 | 0.9091 | 1.0000 | 42 332 2026 |

## Interpretation

- `clean`: best mean accuracy is `0.9394` with `full`.
- `light_noise_g002_p80_b002`: best mean accuracy is `0.9242` with `full`.
- `strong_noise_g005_p30_b005`: best mean accuracy is `0.9091` with `full`.
- `hard_nuisance_v2`: best mean accuracy is `0.7727` with `full`.
- `hard_nuisance_v3_mild`: best mean accuracy is `0.9394` with `full`.

## Files

- Per-run CSV: `experiments\v8_mv4_norm_mixaug_attention_eval_summary_3seed.csv`
- Aggregate CSV: `experiments\v8_mv4_norm_mixaug_attention_eval_aggregate_3seed.csv`
