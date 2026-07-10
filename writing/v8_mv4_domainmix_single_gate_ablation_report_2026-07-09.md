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
| gate0 | clean | 3 | 0.7879 | 0.0262 | 0.7727 | 0.8182 | 42 332 2026 |
| gate1 | clean | 3 | 0.5758 | 0.0525 | 0.5455 | 0.6364 | 42 332 2026 |
| gate2 | clean | 3 | 0.7727 | 0.0787 | 0.6818 | 0.8182 | 42 332 2026 |
| gate0 | light_noise_g002_p80_b002 | 3 | 0.8295 | 0.0301 | 0.8068 | 0.8636 | 42 332 2026 |
| gate1 | light_noise_g002_p80_b002 | 3 | 0.5455 | 0.0455 | 0.5000 | 0.5909 | 42 332 2026 |
| gate2 | light_noise_g002_p80_b002 | 3 | 0.7576 | 0.0694 | 0.6818 | 0.8182 | 42 332 2026 |
| gate0 | strong_noise_g005_p30_b005 | 3 | 0.8523 | 0.0301 | 0.8182 | 0.8750 | 42 332 2026 |
| gate1 | strong_noise_g005_p30_b005 | 3 | 0.5909 | 0.0909 | 0.5000 | 0.6818 | 42 332 2026 |
| gate2 | strong_noise_g005_p30_b005 | 3 | 0.7424 | 0.0525 | 0.6818 | 0.7727 | 42 332 2026 |
| gate0 | hard_nuisance_v2 | 3 | 0.7121 | 0.0262 | 0.6818 | 0.7273 | 42 332 2026 |
| gate1 | hard_nuisance_v2 | 3 | 0.5303 | 0.0262 | 0.5000 | 0.5455 | 42 332 2026 |
| gate2 | hard_nuisance_v2 | 3 | 0.6667 | 0.0262 | 0.6364 | 0.6818 | 42 332 2026 |
| gate0 | hard_nuisance_v3_mild | 3 | 0.8485 | 0.0694 | 0.7727 | 0.9091 | 42 332 2026 |
| gate1 | hard_nuisance_v3_mild | 3 | 0.5455 | 0.0455 | 0.5000 | 0.5909 | 42 332 2026 |
| gate2 | hard_nuisance_v3_mild | 3 | 0.8030 | 0.0525 | 0.7727 | 0.8636 | 42 332 2026 |

## Interpretation

- `clean`: best mean accuracy is `0.7879` with `gate0`.
- `light_noise_g002_p80_b002`: best mean accuracy is `0.8295` with `gate0`.
- `strong_noise_g005_p30_b005`: best mean accuracy is `0.8523` with `gate0`.
- `hard_nuisance_v2`: best mean accuracy is `0.7121` with `gate0`.
- `hard_nuisance_v3_mild`: best mean accuracy is `0.8485` with `gate0`.

## Files

- Per-run CSV: `experiments\v8_mv4_norm_mixaug_attention_eval_summary_3seed.csv`
- Aggregate CSV: `experiments\v8_mv4_norm_mixaug_attention_eval_aggregate_3seed.csv`
