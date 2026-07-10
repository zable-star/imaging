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
| full | clean | 3 | 0.9848 | 0.0262 | 0.9545 | 1.0000 | 42 332 2026 |
| gate0 | clean | 3 | 0.9697 | 0.0525 | 0.9091 | 1.0000 | 42 332 2026 |
| gate1 | clean | 3 | 0.9545 | 0.0000 | 0.9545 | 0.9545 | 42 332 2026 |
| gate2 | clean | 3 | 0.8939 | 0.0262 | 0.8636 | 0.9091 | 42 332 2026 |
| full | light_noise_g002_p80_b002 | 3 | 0.9811 | 0.0328 | 0.9432 | 1.0000 | 42 332 2026 |
| gate0 | light_noise_g002_p80_b002 | 3 | 0.9432 | 0.0710 | 0.8636 | 1.0000 | 42 332 2026 |
| gate1 | light_noise_g002_p80_b002 | 3 | 0.9053 | 0.0512 | 0.8523 | 0.9545 | 42 332 2026 |
| gate2 | light_noise_g002_p80_b002 | 3 | 0.8750 | 0.0710 | 0.8182 | 0.9545 | 42 332 2026 |
| full | strong_noise_g005_p30_b005 | 3 | 0.9205 | 0.0745 | 0.8523 | 1.0000 | 42 332 2026 |
| gate0 | strong_noise_g005_p30_b005 | 3 | 0.7462 | 0.0561 | 0.6818 | 0.7841 | 42 332 2026 |
| gate1 | strong_noise_g005_p30_b005 | 3 | 0.5265 | 0.0237 | 0.5000 | 0.5455 | 42 332 2026 |
| gate2 | strong_noise_g005_p30_b005 | 3 | 0.6098 | 0.0583 | 0.5455 | 0.6591 | 42 332 2026 |

## Interpretation

- `clean`: best mean accuracy is `0.9848` with `full`.
- `light_noise_g002_p80_b002`: best mean accuracy is `0.9811` with `full`.
- `strong_noise_g005_p30_b005`: best mean accuracy is `0.9205` with `full`.

The defensible claim is that the full gate stack gives the strongest or most stable validation performance under the current v8 multiview simulation. Single-gate inputs are still informative, so the result should be presented as added depth-gated evidence rather than complete removal of shape shortcuts.

## Files

- Per-run CSV: `experiments\v8_mv4_norm_mixaug_attention_eval_summary_3seed.csv`
- Aggregate CSV: `experiments\v8_mv4_norm_mixaug_attention_eval_aggregate_3seed.csv`
- Figure PNG: `writing\figures\fig7_mv4_full_stack_vs_single_gate_robustness.png`
- Figure PDF: `writing\figures\fig7_mv4_full_stack_vs_single_gate_robustness.pdf`
