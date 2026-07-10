# v8 mv4 full-stack fusion robustness report

This report compares full-stack fusion modes on the four-view v8 normalized dataset.

| fusion | condition | runs | mean acc | std | min | max | seeds |
|---|---|---:|---:|---:|---:|---:|---|
| attention | clean | 3 | 0.9848 | 0.0262 | 0.9545 | 1.0000 | 42 332 2026 |
| mean | clean | 3 | 0.9848 | 0.0262 | 0.9545 | 1.0000 | 42 332 2026 |
| attention_residual | clean | 3 | 0.9848 | 0.0262 | 0.9545 | 1.0000 | 42 332 2026 |
| attention | light_noise_g002_p80_b002 | 3 | 0.9811 | 0.0328 | 0.9432 | 1.0000 | 42 332 2026 |
| mean | light_noise_g002_p80_b002 | 3 | 0.9848 | 0.0262 | 0.9545 | 1.0000 | 42 332 2026 |
| attention_residual | light_noise_g002_p80_b002 | 3 | 0.9394 | 0.0694 | 0.8636 | 1.0000 | 42 332 2026 |
| attention | strong_noise_g005_p30_b005 | 3 | 0.9205 | 0.0745 | 0.8523 | 1.0000 | 42 332 2026 |
| mean | strong_noise_g005_p30_b005 | 3 | 0.7197 | 0.1025 | 0.6136 | 0.8182 | 42 332 2026 |
| attention_residual | strong_noise_g005_p30_b005 | 3 | 0.7727 | 0.1639 | 0.5909 | 0.9091 | 42 332 2026 |

## Interpretation

- `clean`: best mean accuracy is `0.9848` with `attention`.
- `light_noise_g002_p80_b002`: best mean accuracy is `0.9848` with `mean`.
- `strong_noise_g005_p30_b005`: best mean accuracy is `0.9205` with `attention`.

The fusion result should be used as an engineering ablation. The paper's main claim remains the gate-stack advantage and anti-shortcut validation protocol, not a universal claim about one fusion head.

Files:

- `experiments/v8_mv4_norm_mixaug_full_fusion_eval_summary_3seed.csv`
- `experiments/v8_mv4_norm_mixaug_full_fusion_eval_aggregate_3seed.csv`
- `writing/figures/fig8_mv4_full_stack_fusion_robustness.png`
- `writing/figures/fig8_mv4_full_stack_fusion_robustness.pdf`
