# Domain-mixed full-stack versus single-gate ablation

All values are mean grouped-validation accuracies over seeds 42, 332, and 2026.

| condition | full stack | gate0 | gate1 | gate2 |
|---|---:|---:|---:|---:|
| Clean | 0.9394 | 0.7879 | 0.5758 | 0.7727 |
| Light noise | 0.9242 | 0.8295 | 0.5455 | 0.7576 |
| Strong noise | 0.9091 | 0.8523 | 0.5909 | 0.7424 |
| Hard mild | 0.9394 | 0.8485 | 0.5455 | 0.8030 |
| Hard strong | 0.7727 | 0.7121 | 0.5303 | 0.6667 |

## Interpretation

- The full three-gate stack is the best mean performer under all five evaluation conditions.
- Gate 0 is the strongest single-gate baseline under strong-noise and hard-strong evaluation.
- Gate 1 is consistently weak in the domain-mixed single-gate setting.
- The result supports the manuscript claim that the domain-randomized gate stack retains information beyond the best individual gate.

## Files

- Combined CSV: `experiments/v8_mv4_domainmix_full_vs_single_gate_eval_aggregate_3seed.csv`
- Figure PNG: `writing/figures/fig11_domainmix_full_stack_vs_single_gate.png`
- Figure PDF: `writing/figures/fig11_domainmix_full_stack_vs_single_gate.pdf`
