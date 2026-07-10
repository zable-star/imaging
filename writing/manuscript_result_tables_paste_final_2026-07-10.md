# Paste-ready main result tables

Numbering note: current OLT draft already uses Table 1/2/3 for dataset, shortcut diagnostics, and training settings. These two tables correspond to Sec. 6.6 and 6.7 and should be numbered Table 5 and Table 6 in the full manuscript. If used as a standalone results block, rename them Table 1 / Table 2.

Source:
- experiments/v8_mv4_strategy_comparison_aggregate_2026-07-08.csv
- experiments/v8_mv4_domainmix_full_vs_single_gate_eval_aggregate_3seed.csv

## Table 5. Training-strategy comparison

Table 5. Training-strategy comparison under clean/noisy evaluation and structured hard-nuisance shifts. Values are mean grouped-validation accuracy over three seeds (42/332/2026).

| strategy | clean | light noise | strong noise | hard mild | hard strong |
|---|---:|---:|---:|---:|---:|
| Normal mixaug | 0.9848 | 0.9811 | 0.9205 | 0.5000 | 0.5000 |
| Online nuisance | 0.9697 | 0.7879 | 0.5606 | 0.5000 | 0.5000 |
| Domain mix + strong noise | 0.9394 | 0.9242 | 0.9091 | 0.9394 | 0.7727 |

## Table 6. Domain-mixed full-stack versus single-gate ablation

Table 6. Domain-mixed full-stack versus single-gate ablation under clean, light-noise, strong-noise, hard-mild, and hard-strong evaluation. All models use domain-mixed normal-plus-hard-mild training with strong-noise degradation. Values are mean grouped-validation accuracy over three seeds (42/332/2026).

| condition | full stack | gate0 | gate1 | gate2 |
|---|---:|---:|---:|---:|
| clean | 0.9394 | 0.7879 | 0.5758 | 0.7727 |
| light noise | 0.9242 | 0.8295 | 0.5455 | 0.7576 |
| strong noise | 0.9091 | 0.8523 | 0.5909 | 0.7424 |
| hard mild | 0.9394 | 0.8485 | 0.5455 | 0.8030 |
| hard strong | 0.7727 | 0.7121 | 0.5303 | 0.6667 |

## Camera-ready mean +/- std versions

### Table 5

| strategy | clean | light noise | strong noise | hard mild | hard strong |
|---|---:|---:|---:|---:|---:|
| Normal mixaug | 0.9848 $\pm$ 0.0262 | 0.9811 $\pm$ 0.0328 | 0.9205 $\pm$ 0.0745 | 0.5000 $\pm$ 0.0000 | 0.5000 $\pm$ 0.0000 |
| Online nuisance | 0.9697 $\pm$ 0.0525 | 0.7879 $\pm$ 0.1389 | 0.5606 $\pm$ 0.0694 | 0.5000 $\pm$ 0.0000 | 0.5000 $\pm$ 0.0000 |
| Domain mix + strong noise | 0.9394 $\pm$ 0.0262 | 0.9242 $\pm$ 0.0262 | 0.9091 $\pm$ 0.0000 | 0.9394 $\pm$ 0.0525 | 0.7727 $\pm$ 0.0000 |

### Table 6

| condition | full stack | gate0 | gate1 | gate2 |
|---|---:|---:|---:|---:|
| clean | 0.9394 $\pm$ 0.0262 | 0.7879 $\pm$ 0.0262 | 0.5758 $\pm$ 0.0525 | 0.7727 $\pm$ 0.0787 |
| light noise | 0.9242 $\pm$ 0.0262 | 0.8295 $\pm$ 0.0301 | 0.5455 $\pm$ 0.0455 | 0.7576 $\pm$ 0.0694 |
| strong noise | 0.9091 $\pm$ 0.0000 | 0.8523 $\pm$ 0.0301 | 0.5909 $\pm$ 0.0909 | 0.7424 $\pm$ 0.0525 |
| hard mild | 0.9394 $\pm$ 0.0525 | 0.8485 $\pm$ 0.0694 | 0.5455 $\pm$ 0.0455 | 0.8030 $\pm$ 0.0525 |
| hard strong | 0.7727 $\pm$ 0.0000 | 0.7121 $\pm$ 0.0262 | 0.5303 $\pm$ 0.0262 | 0.6667 $\pm$ 0.0262 |

## Body text

After Table 5: These results show that high clean accuracy alone is insufficient: normal mixaug reaches 0.9848 on clean data but falls to 0.5000 under both hard-mild and hard-strong shifts, whereas domain-mixed training retains 0.9394 and 0.7727 under the same conditions.

After Table 6: After domain-randomized training, the full three-gate stack remains the strongest configuration under every evaluation condition, with the largest absolute gains over the best single gate on clean (+0.1515 versus gate0) and hard-mild (+0.0909 versus gate0) settings.

## Numbering update (2026-07-10, later same day)
In the full manuscript, result tables are now Table 4-9.
- Table 4 single-view full vs single-gate
- Table 5 four-view full vs single-gate
- Table 6 four-view fusion
- Table 7 hard-nuisance boundary
- Table 8 strategy comparison
- Table 9 domainmix full vs single-gate
Updated: sci_manuscript_OLT_target_2026-07-10.md / .docx
