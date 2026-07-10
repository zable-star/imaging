# v8 mv4 domain-randomized training strategy report

This report compares three training strategies for four-view v8 true-3D versus planar-false-target discrimination.
All reported values are grouped-validation mean accuracies over seeds 42, 332, and 2026.

## Aggregate accuracy

| strategy | clean | light noise | strong noise | hard mild | hard strong |
|---|---:|---:|---:|---:|---:|
| Normal mixaug | 0.9848 | 0.9811 | 0.9205 | 0.5000 | 0.5000 |
| Online nuisance | 0.9697 | 0.7879 | 0.5606 | 0.5000 | 0.5000 |
| Domain mix + strong noise | 0.9394 | 0.9242 | 0.9091 | 0.9394 | 0.7727 |

## Interpretation

- `Normal mixaug` is the strongest controlled-simulation baseline under clean/light/strong noise, but it collapses to chance under structured hard-nuisance shifts.
- `Online nuisance` applies structured perturbations during training, but the current post-loading implementation does not recover hard-nuisance performance and substantially weakens strong-noise robustness.
- `Domain mix + strong noise` explicitly trains on both normal and hard-mild domains, while retaining strong-noise augmentation. It recovers hard-nuisance performance and keeps strong-noise accuracy near the original baseline.
- The current best manuscript-safe claim is therefore not that the model is generally robust, but that explicit domain-randomized simulation improves the robustness boundary compared with clean/noisy-only training.

## Main numerical result

Compared with the normal four-view mixed-noise baseline, `Domain mix + strong noise` changes the mean accuracies as follows:

- Clean: -0.0455 absolute accuracy change.
- Light noise: -0.0568 absolute accuracy change.
- Strong noise: -0.0114 absolute accuracy change.
- Hard mild: 0.4394 absolute accuracy change.
- Hard strong: 0.2727 absolute accuracy change.

## Files

- Comparison CSV: `experiments/v8_mv4_strategy_comparison_aggregate_2026-07-08.csv`
- Figure PNG: `writing/figures/fig10_domain_randomization_strategy_comparison.png`
- Figure PDF: `writing/figures/fig10_domain_randomization_strategy_comparison.pdf`
- Domain-mix dataset: `dataset_new/Military_TF_v8_mv4_norm_plus_hardv3_mild`
- Domain-mix manifest: `dataset_new/Military_TF_v8_mv4_norm_plus_hardv3_mild/variant_mixture_manifest.csv`

## Caveat

The hard-nuisance v2 result remains lower than clean/noise validation, and the dataset still contains only 44 selected source military models. This result should be framed as a robustness-boundary improvement in simulation, not real-world deployment robustness.
