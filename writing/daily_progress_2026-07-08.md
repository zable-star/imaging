# Daily progress 2026-07-08

## Hard-nuisance validation

Today the project moved from clean/noisy robustness toward structured nuisance robustness.

New dataset construction script:

```text
dataset_new/build_hard_nuisance_dataset.py
```

The script builds hard-nuisance datasets from the normalized four-view v8 dataset:

```text
dataset_new/Military_TF_v8_mv4_norm
```

It applies the following deterministic nuisance factors to both `true3d` and `flat_false`:

- Low-frequency reflectance texture.
- Weak background scatter.
- Partial rectangular occlusion.
- Shared nuisance key for paired true/false samples.
- Optional per-sample maximum preservation to avoid a max-brightness shortcut.

Generated datasets:

```text
dataset_new/Military_TF_v8_mv4_hard_nuisance_v1
dataset_new/Military_TF_v8_mv4_hard_nuisance_v2
dataset_new/Military_TF_v8_mv4_hard_nuisance_v3_mild
```

v1 was not used as the main report because it introduced a class-level max-brightness difference. v2 and v3 use per-sample max preservation and are used as the formal boundary test.

## Dataset checks

v2 and v3 both contain:

```text
true3d valid samples = 176
flat_false valid samples = 176
total gate images = 1056
low_contrast_images = 0
mean max gray true3d = 180
mean max gray flat_false = 180
```

Gate-stack diagnostics:

| dataset | class | corr maxnorm | mask IoU | absdiff maxnorm |
|---|---|---:|---:|---:|
| v2 | flat_false | 0.5962 | 0.6435 | 0.1645 |
| v2 | true3d | 0.6823 | 0.6555 | 0.0618 |
| v3 mild | flat_false | 0.5859 | 0.6545 | 0.1643 |
| v3 mild | true3d | 0.6753 | 0.6715 | 0.0779 |

## Independent evaluation

Saved four-view attention models trained on `Military_TF_v8_mv4_norm` were evaluated on v2 and v3.

| condition | full stack | gate0 | gate1 | gate2 |
|---|---:|---:|---:|---:|
| hard_nuisance_v2 | 0.5000 | 0.4697 | 0.5000 | 0.4545 |
| hard_nuisance_v3_mild | 0.5000 | 0.4848 | 0.5000 | 0.4545 |

This is a failure-boundary result.

The current models are robust to the clean/light/strong additive-noise protocol, but they do not generalize to structured reflectance, background, and occlusion shifts introduced after training.

## New outputs

```text
scripts/run_v8_mv4_hard_nuisance_eval.ps1
scripts/make_v8_hard_nuisance_boundary_figure.py
experiments/v8_mv4_hard_nuisance_v2_eval_aggregate_3seed.csv
experiments/v8_mv4_hard_nuisance_v3_mild_eval_aggregate_3seed.csv
writing/v8_mv4_hard_nuisance_boundary_report_2026-07-08.md
writing/figures/fig9_hard_nuisance_failure_boundary.png
writing/figures/fig9_hard_nuisance_failure_boundary.pdf
```

## Manuscript update

Updated:

```text
writing/sci_manuscript_v8_gated_false_target_draft_2026-07-07.md
writing/sci_claims_evidence_matrix_2026-07-07.md
```

The hard-nuisance result is written as a limitation and next-step motivation, not as a positive robustness claim.

## Verification

```text
pytest tests/test_hard_nuisance_dataset.py tests/test_train_utils.py tests/test_run_experiments.py tests/test_build_multiview_true_false_dataset.py
23 passed
```

## Current paper-safe story

The defensible story is:

```text
The v8 four-view gate-stack simulation supports the value of complete gated observations under controlled clean/noisy conditions, but structured nuisance shifts remain unsolved. This justifies the next innovation step: nuisance-aware gated simulation and training.
```

Do not claim:

```text
The current model is robust to realistic battlefield reflectance/background/occlusion changes.
```

## Domain-randomized training update

The hard-nuisance failure boundary was followed by an explicit domain-randomized training experiment.

New dataset construction script:

```text
dataset_new/build_variant_mixture_dataset.py
```

Generated mixed-domain dataset:

```text
dataset_new/Military_TF_v8_mv4_norm_plus_hardv3_mild
```

This dataset combines:

```text
dataset_new/Military_TF_v8_mv4_norm
dataset_new/Military_TF_v8_mv4_hard_nuisance_v3_mild
```

Scale:

| class | samples | gate images |
|---|---:|---:|
| true3d | 352 | 1056 |
| flat_false | 352 | 1056 |
| total | 704 | 2112 |

Sample names are prefixed with `domain_norm__` or `domain_hardv3__`. The grouped split now strips both domain and view prefixes before assigning samples to train/validation groups, so normal/hard variants and all yaw views of the same source model remain in the same partition.

## Training strategies tested

Three four-view full-stack strategies are now available:

| strategy | training idea |
|---|---|
| normal mixaug | normal v8 four-view dataset with light mixed noise augmentation |
| online nuisance | normal v8 four-view dataset with online structured reflectance/background/occlusion perturbation |
| domain mix + strong noise | normal + hard-mild domains with strong mixed-noise augmentation |

The final domain-mix strong-noise training command used:

```text
gaussian_noise_std = 0.05
poisson_peak = 30
background_scatter = 0.05
degradation_probability = 0.5
seeds = 42 332 2026
epochs = 20
fusion_mode = attention
```

Training aggregate:

| experiment | mean best val acc | std | min | max | seeds |
|---|---:|---:|---:|---:|---|
| v8_mv4_domainmix_norm_hardv3_strongaug_attention_full_20ep | 0.9280 | 0.0280 | 0.9091 | 0.9602 | 42 332 2026 |

## Strategy comparison result

Independent grouped-validation accuracy:

| strategy | clean | light noise | strong noise | hard mild | hard strong |
|---|---:|---:|---:|---:|---:|
| normal mixaug | 0.9848 | 0.9811 | 0.9205 | 0.5000 | 0.5000 |
| online nuisance | 0.9697 | 0.7879 | 0.5606 | 0.5000 | 0.5000 |
| domain mix + strong noise | 0.9394 | 0.9242 | 0.9091 | 0.9394 | 0.7727 |

Interpretation:

```text
The normal clean/noisy baseline is strong under the original clean/noisy protocol but fails under structured nuisance shifts.
The online nuisance strategy is not sufficient in the current implementation.
Explicit domain mixing between normal and hard-mild simulated domains, combined with strong-noise augmentation, gives the best current robustness-boundary result.
```

This is now the main innovation upgrade after Fig. 9: the paper can tell a stronger story from failure-boundary discovery to domain-randomized simulation/training.

## New domain-randomization outputs

```text
scripts/make_v8_strategy_comparison.py
scripts/run_v8_domainmix_strongaug_eval.ps1
scripts/run_v8_domainmix_single_gate_ablation.ps1
experiments/v8_mv4_strategy_comparison_aggregate_2026-07-08.csv
experiments/v8_mv4_domainmix_norm_hardv3_strongaug_attention_full_20ep/results.csv
experiments/v8_mv4_domainmix_norm_hardv3_strongaug_attention_full_20ep/aggregate_results.csv
experiments/v8_mv4_domainmix_norm_hardv3_strongaug_attention_full_eval_summary_3seed.csv
experiments/v8_mv4_domainmix_norm_hardv3_strongaug_attention_full_eval_aggregate_3seed.csv
writing/v8_mv4_domain_randomization_strategy_report_2026-07-08.md
writing/figures/fig10_domain_randomization_strategy_comparison.png
writing/figures/fig10_domain_randomization_strategy_comparison.pdf
```

## Updated paper-safe story

The defensible story is now:

```text
The v8 four-view gate-stack simulation supports the value of complete gated observations under controlled clean/noisy conditions. Structured nuisance shifts expose a failure boundary. Explicit domain-randomized simulation and strong-noise training improve this boundary, recovering hard-mild accuracy to 0.9394 and hard-strong accuracy to 0.7727 while preserving 0.9091 strong-noise accuracy.
```

Still do not claim:

```text
The current system is deployment-ready or robust to real battlefield decoys.
```

## Domain-mix single-gate smoke

A low-cost single-gate smoke test was run after the formal three-seed full-stack domain-mix result.

Setting:

```text
dataset = dataset_new/Military_TF_v8_mv4_norm_plus_hardv3_mild
seeds = 42 only
epochs = 8
input modes = gate0 / gate1 / gate2
```

Training best validation accuracy:

| input | best val acc |
|---|---:|
| gate0 | 0.8466 |
| gate1 | 0.5682 |
| gate2 | 0.7386 |

Independent smoke evaluation:

| condition | full stack seed42 20ep | gate0 smoke | gate1 smoke | gate2 smoke |
|---|---:|---:|---:|---:|
| clean | 0.9545 | 0.8636 | 0.5455 | 0.8182 |
| light noise | 0.9545 | 0.8636 | 0.5455 | 0.7727 |
| strong noise | 0.9091 | 0.8068 | 0.5000 | 0.7727 |
| hard_nuisance_v3_mild | 1.0000 | 0.8182 | 0.5000 | 0.6818 |
| hard_nuisance_v2 | 0.7727 | 0.6364 | 0.5000 | 0.5455 |

Interpretation:

```text
The smoke trend supports running a formal domain-mix full-stack versus single-gate ablation.
It is not a formal paper result because the single-gate runs use only one seed and 8 epochs.
```

Smoke outputs:

```text
experiments/v8_mv4_domainmix_single_gate_smoke_eval_summary_seed42_8ep.csv
experiments/v8_mv4_domainmix_single_gate_smoke_eval_aggregate_seed42_8ep.csv
writing/v8_mv4_domainmix_single_gate_smoke_report_2026-07-08.md
```
