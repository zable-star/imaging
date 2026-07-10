# Current Mainline Experiments and Best Results

Last updated: 2026-07-10
Primary manuscript: writing/sci_manuscript_OLT_target_2026-07-10.md
Primary plan: writing/OLT_submission_plan_2026-07-10.md

All headline accuracies are grouped-validation mean over seeds 42 / 332 / 2026 unless noted.

## 1. Paper Mainline (freeze this)

One-sentence claim: physics-interpretable laser range-gated imaging simulation + anti-shortcut validation + domain-randomized robustness analysis for true 3D target vs planar false-target discrimination.

| Item | Choice |
|---|---|
| Dataset family | Military TF v8, multi-view mv4, per-gate max-normalized |
| Task | binary true3d vs flat_false |
| Views | 4 yaw views |
| Gates | 3-gate stack (gate0/1/2) |
| Network fusion | attention |
| Training strategy | domain mix + strong noise |
| Formal seeds | 42, 332, 2026 |
| Split | grouped validation |

Main training recipe: v8_mv4_domainmix_norm_hardv3_strongaug_attention_*
Canonical result CSV: experiments/v8_mv4_domainmix_full_vs_single_gate_eval_aggregate_3seed.csv
Strategy CSV: experiments/v8_mv4_strategy_comparison_aggregate_2026-07-08.csv

## 2. Best Results Leaderboard

### A. Strategy comparison (full stack)
Source: experiments/v8_mv4_strategy_comparison_aggregate_2026-07-08.csv

| Strategy | Clean | Light noise | Strong noise | Hard mild | Hard strong | Role |
|---|---:|---:|---:|---:|---:|---|
| Normal mixaug | 0.9848 | 0.9811 | 0.9205 | 0.5000 | 0.5000 | highest clean/noisy, fails structured shift |
| Online nuisance | 0.9697 | 0.7879 | 0.5606 | 0.5000 | 0.5000 | weak / unstable |
| Domain mix + strong noise | 0.9394 | 0.9242 | 0.9091 | 0.9394 | 0.7727 | paper mainline (robust) |

### B. Mainline full-stack vs single-gate (domainmix model)
Source: experiments/v8_mv4_domainmix_full_vs_single_gate_eval_aggregate_3seed.csv

| Condition | Full stack | Gate0 | Gate1 | Gate2 |
|---|---:|---:|---:|---:|
| Clean | 0.9394 +/- 0.0262 | 0.7879 | 0.5758 | 0.7727 |
| Light noise | 0.9242 +/- 0.0262 | 0.8295 | 0.5455 | 0.7576 |
| Strong noise | 0.9091 +/- 0.0000 | 0.8523 | 0.5909 | 0.7424 |
| Hard mild | 0.9394 +/- 0.0525 | 0.8485 | 0.5455 | 0.8030 |
| Hard strong | 0.7727 +/- 0.0000 | 0.7121 | 0.5303 | 0.6667 |

### C. Clean multiview attention model (high clean, non-robust)
Source: experiments/v8_mv4_norm_mixaug_attention_eval_aggregate_3seed.csv

| Input | Clean | Light noise | Strong noise |
|---|---:|---:|---:|
| Full stack | 0.9848 | 0.9811 | 0.9205 |
| Gate0 | 0.9697 | 0.9432 | 0.7462 |
| Gate1 | 0.9545 | 0.9053 | 0.5265 |
| Gate2 | 0.8939 | 0.8750 | 0.6098 |

### D. Fusion-mode comparison
Source: experiments/v8_mv4_norm_mixaug_full_fusion_eval_aggregate_3seed.csv

| Fusion | Clean | Light noise | Strong noise |
|---|---:|---:|---:|
| attention | 0.9848 | 0.9811 | 0.9205 |
| attention_residual | 0.9848 | 0.9394 | 0.7727 |
| mean | 0.9848 | 0.9848 | 0.7197 |

### E. Earlier single-view / pre-mv4 baseline
Source: experiments/v8_per_gate_maxnorm_mixaug_eval_aggregate_3seed.csv

| Condition | Full stack | Gate0 | Gate1 | Gate2 |
|---|---:|---:|---:|---:|
| Clean | 0.9697 | 0.9091 | 0.7121 | 0.8636 |
| Light noise | 0.9545 | 0.8636 | 0.5151 | 0.7727 |
| Strong noise | 0.7424 | 0.6970 | 0.5152 | 0.5152 |

## 3. Current best by use-case

| Use-case | Current best | Mean acc |
|---|---|---|
| Paper main robust model | domainmix + strong noise, full stack, attention | clean 0.9394 / hard mild 0.9394 / hard strong 0.7727 |
| Highest clean/noisy accuracy | normal mixaug attention full stack | 0.9848 / 0.9811 / 0.9205 |
| Best fusion under strong noise | attention | 0.9205 |
| Strongest single-gate (domainmix) | gate0 under strong noise | 0.8523 (still below full 0.9091) |
| Weakest single-gate | gate1 | often 0.53-0.59 |

## 4. Mainline experiment inventory

Tier-0 paper-critical:
1. Domainmix full vs single-gate multi-condition eval
2. Strategy comparison (normal / online / domainmix)
3. Single-gate ablation under domainmix + fig11
4. Anti-shortcut / per-gate maxnorm diagnostics

Tier-1 supporting:
1. Fusion ablation CSV
2. Clean multiview robustness CSV
3. Hard-nuisance boundary aggregates
4. Pre-mv4 progression CSV

Tier-2 archive: ModelNet baseline, physical gate-spacing, military transfer, blender v4/v5/v7 intermediate ablations

## 5. Defensible paper claims

1. Full stack > single gate after domain-randomized training: clean 0.9394 vs best single-gate 0.7879; hard mild 0.9394 vs 0.8485; hard strong 0.7727 vs 0.7121.
2. Domain randomization required for structured nuisance robustness: normal mixaug hard = 0.50; domainmix = 0.9394 / 0.7727.
3. High clean accuracy alone is insufficient: 0.9848 clean collapses on hard nuisance without domainmix.
4. Fusion mode is secondary: clean ties at 0.9848; strong-noise attention 0.9205 > residual 0.7727 > mean 0.7197.
5. Results remain simulation validation (selected models, idealized false targets, simplified physics).

## 6. Source-of-truth map

| Manuscript need | Source |
|---|---|
| Main full vs single-gate table | experiments/v8_mv4_domainmix_full_vs_single_gate_eval_aggregate_3seed.csv |
| Training-strategy table | experiments/v8_mv4_strategy_comparison_aggregate_2026-07-08.csv |
| Fusion ablation | experiments/v8_mv4_norm_mixaug_full_fusion_eval_aggregate_3seed.csv |
| Clean multiview stack vs gate | experiments/v8_mv4_norm_mixaug_attention_eval_aggregate_3seed.csv |
| Claims checklist | writing/sci_claims_evidence_matrix_2026-07-07.md |
| Submission status | writing/OLT_submission_plan_2026-07-10.md |

## 7. Bottom line

- Current mainline model: v8_mv4 domainmix + strongaug + attention + full 3-gate stack
- Current robust SOTA: clean 0.9394, hard mild 0.9394, hard strong 0.7727 (3 seeds)
- Current highest clean SOTA: normal mixaug attention full stack 0.9848 (not robust)
- For OLT writing: freeze domainmix as primary result; keep 0.9848 only as clean/noisy upper-bound contrast
