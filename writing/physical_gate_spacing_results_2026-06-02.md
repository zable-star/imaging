# Gate Spacing Physical Ablation Results

Date: 2026-06-02

Experiment scope: five-class physical-parameter ablation using `chair`, `desk`, `sofa`, `bed`, and `toilet`. The `image2d` abnormal class is excluded from this experiment and remains part of the separate six-class abnormal-input study.

Model: `attention_residual`

Seeds: `42, 332, 2026`

## Accuracy

| Gate spacing | Seed 42 | Seed 332 | Seed 2026 | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 0.9300 | 0.9100 | 0.9200 | 0.9200 | 0.0100 | 0.9100 | 0.9300 |
| default | 0.9500 | 0.9000 | 0.9300 | 0.9267 | 0.0252 | 0.9000 | 0.9500 |
| large | 0.9700 | 0.9300 | 0.9400 | 0.9467 | 0.0208 | 0.9300 | 0.9700 |

## Overall Gate-Level Attention

| Gate spacing | gate_0 | gate_1 | gate_2 |
|---|---:|---:|---:|
| small | 0.3140 | 0.3404 | 0.3456 |
| default | 0.3211 | 0.3402 | 0.3387 |
| large | 0.3299 | 0.3375 | 0.3326 |

## Class-Wise Gate-Level Attention

### Small spacing

| Class | gate_0 | gate_1 | gate_2 | Highest gate |
|---|---:|---:|---:|---|
| bed | 0.2639 | 0.3414 | 0.3947 | gate_2 |
| chair | 0.3111 | 0.3418 | 0.3471 | gate_2 |
| desk | 0.3570 | 0.3454 | 0.2976 | gate_0 |
| sofa | 0.2949 | 0.3472 | 0.3579 | gate_2 |
| toilet | 0.3429 | 0.3264 | 0.3307 | gate_0 |

### Default spacing

| Class | gate_0 | gate_1 | gate_2 | Highest gate |
|---|---:|---:|---:|---|
| bed | 0.2883 | 0.3397 | 0.3720 | gate_2 |
| chair | 0.3161 | 0.3400 | 0.3439 | gate_2 |
| desk | 0.3534 | 0.3487 | 0.2980 | gate_0 |
| sofa | 0.3075 | 0.3435 | 0.3490 | gate_2 |
| toilet | 0.3401 | 0.3292 | 0.3307 | gate_0 |

### Large spacing

| Class | gate_0 | gate_1 | gate_2 | Highest gate |
|---|---:|---:|---:|---|
| bed | 0.3204 | 0.3311 | 0.3485 | gate_2 |
| chair | 0.3255 | 0.3383 | 0.3362 | gate_1 |
| desk | 0.3379 | 0.3471 | 0.3150 | gate_1 |
| sofa | 0.3238 | 0.3404 | 0.3357 | gate_1 |
| toilet | 0.3418 | 0.3306 | 0.3276 | gate_0 |

## Interpretation

The large-spacing setting achieved the highest mean validation accuracy among the three tested spacing conditions. This suggests that, in the current rendering geometry, a wider separation between gate response windows provides more discriminative multi-depth observations than closely overlapping gates.

The attention distribution also changed with spacing. Small spacing placed more overall weight on `gate_2`, whereas large spacing produced a more balanced distribution and increased the contribution of `gate_0`. Class-wise attention indicates that `bed` remains biased toward the far gate, while `desk` and `toilet` rely more strongly on nearer or middle gates depending on spacing.

These results support the manuscript claim that distance-gating parameters affect both recognition accuracy and gate-level discriminative contribution. The next physical experiment should test receiver gate width and laser pulse width to distinguish the effect of gate separation from the effect of gate response-window breadth.

## Paper Placement

Recommended role: main-text physical-parameter ablation or supplementary figure if page space is tight.

Recommended table caption:

```text
Effect of gate spacing on five-class 3D object recognition. Accuracy is reported over three matched random seeds. Attention values indicate gate-level discriminative contribution averaged over validation samples and seeds.
```
