# A Physics-Interpretable Gated-Imaging Simulation and Anti-Shortcut Validation Framework for 3D Target and Planar False-Target Discrimination

> Draft date: 2026-07-08  
> Status: English SCI-style working draft. Initial related-work placeholders have been inserted from the local literature folder and must be checked against final publisher metadata before submission. The current evidence is based on 44 manually selected military 3D models, single-view and four-view Blender simulation, three-gate gated images, per-gate max-normalized anti-shortcut control, hard-nuisance boundary tests, domain-randomized training, and three-seed neural-network validation.

## Abstract

Laser gated imaging records target echoes within controlled temporal range windows and can therefore provide depth-related observations that are not available in ordinary two-dimensional intensity images. This property is potentially useful for discriminating real three-dimensional targets from planar false targets or decoys. However, when gated-image classifiers are trained on simulated data, high classification accuracy may arise from non-physical shortcuts such as black frames, single-gate brightness differences, or rendering artifacts rather than from the intended multi-gate range structure. To address this issue, this study develops a physics-interpretable Blender-based gated-imaging simulation and anti-shortcut validation framework for discriminating real 3D military targets from planar false targets.

In the proposed simulation, true 3D targets are rendered as multi-gate image stacks according to visible depth and reflectance. Planar false targets are constructed by flattening the object geometry to a single camera-depth plane while preserving the visible silhouette. For rectangular laser pulses and rectangular receiver gates, the received gate response is modeled as the temporal overlap length between the two windows, producing a triangular response for equal window widths and a trapezoidal response for unequal widths. Therefore, a planar false target is not simulated as an arbitrary partial slice or manually linear fade; instead, the whole-object silhouette is retained across gates, and its intensity is sampled from the pulse-gate overlap response.

Experiments are conducted on 44 manually selected military 3D models, each paired with a corresponding planar false target. Raw v8 gated images still contain a strong single-gate scalar shortcut: the p99 feature of gate 2 reaches a threshold-classification accuracy of 0.9886. After per-gate maximum normalization, the strongest scalar shortcut is reduced to 0.7955 and changes from an intensity feature to edge/shape statistics. On the single-view dataset, the full three-gate stack achieves mean independent evaluation accuracies of 0.9697, 0.9545, and 0.7424 under clean, light-noise, and strong-noise conditions, respectively. After extending the same simulation to four yaw views per source model and enforcing model-level grouped splitting, the full three-gate stack reaches 0.9848, 0.9811, and 0.9205 under the same three conditions, outperforming the best single-gate baseline on average in all cases. However, the same clean/noisy-trained models collapse to chance accuracy under structured reflectance, background, and occlusion shifts. A domain-randomized training variant combining normal and hard-mild simulated domains with strong-noise augmentation recovers mean accuracies of 0.9394, 0.9242, 0.9091, 0.9394, and 0.7727 under clean, light-noise, strong-noise, hard-mild, and hard-strong evaluations, respectively. These results support the value of complete gate-stack observations in a controlled simulation setting, while also showing that simulation shortcuts, nuisance shifts, and network-design effects must be explicitly diagnosed before drawing physical conclusions.

Keywords: laser gated imaging; range-gated imaging; 3D target recognition; planar false target; Blender simulation; anti-shortcut validation; gate-stack fusion

## 1. Introduction

Military target recognition is often affected by pose variation, limited observation conditions, background clutter, and active deception. Conventional two-dimensional image recognition mainly relies on texture, contour, and local appearance cues. When a planar false target or decoy is visually similar to a real target from a given viewpoint, a single two-dimensional image may not provide enough evidence to determine whether the object has real three-dimensional structure.

Laser gated imaging offers an additional physical cue by controlling the temporal overlap between the transmitted laser pulse and the receiver gate. Different receiver gates correspond to different range windows, so a gated image stack contains both spatial appearance and depth-selective response information [Song2026GRICI]. For a real 3D target, different target parts can appear in different gates due to their nonzero depth distribution. For a planar false target located at a single depth plane, the full visible silhouette should remain more consistent across gates, with only the overall response varying according to the pulse-gate overlap function.

Recent learning-based methods can extract discriminative features from multi-frame or multi-depth observations, but direct training on simulated gated images has a critical risk: the classifier may exploit artifacts introduced by the simulator. For example, if the false target appears only in one gate and the remaining gates are nearly black, a network can solve the task by detecting black frames. If true and false classes differ in absolute brightness, simple scalar features such as maximum intensity or p99 intensity may be sufficient for classification. Such shortcuts can produce high validation accuracy without demonstrating that gated imaging itself provides meaningful physical discrimination [Geirhos2020Shortcut].

This study focuses on building a reproducible simulation and validation chain rather than proposing a complex neural architecture. The central question is:

```text
Can a physically interpretable gated-image stack provide more stable evidence than any individual gate for discriminating real 3D targets from planar false targets, after obvious simulation shortcuts are diagnosed and controlled?
```

The main contributions are:

1. A Blender-based gated-imaging simulation pipeline for selected military 3D models, including true 3D gate-stack generation and planar false-target construction by camera-depth flattening.
2. A rectangular pulse-gate overlap model that gives a physically interpretable gate response for both true 3D targets and planar false targets.
3. An anti-shortcut validation protocol including gate-stack diagnostics, single-gate scalar shortcut analysis, per-gate maximum normalization, single-gate neural ablations, and clean/noisy independent evaluation.
4. A lightweight full-stack neural baseline showing that the complete three-gate sequence outperforms single-gate inputs on average under clean, light-noise, and strong-noise controlled evaluations in both single-view and four-view validation settings.
5. A hard-nuisance boundary test and domain-randomized training protocol showing that explicit simulator-domain variation can improve robustness to structured reflectance, background, and occlusion shifts, while still preserving strong-noise performance.

The current study should be interpreted as a controlled simulation validation. It does not claim deployment-ready military recognition, nor does it claim that one fusion head is universally optimal across all imaging conditions.

## 2. Related Work

### 2.1 Laser range-gated and range-intensity correlation imaging

Laser range-gated imaging uses pulsed illumination and a time-controlled receiver gate to capture returns from selected range intervals. This mechanism suppresses out-of-gate backscatter and background interference, while encoding range-dependent information into gated image intensity. In three-dimensional laser gated range-intensity correlation imaging, range can be decoded from multiple gated images with designed range-intensity profiles. Representative physical models discuss triangular and trapezoidal range-intensity profiles, the role of rectangular laser and gate pulses, photoelectric conversion, shot noise, backscattering noise, and timing jitter [Song2026GRICI].

This line of work provides the physical basis for the present simulation, but its objective is different. Existing GRICI studies mainly aim at range reconstruction or range-precision prediction. The present study instead asks whether a physically interpretable gate stack can help distinguish a real 3D target from a planar false target after obvious simulation shortcuts are removed. Therefore, the rectangular pulse-gate response is used here as a controllable and interpretable imaging model rather than as a full hardware-precision model.

### 2.2 Learning-based 3D range-gated imaging and depth priors

Recent learning-based 3D range-gated imaging methods treat a stack of gated images as input for dense depth estimation. A representative direction combines optical range-gated priors with visual learning: the gate stack provides depth-zone cues, while a transformer-based architecture learns adaptive depth intervals and refines dense depth predictions [Liu2026DepthPriors]. In that framework, AdaBins-style adaptive binning converts continuous depth regression into a classification-regression problem over learned depth bins, while refined optical depth-prior masks and depth-perception priors constrain the learned depth distribution.

These methods support the view that gated images should not be treated as ordinary texture channels. Their value comes from a physical relationship among gate intensity, range window, and depth-dependent response. However, most learning-based 3DRGI work focuses on estimating depth maps, not on detecting planar false targets. It also does not directly address the risk that simulated gated-image classifiers may exploit black-frame or absolute-brightness shortcuts. The present study therefore complements depth-prior learning by adding shortcut diagnostics and single-gate ablations before interpreting binary classification accuracy.

### 2.3 Gated cameras in robust multimodal perception

Gated cameras have also been used as one modality in robust 3D object detection. In adverse-weather multimodal perception, gated near-infrared images can complement RGB, LiDAR, and radar because active gated illumination can remain useful in low-light, foggy, snowy, or rainy scenes [Palladin2024SAMFusion]. Attention-based and transformer-decoder fusion strategies further allow different modalities to be weighted according to distance, visibility, and degradation conditions.

This literature motivates the broader application direction of the present project: the gate stack can be viewed as a physically structured input for robust recognition and future multimodal optical-electronic fusion. Nevertheless, multimodal object detection and planar false-target discrimination are different tasks. Therefore, multimodal fusion work is used here as motivation and future context, not as direct evidence that the current binary classifier is deployment-ready.

### 2.4 Optical computing and diffractive depth-selective processing

A more distant but relevant direction is digital-optical co-design with diffractive decoders. Snapshot 3D image projection using a diffractive decoder shows that learned optical elements can route information to designated axial planes and suppress inter-plane crosstalk in a depth-resolved optical system [Isil2026DiffractiveDecoder]. This is not a direct precedent for the current gated-imaging classifier, but it is relevant to future optical neural-network integration because it demonstrates that depth-selective transformations can be partly implemented by optical propagation and learned physical elements.

Across these directions, a common validation risk remains: a neural model trained on synthetic or controlled data may exploit shortcuts that do not transfer to a different environment [Geirhos2020Shortcut]. Simulation-to-real work addresses part of this problem by increasing simulated variability, for example through domain randomization [Tobin2017DomainRandomization]. In the present study, the same concern motivates a narrower but necessary step: before discussing real-world transfer, the simulated dataset itself must be checked for black-frame and brightness shortcuts.

Taken together, prior work establishes that gated images contain physically meaningful depth cues, that learning-based 3DRGI can benefit from optical priors, and that gated cameras are useful in robust perception. The missing piece addressed by this paper is a controlled simulation and validation protocol for real-3D versus planar-false-target discrimination, with explicit checks against non-physical shortcuts.

## 3. Physical Model and Simulation

### 3.1 Rectangular pulse-gate response

Let the transmitted laser pulse be represented by a rectangular temporal window with width \(T_l\), and let the receiver gate be represented by another rectangular temporal window with width \(T_g\). If the target echo arrives at time \(\tau\) and the center of the \(i\)-th receiver gate is \(c_i\), the normalized response weight can be modeled as the temporal overlap length between the two windows:

\[
W_i(\tau)=
\frac{
\left|\mathrm{rect}_{T_l}(t-\tau)\cap \mathrm{rect}_{T_g}(t-c_i)\right|
}{
\min(T_l,T_g)
}.
\]

When \(T_l=T_g\), the response as a function of relative delay is triangular. When one window is wider than the other, the response becomes trapezoidal with a plateau. In the v8 setting used in this study, the receiver gate width is larger than the laser pulse width:

```text
ReceiverGateWidth = 1.5
LaserPulseWidth = 0.45
AutoGateMargin = 0.12
FlatMinResponse = 0.0
FlatEchoGain = 2.0
```

This formulation is important for planar false-target simulation. A planar false target should not be rendered as a partial object slice in each gate. Instead, because the false target is located at a single depth plane, the same visible silhouette should be modulated by the sampled gate response across the receiver gates [Song2026GRICI].

### 3.2 True 3D target gate stack

For a real 3D target, visible surface points occupy a range of camera depths. The gated image at gate \(i\) can be treated as a depth-weighted rendering:

\[
I_i(x,y)=R(x,y) W_i(\tau(x,y)),
\]

where \(R(x,y)\) denotes the visible reflectance or rendered intensity at pixel \((x,y)\), and \(\tau(x,y)\) is the echo arrival time corresponding to the local visible depth. Since \(\tau(x,y)\) varies across the object, different gates emphasize different depth regions.

In the implementation, Blender is used to load and normalize the 3D model, render top-view gated images, and export three grayscale gate images per sample:

\[
X=[I_0, I_1, I_2].
\]

### 3.3 Planar false-target construction

The planar false target is constructed by flattening the target geometry along the camera-depth direction:

```text
flat_geometry_mode = flatten-camera-depth
flat_geometry_depth_max - flat_geometry_depth_min = 0
```

This operation preserves the visible object silhouette while removing the physical depth distribution. The resulting false target is then rendered through the same rectangular pulse-gate response model. To avoid assigning all false targets to a fixed gate, a round-robin strategy places planar false targets near different gate centers:

```text
FlatTargetGateIndexMode = round-robin
```

Fig. 3 illustrates the intended behavior: the true 3D target changes its visible structure across gates, whereas the planar false target keeps the complete object silhouette and changes primarily in overall response.

## 4. Anti-Shortcut Validation Protocol

### 4.1 Why anti-shortcut controls are necessary

A high classification accuracy alone is not sufficient evidence that gated imaging provides useful physical information. Simulated data may contain artificial cues that are easier to learn than the intended depth structure. The present study therefore treats shortcut diagnosis as a required part of the method rather than as an optional analysis.

### 4.2 Gate-stack diagnostics

For each sample, the similarity among the three gate images is measured using normalized correlation, foreground mask intersection-over-union, and normalized absolute difference. A planar false target should have a more consistent silhouette across gates, while a true 3D target should show depth-related variation.

In the raw v8 dataset, the mean gate-stack diagnostics are:

| class | corr | mask IoU | absdiff | max/mean ratio |
|---|---:|---:|---:|---:|
| flat_false | 0.5848 | 0.5702 | 0.1177 | 103.8464 |
| true3d | 0.6736 | 0.6670 | 0.0934 | 9.3942 |

These values show that the raw simulation still contains intensity-related instability and must be further controlled.

### 4.3 Single-gate scalar shortcut diagnostics

For each gate, simple scalar features are tested using threshold classification. Features include maximum intensity, p95, p99, foreground mean, and edge density. If one scalar feature reaches high accuracy, the dataset contains an obvious shortcut.

In the raw v8 dataset, the strongest scalar shortcuts are:

| gate | strongest scalar feature | threshold acc | true3d mean | flat false mean | Cohen d |
|---:|---|---:|---:|---:|---:|
| 0 | max_value | 0.8636 | 0.3426 | 0.1440 | -2.041 |
| 1 | p95 | 0.8864 | 0.3177 | 0.1597 | -2.291 |
| 2 | p99 | 0.9886 | 0.3055 | 0.1103 | -2.886 |

The gate-2 p99 shortcut is strong enough to make raw-image classification unreliable as physical evidence.

### 4.4 Per-gate maximum normalization

To suppress absolute brightness shortcuts, each gate image is normalized by its own maximum value to a common target range:

```text
dataset_new/normalize_gate_dataset.py
--mode per-gate-max
--target-max 180
--min-source-max 2
```

This produces the controlled dataset:

```text
dataset_new/Military_TrueFalse_Selected44_blender_refl_overlap_w15_m012_v8_per_gate_maxnorm
```

After normalization, the strongest scalar single-gate shortcut decreases from 0.9886 to 0.7955:

| gate | strongest scalar feature | threshold acc | true3d mean | flat false mean | Cohen d |
|---:|---|---:|---:|---:|---:|
| 0 | edge_density | 0.7955 | 0.3664 | 0.1357 | -1.398 |
| 1 | p99 | 0.7386 | 0.6874 | 0.6004 | -0.573 |
| 2 | edge_density | 0.7841 | 0.3499 | 0.1559 | -1.194 |

The remaining separability is mostly related to edge or shape statistics, which is expected because single-gate images can still carry class-specific structure. This motivates neural single-gate ablations.

## 5. Neural Baseline and Training Setup

### 5.1 Input format

Each sample is represented as a stack of three single-channel gate images:

\[
X \in \mathbb{R}^{S \times 1 \times H \times W}, \quad S=3.
\]

The full-stack model processes all three gates, while single-gate baselines use only one selected gate.

### 5.2 Network architecture

The current network is a lightweight electronic baseline rather than the main novelty of the study. A shared CNN encoder extracts features from each gate:

\[
f_i = E(I_i).
\]

Three fusion modes are evaluated:

1. `mean`: average pooling of gate-level features.
2. `attention`: learned gate-level discriminative weighting.
3. `attention_residual`: attention fusion plus a residual projection from concatenated gate features.

The classifier head is a lightweight MLP producing two logits corresponding to `true3d` and `flat_false`. The final class probability can be obtained by softmax during inference, while training uses cross-entropy loss on logits. The current model is intentionally kept lightweight because the goal is to validate gate-stack information and shortcut controls rather than to compete with transformer-based 3DRGI architectures [Liu2026DepthPriors].

### 5.3 Data split and augmentation

The dataset contains 44 selected military 3D models. Each model contributes one true 3D gate stack and one corresponding planar false-target gate stack:

| class | samples | gate images |
|---|---:|---:|
| true3d | 44 | 132 |
| flat_false | 44 | 132 |

To reduce the risk of a fixed-view shape shortcut, a four-view validation dataset is also generated using yaw rotations of 0, 90, 180, and 270 degrees. This produces 176 samples per class and 1056 gate images in total:

| class | samples | gate images |
|---|---:|---:|
| true3d | 176 | 528 |
| flat_false | 176 | 528 |

The four-view dataset is stored as:

```text
dataset_new/Military_TF_v8_mv4_norm
```

In this split, all rendered views of the same source model are assigned to the same partition to avoid cross-view leakage.

Training and validation are split by `sample_id` to avoid leakage from the same source model. Each main experiment is repeated with three random seeds:

```text
42 / 332 / 2026
```

Mixed clean/noisy augmentation is used:

```text
gaussian_noise_std = 0.02
poisson_peak = 80
background_scatter = 0.02
degradation_probability = 0.5
```

Independent evaluation uses three conditions:

| condition | gaussian | poisson peak | background scatter |
|---|---:|---:|---:|
| clean | 0.00 | 0 | 0.00 |
| light noise | 0.02 | 80 | 0.02 |
| strong noise | 0.05 | 30 | 0.05 |

To test whether clean/noisy validation is sufficient, two structured hard-nuisance datasets are also generated from the normalized four-view v8 data. These variants introduce low-frequency reflectance texture, weak background scatter, and partial occlusion while preserving the per-sample maximum intensity. The same nuisance key is used for paired true/false samples derived from the same source model.

A domain-randomized training dataset is then constructed by combining the normal four-view domain and the hard-mild domain:

```text
dataset_new/Military_TF_v8_mv4_norm_plus_hardv3_mild
```

This dataset contains 704 samples and 2112 gate images. Each sample is prefixed with its domain label, for example `domain_norm__...` or `domain_hardv3__...`. During grouped splitting, both the domain prefix and the view prefix are removed before assigning groups, so normal/hard variants and all yaw views of the same source model remain in the same train or validation partition. The domain-randomized strong-noise training setting uses:

```text
gaussian_noise_std = 0.05
poisson_peak = 30
background_scatter = 0.05
degradation_probability = 0.5
```

## 6. Results

### 6.1 Mixed augmentation improves the robustness story

A clean-only full-stack model performs well on clean validation but collapses under noisy independent evaluation:

| evaluation condition | full-stack acc |
|---|---:|
| clean | 1.0000 |
| light noise | 0.5000 |
| strong noise | 0.5000 |

A fully noisy-trained full-stack model over-specializes to the noisy distribution:

| evaluation condition | full-stack acc |
|---|---:|
| clean | 0.6364 |
| light noise | 1.0000 |
| strong noise | 0.5455 |

The mixed clean/noisy augmentation setting better balances clean and degraded conditions and is therefore used as the main evaluation setting.

### 6.2 Full gate stack versus single-gate baselines

Training aggregate over three seeds:

| input | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| Full 3-gate stack | 0.9697 | 0.0525 | 0.9091 | 1.0000 |
| Gate 0 only | 0.9091 | 0.0909 | 0.8182 | 1.0000 |
| Gate 1 only | 0.7273 | 0.0909 | 0.6364 | 0.8182 |
| Gate 2 only | 0.8939 | 0.0525 | 0.8636 | 0.9545 |

Independent evaluation aggregate:

| evaluation condition | input | mean acc | std | min | max |
|---|---|---:|---:|---:|---:|
| clean | Full 3-gate stack | 0.9697 | 0.0525 | 0.9091 | 1.0000 |
| clean | Gate 0 only | 0.9091 | 0.0909 | 0.8182 | 1.0000 |
| clean | Gate 1 only | 0.7121 | 0.0263 | 0.6818 | 0.7273 |
| clean | Gate 2 only | 0.8636 | 0.0455 | 0.8182 | 0.9091 |
| light noise | Full 3-gate stack | 0.9545 | 0.0455 | 0.9091 | 1.0000 |
| light noise | Gate 0 only | 0.8636 | 0.1202 | 0.7273 | 0.9545 |
| light noise | Gate 1 only | 0.5151 | 0.0694 | 0.4545 | 0.5909 |
| light noise | Gate 2 only | 0.7727 | 0.0909 | 0.6818 | 0.8636 |
| strong noise | Full 3-gate stack | 0.7424 | 0.1389 | 0.5909 | 0.8636 |
| strong noise | Gate 0 only | 0.6970 | 0.1144 | 0.5909 | 0.8182 |
| strong noise | Gate 1 only | 0.5152 | 0.0263 | 0.5000 | 0.5455 |
| strong noise | Gate 2 only | 0.5152 | 0.0263 | 0.5000 | 0.5455 |

The full three-gate stack achieves the highest mean accuracy under all three independent evaluation conditions. The result supports the usefulness of complete gate-stack information beyond a single selected range slice.

### 6.3 Fusion-mode comparison

Validation aggregate:

| fusion mode | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| attention | 0.9697 | 0.0525 | 0.9091 | 1.0000 |
| mean | 0.9545 | 0.0788 | 0.8636 | 1.0000 |
| attention_residual | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

Independent evaluation aggregate:

| evaluation condition | fusion mode | mean acc | std | min | max |
|---|---|---:|---:|---:|---:|
| clean | attention | 0.9697 | 0.0525 | 0.9091 | 1.0000 |
| clean | mean | 0.9545 | 0.0788 | 0.8636 | 1.0000 |
| clean | attention_residual | 0.9848 | 0.0263 | 0.9545 | 1.0000 |
| light noise | attention | 0.9545 | 0.0455 | 0.9091 | 1.0000 |
| light noise | mean | 0.9697 | 0.0525 | 0.9091 | 1.0000 |
| light noise | attention_residual | 0.9545 | 0.0455 | 0.9091 | 1.0000 |
| strong noise | attention | 0.7424 | 0.1389 | 0.5909 | 0.8636 |
| strong noise | mean | 0.6515 | 0.0694 | 0.5909 | 0.7273 |
| strong noise | attention_residual | 0.6970 | 0.1721 | 0.5000 | 0.8182 |

The single-view fusion-mode comparison should be interpreted cautiously. `attention_residual` is the strongest validation candidate and has the best clean-test accuracy. `mean` is slightly better under light noise, while `attention` is more robust under strong noise. Thus, the safe claim is that the full gate stack is useful; the final fusion strategy remains an engineering choice.

### 6.4 Four-view validation

The four-view v8 dataset tests whether the gate-stack advantage remains when each source model is rendered from four yaw orientations. The training split is grouped by the underlying source model rather than by the view-labeled sample name, so different views of the same model cannot appear in both training and validation.

Training aggregate over three seeds:

| input | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| Full 3-gate stack | 0.9811 | 0.0328 | 0.9432 | 1.0000 |
| Gate 0 only | 0.9583 | 0.0537 | 0.8977 | 1.0000 |
| Gate 1 only | 0.9356 | 0.0365 | 0.9091 | 0.9773 |
| Gate 2 only | 0.9053 | 0.0174 | 0.8864 | 0.9205 |

Independent evaluation aggregate:

| evaluation condition | Full 3-gate stack | Gate 0 only | Gate 1 only | Gate 2 only |
|---|---:|---:|---:|---:|
| clean | 0.9848 | 0.9697 | 0.9545 | 0.8939 |
| light noise | 0.9811 | 0.9432 | 0.9053 | 0.8750 |
| strong noise | 0.9205 | 0.7462 | 0.5265 | 0.6098 |

The four-view result strengthens the main claim. The full gate stack remains the best mean performer under clean, light-noise, and strong-noise evaluation, and the strong-noise margin over the best single-gate baseline increases substantially. At the same time, single-gate inputs remain informative under clean and light-noise conditions, so the result should be interpreted as evidence that the gate stack adds useful depth-gated information, not as evidence that all single-frame shape shortcuts have been eliminated.

### 6.5 Four-view fusion comparison

The full-stack fusion modes are also compared on the four-view dataset. This experiment checks whether the earlier attention-residual preference remains valid after viewpoint expansion.

Training aggregate over three seeds:

| fusion mode | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| attention | 0.9811 | 0.0328 | 0.9432 | 1.0000 |
| mean | 0.9848 | 0.0262 | 0.9545 | 1.0000 |
| attention_residual | 0.9659 | 0.0495 | 0.9091 | 1.0000 |

Independent evaluation aggregate:

| evaluation condition | attention | mean | attention_residual |
|---|---:|---:|---:|
| clean | 0.9848 | 0.9848 | 0.9848 |
| light noise | 0.9811 | 0.9848 | 0.9394 |
| strong noise | 0.9205 | 0.7197 | 0.7727 |

The four-view fusion result makes the network conclusion more conservative. The attention-residual head is not consistently superior after viewpoint expansion; it performs worse than simple attention under strong noise. Therefore, the neural architecture should be presented as a lightweight electronic baseline and engineering ablation, while the main contribution remains the physics-interpretable simulation and anti-shortcut validation framework.

### 6.6 Hard-nuisance boundary test

A final stress test is added to identify the current robustness boundary. Two hard-nuisance four-view datasets are generated from the normalized v8 dataset by applying deterministic spatial reflectance texture, weak background scatter, and partial occlusion to both `true3d` and `flat_false` samples. The nuisance key is shared for paired true/false samples from the same source model, and per-sample maximum intensity is preserved to avoid introducing a new max-brightness shortcut.

The saved four-view attention models trained on the normalized v8 dataset are evaluated directly on these hard-nuisance datasets:

| condition | Full 3-gate stack | Gate 0 only | Gate 1 only | Gate 2 only |
|---|---:|---:|---:|---:|
| hard nuisance v2 | 0.5000 | 0.4697 | 0.5000 | 0.4545 |
| hard nuisance v3 mild | 0.5000 | 0.4848 | 0.5000 | 0.4545 |

This is a failure-boundary result. The current clean/noisy-trained models remain robust to the additive noise and background-scatter evaluation used in the main v8 protocol, but they do not generalize to structured reflectance, background, and occlusion shifts without additional nuisance-aware training or simulation. This result motivates the next stage of the work: incorporating these nuisance factors directly into the physical simulator and training protocol.

### 6.7 Domain-randomized training improves the robustness boundary

The hard-nuisance failure indicates that the original clean/noisy training distribution does not cover structured reflectance, background, and occlusion shifts. Two follow-up strategies are therefore tested. The first applies structured nuisance perturbations online after loading the normal dataset. The second explicitly mixes two simulated domains: the normal four-view dataset and the hard-mild four-view dataset, while using strong-noise augmentation during training.

The training aggregate of the domain-mixed strong-noise model over three seeds is:

| strategy | mean best val acc | std | min | max |
|---|---:|---:|---:|---:|
| domain mix + strong noise | 0.9280 | 0.0280 | 0.9091 | 0.9602 |

Independent evaluation compares three training strategies:

| strategy | clean | light noise | strong noise | hard mild | hard strong |
|---|---:|---:|---:|---:|---:|
| normal mixaug | 0.9848 | 0.9811 | 0.9205 | 0.5000 | 0.5000 |
| online nuisance | 0.9697 | 0.7879 | 0.5606 | 0.5000 | 0.5000 |
| domain mix + strong noise | 0.9394 | 0.9242 | 0.9091 | 0.9394 | 0.7727 |

The normal mixed-noise model remains the strongest under clean and light-noise evaluations, and it is slightly higher under the original strong-noise condition. However, it collapses to chance under both hard-nuisance datasets. The online nuisance strategy does not recover hard-nuisance performance in its current implementation and also weakens strong-noise robustness. In contrast, the domain-mixed strong-noise model substantially improves the hard-nuisance boundary: hard-mild accuracy increases from 0.5000 to 0.9394, and hard-strong accuracy increases from 0.5000 to 0.7727, while strong-noise accuracy remains close to the original baseline.

This result changes the manuscript story. The correct claim is not that the model is generally robust, but that explicit domain-randomized simulation can improve the robustness boundary compared with clean/noisy-only training. The remaining gap on the hard-strong condition also shows that structured nuisance simulation must be treated as a core part of the imaging model, not as a minor post-processing augmentation.

## 7. Discussion

The current evidence supports a controlled but meaningful conclusion: after brightness shortcuts are diagnosed and reduced, the full three-gate stack provides more stable discriminative information than any single-gate input for separating real 3D targets from planar false targets in the current Blender simulation. This conclusion is supported by both the original single-view setting and the four-view validation setting.

The key methodological point is that simulation design and validation controls are inseparable. If the raw v8 images were used directly, the network could exploit the gate-2 p99 shortcut. If only full-stack accuracy were reported, it would be unclear whether the classifier truly benefited from multi-gate information. The combination of scalar shortcut diagnostics, per-gate normalization, single-gate ablations, and independent noise evaluation provides a more defensible evidence chain.

The hard-nuisance and domain-randomization experiments extend this evidence chain. The hard-nuisance test exposes a failure mode that is invisible under the original clean/noisy protocol. The domain-mixed strong-noise training then shows that the failure is not simply caused by a lack of discriminative information in the gate stack; rather, it reflects a distribution mismatch between the training simulation and structured test conditions. This supports an important methodological claim: for simulated gated-imaging recognition, the physical simulator must include nuisance variability before robustness claims are made.

The work also clarifies the role of the neural network. The current model is not proposed as a novel deep architecture. It is an electronic baseline used to test whether the simulated gated-image stack contains useful discriminative information. This positioning is important for future integration with multimodal optical neural networks: the present pipeline can provide physically structured gate-stack inputs and benchmark electronic fusion strategies before optical front-end integration [Palladin2024SAMFusion, Isil2026DiffractiveDecoder].

## 8. Limitations

The current study has several limitations.

First, the dataset is small. Only 44 manually selected military models are used, so the results are suitable for controlled validation but not for deployment-level claims.

Second, the four-view dataset only covers yaw rotations under the same top-view gated-imaging setup. It reduces a fixed-view shortcut but does not cover arbitrary elevation changes, object articulation, or sensor placement changes. A larger viewpoint protocol is still required before making broad pose-generalization claims.

Third, the hard-nuisance boundary test shows that clean/noisy-trained models are not robust to structured reflectance, background, and occlusion shifts when these factors are introduced after training. Domain-randomized training improves this boundary, but the hard-strong condition remains lower than clean/noise validation. This means that the present result should be interpreted as simulation-domain robustness improvement rather than realistic battlefield robustness.

Fourth, the physical model is simplified. The current simulation uses rectangular laser pulses and rectangular receiver gates. Real systems may include pulse broadening, detector response, atmospheric attenuation, background scattering, material BRDF differences, photoelectron statistics, and timing jitter that are not fully represented here [Song2026GRICI].

Fifth, the planar false target is idealized. It preserves the visible silhouette and collapses depth, but real decoys may have surface texture, partial deformation, nonuniform reflectance, folds, or support structures.

## 9. Conclusion

This study presents a physics-interpretable simulation and validation framework for gated-imaging-based discrimination between 3D targets and planar false targets. True 3D targets are simulated as depth-dependent gate stacks, while planar false targets are generated by camera-depth flattening and rectangular pulse-gate response sampling. The experiments show that raw simulated images can contain strong single-gate intensity shortcuts, and that anti-shortcut controls are necessary before interpreting neural classification results. After per-gate maximum normalization and mixed clean/noisy augmentation, the complete three-gate stack outperforms single-gate baselines on average under clean, light-noise, and strong-noise independent evaluations in both single-view and four-view settings. Hard-nuisance evaluation further shows that clean/noisy robustness does not imply robustness to structured reflectance, background, and occlusion shifts. Domain-mixed strong-noise training improves this robustness boundary, reaching 0.9394 on hard-mild and 0.7727 on hard-strong evaluation while preserving 0.9091 strong-noise accuracy. These results support the value of multi-gate gated imaging and domain-randomized simulation in a controlled setting, and provide a foundation for broader viewpoint validation, larger military model datasets, and integration with multimodal optical neural-network recognition systems.

## References

[Song2026GRICI] Bo Song, Yue Zhang, Xinwei Wang, Liang Sun, Yurun Zhang, and Yan Zhou, "Range precision prediction model for three-dimensional laser gated range-intensity correlation imaging," Optics Express 34(1), 707-717 (2026). DOI: 10.1364/OE.582218.

[Liu2026DepthPriors] Xiaoquan Liu, Di Zhang, Jinming Gao, and Xinwei Wang, "Transformer-based 3D range-gated imaging method with multiple depth priors," Optics & Laser Technology 190, 113234 (2026). DOI: 10.1016/j.optlastec.2025.113234.

[Palladin2024SAMFusion] Edoardo Palladin, Roland Dietze, Praveen Narayanan, Mario Bijelic, and Felix Heide, "SAMFusion: Sensor-Adaptive Multimodal Fusion for 3D Object Detection in Adverse Weather," European Conference on Computer Vision (ECCV), 2024.

[Isil2026DiffractiveDecoder] Cagatay Isil, Alexander Chen, Yuhang Li, F. Onuralp Ardic, Shiqi Chen, Che-Yung Shen, and Aydogan Ozcan, "Snapshot 3D image projection using a diffractive decoder," Light: Science & Applications 15, 270 (2026). DOI: 10.1038/s41377-026-02378-3.

[Geirhos2020Shortcut] Robert Geirhos, Joern-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann, "Shortcut learning in deep neural networks," Nature Machine Intelligence 2, 665-673 (2020). DOI: 10.1038/s42256-020-00257-z.

[Tobin2017DomainRandomization] Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel, "Domain randomization for transferring deep neural networks from simulation to the real world," 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017. DOI: 10.1109/IROS.2017.8202133.

## Figure Plan

| figure | file | purpose |
|---|---|---|
| Fig. 1 | `writing/figures/fig1_gated_imaging_framework.png` | Overall simulation and validation framework |
| Fig. 2 | `writing/figures/fig2_rectangular_overlap_response.png` | Rectangular pulse-gate overlap response |
| Fig. 3 | `writing/figures/fig3_true3d_flatfalse_gate_examples.png` | Example true 3D and planar false gate stacks |
| Fig. 4 | `writing/figures/fig4_scalar_shortcut_control.png` | Raw v8 versus per-gate maxnorm scalar shortcut comparison |
| Fig. 5 | `writing/figures/fig5_full_stack_vs_single_gate_robustness.png` | Full stack versus single-gate robustness |
| Fig. 6 | `writing/figures/fig6_full_stack_fusion_robustness.png` | Fusion-mode comparison |
| Fig. 7 | `writing/figures/fig7_mv4_full_stack_vs_single_gate_robustness.png` | Four-view full stack versus single-gate robustness |
| Fig. 8 | `writing/figures/fig8_mv4_full_stack_fusion_robustness.png` | Four-view full-stack fusion comparison |
| Fig. 9 | `writing/figures/fig9_hard_nuisance_failure_boundary.png` | Hard-nuisance failure boundary |
| Fig. 10 | `writing/figures/fig10_domain_randomization_strategy_comparison.png` | Domain-randomized training strategy comparison |

## Evidence Files

```text
writing/v8_mixaug_robustness_report_2026-07-07.md
writing/figure_captions_v8_mixaug_2026-07-07.md
writing/literature_evidence_matrix_gated_false_target_2026-07-07.md
writing/sci_claims_evidence_matrix_2026-07-07.md
experiments/v8_per_gate_maxnorm_mixaug_eval_aggregate_3seed.csv
experiments/v8_per_gate_maxnorm_full_fusion_mixaug_eval_aggregate_3seed.csv
dataset_new/military_true_false_selected44_blender_refl_overlap_w15_m012_v8_single_gate_feature_separability.csv
dataset_new/military_true_false_selected44_blender_refl_overlap_w15_m012_v8_per_gate_maxnorm_single_gate_feature_separability.csv
writing/multiview_v8_validation_plan_2026-07-07.md
writing/v8_mv4_multiview_robustness_report_2026-07-07.md
writing/v8_mv4_hard_nuisance_boundary_report_2026-07-08.md
writing/v8_mv4_domain_randomization_strategy_report_2026-07-08.md
experiments/v8_mv4_norm_mixaug_attention_eval_summary_3seed.csv
experiments/v8_mv4_norm_mixaug_attention_eval_aggregate_3seed.csv
experiments/v8_mv4_norm_mixaug_full_fusion_eval_summary_3seed.csv
experiments/v8_mv4_norm_mixaug_full_fusion_eval_aggregate_3seed.csv
experiments/v8_mv4_hard_nuisance_v2_eval_aggregate_3seed.csv
experiments/v8_mv4_hard_nuisance_v3_mild_eval_aggregate_3seed.csv
experiments/v8_mv4_strategy_comparison_aggregate_2026-07-08.csv
experiments/v8_mv4_domainmix_norm_hardv3_strongaug_attention_full_eval_summary_3seed.csv
experiments/v8_mv4_domainmix_norm_hardv3_strongaug_attention_full_eval_aggregate_3seed.csv
experiments/v8_mv4_domainmix_norm_hardv3_strongaug_attention_full_20ep/results.csv
experiments/v8_mv4_domainmix_norm_hardv3_strongaug_attention_full_20ep/aggregate_results.csv
scripts/run_v8_multiview_dataset.ps1
scripts/run_v8_mv4_eval_grid.ps1
scripts/run_v8_mv4_fusion_experiments.ps1
scripts/evaluate_v8_mv4_fusion_grid.py
scripts/run_v8_mv4_hard_nuisance_eval.ps1
scripts/run_v8_domainmix_strongaug_eval.ps1
scripts/run_v8_domainmix_single_gate_ablation.ps1
scripts/make_v8_hard_nuisance_boundary_figure.py
scripts/make_v8_strategy_comparison.py
scripts/summarize_eval_grid.py
dataset_new/build_multiview_true_false_dataset.py
dataset_new/build_hard_nuisance_dataset.py
dataset_new/build_variant_mixture_dataset.py
dataset_new/Military_TF_v8_mv4_norm
dataset_new/Military_TF_v8_mv4_norm_plus_hardv3_mild
```

## Next Validation Tasks

1. Extend domain randomization beyond the current normal + hard-mild mixture, including held-out nuisance strengths and nuisance types.
2. Train domain-randomized single-gate baselines if compute time allows, so Fig. 10 can be paired with a full-stack versus single-gate robustness ablation.
3. Extend the view protocol beyond four yaw-only views if compute time allows.
4. Add final Zotero-verified references for military false targets, decoys, and simulation-to-real validation.
5. Convert this Markdown draft into a complete submission-style manuscript with figure captions and reference formatting.

## Citation Tasks Before Submission

The current draft now includes four working references from the local literature folder and two general machine-learning references for shortcut learning and synthetic-to-real validation risk. Before submission, all references must be exported from Zotero or publisher pages in the target journal format. Remaining citation groups:

1. Laser/range-gated imaging principles and applications.
2. 3D target recognition and depth-assisted recognition.
3. False target, decoy, and deception detection.
4. Dataset bias and shortcut diagnostics in scientific imaging.
5. Military decoy, false-target, or deception detection.
6. Synthetic optical-imaging validation and simulation-to-real transfer.
