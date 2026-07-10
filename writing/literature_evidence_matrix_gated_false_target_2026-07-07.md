# Literature evidence matrix for gated false-target discrimination

> Date: 2026-07-07  
> Purpose: connect the local literature folder with the current SCI-style manuscript on laser gated imaging simulation, anti-shortcut validation, and 3D/planar false-target discrimination.

## 1. Overall relevance ranking

| rank | local file | paper | relevance to current work | recommended use |
|---:|---|---|---|---|
| 1 | `E:\wjz\非线性光学\文献\文献4\full.md` | Transformer-based 3D range-gated imaging method with multiple depth priors | Very high | Core related work for gated-image depth priors, three-gate input, AdaBins-style adaptive depth bins, and transformer-based prior integration |
| 2 | `E:\wjz\非线性光学\文献\文献1\full.md` | Range precision prediction model for three-dimensional laser gated range-intensity correlation imaging | Very high | Core physical reference for laser range-gated imaging, GRICI, triangular/trapezoidal range-intensity profiles, and noise-aware imaging physics |
| 3 | `E:\wjz\非线性光学\文献\文献2\full.md` | SAMFusion: Sensor-Adaptive Multimodal Fusion for 3D Object Detection in Adverse Weather | Medium-high | Background reference for gated cameras in adverse conditions and attention/transformer-based multimodal fusion |
| 4 | `E:\wjz\非线性光学\文献\文献3\full.md` | Snapshot 3D image projection using a diffractive decoder | Medium-low for the current experiment; high for future optical-neural direction | Future-work reference for digital-optical co-design, diffractive decoders, optical depth/axial routing, and optical neural network integration |

For the current paper, the most defensible three-paper set is: 文献4 + 文献1 + 文献2. 文献3 should appear only in future-work discussion unless the manuscript is expanded toward optical computing hardware.

## 2. Evidence extracted from each paper

### 2.1 文献1: Range precision prediction model for three-dimensional laser gated range-intensity correlation imaging

Verified bibliographic information:

```text
Bo Song, Yue Zhang, Xinwei Wang, Liang Sun, Yurun Zhang, and Yan Zhou,
"Range precision prediction model for three-dimensional laser gated range-intensity correlation imaging,"
Optics Express 34(1), 707-717 (2026).
DOI: 10.1364/OE.582218
```

Core content:

- Laser range-gated 3D imaging can suppress backscatter and background interference through gate viewing.
- Range-intensity correlation imaging can reconstruct 3D information from at least two gated images with suitable range-intensity profiles.
- The paper explicitly discusses triangular and trapezoidal GRICI methods.
- The physical model includes laser emission, propagation, target reflection, gated detection, photoelectric conversion, shot noise, backscattering noise, and temporal jitter.

How it supports our work:

- It justifies using rectangular laser/gate overlap as a physically interpretable response model.
- It supports the correction that equal-width rectangular windows produce triangular response, while unequal windows can produce trapezoidal response.
- It gives a strong physical basis for saying gated-image intensity is range-dependent, not merely a visual texture signal.
- It also reminds us that the current Blender model is simplified because it does not fully include photoelectron statistics, temporal jitter, atmospheric attenuation, and medium backscatter.

Suggested manuscript use:

- Cite in the physical model section when defining the rectangular pulse-gate overlap response.
- Cite in the limitations section when stating that the current simulation omits detailed photoelectric and propagation noise.
- Do not claim our simulation reaches the range-precision modeling level of this paper.

### 2.2 文献4: Transformer-based 3D range-gated imaging method with multiple depth priors

Verified bibliographic information:

```text
Xiaoquan Liu, Di Zhang, Jinming Gao, and Xinwei Wang,
"Transformer-based 3D range-gated imaging method with multiple depth priors,"
Optics & Laser Technology 190, 113234 (2026).
DOI: 10.1016/j.optlastec.2025.113234
```

Core content:

- The method takes three gated images as input and estimates a refined depth map.
- It argues that purely vision-learning-based 3DRGI methods underuse the optical principle of range-gated imaging.
- It introduces multiple priors: refined optical depth prior, depth perception prior, and adaptive depth intervals.
- It adapts AdaBins: depth estimation is treated as a classification-regression problem over adaptive depth bins.
- Transformer/mViT modules are used to learn adaptive bin widths and range-attention maps.
- The optical-prior consistency idea penalizes predictions whose depth-zone assignment conflicts with the refined optical depth-prior masks.

How it supports our work:

- It is the closest conceptual support for using gate stacks as physically meaningful inputs rather than ordinary image channels.
- It supports our project direction: combine physical gated-imaging priors with neural models instead of relying on black-box visual learning only.
- It suggests a future upgrade path from binary true/false classification to auxiliary depth-zone supervision or gate-consistency loss.

Suggested manuscript use:

- Cite in related work under learning-based 3D range-gated imaging and depth priors.
- Use it to justify why our current anti-shortcut protocol matters: if a model uses gated images but ignores optical consistency, it may learn visual shortcuts.
- In future work, propose an auxiliary prior branch inspired by optical depth-prior masks and adaptive depth bins.

What not to overclaim:

- This paper estimates depth maps; our current task is binary discrimination between real 3D and planar false targets.
- This paper uses transformer-based depth estimation; our current network is a lightweight CNN gate-fusion baseline.

### 2.3 文献2: SAMFusion

Verified bibliographic information:

```text
Edoardo Palladin, Roland Dietze, Praveen Narayanan, Mario Bijelic, and Felix Heide,
"SAMFusion: Sensor-Adaptive Multimodal Fusion for 3D Object Detection in Adverse Weather,"
European Conference on Computer Vision (ECCV), 2024.
```

Core content:

- SAMFusion uses RGB, LiDAR, radar, and NIR gated camera modalities for 3D object detection under adverse weather.
- It emphasizes that gated cameras can provide useful observations under low light, fog, snow, rain, and other degraded conditions.
- It uses attention-based and transformer-decoder fusion to weight different modalities according to distance and visibility.
- It evaluates object detection under adverse weather and reports improved robustness in challenging conditions.

How it supports our work:

- It supports the practical value of gated cameras for robust perception.
- It supports presenting gate-stack fusion as part of a broader robust-perception direction.
- It helps connect this project to the user's later multimodal optical-neural-network project.

Suggested manuscript use:

- Cite in related work as multimodal robust perception using gated cameras.
- Use it to motivate the future integration of our gate-stack simulator with multimodal fusion or optical-electronic networks.
- Do not use it as direct evidence for false-target discrimination, because the task and dataset are different.

### 2.4 文献3: Snapshot 3D image projection using a diffractive decoder

Verified bibliographic information:

```text
Çağatay Işıl, Alexander Chen, Yuhang Li, F. Onuralp Ardic, Shiqi Chen,
Che-Yung Shen, and Aydogan Ozcan,
"Snapshot 3D image projection using a diffractive decoder,"
Light: Science & Applications 15, 270 (2026).
DOI: 10.1038/s41377-026-02378-3
```

Core content:

- The work co-designs a digital encoder and a physical diffractive decoder.
- The system projects different images to multiple axial planes in a single snapshot.
- A learned passive diffractive decoder helps route optical energy to intended depth planes and suppress inter-plane crosstalk.
- The paper emphasizes digital-optical co-design, axial separability, and passive optical computation.

How it supports our work:

- It is not direct support for gated-image simulation or false-target classification.
- It is useful for the future optical neural network direction because it shows how optical propagation and learned physical elements can perform depth-selective transformation.
- It can help frame a long-term path: current electronic gate-stack classifier -> optical/electronic fusion benchmark -> possible diffractive or multimodal optical front end.

Suggested manuscript use:

- Mention only in future work or broader optical computing discussion.
- Avoid using it in the main related-work core unless the manuscript explicitly includes optical neural network hardware as a major theme.

## 3. How these papers sharpen our current story

The current project should be positioned as follows:

```text
Existing 3DRGI work shows that gated images contain range-dependent physical information
and that optical depth priors can improve learning-based depth estimation.
However, for simulated 3D target / planar false-target discrimination, high neural accuracy
can be caused by non-physical shortcuts such as black gates or brightness differences.
Therefore, this work contributes a physics-interpretable Blender simulation and
anti-shortcut validation protocol before claiming gate-stack discrimination ability.
```

This story is stronger than simply saying "we trained a network." It says:

1. The physical cue is real: range-gated imaging encodes range through pulse-gate response.
2. Learning from gate stacks is reasonable: existing work already uses gated images and priors for depth estimation.
3. The gap is specific: existing learning methods usually focus on depth/object detection, not controlled false-target discrimination and simulation shortcut diagnosis.
4. The contribution is defensible: our main novelty is simulation correction plus anti-shortcut validation, with a lightweight network as evidence.

## 4. Concrete upgrade ideas for the next version

Short-term, feasible on the current codebase:

- Add a gate-consistency diagnostic feature: compare inter-gate mask IoU, correlation, and normalized absolute difference between true3d and flat_false samples.
- Add a scalar-shortcut table to every dataset version before network training.
- Train and report full-stack versus single-gate baselines on the multi-view v8 dataset.
- Keep the current network as a baseline and avoid calling it a new architecture.

Medium-term, paper-strengthening upgrade:

- Add an auxiliary "gate-depth-zone consistency" branch inspired by optical depth-prior masks.
- Predict whether each foreground pixel belongs to near/mid/far response zone, then use this as auxiliary supervision for true 3D samples.
- For planar false targets, enforce or measure whole-silhouette gate consistency rather than forcing artificial disappearance.

Long-term, project-fusion direction:

- Use SAMFusion as the conceptual bridge to multimodal robust perception.
- Use diffractive-decoder work as inspiration for optical/electronic co-design, but keep it out of the current main claims unless hardware experiments are added.

## 5. Citation readiness

| citation key | status | action before submission |
|---|---|---|
| `[Song2026GRICI]` | usable | Verify final page range and DOI formatting from Optics Express |
| `[Liu2026DepthPriors]` | usable | Verify final issue/month and DOI formatting from Optics & Laser Technology |
| `[Palladin2024SAMFusion]` | usable | Verify ECCV proceedings citation format |
| `[Isil2026DiffractiveDecoder]` | usable | Verify final publisher formatting before formal reference list |
| `[Geirhos2020Shortcut]` | usable | Use to support shortcut-learning risk in simulated gated-image classification |
| `[Tobin2017DomainRandomization]` | usable | Use to support synthetic-to-real validation and simulation variability discussion |

## 6. Non-local references added to support validation logic

Two general machine-learning references were added to the manuscript because they support the anti-shortcut and synthetic-data validation argument:

```text
Robert Geirhos et al.,
"Shortcut learning in deep neural networks,"
Nature Machine Intelligence 2, 665-673 (2020).
DOI: 10.1038/s42256-020-00257-z
```

```text
Josh Tobin et al.,
"Domain randomization for transferring deep neural networks from simulation to the real world,"
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017.
DOI: 10.1109/IROS.2017.8202133
```

These references should not replace gated-imaging references. Their role is to justify why the paper needs scalar shortcut diagnostics, single-gate ablations, normalization controls, and future multi-view validation.
