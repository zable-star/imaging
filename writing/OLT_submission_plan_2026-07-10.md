# Optics & Laser Technology Submission Plan - 2026-07-10

Target journal: Optics & Laser Technology  
Publisher: Elsevier  
Current manuscript: `writing/sci_manuscript_OLT_target_2026-07-10.md`

## Current Fit

The current paper is a reasonable fit for Optics & Laser Technology if it is framed as an optical imaging simulation and validation study, not as a new neural-network architecture. The strongest story is:

> physics-interpretable laser range-gated imaging simulation + anti-shortcut validation + domain-randomized robustness analysis for true 3D target versus planar false-target discrimination.

This aligns better with OLT than a pure deep-learning journal framing because the central contribution is the gated-imaging model and validation protocol.

## Completed For OLT Version

- Added a target-journal manuscript copy:
  - `writing/sci_manuscript_OLT_target_2026-07-10.md`
- Added Highlights section.
- Converted the reference list to numbered Elsevier-style references.
- Added in-text numbered citations in the related-work section.
- Added declaration, funding, and data/code availability draft sections.
- Kept the main claim simulation-limited and avoided deployment-ready claims.

## Items Requiring Author Confirmation

- Author names, order, affiliations, and corresponding author email.
- Funding statement.
- Declaration of competing interest.
- Whether the selected 3D model assets can be redistributed.
- Whether the GitHub repository can be made public before submission.
- Whether Blender scripts, trained checkpoints, and generated datasets should be released, or only made available on request.

## OLT/Elsevier Preparation Checklist

- [x] Numbered manuscript sections.
- [x] Abstract present.
- [x] Keywords present.
- [x] Highlights present.
- [x] Figures available as separate PNG/PDF files.
- [x] Tables included in the manuscript draft.
- [x] Numbered references and in-text citations added.
- [ ] Final author/title page.
- [ ] Final declaration of competing interest.
- [ ] Final funding statement.
- [ ] Final data availability statement.
- [ ] Check whether OLT requires separate Highlights file at submission.
- [ ] Check whether graphical abstract is optional or useful for this manuscript.
- [ ] Convert manuscript to Word or LaTeX format if needed by the submission system.

## Recommended Next Scientific Improvements

These are not mandatory for an initial advisor-facing draft, but they would strengthen a journal submission:

1. Add one more viewpoint/elevation validation set if time permits.
2. Add a small table comparing this work with range-gated depth-estimation papers and gated-camera fusion papers.
3. Add a more explicit limitation on Blender-to-real transfer.
4. Add a small methods paragraph describing selected military model screening criteria.
5. Add exact Blender version and rendering settings if available.

## Suggested Cover-Letter Angle

This manuscript reports a controlled simulation and validation framework for range-gated optical imaging. Rather than proposing a new neural architecture, it addresses a practical validation problem: simulated gated-image classifiers can exploit brightness and black-frame shortcuts. The work contributes a physically interpretable gated-imaging simulation, a planar false-target construction, scalar shortcut diagnostics, grouped multi-view validation, and domain-randomized robustness tests.

## Files To Attach Or Prepare

- Manuscript draft: `writing/sci_manuscript_OLT_target_2026-07-10.md`
- Figures: `writing/figures/fig1_*.png` to `writing/figures/fig11_*.png`
- Reference verification: `writing/reference_verification_2026-07-09.md`
- Main result CSV: `experiments/v8_mv4_domainmix_full_vs_single_gate_eval_aggregate_3seed.csv`
- Daily progress log: `writing/daily_progress_2026-07-09.md`

## Status Update - 2026-07-10 (presubmission pack)

Completed in this step:
- Expanded references to 17 and rewrote Related Work.
- Built advisor-facing complete draft with tables/figures.
- Added presubmission checklist:
  - `writing/OLT_presubmission_checklist_2026-07-10.md`
- Added cover letter draft:
  - `writing/OLT_cover_letter_2026-07-10.md`

Current decision:
- Advisor review: ready now.
- Journal submit: after blocking author/declaration/reference verification items in the checklist.