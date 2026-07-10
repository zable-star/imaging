# Daily Progress - 2026-07-09

## Completed

- Merged and checked the 3090-returned domain-mixed single-gate results.
- Regenerated missing local checkpoints for:
  - gate1, seed2026
  - gate2, seed2026
- Copied the regenerated gate1/gate2 seed2026 checkpoints back into the 3090 handoff result folder to keep the package complete.
- Verified final evaluation completeness:
  - single-gate evaluation: 45 directories, 45 `eval_summary.json`, 45 `eval_predictions.csv`
  - full-stack evaluation: 15 directories, 15 `eval_summary.json`, 15 `eval_predictions.csv`
- Regenerated Fig. 11:
  - `writing/figures/fig11_domainmix_full_stack_vs_single_gate.png`
  - `writing/figures/fig11_domainmix_full_stack_vs_single_gate.pdf`
- Visually checked Fig. 11: legend, axes, bars, and error bars are readable; no title/legend overlap.
- Updated the manuscript status note to reflect that the domain-mixed single-gate ablation is now complete.
- Verified relevant scripts compile:
  - `scripts/make_v8_domainmix_single_gate_figure.py`
  - `scripts/evaluate_gate_model.py`
  - `scripts/summarize_eval_grid.py`
- Filled manuscript Table 1 with exact dataset counts, source-model group counts, class stack counts, gate-image counts, and grouped split protocol.
- Filled manuscript Table 2 with raw and per-gate max-normalized scalar shortcut diagnostics.
- Filled manuscript Table 3 with final domain-mixed training hyperparameters, degradation settings, hard-nuisance domain parameters, evaluation sample counts, and local software environment.
- Replaced all manuscript `Figure X here` notes with journal-style figure captions and source-file references.
- Created a clean manuscript copy without `placeholders` in the filename:
  - `writing/sci_manuscript_current_2026-07-09.md`
- Verified the six current references against DOI/CrossRef metadata.
- Corrected the Liu depth-prior paper year from 2026 to 2025.
- Added DOI/page information for the SAMFusion ECCV chapter and page information for the Tobin domain-randomization paper.
- Added reference verification record:
  - `writing/reference_verification_2026-07-09.md`

## Final Domain-Mixed Ablation Result

| condition | full stack | gate0 | gate1 | gate2 |
|---|---:|---:|---:|---:|
| clean | 0.9394 | 0.7879 | 0.5758 | 0.7727 |
| light noise | 0.9242 | 0.8295 | 0.5455 | 0.7576 |
| strong noise | 0.9091 | 0.8523 | 0.5909 | 0.7424 |
| hard mild | 0.9394 | 0.8485 | 0.5455 | 0.8030 |
| hard strong | 0.7727 | 0.7121 | 0.5303 | 0.6667 |

Conclusion: under domain-mixed training, the full three-gate stack is the best mean performer across all five evaluation conditions. This supports the current paper claim that the robustness gain is not explained by a single isolated gate.

## Files Ready For Paper Use

- `experiments/v8_mv4_domainmix_full_vs_single_gate_eval_aggregate_3seed.csv`
- `experiments/v8_mv4_domainmix_single_gate_eval_aggregate_3seed.csv`
- `experiments/v8_mv4_domainmix_single_gate_eval_summary_3seed.csv`
- `writing/v8_mv4_domainmix_full_vs_single_gate_ablation_report_2026-07-09.md`
- `writing/figures/fig11_domainmix_full_stack_vs_single_gate.png`
- `writing/figures/fig11_domainmix_full_stack_vs_single_gate.pdf`
- `writing/sci_manuscript_current_placeholders_2026-07-09.md`

## Remaining Work

- Convert the reference list into the exact style required by the target journal.
