# Mask design and validation

## Invariants

- Project masks are boolean and `True` means excluded or masked.
- Reduced images are in assembled detector space. Calibration constants are in
  native panel space until the consuming statistic interprets them.
- Preserve the registered `geometry` and `status_as_mask` floor channels in
  every run-level candidate. They encode detector gaps/outside area and psana
  bad-pixel status respectively.
- Ask `automask_catalog` for current statistics, regularizers, required arrays,
  default parameters, and the canonical pipeline example. Do not copy registry
  details into a prompt from memory.

## Candidate design

Start from the canonical pipeline and make targeted, attributable changes.
Use separate channels for distinct evidence sources or artifact types. Examine
each emitted layer before accepting its contribution to the combined mask.

Protect experiment signal. Bragg peaks and azimuthal rings can be legitimate;
streaks, shadows, saturated regions, and persistent detector artifacts may be
mask candidates. Shape alone is not sufficient to distinguish them, so relate
each layer to the selected-shot image, detector geometry, and run context.

Avoid tuning solely until an overview looks clean. Record why each changed
parameter is physically or statistically plausible, and define a new pipeline
handle for every alternative.

## Validation

Use `validate_mask` to test nearby parameter perturbations and fold/strategy
stability. Review the generated report, arrays, figures, and the recommended
pipeline handle it returns. Validation is descriptive: stability shows robustness
to the tested variations, not agreement with ground truth.

When a human reference exists, evaluation against it is development evidence
and must remain separate from production inference. Report masked fraction,
layer contributions, stability limitations, and visible signal at risk. Escalate
ambiguous scientific judgments rather than converting a weak heuristic into a
confident recommendation.
