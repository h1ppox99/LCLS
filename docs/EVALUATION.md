# Evaluation

Three workflows answer different questions.

## 1. Validate one unlabelled run under perturbations

`validate_mask` keeps the full-run geometry/calibration floor fixed and separates
three descriptive checks: shot-fold perturbations, declared parameter
perturbations, and their interaction. It also reports leave-one-channel-out
effects and removes a channel only when its removal changes none of the evaluated
masks.

```python
from automask.evaluation import (
    MaskValidationDesign,
    ParameterSweep,
    validate_mask,
)

design = MaskValidationDesign(
    sweeps=(
        ParameterSweep(
            "variance.params.k",
            (3.25, 3.75),
            "Check the declared +/-0.25 threshold tolerance",
        ),
        ParameterSweep(
            "variance.field_reg.tv.weight",
            (3.0, 5.0),
            "Check nearby smoothing strengths",
        ),
    ),
    n_folds=10,
)
report = validate_mask(pipeline, run, selection, design, store=store)
report.display()
pipeline = report.recommended_pipeline
```

Supported paths are `<channel>.params.<field>`,
`<channel>.field_reg.<regularizer>.<field>`, and
`<channel>.mask_reg.<regularizer>.<field>`. Each perturbation's values and
one-line reason are explicit; validation does not infer what a scientifically
reasonable perturbation is.

The report contains Markdown, baseline/channel/instability figures, structured
metrics, and optional JSON/NumPy/PNG persistence (`report.save(<dir>)`). It is
deliberately descriptive: stability is robustness to the tested variations, not
agreement with ground truth, and v1 applies no pass/fail thresholds.

## 2. Choose default parameters with labels

Labelled evaluation uses only masks actually present in
`automask/data/masks/`, split chronologically into fit and validation sets. Tune
only on the fit runs, rank candidates by residual IoU (pixels added beyond the
shared floor), then measure the chosen configuration once on validation.
Validation performance is the reported estimate and must not be used to revise
parameters.

The historical run-475 human mask is not present in this checkout, so the
default labelled sets are currently empty. Recover and package the original
mask before using this workflow; do not substitute a generated mask as ground
truth.

```python
from automask.evaluation import FIT_RUNS, evaluate

# Once reference masks are packaged, explicit runs score the fit set and no
# explicit runs use the held-out validation set.
result = evaluate(pipeline, runs=FIT_RUNS)
result = evaluate(pipeline)
```

## 3. Check fold consistency only

`evaluate_consistency(pipeline, run, selection, store)` reports, for the
requested number of real-shot folds:

- all pairwise agreements between folds;
- each fold's agreement with the full-shot mask;
- both round-robin and chronological folds.

These are reproducibility diagnostics, not a substitute ground truth or a scalar
score. See [CONSISTENCY.md](CONSISTENCY.md) for the fixed procedure.

Synthetic artifact injection (`automask.research.synthetic`) is an auxiliary stress test;
it is not used to choose production defaults or to claim held-out performance.
