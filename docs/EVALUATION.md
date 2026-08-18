# Evaluation

There are three evaluation workflows and they answer different questions.

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
# report.save("automask/outputs/validation/run0475")
```

Supported paths are `<channel>.params.<field>`,
`<channel>.field_reg.<regularizer>.<field>`, and
`<channel>.mask_reg.<regularizer>.<field>`.
Values and the one-line reason are explicit; validation does not infer what a
scientifically reasonable perturbation is.

The report contains Markdown, baseline/channel/instability figures, structured
metrics, and optional JSON/NumPy/PNG persistence. It is deliberately descriptive:
stability is not ground truth, and v1 applies no confidence intervals or numeric
pass/fail thresholds.

## 2. Choose default parameters with labels

The XTC runs are split chronologically:

- fit: runs 378 and 389;
- validation: runs 396 and 475.

Sweep channels, regularizers, and their parameters only on the fit
runs. Rank candidates primarily by residual IoU, which measures the pixels added
beyond the shared geometry/calibration floor. After choosing one configuration,
run it once on validation with `eval.phase=validate`. Validation performance is
the reported estimate; it must not be used to revise parameters.

```bash
python -m automask.studies.sweep_hyperparameters -m \
  stat=variance stat.params.k=2,2.5,3,3.5

python -m automask.studies.sweep_hyperparameters \
  experiment=production eval.phase=validate
```

The labelled API is `automask.evaluation.evaluate(pipeline, runs)`. With no
explicit runs it evaluates on the held-out validation set.

## 3. Check fold consistency only

For the smaller legacy dictionary API, call
`automask.evaluation.evaluate_consistency(pipeline, run, selection, store)` to report:

- all pairwise agreements between the requested number of real-shot folds;
- each fold's agreement with the full-shot mask;
- the same diagnostics for round-robin and chronological folds;

These are diagnostics, not a substitute ground truth or a scalar score. See
`CONSISTENCY.md` for the fixed procedure. Physical diagnostics remain separate
because their scientific assumptions differ from reproducibility.

Synthetic artifact injection remains an auxiliary stress test under
`automask.synthetic`. It is not used to choose production defaults or to claim
held-out performance.
