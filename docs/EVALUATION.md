# Evaluation

There are two evaluation workflows and they answer different questions.

## 1. Choose default parameters with labels

The XTC runs are split chronologically:

- fit: runs 378 and 389;
- validation: runs 396 and 475.

Sweep channels, regularizers, combiners, and their parameters only on the fit
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

## 2. Check a new mask without labels

Production cannot use IoU. Call
`automask.evaluation.evaluate_runtime(pipeline, run)` instead. It reports:

- a distribution of stability across complementary real-shot folds;
- stability between chronological halves of the run;
- azimuthal consistency against repeated size-matched random controls;
- masked fraction and floor containment as structural outputs.

These are diagnostics and gates, not a substitute ground truth or a scalar score.
They tell the agent whether a mask is unstable, violates the masking contract,
or fails to improve a relevant physical consistency check. See
`RUNTIME_EVALUATION.md` for assumptions and statistical definitions.

Synthetic artifact injection remains an auxiliary stress test under
`automask.synthetic`. It is not used to choose production defaults or to claim
held-out performance.
