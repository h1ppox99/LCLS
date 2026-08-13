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

Production cannot use reference-mask IoU. Call
`automask.evaluation.evaluate_consistency(pipeline, run, selection, store)` to report:

- all pairwise agreements between the requested number of real-shot folds;
- each fold's agreement with the full-shot mask;
- stability between chronological halves of the run;

These are diagnostics, not a substitute ground truth or a scalar score. See
`CONSISTENCY.md` for the fixed procedure. Physical diagnostics remain separate
because their scientific assumptions differ from reproducibility.

Synthetic artifact injection remains an auxiliary stress test under
`automask.synthetic`. It is not used to choose production defaults or to claim
held-out performance.
