# Mask consistency

Consistency evaluation measures whether a pipeline finds the same evidence from
different real shots. It uses no reference mask and evaluates both fold
strategies:

- `round_robin`: interleaved shots from across the run;
- `chronological`: contiguous time-ordered groups.

```python
from automask.evaluation import evaluate_consistency

result = evaluate_consistency(
    pipeline, run=475, selection=selection, store=store, n_folds=10)
```

`ImageStore.folds` defaults to 10 folds and accepts `strategy="round_robin"` or
`strategy="chronological"`. It caches
every required reduction with the same content-addressed convention as full-run
images. Calibration constants are shared because they do not depend on shots.

The result contains the same pairwise fold IoUs and fold-versus-full IoUs under
`result["round_robin"]` and `result["chronological"]`.
IoU excludes the fixed geometry/calibration floor, which would otherwise inflate
agreement without measuring the reproducibility of detected evidence.

For model perturbations, instability maps, channel ablations, and Markdown/image
reporting, use `automask.evaluation.validate_mask`; `evaluate_consistency` remains
the compact backward-compatible data-only interface.

Mean/std and median/MAD are supported. A cold evaluation may need an XTC pass
because an existing pooled 2D cache has already lost shot identity; subsequent
calls reuse the split cache.
