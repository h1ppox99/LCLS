# Mask consistency

Consistency evaluation measures whether a pipeline finds the same evidence from
different real shots. It uses no reference mask, and folding is always
round-robin.

```python
from automask.evaluation import evaluate_consistency

result = evaluate_consistency(
    pipeline, run=475, selection=selection, store=store, n_folds=10)
```

`ImageStore` assigns selected shots to the requested number of folds and caches
every required reduction with the same content-addressed convention as full-run
images. Calibration constants are shared because they do not depend on shots.

The result contains all pairwise fold IoUs, one fold-versus-full IoU per fold,
their means and standard deviations, and first-half versus second-half IoU for drift.
IoU excludes the fixed geometry/calibration floor, which would otherwise inflate
agreement without measuring the reproducibility of detected evidence.

Mean/std and median/MAD are supported. A cold evaluation may need an XTC pass
because an existing pooled 2D cache has already lost shot identity; subsequent
calls reuse the split cache.
