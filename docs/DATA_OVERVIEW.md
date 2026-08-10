# Data overview

This repository contains the recovered code plus a local mirror of selected
`xppl1016922` data.  The large data directories are intentionally ignored by
Git and may be absent in a clone.

| path | purpose |
| --- | --- |
| `xtc/` | Raw XTC streams for psana. Run 475 is complete; run 389 contains only truncated stream `s00` (~4,606 decoded events). |
| `calib/` | Detector calibration constants; macOS-recovered colon characters are stored as U+F022. |
| `automask/outputs/cache/features/` | Regenerable feature cache populated from XTC. |

## Prewarm production features

```bash
# Features are computed on demand by the FeatureStore (automask.features): a
# feature = a reduction over a ShotSelection, resolved from the warm .npy cache
# or, on a miss, from raw XTC. The evaluation loop materializes only the features
# a pipeline's stats declare (Pipeline.features_needed). Prewarming is optional --
# it just makes the first evaluation numpy-only instead of paying one XTC pass:
source psana_env.sh
python -m automask.producers.build_features --run 389 475   # prewarm the catalogue
# Robust median/MAD lit features are just other reductions in the same catalogue
# (features/base.py: mean|std|median|mad) -- no separate producer.
```

See [PSANA_XTC.md](PSANA_XTC.md) for raw-data limits.
