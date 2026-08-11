# Data overview

This repository contains the recovered code plus a local mirror of selected
`xppl1016922` data.  The large data directories are intentionally ignored by
Git and may be absent in a clone.

| path | purpose |
| --- | --- |
| `xtc/` | Raw XTC streams for psana. Run 475 is complete; run 389 contains only truncated stream `s00` (~4,606 decoded events). |
| `calib/` | Detector calibration constants; macOS-recovered colon characters are stored as U+F022. |
| `automask/data/masks/` | The hand-drawn reference masks — the only frozen input left, and the only thing psana cannot recompute. |
| `automask/outputs/cache/images/` | Regenerable image cache populated from XTC. |

## Prewarm production images

```bash
# Selected-shot images are computed on demand by ImageStore: each is a reduction
# over a ShotSelection, resolved from the warm .npy cache or, on a miss, raw XTC.
# The evaluation loop materializes only the reductions and calibration constants
# declared by Pipeline.needs(). Prewarming is optional --
# it just makes the first evaluation numpy-only instead of paying one XTC pass:
source psana_env.sh
python -m automask.producers.build_images --run 389 475
# Robust median/MAD images use the same ImageStore API -- no separate producer.
```

See [PSANA_XTC.md](PSANA_XTC.md) for raw-data limits.
