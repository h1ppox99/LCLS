# Data overview

This repository contains the recovered code plus a local mirror of selected
`xppl1016922` data.  The large data directories are intentionally ignored by
Git and may be absent in a clone.

| path | purpose |
| --- | --- |
| `hdf5/smalldata/` | Per-event scalar data, detector geometry, calibration masks, and calibrated run sums for runs 389 and 475. |
| `xtc/` | Raw XTC streams for psana. Run 475 is complete; run 389 contains only truncated stream `s00` (~4,606 decoded events). |
| `calib/` | Detector calibration constants; macOS-recovered colon characters are stored as U+F022. |
| `automask/data/` | Regenerable NumPy working dataset; build it with the producer commands below. |

Use `automask.io.lcls_xpp.SmallData` for small-data access without psana.  Its
assembled image convention differs from the frozen automask dataset: the
automask producers deliberately save `(1064, 1030)` arrays indexed as
`[ix, iy]`, matching the recovered hand-mask reference.

## Rebuild the masking inputs

```bash
# Reproduces the lab notebook baseline from 100 IPM-filtered XTC frames.
python -m automask.producers.baseline_mask --source xtc --run 475

# Consumes that baseline and freezes images/masks for the pipeline.
python -m automask.producers.extract_dataset

# Builds features from raw XTC via ShotSelection (needs the psana env): umean/ustd
# from x-ray-on shots, mean from x-ray-off shots. No small-data dependency.
source psana_env.sh
python -m automask.producers.build_features --run 389 475 --n-shots 500
# Alternative lit features via robust median/MAD instead of mean/std:
python -m automask.producers.normalized_median --run 475 --n 800
```

The HDF5-only `--source smalldata` mode is a diagnostic approximation, not a
replacement for the lab baseline. There is no verified run-389 reference mask;
its real-mask evaluation falls back to the run-475 target. See
[PSANA_XTC.md](PSANA_XTC.md) for raw-data limits.
