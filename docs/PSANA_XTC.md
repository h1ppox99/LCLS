# Reading raw XTC with psana

The raw XTC reader is optional after the frozen NumPy dataset has been built.
Complete run-475 XTC is required to reproduce the notebook reference exactly.

```bash
source psana_env.sh
python -m automask.producers.baseline_mask --source xtc --run 475 --n-images 100
# Optional: prewarm the FeatureStore cache so evaluation stays numpy-only.
python -m automask.producers.build_features --run 389 475
```

`psana_env.sh` sets the `SIT_*` variables and activates the expected psana
environment. `automask.io.read_xtc.open_local_run()` opens explicit stream files
and configures calibration lookup.

Run 389 is incomplete locally: only truncated stream `s00` is available. It
cannot produce a complete bright-only sum or a verified run-specific reference
mask. Use `--source smalldata` only for diagnostic images, never as a reference.
