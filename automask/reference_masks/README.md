# Reference masks

The non-regenerable human reference masks used by labelled evaluation. Binary
masks here are intentionally allowed by `.gitignore`; generated reductions
belong in the cache instead.

Format: boolean assembled arrays, `True == masked`, on the Jungfrau1M assembled
canvas — shape `(1064, 1030)`.

Naming: `reference_mask_run<run>.npy` (e.g. `reference_mask_run475.npy`).
`evaluation.reference_mask(run)` loads the file for that run; files under any
other name are ignored.
