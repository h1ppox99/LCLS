# studies/archive — pre-refactor exploratory scripts

These scripts predate the `stats/` · `regularization/` · `combine/` reorganization
and import the **removed `methods.py`** module (old API: `method_variance`,
`load_all`, `close_open`, `defectiveness_fields`, …). They are kept for reference —
the ideas, grids, and figures they document — but **do not run as-is**.

What replaced each:

| archived script | replaced by |
|---|---|
| `tv_variance_sweep.py`, `variance_sweep.py` | `scripts/sweep_variance.sh` → `studies/sweep_hyperparameters.py` |
| `tv_window_median_sweep.py`, `kw_sweep.py` | `scripts/sweep_window_median.sh`, `scripts/sweep_blackhat.sh` |
| `fusion_stats.py` | `combine/weighted_sum.py` + `combine/mahalanobis.py`; `scripts/sweep_fusion.sh` |
| `radial_median_subtract.py` | `stats/radial_median.py` |
| `pyfai_sigma_clip.py` | `stats/azimuthal_sigma.py` (numpy-only hand-rolled version) |
| `window_median_visual.py`, `tv_blackhat_visual.py`, `window_median_subtract.py`, `shape_priors_variance.py`, `tv_shape_robustness.py`, `intensity_bands.py` | diagnostics; port to the `Sample`/`Pipeline` API in `masking.py` if revived |

To revive one: swap `from methods import ...` for the equivalent in `masking.py`
(or the relevant `stats/`·`regularization/`·`combine/` module) and load runs via
`evaluation.load_sample(run)` instead of the old `load_all`.
