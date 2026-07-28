---
name: xray-masking-status-baseline
description: Static bad-pixel mask from the detector's pixel_status calibration array. The floor every other mask is unioned onto. Use as the always-on baseline before any other masking method.
category: masking
role: signal-independent (baseline)
gate: always — the floor every other mask layer is unioned onto
status: wired
---

# Masking · 00 — Static status mask from calibration

## Principle

The calibration store ships a per-pixel `pixel_status` array flagging factory-known bad
pixels. Collapse across gain stages and treat any flagged pixel as bad.

```python
status  = (load_ndarr(f'{CAL}/pixel_status/362-end.data', (3,2,512,1024)) != 0).any(0)
good_px = ~status                      # (2,512,1024) bool, True = usable
```

## Parameters

None to tune — it is a lookup. Shape `(3, 2, 512, 1024)` collapses over the 3 gain stages
with `.any(0)`.

## When to use

Always. This is the floor that methods [01](01_geometry_gap.md)/[02](02_pyfai_azimuthal_sigmaclip.md)/[03](03_rmm_dark_based.md)/[04](04_rmm_feature6_light.md)
are unioned onto.

## Trade-offs

Static only — misses pixels that go bad between calibrations, and misses signal-dependent
outliers. Necessary but never sufficient on its own.

## Outputs

Baseline mask layer, unioned into the run mask. Used in `accumulate_calib.py` and
`weighted_sum_v2.py` (`good_px`).

## Links

Part of: [masking](README.md). The floor for
[01](01_geometry_gap.md) / [02](02_pyfai_azimuthal_sigmaclip.md) /
[03](03_rmm_dark_based.md) / [04](04_rmm_feature6_light.md).
