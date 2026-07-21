# Public diffraction corpus and active mask review

This directory contains 1,000 detector-native diffraction images sampled from
six CC0 CXIDB deposits. Arrays are NumPy `.npy` files and are **not resized or
intensity-normalized**. Two-panel pnCCD shots are represented as a vertical
top/bottom panel stack. See `manifest.jsonl` for image-level provenance and
`sources.json` for archive hashes, DOIs, licenses, and download URLs.

## Rebuild the corpus

From the repository root:

```bash
python src/automask/corpus.py --download
```

Existing archives in `.downloads/` are reused. The deterministic seed and
quotas in `corpus.py` reproduce the same selection.

## Review proposed masks

Provide a callable accepting one 2-D array and returning a same-shaped boolean
array (`True == masked`):

```bash
python src/automask/review.py --masker your_module:mask_image
```

Click **Yes** / press `Y` to accept, **No** / press `N` to reject, or `Q` to
quit. Each proposal is snapshotted in `proposals/`; append-only decisions and
code fingerprints go to `labels/verdicts.csv`. Only accepted proposals are
copied into `reference_masks/`.

The included adapter is only a UI smoke test:

```bash
python src/automask/review.py --masker reviewer_example:mask_image --limit 10
```

To benchmark a later candidate against accepted references:

```bash
python src/automask/evaluate_reference.py --masker your_module:mask_image
```

An accepted proposal validates that exact mask snapshot. A rejection is useful
for acceptance-rate tracking, but cannot provide pixelwise ground truth without
an additional mask-editing/annotation step.