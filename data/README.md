# Data directory

`labels/curation.csv` is the recovered review ledger. Large generated and
downloaded arrays are intentionally ignored by Git.

The active masking dataset is regenerated under `automask/data/`, not here:

```bash
python -m automask.producers.baseline_mask --run 475
python -m automask.producers.extract_dataset
python -m automask.producers.build_features
```

The former public-corpus download and reference-evaluation scripts were not
recovered; do not rely on the old `src/automask/...` commands.
