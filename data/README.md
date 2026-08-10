# Data directory

`labels/curation.csv` is the recovered review ledger. Large generated and
downloaded arrays are intentionally ignored by Git.

The production feature cache is regenerated from XTC:

```bash
python -m automask.producers.build_features
```

The former public-corpus download and reference-evaluation scripts were not
recovered; do not rely on the old `src/automask/...` commands.
