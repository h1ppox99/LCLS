The reader code is in `src/io/` and is **data access only**:

| file | what it does |
|------|--------------|
| `io/lcls_xpp.py` | read the small‑data HDF5 and the psana calib files as plain numpy — no psana needed |
| `io/read_xtc.py` | pull calibrated per‑event Jungfrau frames + scalars out of the raw XTC (needs psana) |
| `io/setup_psdm_layout.py` | build a psana‑readable symlink tree (real‑colon calib names) |
| `requirements.txt` | `pip install -r requirements.txt` (numpy, h5py, scipy, scikit-image, matplotlib) |