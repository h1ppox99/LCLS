#!/usr/bin/env python3
"""
read_normalized_median.py -- per-shot-normalized, robust per-pixel reference.

Answers: does normalizing each shot by its intensity and taking a ROBUST
per-pixel statistic (median + MAD) give a cleaner reference for masking than the
raw run-sum in the small-data?

For each event we take the calibrated Jungfrau1M frame, divide by that shot's i0
(ipm2 = XPP-SB2-BMMON TotalIntensity), and rescale to a reference intensity so
values stay in physical units.  Over the kept shots we compute, per pixel:
    median_ref  -- robust reference image (outlier-immune, unlike the sum)
    mad_ref     -- robust spread (1.4826 * median abs deviation ~ std), the
                   normalized robust analogue of the RMS bad-pixel feature.

Needs psana (reads the XTC). Output HDF5 -> automask/outputs/cache/.
Run (env already set, or `source src/psana_env.sh` first):
    python normalized_median.py --n 800
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import h5py
import psana

HERE = os.path.dirname(os.path.abspath(__file__))            # .../automask/producers
AUTOMASK = os.path.dirname(HERE)                             # .../automask
CACHE = os.path.join(AUTOMASK, "outputs", "cache")          # generated HDF5 lands here
sys.path.insert(0, os.path.join(os.path.dirname(AUTOMASK), "io"))  # src/io -> readers
from read_xtc import open_local_run, JUNGFRAU_NAME

SMALLDATA = "/Data/hippolyte.wallaert/LCLS/hdf5/smalldata"
ASM = (1064, 1030)


def assemble(panel, ix, iy):
    out = np.zeros(ASM, dtype=panel.dtype)
    out[ix.astype(np.int64), iy.astype(np.int64)] = panel
    return out
