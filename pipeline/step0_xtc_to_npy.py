#!/usr/bin/env python3
"""Step 0 (deterministic, no agent): XTC -> numpy.

Walks all XTC streams, merges events into global time order, and writes:

  npy/frames_raw.npy   (N, 2, 512, 1024) uint16 memmap — raw frames, gain bits intact.
                       Calibration is applied later (step2) from calib/*.npy so the
                       stored data stays lossless and half the size of float32.
  npy/shot_table.npz   per-shot scalars, all aligned to the same global order:
                       fiducial, sec, nsec, stream, ipm2, xray, laser,
                       det_total (sum of calibrated keV over good pixels),
                       photon_px (# good pixels in 7-12 keV), nswitch (# non-G0 pixels)
  npy/README_npy.md    column documentation for downstream agents.

Usage:  python3 pipeline/step0_xtc_to_npy.py [--xtc-glob 'xppl1016922-r0475-s0*-c00.xtc']
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xtclib import FRAME_SHAPE, calibrate, parse_event, read_dgram, scan_headers

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xtc-glob", default="xppl1016922-r0475-s0*-c00.xtc")
    args = ap.parse_args()

    paths = sorted(glob.glob(str(ROOT / args.xtc_glob)))
    if not paths:
        print(f"no XTC files match {args.xtc_glob} under {ROOT}", file=sys.stderr)
        return 1

    t0 = time.time()
    # Pass A - header-only scan, then global time sort
    index = []  # (sec, nsec, fid, stream_idx, offset)
    for si, p in enumerate(paths):
        n = 0
        for off, total, sec, nsec, fid in scan_headers(p):
            index.append((sec, nsec, fid, si, off))
            n += 1
        print(f"[scan] {Path(p).name}: {n} events", flush=True)
    index.sort(key=lambda t: (t[0], t[1], t[2]))
    N = len(index)
    print(f"[scan] total {N} events, {time.time()-t0:.0f}s", flush=True)

    npy_dir = ROOT / "npy"
    npy_dir.mkdir(exist_ok=True)
    frames = np.lib.format.open_memmap(
        npy_dir / "frames_raw.npy", mode="w+", dtype=np.uint16, shape=(N, *FRAME_SHAPE)
    )

    ped = np.load(ROOT / "calib/ped.npy")
    gain = np.load(ROOT / "calib/gain.npy")
    good = ~np.load(ROOT / "calib/status_bad.npy")

    cols = {k: np.zeros(N, d) for k, d in [
        ("fiducial", np.int64), ("sec", np.int64), ("nsec", np.int64), ("stream", np.int16),
        ("ipm2", np.float64), ("xray", np.int8), ("laser", np.int8),
        ("det_total", np.float64), ("photon_px", np.int32), ("nswitch", np.int32),
    ]}

    files = [open(p, "rb") for p in paths]
    try:
        for i, (sec, nsec, fid, si, off) in enumerate(index):
            ev = parse_event(read_dgram(files[si], off))
            if ev["frame_raw"] is None:
                print(f"[warn] event {i} (stream {si} @{off}): no Jungfrau leaf", file=sys.stderr)
                continue
            frames[i] = ev["frame_raw"]
            calib = calibrate(ev["frame_raw"], ped, gain)
            cols["fiducial"][i], cols["sec"][i], cols["nsec"][i], cols["stream"][i] = fid, sec, nsec, si
            cols["ipm2"][i], cols["xray"][i], cols["laser"][i] = ev["ipm2"], ev["xray"], ev["laser"]
            cols["det_total"][i] = float(calib[good].sum())
            cols["photon_px"][i] = int(((calib >= 7) & (calib < 12) & good).sum())
            cols["nswitch"][i] = int((ev["frame_raw"] >> 14 != 0).sum())
            if (i + 1) % 500 == 0:
                print(f"[convert] {i+1}/{N}  ({time.time()-t0:.0f}s)", flush=True)
    finally:
        for f in files:
            f.close()
        frames.flush()

    np.savez(npy_dir / "shot_table.npz", **cols)
    (npy_dir / "README_npy.md").write_text(
        "# npy/ — deterministic XTC conversion (step 0 output)\n\n"
        f"- `frames_raw.npy` — ({N}, 2, 512, 1024) uint16 **memmap** "
        "(`np.load(..., mmap_mode='r')`). Raw Jungfrau words: high 2 bits = gain mode "
        "(00 G0 / 01 G1 / 11 G2), low 14 bits = ADC. Calibrate with "
        "`(adc - calib/ped.npy[mode]) / calib/gain.npy[mode]` -> keV (see pipeline/xtclib.py).\n"
        "- `shot_table.npz` — per-shot columns, global time order (same order as frames):\n"
        "  fiducial, sec, nsec, stream, ipm2 (XPP-SB2-BMMON sum; offset NOT subtracted),\n"
        "  xray (1 = beam, from EVR code 137/162), laser (1 = on, code 90/91),\n"
        "  det_total (summed calibrated keV over good pixels),\n"
        "  photon_px (# good pixels in 7-12 keV), nswitch (# non-G0 pixels, stuck-pixel proxy).\n"
        "- ipm2 zero offset: estimate as median ipm2 over xray==0 shots.\n"
    )
    meta = {"n_events": N, "streams": [Path(p).name for p in paths],
            "elapsed_s": round(time.time() - t0, 1)}
    (npy_dir / "step0_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {N} events -> npy/ in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
