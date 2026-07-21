#!/usr/bin/env python3
"""
extract_dataset.py -- freeze a self-contained masking dataset from the LCLS data.

This is the ONE producer step for the automated-masking project.  It reads the
heavy, experiment-specific sources (small-data HDF5) once and writes small,
plain .npy arrays into ./data so that all masking development downstream needs
only numpy -- no psana, no h5py, no LCLS filesystem.

Sources (read-only, never modified):
  hdf5/smalldata/xppl1016922_Run{389,475}.h5
      Sums/jungfrau1M_alcove_calib*           full run-sum detector images
      UserDataCfg/jungfrau1M_alcove/{ix,iy}   panel->assembled pixel mapping
      UserDataCfg/jungfrau1M_alcove/{mask,cmask,statusMask}   production masks
  automask/data/masks/human_Mask_source.npy   the human hand-drawn mask (target)

Conventions in the frozen dataset (uniform, unlike the raw sources):
  * every mask is bool with  True == MASKED (excluded)  -- matches Mask.npy.
    (small-data masks are the opposite, 1==good; we invert them here.)
  * every array is saved in BOTH forms:
      - panel  : (2, 512, 1024)   raw Jungfrau1M geometry
      - asm    : (1064, 1030)     psana-assembled image (what Mask.npy is)

Run:  python -m automask.producers.extract_dataset
"""
from __future__ import annotations
import os, json, datetime
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))               # .../automask/producers
AUTOMASK = os.path.dirname(HERE)                                # .../automask
ROOT = os.path.dirname(os.path.dirname(AUTOMASK))              # .../LCLS
SMALLDATA = os.path.join(ROOT, "hdf5", "smalldata")
HUMAN_MASK = os.path.join(AUTOMASK, "data", "masks", "human_Mask_source.npy")

IMG_DIR = os.path.join(AUTOMASK, "data", "images")
MSK_DIR = os.path.join(AUTOMASK, "data", "masks")
DET = "jungfrau1M_alcove"
ASM_SHAPE = (1064, 1030)
RUNS = (389, 475)


def sd_path(run: int) -> str:
    return os.path.join(SMALLDATA, f"xppl1016922_Run{run:04d}.h5")


def assemble(panel: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Map a (2,512,1024) panel array into the (1064,1030) assembled image."""
    out = np.zeros(ASM_SHAPE, dtype=panel.dtype)
    out[ix.astype(np.int64), iy.astype(np.int64)] = panel
    return out


def main() -> None:
    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "detector": DET,
        "assembled_shape": list(ASM_SHAPE),
        "mask_convention": "bool, True == masked (excluded)",
        "runs": RUNS,
        "n_events": {},        # events per run (normalizer N for mean/variance)
        "images": {},
        "masks": {},
    }

    # ---- human hand-drawn mask (the primary target) -----------------------
    human = np.load(HUMAN_MASK).astype(bool)              # already True==masked, asm
    np.save(os.path.join(MSK_DIR, "human_Mask_asm.npy"), human)
    manifest["masks"]["human_Mask_asm"] = {
        "shape": list(human.shape), "frac_masked": float(human.mean()),
        "source": "automask/data/masks/human_Mask_source.npy",
        "kind": "hand-drawn (bad pixels + geometry)"}

    for run in RUNS:
        with h5py.File(sd_path(run), "r") as f:
            g = f[f"UserDataCfg/{DET}"]
            ix, iy = g["ix"][()], g["iy"][()]
            manifest["n_events"][str(run)] = int(f["ipm2/sum"].shape[0])

            # --- sum images (fully calibrated run sums) --------------------
            for tag in ("calib", "calib_dropped", "calib_dropped_square"):
                key = f"Sums/{DET}_{tag}"
                if key not in f:
                    continue
                panel = f[key][()].astype(np.float32)
                np.save(os.path.join(IMG_DIR, f"sum_{tag}_run{run:04d}_panel.npy"), panel)
                asm = assemble(panel, ix, iy)
                np.save(os.path.join(IMG_DIR, f"sum_{tag}_run{run:04d}_asm.npy"), asm)
                manifest["images"][f"sum_{tag}_run{run:04d}"] = {
                    "panel_shape": list(panel.shape), "asm_shape": list(asm.shape),
                    "source": f"{os.path.basename(sd_path(run))}:{key}"}

            # --- production / status masks (invert: raw is 1==good) --------
            for nm in ("mask", "cmask", "statusMask"):
                raw = g[nm][()]                          # (2,512,1024) uint8, 1==good
                masked = raw == 0                        # True == masked
                np.save(os.path.join(MSK_DIR, f"{nm}_run{run:04d}_panel.npy"), masked)
                asm = assemble(masked, ix, iy).astype(bool)
                np.save(os.path.join(MSK_DIR, f"{nm}_run{run:04d}_asm.npy"), asm)
                manifest["masks"][f"{nm}_run{run:04d}_panel"] = {
                    "shape": list(masked.shape), "frac_masked": float(masked.mean()),
                    "source": f"{os.path.basename(sd_path(run))}:UserDataCfg/{DET}/{nm}",
                    "kind": {"mask": "production mask (bad px)",
                             "cmask": "combined mask (bad px)",
                             "statusMask": "pixel_status-derived (bad px)"}[nm]}

    with open(os.path.join(AUTOMASK, "data", "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    n_img = len([n for n in os.listdir(IMG_DIR) if n.endswith(".npy")])
    n_msk = len([n for n in os.listdir(MSK_DIR) if n.endswith(".npy")])
    print(f"[done] wrote {n_img} image arrays -> {IMG_DIR}")
    print(f"[done] wrote {n_msk} mask arrays  -> {MSK_DIR}")
    print(f"[done] manifest -> {os.path.join(AUTOMASK, 'data', 'manifest.json')}")


if __name__ == "__main__":
    main()
