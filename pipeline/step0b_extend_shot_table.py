#!/usr/bin/env python3
"""Step 0b (deterministic, run once after step 0): extend shot_table.npz with
every additional per-shot scalar channel present in the XTC streams.

step 0 extracts only ipm2 + EVR flags. The raw XTC carries far more per-shot
diagnostics; this script decodes them all, aligns them to the existing
shot_table order (verified bit-exact against the ipm2 column), and merges the
new columns into npy/shot_table.npz in place. frames_raw.npy is not touched.

Channels decoded per L1Accept (names as stored):
  gasdet_f11/f12/f21/f22/f63/f64   FEE gas detectors [mJ]           (TypeId 14)
  ebeam_*                          e-beam params incl. photon_energy_ev,
                                   damage mask (TypeId 15, BldDataEBeamV7;
                                   all-zero fields LTUpos/ang + XTCAV skipped)
  pcav_t1_ps/t2_ps/q1_pc/q2_pc     phase cavity timing               (TypeId 16)
  ipmfex{22,23,26,28}_{ch0..3,sum,xpos,ypos}
                                   IPM/PIM diode boxes, IpmFex f32   (TypeId 31)
  bmmon43_{sum,xpos,ypos}          upstream BMMON wave8              (TypeId 98)
  bmmon4b_{xpos,ypos}              XPP-SB2 BMMON positions (sum == ipm2)
  bmmon4c_{sum,xpos,ypos}          second XPP BMMON wave8

Usage:  python3 pipeline/step0b_extend_shot_table.py
"""

from __future__ import annotations

import argparse
import glob
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xtclib import read_dgram, scan_headers, walk_leaves

ROOT = Path(__file__).resolve().parent.parent

EBEAM_FIELDS = [  # (index in the 20-double BldDataEBeamV7 block, column name)
    (0, "ebeam_charge_nc"),
    (1, "ebeam_l3_mev"),
    (6, "ebeam_pkcurr_bc2"),
    (7, "ebeam_energy_bc2"),
    (8, "ebeam_pkcurr_bc1"),
    (9, "ebeam_energy_bc1"),
    (10, "ebeam_und_posx"),
    (11, "ebeam_und_posy"),
    (12, "ebeam_und_angx"),
    (13, "ebeam_und_angy"),
    (16, "ebeam_dump_charge"),
    (17, "photon_energy_ev"),
    (18, "ebeam_ltu250"),
    (19, "ebeam_ltu450"),
]
IPMFEX_SRCS = (0x22, 0x23, 0x26, 0x28)
IPMFEX_FIELDS = ("ch0", "ch1", "ch2", "ch3", "sum", "xpos", "ypos")

README_MARKER = "## Extended monitor columns (step0b)"

README_SECTION = f"""
{README_MARKER}

`step0b_extend_shot_table.py` merges every remaining per-shot scalar in the XTC
into `shot_table.npz` (same global time order; alignment verified bit-exact
against `ipm2`). Groups:

- `gasdet_f11/f12/f21/f22/f63/f64` — FEE gas detectors [mJ], upstream of all
  beamline optics. Decorrelated from the in-hutch monitors by spectral jitter
  (r ~= 0.69 vs ipm2): sanity anchors, not verdict monitors.
- `photon_energy_ev` — per-shot machine estimate of photon energy (BldDataEBeamV7).
  RELATIVE use only (jitter/drift): its absolute scale disagrees with the
  LaB6-ring-fitted geometry by ~+0.38% on Run0475. `ebeam_damage` != 0 marks
  shots where some machine fields are invalid (the always-invalid LTUpos/XTCAV
  fields make it nonzero on every Run0475 shot — judge per use, don't gate on it).
- `ebeam_charge_nc, ebeam_l3_mev, ebeam_pkcurr_bc1/bc2, ebeam_energy_bc1/bc2,
  ebeam_und_posx/posy/angx/angy, ebeam_dump_charge, ebeam_ltu250/ltu450` —
  accelerator diagnostics (weak direct value for normalization; kept for
  drift attribution studies).
- `pcav_t1_ps, pcav_t2_ps, pcav_q1_pc, pcav_q2_pc` — phase-cavity arrival-time
  diagnostics (pump-probe timing, not flux).
- `ipmfex22_*, ipmfex23_*, ipmfex26_*, ipmfex28_*` — four IPM/PIM diode boxes
  (IpmFex: 4 diode channels + sum + x/y position). Channel health on Run0475:
  `ipmfex22_ch1` dead (always 0); `ipmfex26_ch1` saturated (~1.21 const, poisons
  `ipmfex26_sum`); `ipmfex23` near saturation with x/ypos pinned at ~1.
  Healthy high-correlation channels: `ipmfex22_sum`, `ipmfex26_ch0+ch3`,
  `ipmfex28_sum` (all r ~= 0.90 vs det_total). Source identities per the
  pdsdata BldInfo enum (to confirm against the elog): 0x22 XppMonPim0,
  0x23 XppMonPim1, 0x26 XppSb3Pim, 0x28 XppEnds_Ipm0.
- `bmmon43_sum/xpos/ypos` — upstream (HX2-SB1) BMMON wave8: anchor only.
- `bmmon4b_xpos/ypos` — beam position at the ipm2 device (XPP-SB2 BMMON);
  `bmmon4b` sum IS the existing `ipm2` column (not duplicated).
- `bmmon4c_sum/xpos/ypos` — second XPP BMMON (likely SB3). Its xpos/ypos read
  identically on Run0475 — decode/device quirk, treat positions as opaque.

See the `skills/normalization/` monitor menu (README comparison table + one
file per reference parameter) for the evidence-based choice among these monitors.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xtc-glob", default="xppl1016922-r0475-s0*-c00.xtc")
    args = ap.parse_args()

    paths = sorted(glob.glob(str(ROOT / args.xtc_glob)))
    if not paths:
        print(f"no XTC files match {args.xtc_glob} under {ROOT}", file=sys.stderr)
        return 1
    table_path = ROOT / "npy/shot_table.npz"
    old = dict(np.load(table_path).items())
    N_old = len(old["ipm2"])

    t0 = time.time()
    index = []
    for si, p in enumerate(paths):
        for off, _total, sec, nsec, fid in scan_headers(p):
            index.append((sec, nsec, fid, si, off))
    index.sort(key=lambda t: (t[0], t[1], t[2]))
    N = len(index)
    if N_old != N:
        print(f"event count {N} != shot_table rows {N_old} — rerun step 0 first", file=sys.stderr)
        return 1

    cols: dict[str, np.ndarray] = {}
    for f in ("f11", "f12", "f21", "f22", "f63", "f64"):
        cols[f"gasdet_{f}"] = np.full(N, np.nan)
    for _, name in EBEAM_FIELDS:
        cols[name] = np.full(N, np.nan)
    cols["ebeam_damage"] = np.full(N, -1, np.int64)
    for f in ("t1_ps", "t2_ps", "q1_pc", "q2_pc"):
        cols[f"pcav_{f}"] = np.full(N, np.nan)
    for s in IPMFEX_SRCS:
        for f in IPMFEX_FIELDS:
            cols[f"ipmfex{s:02x}_{f}"] = np.full(N, np.nan)
    for pref in ("bmmon43", "bmmon4c"):
        for f in ("sum", "xpos", "ypos"):
            cols[f"{pref}_{f}"] = np.full(N, np.nan)
    for f in ("xpos", "ypos"):
        cols[f"bmmon4b_{f}"] = np.full(N, np.nan)
    bm4b_sum = np.full(N, np.nan)  # alignment check only, not stored

    files = [open(p, "rb") for p in paths]  # noqa: SIM115 - closed together below
    try:
        for i, (_sec, _nsec, _fid, si, off) in enumerate(index):
            buf = read_dgram(files[si], off)
            leaves: list = []
            walk_leaves(buf, 20, leaves)
            for tid, sphy, poff, psize in leaves:
                if tid == 14 and psize >= 48:
                    v = struct.unpack("<6d", buf[poff : poff + 48])
                    for j, f in enumerate(("f11", "f12", "f21", "f22", "f63", "f64")):
                        cols[f"gasdet_{f}"][i] = v[j]
                elif tid == 15 and psize >= 164:
                    (dmg,) = struct.unpack("<I", buf[poff : poff + 4])
                    v = struct.unpack("<20d", buf[poff + 4 : poff + 164])
                    cols["ebeam_damage"][i] = dmg
                    for j, name in EBEAM_FIELDS:
                        cols[name][i] = v[j]
                elif tid == 16 and psize >= 32:
                    v = struct.unpack("<4d", buf[poff : poff + 32])
                    for j, f in enumerate(("t1_ps", "t2_ps", "q1_pc", "q2_pc")):
                        cols[f"pcav_{f}"][i] = v[j]
                elif tid == 31 and sphy in IPMFEX_SRCS and psize >= 28:
                    v = struct.unpack("<7f", buf[poff : poff + 28])
                    for j, f in enumerate(IPMFEX_FIELDS):
                        cols[f"ipmfex{sphy:02x}_{f}"][i] = v[j]
                elif tid == 98 and psize >= 24:
                    v = struct.unpack("<3d", buf[poff : poff + 24])
                    if sphy == 0x43:
                        cols["bmmon43_sum"][i], cols["bmmon43_xpos"][i], cols["bmmon43_ypos"][i] = v
                    elif sphy == 0x4B:
                        bm4b_sum[i] = v[0]
                        cols["bmmon4b_xpos"][i], cols["bmmon4b_ypos"][i] = v[1], v[2]
                    elif sphy == 0x4C:
                        cols["bmmon4c_sum"][i], cols["bmmon4c_xpos"][i], cols["bmmon4c_ypos"][i] = v
            if (i + 1) % 800 == 0:
                print(f"[decode] {i + 1}/{N}  ({time.time() - t0:.0f}s)", flush=True)
    finally:
        for f in files:
            f.close()

    # alignment proof: this pass must reproduce step 0's ipm2 column exactly
    diff = np.nanmax(np.abs(bm4b_sum - old["ipm2"]))
    if not (diff < 1e-9):
        print(f"ALIGNMENT FAILED: bmmon4b sum vs ipm2 max|diff| = {diff}", file=sys.stderr)
        return 1

    overlap = set(cols) & set(old)
    if overlap:
        print(f"[note] replacing existing columns: {sorted(overlap)}")
    merged = {**old, **cols}
    tmp = table_path.with_name("shot_table_tmp.npz")  # np.savez enforces .npz suffix
    np.savez(tmp, **merged)
    tmp.replace(table_path)

    readme = ROOT / "npy/README_npy.md"
    text = readme.read_text()
    if README_MARKER in text:
        text = text[: text.index(README_MARKER)].rstrip() + "\n"
    readme.write_text(text.rstrip() + "\n" + README_SECTION)

    print(
        f"[done] shot_table.npz: {len(old)} -> {len(merged)} columns "
        f"({len(cols)} added), alignment max|diff| = {diff:.1e}, "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
