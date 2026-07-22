"""Minimal pure-Python LCLS-1 XTC parser for the XTC_Agent pipeline.

Self-contained (no psana). Knows just enough to pull, per L1Accept event:
- the Jungfrau1M raw frame  (TypeId 108 leaf, (2,512,1024) uint16)
- ipm2                      (TypeId 98 / BldDataBeamMonitor, src phy 0x4b,
                             float64 at payload offset 0 — located by value-match
                             against smalldata and verified on Run0475)
- EVR event codes           (TypeId 19 / EvrData v4)
  xray on  = code 137 present (162 = dropped shot)
  laser on = code 90  present (91  = laser off)

Datagram layout (LCLS-1): 40 B header = ClockTime(nsec,sec) + TimeStamp(st0,st1)
+ env; service = (st0>>24)&0x1f (12 = L1Accept); fiducial = st1 & 0x1ffff;
total dgram size = 20 + extent(u32 at byte 36).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

NPIX = 2 * 512 * 1024
FRAME_SHAPE = (2, 512, 1024)
L1ACCEPT = 12
TID_CONTAINER, TID_EVR, TID_BMMON, TID_JUNGFRAU = 1, 19, 98, 108
SRC_IPM2 = 0x4B  # XPP-SB2-BMMON
CODE_XRAY_ON, CODE_XRAY_DROP, CODE_LASER_ON = 137, 162, 90


def scan_headers(path: str | Path):
    """Header-only walk. Yields (offset, total, sec, nsec, fiducial) per L1Accept."""
    with open(path, "rb") as f:
        off = 0
        while True:
            f.seek(off)
            hdr = f.read(40)
            if len(hdr) < 40:
                return
            nsec, sec, st0, st1 = struct.unpack("<IIII", hdr[0:16])
            (ext,) = struct.unpack("<I", hdr[36:40])
            total = 20 + ext
            if total < 40:
                return
            if (st0 >> 24) & 0x1F == L1ACCEPT:
                yield off, total, sec, nsec, st1 & 0x1FFFF
            off += total


def read_dgram(f, offset: int) -> bytes:
    f.seek(offset)
    hdr = f.read(40)
    (ext,) = struct.unpack("<I", hdr[36:40])
    f.seek(offset)
    return f.read(20 + ext)


def walk_leaves(buf: bytes, off: int, out: list):
    """Collect (tid, src_phy, payload_off, payload_size) for every non-container Xtc."""
    if off + 20 > len(buf):
        return
    _, sphy = struct.unpack("<II", buf[off + 4 : off + 12])
    contains, ext = struct.unpack("<II", buf[off + 12 : off + 20])
    tid = contains & 0xFFFF
    if ext < 20:
        return
    if tid == TID_CONTAINER:
        p = off + 20
        end = off + ext
        while p + 20 <= end:
            (ce,) = struct.unpack("<I", buf[p + 16 : p + 20])
            if ce < 20:
                break
            walk_leaves(buf, p, out)
            p += ce
    else:
        out.append((tid, sphy, off + 20, ext - 20))


def parse_event(buf: bytes):
    """One L1Accept dgram -> dict(frame_raw, ipm2, xray, laser, ncodes)."""
    leaves: list = []
    walk_leaves(buf, 20, leaves)
    frame = None
    ipm2 = np.nan
    xray = laser = -1
    for tid, sphy, poff, psize in leaves:
        if tid == TID_JUNGFRAU and psize >= NPIX * 2:
            hb = psize - NPIX * 2  # device header precedes the pixels
            frame = np.frombuffer(buf, "<u2", count=NPIX, offset=poff + hb).reshape(FRAME_SHAPE)
        elif tid == TID_BMMON and sphy == SRC_IPM2 and psize >= 8:
            (ipm2,) = struct.unpack("<d", buf[poff : poff + 8])
        elif tid == TID_EVR:
            (n,) = struct.unpack("<I", buf[poff : poff + 4])
            codes = {
                struct.unpack("<I", buf[poff + 4 + 12 * i + 8 : poff + 4 + 12 * i + 12])[0]
                for i in range(n)
            }
            xray = int(CODE_XRAY_ON in codes and CODE_XRAY_DROP not in codes)
            laser = int(CODE_LASER_ON in codes)
    return {"frame_raw": frame, "ipm2": ipm2, "xray": xray, "laser": laser}


def calibrate(raw: np.ndarray, ped: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """(ADC - ped[gain mode]) / gain[gain mode], float32 keV-equivalent."""
    gb = raw >> 14
    adc = (raw & 0x3FFF).astype(np.float32)
    mode = np.zeros(FRAME_SHAPE, np.intp)
    mode[gb == 1] = 1
    mode[gb == 3] = 2
    pedm = np.take_along_axis(ped, mode[None], 0)[0]
    gm = np.take_along_axis(gain, mode[None], 0)[0]
    return (adc - pedm) / gm
