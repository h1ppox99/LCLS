#!/usr/bin/env python3
"""
setup_psdm_layout.py -- make the local xppl1016922 data readable by psana.

psana finds data through the environment variable SIT_PSDM_DATA and a *fixed*
directory layout:

    $SIT_PSDM_DATA/<instrument>/<experiment>/xtc/*.xtc
    $SIT_PSDM_DATA/<instrument>/<experiment>/calib/<Type::CalibV1>/<Src>/<ctype>/...

Our local copy differs in two ways, so psana can't read it directly:

  1. It lives at   <ROOT>/xtc  and  <ROOT>/calib   (flat), not under
     xpp/xppl1016922/.
  2. Every ':' in the calib directory names was replaced by U+F022 when the
     data was copied on macOS (macOS forbids ':' in filenames).  psana looks
     for the real-colon names (e.g. "Jungfrau::CalibV1",
     "XppEndstation.0:Jungfrau.0"), which don't exist on disk.

Linux happily allows ':' in filenames, so this script builds a light-weight
*symlink farm* with the correct psana layout and the correct real-colon names,
pointing back at the real files.  Nothing is copied; it costs ~no disk.

Run once:

    python setup_psdm_layout.py

then (in the psana env):

    export SIT_PSDM_DATA=/Data/hippolyte.wallaert/psdm
    python read_xtc.py

Re-running is safe (idempotent): it rebuilds the links.
"""
from __future__ import annotations
import os
import shutil

COLON = chr(0xF022)                       # the on-disk stand-in for ':'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../LCLS (automask/io/ -> LCLS)
INSTRUMENT = "xpp"
EXPERIMENT = "xppl1016922"
# Where to build the psana-style tree.  On /Data (lots of free space), NOT home.
PSDM = os.environ.get("SIT_PSDM_DATA",
                      os.path.join(os.path.dirname(ROOT), "psdm"))


def _fix(name: str) -> str:
    """on-disk name (U+F022) -> real psana name (':')."""
    return name.replace(COLON, ":")


def _link(src: str, dst: str):
    """Create/refresh a symlink dst -> src, making parent dirs as needed."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst):
            os.unlink(dst)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.unlink(dst)
    os.symlink(src, dst)


def build():
    exp_dir = os.path.join(PSDM, INSTRUMENT, EXPERIMENT)
    os.makedirs(exp_dir, exist_ok=True)

    # 1) xtc: a single directory symlink is enough (filenames have no colons).
    xtc_src = os.path.join(ROOT, "xtc")
    _link(xtc_src, os.path.join(exp_dir, "xtc"))
    print(f"  xtc   -> {xtc_src}")

    # 2) calib: rebuild the tree with real-colon names, symlinking the
    #    constant-type dirs (pedestals, pixel_gain, geometry, ...) which have
    #    no colons in their own names or their .data files.
    calib_src = os.path.join(ROOT, "calib")
    calib_dst = os.path.join(exp_dir, "calib")
    n = 0
    for dtype in os.listdir(calib_src):                    # e.g. Jungfrau<U+F022><U+F022>CalibV1
        dtype_p = os.path.join(calib_src, dtype)
        if not os.path.isdir(dtype_p) or COLON not in dtype:
            continue                                       # skip pedestal_workdir etc.
        for src in os.listdir(dtype_p):                    # e.g. XppEndstation.0<U+F022>Jungfrau.0
            src_p = os.path.join(dtype_p, src)
            if not os.path.isdir(src_p):
                continue
            for ctype in os.listdir(src_p):                # pedestals, pixel_gain, geometry, ...
                ctype_p = os.path.join(src_p, ctype)
                if not os.path.isdir(ctype_p):
                    continue
                dst = os.path.join(calib_dst, _fix(dtype), _fix(src), ctype)
                _link(ctype_p, dst)
                n += 1
    print(f"  calib -> {n} constant-type dirs linked with real-colon names")

    print(f"\nSIT_PSDM_DATA layout ready at: {PSDM}")
    print(f"  export SIT_PSDM_DATA={PSDM}")
    print(f"  dataset string:  exp={EXPERIMENT}:run=475:dir={os.path.join(exp_dir,'xtc')}")
    return PSDM


if __name__ == "__main__":
    build()
