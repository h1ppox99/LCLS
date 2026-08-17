#!/usr/bin/env python3
"""
dev/setup_psdm_layout.py -- make the local data mirror readable by psana.

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

    python -m automask.dev.setup_psdm_layout

then (in the psana env):

    export SIT_PSDM_DATA=/home/groups/darve/hippowal/psdm
    python -m automask.producers.build_images --run ${RUN}

Re-running is safe (idempotent): it rebuilds the links.
"""
from __future__ import annotations
import os
import shutil

COLON = chr(0xF022)                       # the on-disk stand-in for ':'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INSTRUMENT = "xpp"
EXPERIMENT = "xppl1016922"
# Where to build the psana-style tree. Prefer writable group storage, not home.
# psana_env.sh sets SIT_PSDM_DATA from PSANA_PSDM in psana_env.local.
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

    # 2) calib. Two cases, and getting this wrong fails SILENTLY -- psana falls
    #    back to uncalibrated frames rather than erroring.
    calib_src = os.path.join(ROOT, "calib")
    calib_dst = os.path.join(exp_dir, "calib")
    entries = os.listdir(calib_src)

    # 2a) The tree already has real colons -- it was copied on Linux, or `calib`
    #     is a symlink to group storage that never went through macOS. Nothing
    #     to rename, so one directory symlink is enough, exactly like xtc.
    if not any(COLON in name for name in entries):
        _link(calib_src, calib_dst)
        print(f"  calib -> {calib_src} (real-colon names, linked whole)")
        print(f"\nSIT_PSDM_DATA layout ready at: {PSDM}")
        print(f"  export SIT_PSDM_DATA={PSDM}")
        return PSDM

    # 2b) macOS copy: rebuild the tree with real-colon names, symlinking the
    #    constant-type dirs (pedestals, pixel_gain, geometry, ...) which have
    #    no colons in their own names or their .data files.
    n = 0
    for dtype in entries:                                  # e.g. Jungfrau<U+F022><U+F022>CalibV1
        dtype_p = os.path.join(calib_src, dtype)
        if not os.path.isdir(dtype_p) or "CalibV1" not in dtype:
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
    if n == 0:
        raise RuntimeError(
            f"no calibration constants linked from {calib_src}. psana would "
            f"fall back to UNCALIBRATED frames without saying so, which is why "
            f"this is an error and not a warning. Found: {sorted(entries)[:6]}")
    print(f"  calib -> {n} constant-type dirs linked with real-colon names")

    print(f"\nSIT_PSDM_DATA layout ready at: {PSDM}")
    print(f"  export SIT_PSDM_DATA={PSDM}")
    print(f"  dataset string:  exp={EXPERIMENT}:run=<RUN>:dir={os.path.join(exp_dir,'xtc')}")
    return PSDM


def check(run: int = 475) -> bool:
    """Prove the layout works: open the run and pull ONE calibrated frame.

    `det.calib()` returning None is the failure this whole script exists to
    prevent, and psana does not raise for it -- a mis-wired calib directory
    yields uncalibrated data or None with no error anywhere. So the check is an
    actual frame, not the presence of a directory.
    """
    import numpy as np
    from automask.io.read_xtc import JUNGFRAU_NAME, calib_dir, local_run_source

    ok = True
    cdir = calib_dir()
    print(f"  SIT_PSDM_DATA = {os.environ.get('SIT_PSDM_DATA', '(unset)')}")
    print(f"  calib dir     = {cdir}  exists={os.path.isdir(cdir)}")
    if not os.path.isdir(cdir):
        print("  -> psana will silently return uncalibrated frames. Fix SIT_PSDM_DATA.")
        ok = False

    for name in ("xtc", "calib"):
        p = os.path.join(ROOT, name)
        target = os.path.realpath(p)
        print(f"  <repo>/{name:<6}-> {target}  exists={os.path.exists(target)}")
        ok &= os.path.exists(target)

    import psana
    source = local_run_source(run)
    ds = source.open()
    files = source.files
    print(f"  run {run}: {len(files)} stream(s)")
    det = psana.Detector(JUNGFRAU_NAME)
    for evt in ds.events():
        frame = det.calib(evt)
        if frame is None:
            print("  -> det.calib() returned None: calibration is NOT wired up.")
            return False
        frame = np.asarray(frame)
        print(f"  det.calib() -> shape {frame.shape}, mean {frame.mean():.3f}, "
              f"finite {np.isfinite(frame).all()}")
        break
    else:
        print(f"  -> no events decoded from run {run}")
        return False
    print("  OK" if ok else "  PROBLEMS ABOVE")
    return ok


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="after building, open a run and pull one calibrated frame")
    ap.add_argument("--run", type=int, default=475)
    args = ap.parse_args()
    build()
    if args.check:
        print("\n=== checking ===")
        sys.exit(0 if check(args.run) else 1)
