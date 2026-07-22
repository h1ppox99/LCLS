#!/usr/bin/env python3
"""Step 2 (deterministic executor): apply the agent's decisions.json, then accumulate.

Reads outputs/<run>/decisions.json (written by the reduction agent), builds the
keep-mask and per-shot weights from npy/shot_table.npz, streams frames from the
npy/frames_raw.npy memmap, calibrates each kept frame, and writes:

  outputs/<run>/sum_calib.npy      (2, 512, 1024) float64 — (weighted) calib sum
  outputs/<run>/sum_assembled.npy  (1064, 1030)  float64 — psana geometry via calib/ix,iy
  outputs/<run>/sum.png            robust-scaled preview for the mask agent
  outputs/<run>/accumulate_log.json  counts + settings, the provenance record

decisions.json schema (all fields required; "rationale" free text):
{
  "selection": {
    "require_xray_on": true,
    "low_ipm_exclusion":  {"run": true,  "threshold": 3000.0, "rationale": "..."},
    "high_ipm_exclusion": {"run": false, "percentile": 99.0,  "rationale": "..."}
  },
  "normalization": {
    "run": true, "monitor": "ipm2", "form": "per_shot",   # or "ratio_of_sums" / no-op
    "offset": "auto",                                      # or a number
    "rationale": "..."
  }
}

"monitor" may name any per-shot column of npy/shot_table.npz (see npy/README_npy.md;
e.g. "ipmfex22_sum") — the skills/normalization/ monitor menu (one file per
reference parameter) governs that choice. "offset" is always the ipm2 offset (selection thresholds act on ipm2c);
a non-ipm2 monitor gets its own zero offset from the optional "monitor_offset"
field ("auto" = median over x-ray-off shots, the default, or a number).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xtclib import calibrate

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="outputs/<run> dir holding decisions.json")
    args = ap.parse_args()
    out = Path(args.out_dir)
    dec = json.loads((out / "decisions.json").read_text())

    t = np.load(ROOT / "npy/shot_table.npz")
    ipm2, xray = t["ipm2"], t["xray"]
    N = len(ipm2)

    sel = dec["selection"]
    keep = np.ones(N, bool)
    if sel.get("require_xray_on", True):
        keep &= xray == 1

    off_cfg = dec["normalization"].get("offset", "auto")
    offset = float(np.median(ipm2[xray == 0])) if off_cfg == "auto" else float(off_cfg)
    ipm2c = ipm2 - offset

    low = sel["low_ipm_exclusion"]
    if low["run"]:
        keep &= ipm2c > float(low["threshold"])
    high = sel["high_ipm_exclusion"]
    if high["run"]:
        keep &= ipm2 < np.percentile(ipm2[xray == 1], float(high["percentile"]))

    norm = dec["normalization"]
    weights = np.ones(N)
    mono_name = norm.get("monitor", "ipm2")
    mono_offset = offset
    if norm["run"]:
        if mono_name != "ipm2" and mono_name in t.files:
            raw = t[mono_name].astype(float)
            moff = norm.get("monitor_offset", "auto")
            mono_offset = (float(np.nanmedian(raw[xray == 0])) if moff == "auto"
                           else float(moff))
            mono = raw - mono_offset
        else:
            if mono_name != "ipm2":
                print(f"[warn] monitor {mono_name} not in shot_table; falling back to ipm2",
                      file=sys.stderr)
                mono_name = "ipm2"
            mono = ipm2c.copy()
        bad = keep & ~(mono > 0)          # also catches NaN (NaN > 0 is False)
        if bad.any():                       # normalization must never divide by <=0
            keep &= mono > 0
            print(f"[guard] dropped {int(bad.sum())} kept shots with monitor <= 0", file=sys.stderr)
        if norm.get("form", "per_shot") == "per_shot":
            weights[keep] = mono[keep].mean() / mono[keep]
        # ratio_of_sums: accumulate plain, record divisor in the log instead

    ped = np.load(ROOT / "calib/ped.npy")
    gain = np.load(ROOT / "calib/gain.npy")
    frames = np.load(ROOT / "npy/frames_raw.npy", mmap_mode="r")

    t0 = time.time()
    s = np.zeros((2, 512, 1024))
    kept_idx = np.where(keep)[0]
    for k, i in enumerate(kept_idx):
        s += weights[i] * calibrate(np.asarray(frames[i]), ped, gain)
        if (k + 1) % 500 == 0:
            print(f"[accumulate] {k+1}/{len(kept_idx)} ({time.time()-t0:.0f}s)", flush=True)

    np.save(out / "sum_calib.npy", s)
    ix = np.load(ROOT / "calib/ix.npy").astype(np.intp)
    iy = np.load(ROOT / "calib/iy.npy").astype(np.intp)
    img = np.zeros((1064, 1030))
    img[ix, iy] = s
    np.save(out / "sum_assembled.npy", img)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nz = img[img != 0]
    fig, ax = plt.subplots(figsize=(9, 9.5))
    im = ax.imshow(img, cmap="viridis", vmin=0, vmax=np.percentile(nz, 99.5),
                   interpolation="nearest")
    ax.set_title(f"{out.name}: sum of {len(kept_idx)} kept shots"
                 f" ({'normalized' if norm['run'] else 'plain'})")
    plt.colorbar(im, ax=ax, shrink=0.85, label="summed keV")
    fig.savefig(out / "sum.png", dpi=110, bbox_inches="tight")

    log = {
        "n_events": int(N),
        "kept": int(keep.sum()),
        "rejected": int(N - keep.sum()),
        "ipm2_offset": offset,
        "monitor": mono_name if norm["run"] else None,
        "monitor_offset": mono_offset if norm["run"] else None,
        "weights_range": [float(weights[keep].min()), float(weights[keep].max())]
        if keep.any() else None,
        "ratio_of_sums_divisor": float(mono[keep].sum())
        if norm["run"] and norm.get("form") == "ratio_of_sums" else None,
        "sum_total_keV": float(s.sum()),
        "elapsed_s": round(time.time() - t0, 1),
        "decisions_echo": dec,
    }
    (out / "accumulate_log.json").write_text(json.dumps(log, indent=2))
    print(f"[done] kept {log['kept']}/{N}, sum written to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
