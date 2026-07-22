"""QA check on 1D radial averages: peak position + smoothness.

For each .npz in --curves-dir, find the strongest peak in the search window,
check it lies within the expected CO2 range, score smoothness via the
second-derivative L2 norm.

Usage:
  python /app/.claude/skills/qa/scripts/qa_check.py \\
    --curves-dir /outputs/<run>/curves \\
    --output /outputs/<run>/qa_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def find_peak(q, intensity, q_lo, q_hi):
    sel = (q >= q_lo) & (q <= q_hi)
    if not np.any(sel):
        return None
    q_sel, i_sel = q[sel], intensity[sel]
    valid = ~np.isnan(i_sel)
    if not np.any(valid):
        return None
    idx = int(np.argmax(i_sel[valid]))
    return float(q_sel[valid][idx]), float(i_sel[valid][idx])


def smoothness(intensity):
    """Second-derivative L2 norm, normalized by intensity range. Lower = smoother."""
    valid = ~np.isnan(intensity)
    ic = intensity[valid]
    if len(ic) < 3:
        return float("nan")
    d2 = np.diff(ic, n=2)
    rng = float(np.nanmax(ic) - np.nanmin(ic))
    if rng <= 0:
        return float("nan")
    return float(np.linalg.norm(d2) / rng)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--curves-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    # Peak windows must come from the manifest. No defaults — passing
    # campaign-specific values via hardcoded fallback was the source of
    # a prior validation failure.
    p.add_argument("--peak-min", type=float, required=True,
                   help="lower bound of expected peak window (Å⁻¹); from manifest")
    p.add_argument("--peak-max", type=float, required=True,
                   help="upper bound of expected peak window (Å⁻¹); from manifest")
    p.add_argument("--search-lo", type=float, required=True,
                   help="search window low (Å⁻¹); typically peak-min minus a margin")
    p.add_argument("--search-hi", type=float, required=True,
                   help="search window high (Å⁻¹); typically peak-max plus a margin")
    args = p.parse_args()

    curves = sorted(args.curves_dir.glob("*.npz"))
    rows = []
    pass_n = 0
    for npz in curves:
        data = np.load(npz)
        q, intensity = data["q"], data["I"]
        peak = find_peak(q, intensity, args.search_lo, args.search_hi)
        sm = smoothness(intensity)
        if peak is None:
            verdict = "FAIL: no peak in search window"
            qpeak = float("nan")
        else:
            qpeak, _ = peak
            if args.peak_min <= qpeak <= args.peak_max:
                verdict = "PASS"
                pass_n += 1
            else:
                verdict = f"FAIL: peak at {qpeak:.4f} outside [{args.peak_min}, {args.peak_max}]"
        rows.append({"file": npz.name, "qpeak": qpeak, "smoothness": sm, "verdict": verdict})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# QA report",
        "",
        f"Expected main peak: [{args.peak_min}, {args.peak_max}] Å⁻¹ (liquid CO2).",
        f"Search window: [{args.search_lo}, {args.search_hi}] Å⁻¹.",
        "",
        f"**{pass_n} / {len(rows)} passed.**",
        "",
        "| File | peak q (Å⁻¹) | smoothness | verdict |",
        "|------|---------------|------------|---------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['file']} | {r['qpeak']:.4f} | {r['smoothness']:.4f} | {r['verdict']} |"
        )
    args.output.write_text("\n".join(lines) + "\n")

    # Also write a JSON summary for the agent to consume programmatically
    summary = {"pass_n": pass_n, "total": len(rows), "rows": rows}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.output}: {pass_n}/{len(rows)} passed")
    return 0 if pass_n == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
