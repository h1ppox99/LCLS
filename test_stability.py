#!/usr/bin/env python3
"""Stability test for the two-agent pipeline.

Runs the entrypoint K times into outputs/trial_XX/, then compares across trials:
- decisions.json         (identical? which fields diverge?)
- accumulate_log.json    (kept counts, sum totals)
- mask_assembled.npy     (pixel counts, pairwise IoU)
- masked_sum.npy         (pairwise max relative difference on shared pixels)

Writes outputs/stability_report.md.

Usage:
  python3 test_stability.py --trials 3 --no-llm     # mechanics only, deterministic
  python3 test_stability.py --trials 3              # full agent runs (needs creds)
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def run_trial(k: int, no_llm: bool) -> Path:
    run_id = f"trial_{k:02d}" + ("_nollm" if no_llm else "")
    cmd = [sys.executable, str(ROOT / "agent/entrypoint.py"), "--run-id", run_id]
    if no_llm:
        cmd.append("--no-llm")
    print(f"=== trial {k}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    return OUT / run_id


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for key, v in d.items():
        p = f"{prefix}.{key}" if prefix else key
        if isinstance(v, dict):
            out.update(flat(v, p))
        elif key != "rationale":            # free text may differ legitimately
            out[p] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    dirs = [run_trial(k, args.no_llm) for k in range(1, args.trials + 1)]

    decisions = [flat(json.loads((d / "decisions.json").read_text())) for d in dirs]
    logs = [json.loads((d / "accumulate_log.json").read_text()) for d in dirs]
    masks = [np.load(d / "mask_assembled.npy") for d in dirs]
    sums = [np.load(d / "masked_sum.npy") for d in dirs]

    lines = [f"# Stability report — {len(dirs)} trials "
             f"({'no-llm baseline' if args.no_llm else 'LLM agents'})\n"]

    keys = sorted(set(itertools.chain(*[d.keys() for d in decisions])))
    diverging = [k for k in keys if len({json.dumps(d.get(k), default=str) for d in decisions}) > 1]
    lines.append("## Decisions")
    lines.append(f"- fields compared (rationales excluded): {len(keys)}")
    if diverging:
        lines.append(f"- **DIVERGING fields: {diverging}**")
        for k in diverging:
            lines.append(f"    - `{k}`: " + " | ".join(str(d.get(k)) for d in decisions))
    else:
        lines.append("- all decision fields identical across trials ✓")

    lines.append("\n## Accumulation")
    kept = [lg["kept"] for lg in logs]
    tot = [lg["sum_total_keV"] for lg in logs]
    lines.append(f"- kept shots per trial: {kept}"
                 + (" ✓ identical" if len(set(kept)) == 1 else " **DIVERGES**"))
    rel = (max(tot) - min(tot)) / abs(np.mean(tot)) if np.mean(tot) else 0.0
    lines.append(f"- sum_total_keV spread: {rel*100:.4f}% "
                 + ("✓" if rel < 1e-9 else "(nonzero — weights or kept-set differ)"))

    lines.append("\n## Mask")
    npx = [int(m.sum()) for m in masks]
    lines.append(f"- masked pixels per trial: {npx}")
    for (i, a), (j, b) in itertools.combinations(enumerate(masks, 1), 2):
        iou = (a & b).sum() / max((a | b).sum(), 1)
        lines.append(f"- IoU(trial{i}, trial{j}) = {iou:.4f}"
                     + (" ✓" if iou > 0.99 else ""))

    lines.append("\n## Masked sum (endpoint)")
    for (i, a), (j, b) in itertools.combinations(enumerate(sums, 1), 2):
        both = np.isfinite(a) & np.isfinite(b)
        denom = np.abs(a[both]).mean()
        d = np.abs(a[both] - b[both]).max() / denom if denom else 0.0
        lines.append(f"- max relative diff on shared pixels (t{i},t{j}): {d:.3e}")

    verdict = "STABLE" if not diverging and len(set(kept)) == 1 and \
        all((a & b).sum() / max((a | b).sum(), 1) > 0.99
            for a, b in itertools.combinations(masks, 2)) else "UNSTABLE — see above"
    lines.append(f"\n## Verdict: **{verdict}**")

    report = OUT / "stability_report.md"
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[report] {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
