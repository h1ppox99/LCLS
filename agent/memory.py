#!/usr/bin/env python3
"""Workflow memory — durable measured findings, injected into agent prompts.

memory/decisions.jsonl holds one JSON object per finding. entrypoint.py calls
prompt_block(phase) and pastes the result into that phase's prompt, so an agent
sees the memory before it opens any skill file.

Standalone:
  python3 agent/memory.py               # show every active entry
  python3 agent/memory.py --phase mask  # what the mask agent will be told
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "memory" / "decisions.jsonl"


def load(phase: str | None = None, include_retired: bool = False) -> list[dict]:
    if not STORE.exists():
        return []
    out = []
    for line in STORE.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if not include_retired and e.get("status") != "active":
            continue
        if phase and e.get("phase") != phase:
            continue
        out.append(e)
    return out


def prompt_block(phase: str) -> str:
    """The text pasted into a phase prompt. Empty when there is nothing to say."""
    entries = load(phase)
    if not entries:
        return ""
    parts = [
        "\n\nWORKFLOW MEMORY — measured findings from earlier runs of this "
        "pipeline. These are established by measurement, not guesses; follow them "
        "unless this run's evidence contradicts them, and say so explicitly if it "
        "does.\n"
    ]
    for e in entries:
        parts.append(
            f"[{e['id']}] {e['title']}\n"
            f"  RULE: {e['rule']}\n"
            f"  EVIDENCE: {e['evidence']}\n"
            f"  SCOPE: {e['scope']}\n"
        )
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["reduction", "mask", "verify"])
    ap.add_argument("--all", action="store_true", help="include retired entries")
    args = ap.parse_args()
    entries = load(args.phase, include_retired=args.all)
    if not entries:
        print("(no memory entries)")
        return 0
    for e in entries:
        print(f"── {e['id']}  [{e['phase']}]  {e['date']}  ({e['confidence']})")
        print(f"   {e['title']}")
        print(f"   RULE     : {e['rule']}")
        print(f"   EVIDENCE : {e['evidence']}")
        if e.get("why"):
            print(f"   WHY      : {e['why']}")
        print(f"   SCOPE    : {e['scope']}")
        if e.get("conflict_resolved"):
            print(f"   SETTLES  : {e['conflict_resolved']}")
        print(f"   SOURCE   : {', '.join(e['source'])}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
