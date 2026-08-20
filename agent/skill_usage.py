#!/usr/bin/env python3
"""Skill-usage reporter — what did the agents actually read, and when?

Parses outputs/<run>/events.jsonl (the SDK message stream written by
entrypoint.py) and reconstructs which skill files each phase consumed, in
order, with timestamps. Writes:

  skill_usage.md    human-readable timeline + coverage tables
  skill_usage.json  machine-readable (same content, for cross-run comparison)
  skill_usage.png   timeline figure: skill file vs elapsed time, by phase

Called automatically at the end of every agent run; also runnable standalone:

  python3 agent/skill_usage.py --run-id agent_trial_07
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL_RE = re.compile(r"skills/([\w.\-]+)/(?:methods/)?([\w.\-]+\.md)")
PHASE_ORDER = ["reduction", "center", "mask", "verify"]
# dataviz reference palette (light surface), slots 1-3 validate all-pairs
PHASE_COLOR = {"reduction": "#2a78d6", "center": "#8b6fcb", "mask": "#eb6834", "verify": "#1baf7a"}
READ_TOOLS = {"Read"}  # actual consumption
DISCOVER_TOOLS = {"Glob", "Grep", "LS"}  # looking around
DELEGATE_TOOLS = {"Agent", "Task"}  # handed to a subagent


def catalogue() -> dict[str, list[str]]:
    """Every skill .md in the repo, grouped by category folder."""
    out: dict[str, list[str]] = {}
    for p in sorted((ROOT / "skills").rglob("*.md")):
        rel = p.relative_to(ROOT / "skills")
        cat = rel.parts[0] if len(rel.parts) > 1 else "_index"
        out.setdefault(cat, []).append(p.name)
    return out


def parse_events(run_dir: Path) -> dict:
    ev = run_dir / "events.jsonl"
    if not ev.exists():
        raise FileNotFoundError(f"no events.jsonl in {run_dir}")
    records = [json.loads(line) for line in ev.read_text().splitlines() if line.strip()]
    tools = [r for r in records if r.get("type") == "tool_use"]
    if not records:
        raise ValueError("empty events.jsonl")

    t0 = datetime.fromisoformat(records[0]["ts"])
    t_end = datetime.fromisoformat(records[-1]["ts"])

    events, discovery, delegation = [], [], []
    for r in tools:
        blob = json.dumps(r.get("input") or {})
        hits = SKILL_RE.findall(blob)
        if not hits:
            continue
        ts = datetime.fromisoformat(r["ts"])
        row = {
            "ts": r["ts"],
            "elapsed_s": round((ts - t0).total_seconds(), 1),
            "phase": r.get("phase", "?"),
            "tool": r["tool"],
        }
        if r["tool"] in READ_TOOLS:
            cat, name = hits[0]
            events.append({**row, "category": cat, "file": name})
        elif r["tool"] in DISCOVER_TOOLS:
            inp = r.get("input") or {}
            discovery.append({**row, "pattern": inp.get("pattern") or inp.get("path", "")})
        elif r["tool"] in DELEGATE_TOOLS:
            delegation.append({**row, "skills_named": sorted({f"{c}/{n}" for c, n in hits})})

    # phase spans, from all events (not just skill ones)
    spans = {}
    for r in records:
        ph = r.get("phase")
        if not ph:
            continue
        e = round((datetime.fromisoformat(r["ts"]) - t0).total_seconds(), 1)
        lo, hi = spans.get(ph, (e, e))
        spans[ph] = (min(lo, e), max(hi, e))

    cat = catalogue()
    used = {c: sorted({e["file"] for e in events if e["category"] == c}) for c in cat}
    coverage = {
        c: {
            "available": len(cat[c]),
            "used": len(used.get(c, [])),
            "used_files": used.get(c, []),
            "unused_files": [f for f in cat[c] if f not in used.get(c, [])],
        }
        for c in cat
    }

    return {
        "run": run_dir.name,
        "started": records[0]["ts"],
        "ended": records[-1]["ts"],
        "duration_s": round((t_end - t0).total_seconds(), 1),
        "n_tool_calls": len(tools),
        "n_skill_reads": len(events),
        "n_distinct_skills": len({(e["category"], e["file"]) for e in events}),
        "phase_spans_s": spans,
        "reads": events,
        "discovery": discovery,
        "delegation": delegation,
        "coverage": coverage,
    }


# ---------------------------------------------------------------- markdown


def write_markdown(rep: dict, dest: Path) -> None:
    L = []
    A = L.append
    mins = rep["duration_s"] / 60
    A(f"# Skill usage — `{rep['run']}`\n")
    A(
        f"The agents read **{rep['n_distinct_skills']} distinct skill files** "
        f"({rep['n_skill_reads']} read calls) out of {rep['n_tool_calls']} tool calls, "
        f"over {mins:.1f} min.\n"
    )

    A("| Phase | Window (s) | Skill files read |")
    A("|---|---|---|")
    for ph in PHASE_ORDER:
        if ph not in rep["phase_spans_s"]:
            continue
        lo, hi = rep["phase_spans_s"][ph]
        n = len({e["file"] for e in rep["reads"] if e["phase"] == ph})
        A(f"| {ph} | {lo:.0f} – {hi:.0f} | {n} |")
    A("")

    A("## Timeline — what was read, in order\n")
    A("| # | t (s) | clock | Phase | Skill file |")
    A("|---|---|---|---|---|")
    for i, e in enumerate(rep["reads"], 1):
        A(
            f"| {i} | {e['elapsed_s']:.0f} | {e['ts'][11:19]} | {e['phase']} | "
            f"`{e['category']}/{e['file']}` |"
        )
    A("")

    A("## Coverage — which skills the run touched\n")
    A("| Category | Used | Available | Unused |")
    A("|---|---|---|---|")
    for c, v in sorted(rep["coverage"].items()):
        unused = ", ".join(f"`{u}`" for u in v["unused_files"]) or "—"
        A(f"| {c} | {v['used']} | {v['available']} | {unused} |")
    A("")
    A(
        "An unused file is not a failure — most categories are menus where the agent "
        "is expected to pick. It is a failure only when a file the run's decisions "
        "depended on was never opened.\n"
    )

    if rep["discovery"]:
        A("## Discovery calls (looking around, not reading)\n")
        A("| t (s) | Phase | Tool | Pattern |")
        A("|---|---|---|---|")
        for d in rep["discovery"]:
            A(f"| {d['elapsed_s']:.0f} | {d['phase']} | {d['tool']} | `{d['pattern']}` |")
        A("")

    if rep["delegation"]:
        A("## Delegated to subagents\n")
        for d in rep["delegation"]:
            A(
                f"- t={d['elapsed_s']:.0f}s ({d['phase']}): {d['tool']} naming "
                + ", ".join(f"`{s}`" for s in d["skills_named"])
            )
        A("")

    A(
        f"<sub>Generated by `agent/skill_usage.py` from `{rep['run']}/events.jsonl`; "
        f"run started {rep['started'][:19].replace('T', ' ')}Z.</sub>"
    )
    dest.write_text("\n".join(L))


# ---------------------------------------------------------------- figure


def write_figure(rep: dict, dest: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    SURF, PAGE = "#fcfcfb", "#f9f9f7"
    INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
    GRID, AXIS = "#e1e0d9", "#c3c2b7"

    reads = rep["reads"]
    if not reads:
        return
    # y order: category, then filename — reversed so the first-read sits on top
    labels = sorted({f"{e['category']}/{e['file']}" for e in reads})
    ypos = {lab: i for i, lab in enumerate(reversed(labels))}

    fig, (ax, axc) = plt.subplots(
        1,
        2,
        figsize=(14, 0.42 * len(labels) + 3.2),
        facecolor=PAGE,
        gridspec_kw={"width_ratios": [3.3, 1], "wspace": 0.06},
    )
    ax.set_facecolor(SURF)
    axc.set_facecolor(SURF)

    # phase bands behind everything
    for ph, (lo, hi) in rep["phase_spans_s"].items():
        c = PHASE_COLOR.get(ph, MUTED)
        ax.axvspan(lo, hi, color=c, alpha=0.06, zorder=0)
        ax.text(
            (lo + hi) / 2, len(labels) - 0.35, ph, ha="center", va="bottom", fontsize=10, color=c
        )

    ax.grid(axis="x", color=GRID, lw=0.7, zorder=1)
    ax.set_axisbelow(True)

    for i, e in enumerate(reads, 1):
        lab = f"{e['category']}/{e['file']}"
        c = PHASE_COLOR.get(e["phase"], MUTED)
        ax.scatter(
            e["elapsed_s"], ypos[lab], s=95, zorder=4, color=c, edgecolor=SURF, linewidth=1.2
        )  # 2px surface ring
        ax.annotate(
            str(i),
            (e["elapsed_s"], ypos[lab]),
            xytext=(9, 5),
            textcoords="offset points",
            fontsize=7.5,
            color=c,
            zorder=5,
        )  # read order

    # connect reads within a phase to show reading order
    for ph in PHASE_ORDER:
        seq = [e for e in reads if e["phase"] == ph]
        if len(seq) > 1:
            ax.plot(
                [e["elapsed_s"] for e in seq],
                [ypos[f"{e['category']}/{e['file']}"] for e in seq],
                color=PHASE_COLOR.get(ph, MUTED),
                lw=1.2,
                alpha=0.45,
                zorder=3,
            )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(reversed(labels)), fontsize=9, color=INK)
    ax.set_ylim(-0.8, len(labels) - 0.05)
    ax.set_xlabel("elapsed time since run start (s)", color=INK2)
    ax.set_xlim(left=-rep["duration_s"] * 0.02)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)

    ax.set_title(
        f"Skill files read during {rep['run']} — "
        f"{rep['n_distinct_skills']} distinct files, "
        f"{rep['duration_s'] / 60:.1f} min "
        "(numbers = read order)",
        fontsize=12,
        color=INK,
        pad=14,
        loc="left",
    )
    handles = [
        Line2D([], [], marker="o", ls="", markersize=9, color=PHASE_COLOR[p], label=p)
        for p in PHASE_ORDER
        if p in rep["phase_spans_s"]
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.95)

    # ---- right panel: per-category coverage -----------------------------
    cats = [c for c in sorted(rep["coverage"]) if c != "_index"]
    yy = range(len(cats))
    avail = [rep["coverage"][c]["available"] for c in cats]
    used = [rep["coverage"][c]["used"] for c in cats]
    axc.barh(list(yy), avail, height=0.55, color=GRID, zorder=2)
    axc.barh(list(yy), used, height=0.55, color="#2a78d6", zorder=3)
    for i, (u, a) in enumerate(zip(used, avail, strict=True)):
        axc.text(
            a + max(avail) * 0.04, i, f"{u}/{a}", va="center", fontsize=9, color=INK if u else MUTED
        )
    axc.set_yticks(list(yy))
    axc.set_yticklabels(cats, fontsize=9, color=INK)
    axc.set_xlim(0, max(avail) * 1.30)
    axc.set_xticks([])
    axc.invert_yaxis()
    axc.grid(False)
    for s in ("top", "right", "bottom", "left"):
        axc.spines[s].set_visible(False)
    axc.set_title("read / available", fontsize=10, color=INK2, pad=14, loc="left")

    # tight_layout cannot handle the two-panel + long-label combination; set the
    # left margin explicitly and let bbox_inches finish the job
    fig.subplots_adjust(left=0.29, right=0.985, top=0.90, bottom=0.085)
    fig.savefig(dest, dpi=150, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)


def generate(run_dir: Path) -> dict:
    rep = parse_events(run_dir)
    (run_dir / "skill_usage.json").write_text(json.dumps(rep, indent=2))
    write_markdown(rep, run_dir / "skill_usage.md")
    try:
        write_figure(rep, run_dir / "skill_usage.png")
    except Exception as exc:  # a figure must never fail a run
        print(f"[skill_usage] figure skipped: {exc}")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_dir = ROOT / "outputs" / args.run_id
    rep = generate(run_dir)
    print(
        f"[skill_usage] {rep['n_distinct_skills']} distinct skills, "
        f"{rep['n_skill_reads']} reads -> {run_dir}/skill_usage.{{md,json,png}}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
