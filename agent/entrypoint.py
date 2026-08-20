"""XTC_Agent entrypoint — rewritten SDK loop for the npy-based two-agent workflow.

Pipeline (see PIPELINE.md):
  step 0 (code, run once)   : XTC -> npy/frames_raw.npy + npy/shot_table.npz
  step 1 (REDUCTION AGENT)  : reads shot_table + skills/{selection,normalization},
                              autonomously decides low/high-ipm exclusions and
                              whether to normalize -> writes decisions.json,
                              then runs step2_accumulate.py (calib + weighted sum).
  step 2 (code)             : executed by the reduction agent via Bash.
  optional center submodule : reuses a context-specified image center; otherwise
                              an agent estimates it from the LaB6 rings.
  step 3 (MASK AGENT)       : first receives aligned sum + pedestal diagnostics,
                              then draws mask_assembled.npy and applies it.
  endpoint                  : outputs/<run>/masked_sum.{npy,png}

Runs locally (no Docker). Credentials: ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL /
ANTHROPIC_MODEL from env or a .env file at the XTC_Agent root.

--no-llm mode replays the skills' default decision rules in plain Python — same
file contracts, zero API calls — for testing the mechanical stability of the
pipeline separately from agent-decision stability.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


# ---------------------------------------------------------------- event logging


def _serialize_message(message) -> list[dict]:
    cls = type(message).__name__
    ts = datetime.now(UTC).isoformat()
    if cls == "SystemMessage":
        return [{"ts": ts, "type": "system", "subtype": getattr(message, "subtype", None)}]
    if cls == "ResultMessage":
        return [
            {
                "ts": ts,
                "type": "result",
                "subtype": getattr(message, "subtype", None),
                "is_error": getattr(message, "is_error", False),
                "result": getattr(message, "result", None),
                "duration_ms": getattr(message, "duration_ms", None),
            }
        ]
    if cls in ("AssistantMessage", "UserMessage"):
        records = []
        for block in getattr(message, "content", []) or []:
            btype = type(block).__name__
            if btype == "TextBlock":
                records.append({"ts": ts, "type": "assistant_text", "text": block.text})
            elif btype == "ThinkingBlock":
                records.append(
                    {"ts": ts, "type": "thinking", "text": getattr(block, "thinking", "")}
                )
            elif btype == "ToolUseBlock":
                records.append(
                    {
                        "ts": ts,
                        "type": "tool_use",
                        "tool_use_id": getattr(block, "id", None),
                        "tool": getattr(block, "name", None),
                        "input": getattr(block, "input", None),
                    }
                )
            elif btype == "ToolResultBlock":
                content = getattr(block, "content", None)
                if isinstance(content, list):
                    content = "\n".join(
                        c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content
                    )
                records.append(
                    {
                        "ts": ts,
                        "type": "tool_result",
                        "tool_use_id": getattr(block, "tool_use_id", None),
                        "is_error": getattr(block, "is_error", False),
                        "content": content,
                    }
                )
        return records
    return []


async def _run_phase(phase: str, prompt: str, out_dir: Path) -> None:
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query

    trace_file = out_dir / "trajectory.jsonl"

    async def post_tool_hook(input_data, tool_use_id, context):
        with trace_file.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "phase": phase,
                        "tool": input_data.get("tool_name", ""),
                        "input": input_data.get("tool_input"),
                    },
                    default=str,
                )
                + "\n"
            )
        return {}

    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        cwd=str(ROOT),
        permission_mode="bypassPermissions",
        model=os.environ.get("ANTHROPIC_MODEL"),
        hooks={"PostToolUse": [HookMatcher(hooks=[post_tool_hook])]},
    )
    with (out_dir / "events.jsonl").open("a") as events_f:
        async for message in query(prompt=prompt, options=options):
            print(f"[{phase}] {type(message).__name__}", file=sys.stderr, flush=True)
            for rec in _serialize_message(message):
                rec["phase"] = phase
                events_f.write(json.dumps(rec, default=str) + "\n")
                events_f.flush()


# ---------------------------------------------------------------- agent prompts


def _memory_block(phase: str) -> str:
    """Durable measured findings for this phase (memory/decisions.jsonl)."""
    try:
        from agent.memory import prompt_block

        return prompt_block(phase)
    except Exception:  # memory must never break a run
        return ""


def _feedback_block(feedback: dict | None) -> str:
    if not feedback:
        return ""
    return f"""

PREVIOUS ITERATION FAILED VERIFICATION (iteration {feedback.get("iteration", "?")}).
Verifier feedback — you must address it explicitly in your rationale:
{feedback.get("message", "")}
"""


def reduction_prompt(out_dir: Path, feedback: dict | None = None) -> str:
    # Minimal-hint variant (the original step-by-step prompt is preserved in
    # outputs/reduction_ipm_decision_report.html and the agent_trial_* events).
    return f"""You are the REDUCTION agent for LCLS run 475 (Jungfrau1M, LaB6).
Working directory: {ROOT}

Available inputs: npy/shot_table.npz (per-shot scalars, columns documented in
npy/README_npy.md) and npy/frames_raw.npy (raw frame memmap — never load it
fully into memory; do not touch the .xtc files).

The skill files under skills/selection/ and skills/normalization/ govern your
decisions. Decide what to apply and with which parameters, then write
{out_dir}/decisions.json following the schema documented at the top of
pipeline/step2_accumulate.py, and run:

  python3 pipeline/step2_accumulate.py --out-dir {out_dir}

Every decision, including any skip, must carry a "rationale".
{_memory_block("reduction")}{_feedback_block(feedback)}"""


def center_prompt(out_dir: Path) -> str:
    return f"""You are the optional IMAGE-CENTER submodule for a Jungfrau1M run.
Working directory: {ROOT}

This is deliberately a small decision task between reduction and masking. Read
{out_dir}/center_context.json first. Your gate is only whether an assembled-image
center has already been specified in the current run context.

- If specified=true: reuse that exact row/col. Write {out_dir}/image_center.json
  with schema_version=2, coordinate_system="assembled_row_col", decision="reuse",
  activated=false, specified_in_context=true, center={{"row": ..., "col": ...}},
  source copied from the context, confidence="specified", and a short rationale.
- If specified=false: activate center calibration. Read
  skills/center/01_lab6_concentric_rings.md, open {out_dir}/sum.png, then run:

    python3 pipeline/step2c_estimate_center.py --out-dir {out_dir} --source agent_estimate

  The Agent must assign control points to at least two indexed LaB6 ring orders.
  pyFAI supplies the physical relationship between all LaB6 orders; the historical
  367/531/734 px diffuse anchors must never enter a center constraint. After fitting,
  a separate validator predicts every in-field pyFAI ring and independently checks
  the image pixels at each location. Open image_center_overlay.png and
  ring_validation_overlay.png, and read image_center.json.verification plus
  ring_validation.json.

  Apply this bounded loop (maximum two revisions):
  * verdict=accept: require ring_validation.all_in_field_match=true, inspect both
    overlays, preserve the accepted center, and stop;
  * verdict=revise: use failure_type/evidence to change only ridge-detection or
    evidence-based search bounds or the evidence-backed ring-order assignment,
    rerun the executor, and record the revision. A ring_semantics_failure means
    the proposed order assignment is contradicted by other predicted rings;
  * verdict=escalate: do not turn candidate_center into center. Preserve
    center_human_review.md and stop for a human researcher.

  Insufficient angular coverage of an outer partial ring is uncertainty evidence,
  not an escalation condition. Do not escalate solely because an arc is short.
  Since a short arc also makes its independent single-circle center unstable, do
  not use the resulting small independent-center drift as an indirect reason to
  escalate. Base the semantic decision on pyFAI cross-order agreement and the
  independent all-ring location validator.

  You may write diagnostic code only inside {out_dir}. Do not alter wavelength,
  lattice constant, reflection identity, or upstream artifacts to force a fit.

For an activated fit, finish only when image_center.json, image_center_overlay.png,
ring_validation.json, and ring_validation_overlay.png exist. A reused specified
center needs only the JSON artifact.
Do not perform masking or verification in this submodule."""


def mask_prompt(out_dir: Path, feedback: dict | None = None) -> str:
    return f"""You are the MASK agent for a Jungfrau1M scattering dataset.
Working directory: {ROOT}

Aligned inputs (all use assembled row/col coordinates):
- {out_dir}/sum_assembled.npy and sum.png: summed, calibrated shots;
- {out_dir}/pedestal_g0_assembled.npy: gain-stage-0 pedestal;
- {out_dir}/pedestal_residual_z.npy: locally normalized spatial pedestal residual;
- {out_dir}/pedestal.png: pedestal, residual candidates, and their sum-image overlay;
- {out_dir}/pedestal_anomaly_mask.npy and pedestal_components.json: compact,
  spatially coherent pedestal candidates and their measured components.
The reduction decisions are in {out_dir}/decisions.json.
The image-center decision is in {out_dir}/image_center.json. Read it and use its
assembled (row, col) coordinates for every radial or beamstop-dependent layer.

Your job — draw the detector mask ON THIS SUMMED IMAGE, guided by the masking
skill set: read skills/masking/README.md and its method files first.
Your FIRST evidence action must be to open both sum.png and pedestal.png. Compare
the same assembled coordinates. Do not inspect only the summed image.

Minimum layers to consider (justify any you skip):
- zero/dead mask: pixels <= 0 in the sum (dead pixels, panel gap) + a small
  binary dilation to cover unreliable gap-edge pixels;
- known-bad pixels: calib/status_bad.npy maps to assembled space via
  calib/ix.npy, calib/iy.npy;
- pedestal anomalies (Mask Step 2): inspect pedestal.png and
  pedestal_components.json for compact dots, blobs, blocks, or other coherent
  residuals. The candidate mask uses local-median background removal, spatial
  pooling, robust per-ASIC z scores, and connected components; it is evidence,
  not automatic truth. Compare each candidate against the aligned sum, decide
  whether it is detector-correlated, and explicitly include or skip this layer.
  Moderate summed intensity is NOT a reason to skip a feature when the pedestal
  has a coherent anomaly at the same coordinate;
- beamstop shadow and any geometry features you can SEE in the image (inspect
  crops/percentiles with small Python scripts and relate them to the center artifact);
- double-width ASIC-boundary lines if they bias the image.

Then:
1. Write {out_dir}/mask_assembled.npy — (1064, 1030) bool, True = masked.
2. Run:  python3 pipeline/step3_apply_mask.py --out-dir {out_dir}
3. Write {out_dir}/mask_rationale.md — layer-by-layer: what you masked, why,
   and the pixel count each layer added.
The masked_sum.png it produces is the pipeline endpoint.
{_memory_block("mask")}{_feedback_block(feedback)}"""


def verify_prompt(out_dir: Path, iteration: int = 1) -> str:
    return f"""You are the VERIFIER agent for LCLS run 475 (Jungfrau1M, LaB6).
Working directory: {ROOT}

The pipeline endpoint to judge is {out_dir}/masked_sum.npy (upstream context:
{out_dir}/decisions.json, {out_dir}/mask_rationale.md).

Steps:
1. Run:  python3 pipeline/step4_iq.py --out-dir {out_dir}
   (deterministic: writes iq.npy, iq.png, iq_metrics.json — do not compute the
   physics yourself; judge the metrics it produces).
2. Read skills/verification/README.md and EVERY criterion file in
   skills/verification/. Each criterion carries a machine-readable ```json
   threshold block. You may open iq.png for a visual sanity check.
3. Judge every criterion against {out_dir}/iq_metrics.json and write
   {out_dir}/verify_report.json exactly following the schema in
   skills/verification/README.md, with "iteration": {iteration}.

Rules:
- Every pass/fail must cite measured value vs threshold.
- One failed sub-check fails the criterion; one failed criterion fails the run.
- On overall "fail": feedback.target_phase must be "reduction" or "mask"
  (choose the most likely responsible phase per the criterion's routing notes)
  and feedback.message must be concrete and actionable — it will be injected
  verbatim into that agent's prompt on the next iteration.
- On overall "pass": feedback is null.
- Do not modify thresholds, skills, or any upstream artifact.{_memory_block("verify")}"""


# ---------------------------------------------------------------- no-llm baseline


def baseline_decisions(out_dir: Path) -> None:
    """Default decisions straight from the skills' rules (no LLM)."""
    import numpy as np

    t = np.load(ROOT / "npy/shot_table.npz")
    ipm2, xray = t["ipm2"], t["xray"]
    offset = float(np.median(ipm2[xray == 0]))
    dec = {
        "selection": {
            "require_xray_on": True,
            "low_ipm_exclusion": {
                "run": True,
                "threshold": 3000.0,
                "rationale": "baseline: normalization planned -> plateau threshold "
                f"(offset {offset:+.1f} subtracted); near-zero pile present",
            },
            "high_ipm_exclusion": {
                "run": False,
                "percentile": 99.0,
                "rationale": "baseline: det/ipm2 linear to the top on Run0475 -> skip",
            },
        },
        "normalization": {
            "run": True,
            "monitor": "ipm2",
            "form": "per_shot",
            "offset": "auto",
            "rationale": "baseline: bin CV 6.03%->3.55% on Run0475 -> normalize",
        },
    }
    (out_dir / "decisions.json").write_text(json.dumps(dec, indent=2))


def _center_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text())
    if artifact.get("coordinate_system") != "assembled_row_col":
        raise ValueError(f"unsupported center coordinate system in {path}")
    return artifact


def _read_center(path: Path) -> tuple[float, float]:
    artifact = _center_artifact(path)
    center = artifact.get("center") or {}
    if not center:
        decision = artifact.get("decision", "unknown")
        raise ValueError(f"no accepted center in {path} (decision={decision})")
    row, col = float(center["row"]), float(center["col"])
    if not (math.isfinite(row) and math.isfinite(col)):
        raise ValueError(f"non-finite center in {path}")
    return row, col


def center_outcome(path: Path) -> str:
    artifact = _center_artifact(path)
    decision = artifact.get("decision")
    if decision in ("estimate", "reuse") and artifact.get("center"):
        _read_center(path)
        return "accept"
    if decision in ("revise", "escalate"):
        return decision
    raise ValueError(f"unrecognized center decision in {path}: {decision!r}")


def write_center_context(out_dir: Path, supplied: list[float] | None) -> dict:
    """Expose only the fact the center gate is allowed to use."""
    existing = out_dir / "image_center.json"
    if supplied is not None:
        context = {
            "specified": True,
            "center": {"row": float(supplied[0]), "col": float(supplied[1])},
            "source": "cli --image-center",
        }
    elif existing.exists():
        artifact = _center_artifact(existing)
        if artifact.get("center") and artifact.get("decision") in ("estimate", "reuse"):
            row, col = _read_center(existing)
            context = {
                "specified": True,
                "center": {"row": row, "col": col},
                "source": "existing accepted run artifact",
            }
        else:
            context = {
                "specified": False,
                "center": None,
                "source": None,
                "previous_center_outcome": {
                    "decision": artifact.get("decision"),
                    "candidate_center": artifact.get("candidate_center"),
                    "verification": artifact.get("verification"),
                },
            }
    else:
        context = {"specified": False, "center": None, "source": None}
    (out_dir / "center_context.json").write_text(json.dumps(context, indent=2) + "\n")
    return context


def baseline_center(out_dir: Path, context: dict, run_py) -> None:
    """No-LLM replay of the same specified/reuse gate."""
    if context["specified"]:
        artifact = {
            "schema_version": 2,
            "coordinate_system": "assembled_row_col",
            "decision": "reuse",
            "activated": False,
            "specified_in_context": True,
            "center": context["center"],
            "source": context["source"],
            "confidence": "specified",
            "verification": {
                "verdict": "accept",
                "failure_type": None,
                "message": "Center was specified in the current run context.",
            },
            "rationale": "The current run context already specifies the image center.",
        }
        (out_dir / "image_center.json").write_text(json.dumps(artifact, indent=2) + "\n")
    else:
        run_py(
            "pipeline/step2c_estimate_center.py",
            "--out-dir",
            str(out_dir),
            "--source",
            "deterministic_no_llm",
        )


def center_stop_code(out_dir: Path) -> int | None:
    """Return a terminal code for revise/escalate, or None for an accepted center."""
    path = out_dir / "image_center.json"
    outcome = center_outcome(path)
    if outcome == "escalate":
        artifact = _center_artifact(path)
        verification = artifact.get("verification") or {}
        print(
            f"[center] HUMAN REVIEW REQUIRED: {verification.get('message', '')} "
            f"-> {out_dir / 'center_human_review.md'}",
            file=sys.stderr,
        )
        return 6
    if outcome == "revise":
        artifact = _center_artifact(path)
        verification = artifact.get("verification") or {}
        print(
            f"[center] revision required: {verification.get('message', '')}",
            file=sys.stderr,
        )
        return 7
    return None


def baseline_mask(out_dir: Path) -> None:
    """Geometry/status/pedestal baseline mask (no LLM)."""
    import numpy as np
    from scipy.ndimage import binary_dilation

    img = np.load(out_dir / "sum_assembled.npy")
    gap = img == 0  # unfilled assembled pixels
    med = np.median(img[~gap])
    mad = np.median(np.abs(img[~gap] - med))
    dead = ~gap & (img < med - 5 * 1.4826 * mad)  # NOT plain <=0: unclipped sums
    mask = binary_dilation(gap | dead, structure=np.ones((5, 5)))
    ix = np.load(ROOT / "calib/ix.npy").astype(np.intp)
    iy = np.load(ROOT / "calib/iy.npy").astype(np.intp)
    status = np.load(ROOT / "calib/status_bad.npy")
    smap = np.zeros_like(mask)
    smap[ix, iy] = status
    mask |= smap
    pedestal_layer = np.load(out_dir / "pedestal_anomaly_mask.npy").astype(bool)
    pedestal_added = int((pedestal_layer & ~mask).sum())
    mask |= pedestal_layer
    center_row, center_col = _read_center(out_dir / "image_center.json")
    row, col = int(round(center_row)), int(round(center_col))
    mask[max(0, row - 22) : min(mask.shape[0], row + 14), max(0, col + 65) :] = (
        True  # beamstop shadow band
    )
    mask[max(0, row + 8) :, : min(mask.shape[1], col + 265)] = True
    np.save(out_dir / "mask_assembled.npy", mask)
    (out_dir / "mask_rationale.md").write_text(
        "# Baseline mask (no-llm mode)\n\nzero-mask(<=0)+5x5 dilation, status_bad map, "
        f"pedestal coherent-anomaly layer ({pedestal_added} newly masked pixels), "
        f"beamstop layers relative to center (row={center_row:.1f}, col={center_col:.1f}). "
        "Rule-based; no agent judgment.\n"
    )


def baseline_verify(out_dir: Path, iteration: int = 1) -> dict:
    """Rule-based verifier (no LLM): apply skills/verification thresholds verbatim."""
    import re

    md = (ROOT / "skills/verification/01_iq_quality.md").read_text()
    th = json.loads(re.search(r"```json\n(.*?)```", md, re.S).group(1))
    m = json.loads((out_dir / "iq_metrics.json").read_text())

    checks, fail_msgs, fail_target = [], [], None

    ring_ok, ring_meas = True, {}
    for r in m["rings"]:
        exp = r["expected_px"]
        c_min = th["ring_contrast_min"][str(exp)]
        ok = (
            r.get("found")
            and abs(r.get("delta_px", 99)) <= th["ring_delta_px_max"]
            and (r.get("contrast") or -1) >= c_min
        )
        ring_meas[str(exp)] = {"found_px": r.get("found_px"), "contrast": r.get("contrast")}
        if not ok:
            ring_ok = False
            violations = []
            if not r.get("found"):
                violations.append("not found")
            if abs(r.get("delta_px", 99)) > th["ring_delta_px_max"]:
                violations.append(
                    f"delta_px={r.get('delta_px')} exceeds ±{th['ring_delta_px_max']}"
                )
            if (r.get("contrast") or -1) < c_min:
                violations.append(f"contrast={r.get('contrast')} below {c_min}")
            fail_msgs.append(f"ring {exp}: " + ", ".join(violations))
    checks.append(
        {
            "id": "iq_rings",
            "pass": ring_ok,
            "measured": ring_meas,
            "threshold": {
                "delta_px_max": th["ring_delta_px_max"],
                "contrast_min": th["ring_contrast_min"],
            },
            "note": "all rings present with per-ring contrast"
            if ring_ok
            else "ring presence/contrast violation",
        }
    )
    if not ring_ok:
        fail_target = fail_target or "reduction"

    b = m["background"]
    bg_ok = (
        b["neg_bin_fraction"] <= th["neg_bin_fraction_max"]
        and b["rel_noise"] <= th["rel_noise_max"]
        and b["bump_max_sigma"] <= th["bump_max_sigma_max"]
    )
    checks.append(
        {
            "id": "iq_background",
            "pass": bg_ok,
            "measured": {k: b[k] for k in ("neg_bin_fraction", "rel_noise", "bump_max_sigma")},
            "threshold": {
                "neg_bin_fraction_max": th["neg_bin_fraction_max"],
                "rel_noise_max": th["rel_noise_max"],
                "bump_max_sigma_max": th["bump_max_sigma_max"],
            },
            "note": "background sane" if bg_ok else "background violation",
        }
    )
    if not bg_ok:
        fail_target = fail_target or "mask"
        fail_msgs.append(
            f"background: neg_bin_fraction={b['neg_bin_fraction']} "
            f"(max {th['neg_bin_fraction_max']}), rel_noise={b['rel_noise']} "
            f"(max {th['rel_noise_max']}), bump_max_sigma={b['bump_max_sigma']} "
            f"(max {th['bump_max_sigma_max']})"
        )

    cov = m["coverage"]
    cov_ok = (cov.get("panel_mask_fraction") or 0) <= th["panel_mask_fraction_max"] and cov[
        "valid_bins"
    ] >= th["valid_bins_min"]
    checks.append(
        {
            "id": "iq_coverage",
            "pass": cov_ok,
            "measured": {
                "panel_mask_fraction": cov.get("panel_mask_fraction"),
                "valid_bins": cov["valid_bins"],
            },
            "threshold": {
                "panel_mask_fraction_max": th["panel_mask_fraction_max"],
                "valid_bins_min": th["valid_bins_min"],
            },
            "note": "coverage in bounds" if cov_ok else "coverage violation",
        }
    )
    if not cov_ok:
        fail_target = fail_target or "mask"
        fail_msgs.append(
            f"coverage: panel_mask_fraction={cov.get('panel_mask_fraction')} "
            f"(max {th['panel_mask_fraction_max']}), valid_bins={cov['valid_bins']} "
            f"(min {th['valid_bins_min']})"
        )

    overall = "pass" if all(c["pass"] for c in checks) else "fail"
    report = {
        "iteration": iteration,
        "overall": overall,
        "criteria": checks,
        "feedback": None
        if overall == "pass"
        else {"target_phase": fail_target, "message": "; ".join(fail_msgs)},
    }
    (out_dir / "verify_report.json").write_text(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------- main


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="run475")
    ap.add_argument(
        "--no-llm",
        action="store_true",
        help="rule-based decisions + mask + verify; tests pipeline mechanics only",
    )
    ap.add_argument(
        "--phase",
        choices=["all", "reduction", "center", "mask", "verify"],
        default="all",
        help="resume a single phase (e.g. after a gateway drop)",
    )
    ap.add_argument(
        "--image-center",
        nargs=2,
        type=float,
        metavar=("ROW", "COL"),
        help="center already specified in assembled row/col coordinates; "
        "the optional estimation path is then skipped",
    )
    ap.add_argument(
        "--max-iters",
        type=int,
        default=1,
        help="verify-feedback loop: on FAIL, re-run the phase named by the "
        "verifier's feedback, up to N iterations (1 = single pass)",
    )
    args = ap.parse_args()

    _load_dotenv()
    out_dir = OUTPUTS / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (ROOT / "npy/shot_table.npz").exists():
        print("run step 0 first: python3 pipeline/step0_xtc_to_npy.py", file=sys.stderr)
        return 1

    import shutil
    import subprocess

    def run_py(script: str, *a: str) -> None:
        subprocess.run([sys.executable, str(ROOT / script), *a], check=True)

    def archive_iter(it: int) -> None:
        h = out_dir / "history" / f"iter_{it:02d}"
        h.mkdir(parents=True, exist_ok=True)
        for f in (
            "decisions.json",
            "accumulate_log.json",
            "mask_rationale.md",
            "_mask_layer_counts.json",
            "mask_log.json",
            "iq_metrics.json",
            "verify_report.json",
            "iq.png",
            "masked_sum.png",
            "pedestal.png",
            "pedestal_components.json",
            "center_context.json",
            "image_center.json",
            "image_center_overlay.png",
            "ring_validation.json",
            "ring_validation_overlay.png",
            "center_human_review.md",
            "center_loop.jsonl",
        ):
            p = out_dir / f
            if p.exists():
                shutil.copy2(p, h / f)

    feedback: dict | None = None
    report: dict | None = None
    for it in range(1, max(1, args.max_iters) + 1):
        target = (feedback or {}).get("target_phase")  # None on iteration 1
        do_reduction = args.phase in ("all", "reduction") and (it == 1 or target == "reduction")
        do_center = (args.phase in ("all", "center") or args.image_center is not None) and (
            it == 1 or target == "reduction"
        )
        do_mask = args.phase in ("all", "mask") and (it == 1 or target in ("reduction", "mask"))
        do_verify = args.phase in ("all", "verify")

        if args.no_llm:
            if do_reduction:
                baseline_decisions(out_dir)
                run_py("pipeline/step2_accumulate.py", "--out-dir", str(out_dir))
            if do_center:
                if not (out_dir / "sum_assembled.npy").exists():
                    print(
                        "[fatal] no sum_assembled.npy — center calibration needs reduction",
                        file=sys.stderr,
                    )
                    return 2
                context = write_center_context(out_dir, args.image_center)
                baseline_center(out_dir, context, run_py)
                stop_code = center_stop_code(out_dir)
                if stop_code is not None:
                    return stop_code
            if do_mask:
                if not (out_dir / "image_center.json").exists():
                    print(
                        "[fatal] no image_center.json — run the center phase first", file=sys.stderr
                    )
                    return 2
                stop_code = center_stop_code(out_dir)
                if stop_code is not None:
                    return stop_code
                run_py("pipeline/step2b_pedestal_diagnostics.py", "--out-dir", str(out_dir))
                baseline_mask(out_dir)
                run_py("pipeline/step3_apply_mask.py", "--out-dir", str(out_dir))
            if do_verify:
                if not (out_dir / "image_center.json").exists():
                    print(
                        "[fatal] no image_center.json — run the center phase first", file=sys.stderr
                    )
                    return 2
                stop_code = center_stop_code(out_dir)
                if stop_code is not None:
                    return stop_code
                run_py("pipeline/step4_iq.py", "--out-dir", str(out_dir))
                report = baseline_verify(out_dir, iteration=it)
        else:
            if do_reduction:
                fb = feedback if target == "reduction" else None
                await _run_phase("reduction", reduction_prompt(out_dir, fb), out_dir)
                if not (out_dir / "sum_assembled.npy").exists():
                    print(
                        "[fatal] reduction phase did not produce sum_assembled.npy", file=sys.stderr
                    )
                    return 2
            if do_center:
                if not (out_dir / "sum_assembled.npy").exists():
                    print(
                        "[fatal] no sum_assembled.npy — center calibration needs reduction",
                        file=sys.stderr,
                    )
                    return 2
                write_center_context(out_dir, args.image_center)
                await _run_phase("center", center_prompt(out_dir), out_dir)
                cp = out_dir / "image_center.json"
                if not cp.exists():
                    print(
                        "[fatal] center submodule did not produce image_center.json",
                        file=sys.stderr,
                    )
                    return 2
                try:
                    stop_code = center_stop_code(out_dir)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    print(f"[fatal] invalid image_center.json: {exc}", file=sys.stderr)
                    return 2
                if stop_code is not None:
                    return stop_code
            if do_mask:
                if not (out_dir / "sum_assembled.npy").exists():
                    print(
                        "[fatal] no sum_assembled.npy — run the reduction phase first",
                        file=sys.stderr,
                    )
                    return 2
                if not (out_dir / "image_center.json").exists():
                    print(
                        "[fatal] no image_center.json — run the center phase first", file=sys.stderr
                    )
                    return 2
                stop_code = center_stop_code(out_dir)
                if stop_code is not None:
                    return stop_code
                run_py("pipeline/step2b_pedestal_diagnostics.py", "--out-dir", str(out_dir))
                fb = feedback if target == "mask" else None
                await _run_phase("mask", mask_prompt(out_dir, fb), out_dir)
                if not (out_dir / "masked_sum.npy").exists():
                    print("[fatal] mask phase did not produce masked_sum.npy", file=sys.stderr)
                    return 3
            if do_verify:
                if not (out_dir / "masked_sum.npy").exists():
                    print("[fatal] no masked_sum.npy — nothing to verify", file=sys.stderr)
                    return 3
                if not (out_dir / "image_center.json").exists():
                    print(
                        "[fatal] no image_center.json — run the center phase first", file=sys.stderr
                    )
                    return 2
                stop_code = center_stop_code(out_dir)
                if stop_code is not None:
                    return stop_code
                await _run_phase("verify", verify_prompt(out_dir, it), out_dir)
                rp = out_dir / "verify_report.json"
                if not rp.exists():
                    print(
                        "[fatal] verify phase did not produce verify_report.json", file=sys.stderr
                    )
                    return 4
                report = json.loads(rp.read_text())

        if report is None:  # no verify phase in this invocation
            break
        print(f"[verify] iteration {it}: {report['overall'].upper()}", file=sys.stderr)
        archive_iter(it)
        if report["overall"] == "pass":
            break
        feedback = dict(report.get("feedback") or {})
        if feedback.get("target_phase") not in ("reduction", "mask"):
            feedback["target_phase"] = "mask"
        feedback["iteration"] = it
        if it == args.max_iters:
            print(f"[loop] still FAILING after {it} iteration(s)", file=sys.stderr)

    try:  # provenance: which skills were consulted
        from agent.skill_usage import generate as _skill_usage

        u = _skill_usage(out_dir)
        print(
            f"[skills] {u['n_distinct_skills']} distinct skill files read "
            f"({u['n_skill_reads']} reads) -> {out_dir}/skill_usage.md",
            file=sys.stderr,
        )
    except Exception as exc:  # reporting must never fail a run
        print(f"[skills] usage report skipped: {exc}", file=sys.stderr)

    if (out_dir / "image_center.json").exists():
        row, col = _read_center(out_dir / "image_center.json")
        print(f"[center] row={row:.3f}, col={col:.3f} -> {out_dir}/image_center.json")
    if (out_dir / "masked_sum.png").exists():
        print(f"[endpoint] {out_dir}/masked_sum.png")
    if report is not None:
        print(f"[verdict] {report['overall']}")
        return 0 if report["overall"] == "pass" else 5
    return 0


def cli() -> None:
    """Synchronous console entry point for the asynchronous orchestrator."""
    raise SystemExit(asyncio.run(main()))


if __name__ == "__main__":
    cli()
