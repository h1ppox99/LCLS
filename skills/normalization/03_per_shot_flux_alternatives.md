---
name: xray-normalization-per-shot-flux-alternatives
description: Per-shot flux normalization using the IN-HUTCH alternative monitors — four IPM/PIM diode boxes (ipmfex22/23/26/28 - 4 diode channels + sum + x/y each) and the second XPP BMMON (bmmon4c_sum). Same division as method 01, different reference. Three of them tie ipm2 on Run0475 (r ~= 0.91, bin-CV 3.4-4.1% vs ipm2's 3.42%) — true redundancy. Carries the mandatory channel-health screen (dead ch, saturated ch that poisons a box sum, pinned positions) and the cross-check protocol (ratio drift vs ipm2). Choose as cross-check or fallback verdict; keep ipm2 as verdict while healthy.
---

# Normalization · Method 3 — Per-shot flux via in-hutch alternative monitors

**Same operation as [method 01](01_per_shot_flux_ipm2.md), different reference.**
Part of the [monitor menu](README.md). Columns from `pipeline/step0b_extend_shot_table.py`.

## The references

| Columns | Device (per BldInfo enum — confirm vs elog) | Notes |
|---|---|---|
| `ipmfex22_{ch0..3,sum,xpos,ypos}` | XppMonPim0 diode box | ch1 dead (always 0) |
| `ipmfex23_*` | XppMonPim1 diode box | near saturation; x/ypos pinned ≈ 1 |
| `ipmfex26_*` | XppSb3Pim diode box | **ch1 saturated (~1.21 const)** |
| `ipmfex28_*` | XppEnds_Ipm0 diode box (hutch/user) | all 4 channels healthy |
| `bmmon4c_{sum,xpos,ypos}` | second XPP BMMON wave8 (likely SB3) | positions decode suspect — treat as opaque |

Different electronics from ipm2 (diode boxes vs wave8) — which is exactly what makes
them useful: shared-mode failures become detectable.

## Mandatory channel-health screen (before ever dividing)

1. **Dead channel** — always reads 0 (`ipmfex22_ch1`): harmless inside a sum, but know it.
2. **Saturated channel** — pinned near a constant (`ipmfex26_ch1` ≈ 1.21): **poisons the
   box sum** — it drags `ipmfex26_sum` from r 0.91 (healthy channels) to 0.61. Use the
   healthy channels (`ipmfex26_ch0`, `ipmfex26_ch3`) individually instead.
3. **Pinned positions** — x/ypos stuck at ±1 (`ipmfex23`): the fex is not computing;
   distrust the whole box.

Never use a box's `sum` without this look. This is the single most damaging trap in
the monitor menu.

## Evidence on Run0475 (1 781 plateau-kept shots, 8 chronological bins)

Unnormalized **9.33 %**; ipm2 (method 01) **3.42 %**.

| Reference | r(det_total) | bin-CV after norm | Health |
|---|---|---|---|
| `ipmfex22_sum` | 0.909 | **3.41 %** | dead ch1 harmless — best alternative |
| `ipmfex22_ch0`+`ch2` | 0.912 | 3.54 % | hand-built from healthy channels |
| `bmmon4c_sum` | 0.639 | 3.94 % | sum usable, positions opaque |
| `ipmfex28_sum` | 0.905 | 4.06 % | fully healthy box |
| `ipmfex26_ch0`+`ch3` | 0.909 | 4.09 % | must exclude ch1 |
| `ipmfex23_sum` | 0.562 | 4.37 % | near saturation — avoid |
| `ipmfex26_sum` (naive) | 0.606 | 5.89 % | **the saturated-channel trap, measured** |

## The cross-check protocol (this reference's main job)

Designate ONE healthy alternative (Run0475: `ipmfex22_sum`) as the standing
cross-check of the verdict monitor and record, per run:

- **per-shot ratio CV** of ipm2/alt (Run0475: 17.7 %) and its **chronological-bin
  drift** (Run0475: 5.3 % p2p) — two *good* monitors disagree at the few-percent
  level, which bounds the monitor-systematic floor at ~2 % per device;
- **alarm rule**: a step or trend ≫ the documented p2p, or the verdict's r(det)
  collapsing while the alternative's holds, or coverage dropping → declare the
  verdict monitor broken, promote the alternative, flag the run.

Dual-monitor form (geometric mean of ipm2 and ipmfex22): bin-CV 3.29 % vs 3.41–3.42 %
single — real but marginal; use only if the provenance complexity is worth ~0.1 %.

## When to choose this reference

- **Cross-check** (always, it is nearly free) — the protocol above.
- **Fallback verdict** when ipm2 fails its health screen: `ipmfex22_sum` ties it
  statistically.
- **Not** as a routine replacement while ipm2 is healthy: every documented threshold
  (low cut 200 / plateau 3000, offsets, baseline conventions) is calibrated on ipm2 —
  switching re-defines them for no measurable gain. If you do switch, re-derive the
  low-cut/plateau against the new column and sanity-check the resulting weight range
  (a per-shot weight of 5.8 means the cut and the monitor no longer match).

## Execution

```json
"normalization": { "run": true, "monitor": "ipmfex22_sum", "form": "per_shot",
                   "offset": "auto", "rationale": "..." }
```

Executable today — `step2_accumulate.py` resolves any shot_table column and its own
x-ray-off offset (smoke-tested: monitor_offset 8.8e-4, 1 781 shots, weights finite).
