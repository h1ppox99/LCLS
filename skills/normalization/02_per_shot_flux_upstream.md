---
name: xray-normalization-per-shot-flux-upstream
description: Per-shot flux normalization using the UPSTREAM machine-level monitors — the FEE gas detectors (GMD, gasdet_f11/f12 before and f21/f22 after the gas attenuator, in mJ) or the SB1 BMMON (bmmon43_sum). Same division operation as method 01, different reference. On Run0475 they only remove ~1/3 of the drift (bin-CV 9.33% -> 6.5-6.9% vs 3.4% for ipm2) because spectral jitter through the mono/attenuators decorrelates them from the sample (r ~= 0.69 vs ipm2). Choose them as anchors or when every in-hutch monitor is dead — never as the verdict monitor while one is alive.
---

# Normalization · Method 2 — Per-shot flux via upstream machine monitors

**Same operation as [method 01](01_per_shot_flux_ipm2.md)** — offset-correct, divide each
shot by its own reference — **different reference parameter.** Part of the
[monitor menu](README.md). Columns from `pipeline/step0b_extend_shot_table.py`.

## The reference

| Column(s) | Device | Where it sits |
|---|---|---|
| `gasdet_f11`, `gasdet_f12` | FEE gas detector (GMD), PMT pair 1 [mJ] | machine exit, **before** the gas attenuator, before all beamline optics |
| `gasdet_f21`, `gasdet_f22` | GMD PMT pair 2 [mJ] | **after** the gas attenuator |
| `gasdet_f63`, `gasdet_f64` | auxiliary GMD channels | low-signal range |
| `bmmon43_sum` (+xpos/ypos) | SB1 BMMON wave8 | HX2, after FEE, before the XPP optics |

These measure **what the machine emitted**, not what reached the sample: between them
and the detector sit the gas/solid attenuators, the mono, and the slits — and the
SASE spectrum jitters shot-to-shot, so the transmitted fraction varies even at
constant pulse energy.

## Evidence on Run0475 (1 781 plateau-kept shots, 8 chronological bins)

Unnormalized bin-CV **9.33 %**; ipm2 reference (method 01) reaches **3.42 %**.

| Reference | r(det_total) | r(ipm2) | bin-CV after norm |
|---|---|---|---|
| `gasdet_f11` | 0.504 | 0.69 | 6.53 % |
| mean of f11/f12/f21/f22 | 0.505 | 0.69 | 6.57 % — averaging the four gains nothing |
| `bmmon43_sum` | 0.505 | 0.69 | 6.93 % |

All offsets on x-ray-off shots are ≈ 0 (subtract the x-ray-off median regardless).
Coverage 99.9 %. The three behave identically because they share the same limitation:
they cannot see what the optics did to *this* shot's spectrum.

## When to choose this reference

- **Every in-hutch monitor is dead or absent** (salvage mode): 6.5 % beats 9.3 %.
- **Gross-failure anchor**: a run where det_total correlates with the GMD but not with
  ipm2 localizes the fault to the in-hutch monitor, not the beam.
- **Cross-run sanity**: machine-level pulse energy is comparable across runs and hutches.

**Never the verdict monitor while an in-hutch monitor is alive** — you would keep
3 percentage points of removable drift.

## Trade-offs & extras

- Decorrelation is physics, not noise — no amount of averaging PMTs fixes it (measured).
- `f21/f11` tracks the **gas-attenuator transmission** per shot — a useful cross-run
  diagnostic even when this reference is not used for division.
- If dividing here, re-derive the low-cut against THIS column (the 01a threshold and
  plateau are defined on ipm2, not on mJ).

## Execution

```json
"normalization": { "run": true, "monitor": "gasdet_f11", "form": "ratio_of_sums",
                   "offset": "auto", "rationale": "..." }
```

`step2_accumulate.py` accepts any shot_table column as `monitor` and derives that
column's own x-ray-off offset (`monitor_offset: "auto"`).
