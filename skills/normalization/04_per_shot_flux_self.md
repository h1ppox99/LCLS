---
name: xray-normalization-per-shot-flux-self
description: Self-normalization — dividing each shot by the DETECTOR'S OWN summed intensity (det_total, or a signal-free q-band integral) instead of an independent monitor. The last-resort reference - it removes ALL multiplicative per-shot variation including the physics you may be trying to measure, so it is legitimate only for pure line-shape studies or monitor-dead salvage, and NEVER for pump-probe or any intensity observable. On Run0475 the raw det_total column is additionally contaminated (99/1781 plateau-kept shots have det_total <= 0 from stuck/overflow pixels) — a usable self-reference needs a cleaned band integral, not the raw column.
category: normalization
role: reference — the detector itself (self-normalization), last resort
gate: monitor-dead salvage or shape-only analyses; never pump–probe or intensity observables
status: wired
---

# Normalization · 04 — Self-normalization by the detector's own intensity

**Same operation as [method 01](01_per_shot_flux_ipm2.md), but the reference is the
data itself.** **Last resort.**

## Principle

### The reference

| Column / quantity | What it is |
|---|---|
| `det_total` | summed calibrated keV over good pixels, computed at step 0 |
| a signal-free q-band integral | sum over a masked, ring-free radial band (compute yourself from the frames) — the refined variant |

### The trap built into it

Dividing by the detector's own total forces every shot to the same integrated
intensity. That removes flux drift *by construction* — and with it **every real
per-shot intensity change**: pump-induced signal, structure-factor evolution,
damage, anything. The bin-CV test from method 01 is meaningless here (the ratio
det/det ≡ 1, CV → 0 artificially); a perfect score is the warning, not the
validation. Self-normalization converts the data from "scattering per incident
photon" to "scattering shape only".

## Evidence (Run0475)

Why even the last resort needs care:

- Monitors exist and are healthy → the decision here would be **SKIP**.
- The raw column is contaminated: **99 / 1 781 plateau-kept shots have
  `det_total` ≤ 0** (min −1.09×10⁵ vs median +1.67×10⁵) — stuck/gain-switch overflow
  pixels leak through the status mask into the step-0 sum. Dividing by raw
  `det_total` would silently drop those 99 shots at the step-2 guard, and the
  surviving totals still carry the contamination (per-shot CV 82 %).
- A usable self-reference must therefore be a **cleaned band integral**: apply the
  mask first, integrate a ring-free band (e.g. the background windows
  150–330 / 410–500 px used by the verifier), and divide by that. This is closer to
  a true flux proxy because the band is signal-free — but it requires the mask
  phase's output, i.e. a second pass.

## When to use

- **Monitor-dead salvage**: no monitor column is alive and the science survives on
  relative line *shape* (peak positions, widths, ratios of features).
- **Static, isotropic samples** where only the normalized S(q) shape is reported.
- As a **diagnostic ratio**: det_total/monitor per shot is the residual series that
  methods 01/03 use for their health checks — self-normalization's honest use.

## Do not

- **Pump–probe / any intensity observable** — it divides out the signal itself.
- Any analysis comparing absolute or relative *levels* between shot groups.
- Runs where the sample's total scattering genuinely varies (concentration series,
  damage progression).

## Outputs

Executed via `decisions.json`:

```json
"normalization": { "run": true, "monitor": "det_total", "form": "ratio_of_sums",
                   "offset": 0, "rationale": "salvage: all monitors dead; shape-only analysis" }
```

Mechanically executable today (`det_total` is a shot_table column; the step-2 guard
drops non-positive values — record the 99-shot loss). The band-integral variant needs
a small agent-written script at mask time; record the band and the mask version used.

## Links

Part of: the [monitor menu](README.md). Same operation as
[01](01_per_shot_flux_ipm2.md). The honest use of det_total/monitor ratios is the
health checks in [01](01_per_shot_flux_ipm2.md) and
[03](03_per_shot_flux_alternatives.md). Band integral depends on the
[masking](../masking/README.md) phase's output.
