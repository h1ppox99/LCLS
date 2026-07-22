---
name: xray-normalization
description: Normalization skill category — per-shot flux normalization as ONE operation (offset-correct, divide each shot by its own reference) with a MENU of reference parameters, one file per reference class - 01 ipm2 (the default verdict monitor), 02 upstream machine monitors (GMD gas detectors, SB1 BMMON), 03 in-hutch alternatives (diode boxes, SB3 BMMON), 04 the detector itself (self-normalization, last resort). The agent picks the reference by evidence (correlation + bin-CV test + channel health), records the choice, and step 2 executes any shot_table column via decisions.json "monitor".
---

# Normalization (category index)

**Goal.** Put shots on a common scale: offset-correct a per-shot reference, divide
each shot by it (per-shot division or ratio-of-sums — see
[01](01_per_shot_flux_ipm2.md)). Normalization *rescales*; it never removes pixels
(that is [masking](../masking/README.md)) and never removes shots (that is
[selection](../selection/README.md)).

**One operation, a menu of reference parameters.** Each file covers one reference
class — what the device measures, where it sits in the beam path, its Run0475
evidence, and when to pick it:

| File | Reference class | Position in beam path | Role |
|---|---|---|---|
| [01_per_shot_flux_ipm2.md](01_per_shot_flux_ipm2.md) | `ipm2` — SB2 BMMON wave8 | last monitor before the sample optics | **default verdict monitor**; also carries the run/skip decision and the bin-CV evidence test |
| [02_per_shot_flux_upstream.md](02_per_shot_flux_upstream.md) | GMD gas detectors (`gasdet_*`), SB1 BMMON (`bmmon43_sum`) | machine exit, before all beamline optics | anchor / salvage — never verdict while an in-hutch monitor lives |
| [03_per_shot_flux_alternatives.md](03_per_shot_flux_alternatives.md) | diode boxes (`ipmfex22/23/26/28_*`), SB3 BMMON (`bmmon4c_sum`) | in-hutch, around the sample | **cross-check** + fallback verdict; carries the channel-health screen |
| [04_per_shot_flux_self.md](04_per_shot_flux_self.md) | the detector itself (`det_total`, band integrals) | the data | last resort — shape-only analyses; never pump–probe |

## The comparison table (Run0475, 1 781 plateau-kept shots, 8 chronological bins)

Unnormalized chronological bin-CV: **9.33 %**.

| Reference (column) | r(det_total) | bin-CV after norm | Health / note |
|---|---|---|---|
| `ipm2` | 0.895 | **3.42 %** | clean; offset +4.5 — the verdict |
| `ipmfex22_sum` | 0.909 | **3.41 %** | dead ch1 harmless — the cross-check |
| `ipmfex22_ch0+ch2` | 0.912 | 3.54 % | hand-built healthy pair |
| `bmmon4c_sum` | 0.639 | 3.94 % | positions opaque, sum usable |
| `ipmfex28_sum` | 0.905 | 4.06 % | fully healthy box |
| `ipmfex26_ch0+ch3` | 0.909 | 4.09 % | ch1 must be excluded |
| `ipmfex23_sum` | 0.562 | 4.37 % | near saturation — avoid |
| `ipmfex26_sum` (naive) | 0.606 | 5.89 % | saturated-ch trap, measured |
| `gasdet_f11` | 0.504 | 6.53 % | upstream — anchor only |
| `bmmon43_sum` | 0.505 | 6.93 % | upstream — anchor only |
| `det_total` (self) | 1 by construction | 0 by construction | the perfect score IS the warning |

## Choosing (agent decides, records the choice)

- **Default verdict: `ipm2` (01) while it passes its health look** — every documented
  threshold (low cut 200 / plateau 3000, offsets, baseline conventions) is calibrated
  on it.
- **Always designate one cross-check from 03** (different electronics) and record the
  ipm2/alt ratio drift (Run0475: 5.3 % p2p — the monitor-systematic floor).
- **Verdict monitor fails its screen** → promote the best healthy alternative from 03,
  re-derive the low-cut/plateau against the new column, flag the run.
- **All in-hutch monitors dead** → upstream reference (02), accepting that only ~⅓ of
  the drift is removable.
- **No monitor at all, shape-only science** → self-normalization (04), with its
  contamination caveats.

## Golden rules

1. **Normalize per-shot before summing** ([01](01_per_shot_flux_ipm2.md)) — summing
   then dividing once is only valid when the incident flux is constant.
2. **Health-check before dividing** ([03](03_per_shot_flux_alternatives.md)) — a dead
   or saturated channel inside a box sum silently poisons every weight.
3. **[Low-ipm exclusion](../selection/01a_low_ipm_exclusion.md) runs first, against the
   chosen reference** — near-zero readings are noise, often negative; a divide guard
   does not catch −0.0002, and a cut defined on ipm2 does not protect a different column.
4. **A reference downstream of the sample divides out the physics** — upstream-of-sample
   is a hard requirement for pump–probe ([01](01_per_shot_flux_ipm2.md) §monitor choice,
   [04](04_per_shot_flux_self.md) for the extreme case).

**Data prerequisite.** References in 02–04 need the extended shot_table:

```bash
python3 pipeline/step0b_extend_shot_table.py   # ~3 s, npy/shot_table.npz 10 -> 71 columns
```

Column documentation: `npy/README_npy.md` ("Extended monitor columns"). Execution:
`decisions.json → normalization.monitor` accepts any shot_table column;
`step2_accumulate.py` derives that column's own x-ray-off offset automatically.
