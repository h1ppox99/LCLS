# xppl1016922 — experiment data

A local mirror of LCLS experiment `xppl1016922` (XPP instrument, SLAC) and a
description of what a single shot records. The large data directories are
gitignored and may be absent in a clone.

## Repository contents

| path | contents |
| --- | --- |
| `xtc/` | Raw XTC streams for psana (runs below). |
| `calib/` | Detector calibration constants; macOS-recovered colons stored as U+F022. |
| `automask/data/masks/` | Hand-drawn reference masks — the only frozen input psana cannot recompute. |
| `automask/cache/` | Regenerable image + profile cache built from XTC (gitignored). |

## Runs present locally

| run | streams on disk | state |
| --- | --- | --- |
| 475 | s00–s04 (all 5) | complete — the reference run, 3 201 events |
| 389 | s03 only | truncated; open by explicit path, not the run resolver |
| 378 | s00 only | partial |
| 396 | s00 only | partial |

Run 389's single truncated stream yields no complete sum or run-specific
reference mask. See [PSANA_XTC.md](PSANA_XTC.md) for opening partial runs.

---

## Beam branches: CC and VCC

The beam is one x-ray source split into two branches, CC and VCC; there is no
pump laser. The per-shot branch state is carried by two analog voltage lines, not
an event code. Source `BldInfo(XPP-AIN-01)` (`Bld.BldDataAnalogInputV1`), 16
channels of `channelVoltages()` on every shot, mirrored in small data as
`ai/ch00 … ai/ch15`.

| channel | small data | branch | open |
|---|---|---|---|
| `ch02` | `ai/ch02` | CC | `> 2 V` |
| `ch03` | `ai/ch03` | VCC | `> 2 V` |

The lines are binary — ~0.045 V low, ~5.05 V high, nothing between (run 389:
3 596 low, 36 407 high, 0 elsewhere); any threshold in 0.5–4.5 V is equivalent.
The other 14 channels are unlabelled analog noise near 0 V.

The EVR config carries stock XPP labels that do not describe this experiment:
`EvrData.ConfigV7` names code 90 `'Laser on'`, 91 `'Laser off'`, 137 `'Beam On'`,
94 `'PP'`, 95 `'Readout Seq'`. Codes 90/91 fire on a fixed 2/3–1/3 pattern
regardless of branch and are not a pump-probe state.

Observed states:

| | run 389 (40 003 ev) | run 475 (3 201 ev) |
|---|---|---|
| `ch02` (CC) | 5.049 V, 100 % open | 5.049 V, 100 % open |
| `ch03` (VCC) | bimodal, 91.0 % open, 20 transitions | 0.045 V, 0 % open |

CC is stuck open in both local runs, so all branch information rides on VCC.
Run 389's 20 VCC transitions are 10 contiguous closed blocks of ~360 events every
~3 600 — a periodic chop, not per-shot alternation. The two lab datasets are
defined by VCC alone (`vcc < 2` vs `vcc > 2`).

Monitor means over VCC-open vs VCC-closed shots (run 389) place each monitor on
one side of the split:

| monitor | ratio | side |
|---|---|---|
| `gas_detector/f_11_ENRC` | 0.990 | upstream |
| `ipm_hx2/sum` (HX2-SB1) | 0.989 | upstream |
| `lomdiode/sum` (XppMon_Pim1) | 0.983 | upstream |
| `ipm2/sum` (XPP-SB2) | 0.978 | upstream |
| `diode2/sum` (XppSb3_Pim) | 0.978 | upstream |
| `ipm3/sum` (XPP-SB3) | 0.971 | upstream |
| `lombpm/sum` (XppMon_Pim0) | 2.412 | downstream |
| `diodeU/sum` (XppEnds_Ipm0) | 1.668 | downstream |
| Jungfrau `azav` sum | 2.039 | downstream |

Only `diodeU` and `lombpm` track the branch flux the Jungfrau sees; `ipm2` is
upstream and does not.

---

## Per-shot quantities

All are per-shot and present in both XTC and small data unless noted.

**Beam monitors (BMMON)** — `Bld.BldDataBeamMonitorV1`: `TotalIntensity()`,
`X_Position()`, `Y_Position()`, `peakA[16]`, `peakT[16]`; small data stores `sum`,
`xpos`, `ypos`, `peaks[16]`.

| small data | source | position |
|---|---|---|
| `ipm_hx2/` | `HX2-SB1-BMMON` | upstream |
| `ipm2/` | `XPP-SB2-BMMON` | upstream |
| `ipm3/` | `XPP-SB3-BMMON` | upstream |

**Diodes / IPMs** — `Lusi.IpmFexV1` (`channel[4]`, `sum`, `xpos`, `ypos`) with a
raw `Ipimb.DataV2` companion (`channelN`, `channelNVolts`, `channelNps`,
`channelNpsVolts`, `triggerCounter`, `config0..2`).

| small data | source | position |
|---|---|---|
| `diodeU/` | `XppEnds_Ipm0` | endstation, downstream |
| `diode2/` | `XppSb3_Pim` | upstream |
| `lomdiode/` | `XppMon_Pim1` | LODCM |
| `lombpm/` | `XppMon_Pim0` | downstream |

Channels the lab uses :

| name | field | run 389 median | run 475 median |
|---|---|---|---|
| `d1` | `diode2/channels[:, 0]` | 0.0539 | 0.0056 |
| `d5` | `diodeU/channels[:, 3]` | 0.0526 | 0.0000 |
| `d6` | `lombpm/channels[:, 2]` | 0.2065 | 0.0077 |
| `sample_diode` | `diodeU/channels[:, 0]` | 0.0178 | 0.0019 |
| `ipm2` | `ipm2/sum` | — | — |

`sample_diode` is the normalizer (the pipeline divides each assembled image by it,
then clips negatives). `d1` is a quality gate, kept only when `0.4 ≤ d1 ≤ 0.45`;
that band is run-specific — it keeps 179 events (0.45 %) of run 389 and 0 of run
475, whose `d1` never exceeds 0.157.

**FEE gas detector** — `Bld.BldDataFEEGasDetEnergy`: `f_11/12/21/22/63/64_ENRC`,
pulse energy in mJ, upstream of everything.

**Detector pixel sum** — not in small data per shot. Available by decoding XTC
frames (`iter_calibrated`), or as `jungfrau1M_alcove/azav_azav` (3201, 1, 123),
the per-shot azimuthal average (correlates 0.879 with `ipm2` on run 475);
`Sums/jungfrau1M_alcove_calib` is run-level only.

**FEE spectrometer** — `FEE-SPEC0` / `feeBld/hproj` (2048-bin) is present on only
77.6 % of events with large negative per-shot sums (median ≈ −3.6e7); unusable
without calibration.

**EPICS** — 383 PVs in `EpicsArch.0:NoDevice.0`, per event via
`ds.env().epicsStore()`. No PV name contains "CC"/"VCC" (only `ccm_*`, the
channel-cut monochromator, and `vac_Gcc_*` vacuum gauges); the branch state exists
only as `ai/ch02`/`ai/ch03`. Small data saves 43 PVs per event and 383 once per
run under `epicsOnce/`. Delay is `epicsAll/delay`, converted
`delay_ps = 0.9376321852982434 * (delay − 3.1)`; both local runs are single-delay
(389 at 2.999 ps, 475 at 4.998 ps).

---

## psana access notes

- `damage/<det> == 1` means the data is **present**, not damaged; absent detectors
  read 0 (`evr1`, `zyla_sb5`, `ControlData`, `scan`).
- `psana.Detector('XPP-AIN-01').get(evt)` returns `None` (the `DdlDetector`
  wrapper cannot handle this type). Use the raw accessor
  `evt.get(psana.Bld.BldDataAnalogInputV1, psana.Source("BldInfo(XPP-AIN-01)")).channelVoltages()`;
  the same applies to `Lusi.IpmFexV1`.
- `ipm3/sum` goes negative (run 389 mean ≈ −6 800, run 475 min ≈ −29 000); not a
  usable normalizer without baseline investigation.
- EVR `desc()` strings live in `EvrData.ConfigV7` (config store), not the
  per-event `DataV4`; they are stock XPP labels.
