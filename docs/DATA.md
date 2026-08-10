# Per-shot data in the xppl1016922 XTC

What is actually recorded for a single shot, where it comes from, and what the
values mean. Everything below was verified by reading the local run 475 XTC
(all 5 streams) and the run 389 / 475 small-data files — no values are quoted
from documentation or assumed from convention.

Companion docs: `docs/DATA_OVERVIEW.md` (data layout), `docs/PSANA_XTC.md` (how to
open the files).

---

## 1. There is no laser. There are two x-ray branches: CC and VCC.

This experiment has **one x-ray beam split into two branches**. There is no pump
laser. Which branch is on for a given shot is recorded as **two analog voltage
lines** that are thresholded, not as an event code.

The EVR config in the XTC *does* carry laser-sounding labels — `EvrData.ConfigV7`
names code 90 `'Laser on'`, code 91 `'Laser off'`, code 137 `'Beam On'`, code 94
`'PP'`, code 95 `'Readout Seq'`. **These labels are the stock XPP timing
configuration and do not describe this experiment.** Code 90/91 fire on a fixed
2/3–1/3 pattern in both local runs regardless of branch state. Do not read them
as a pump-probe state.

### The voltage lines

Source `BldInfo(XPP-AIN-01)`, type `Bld.BldDataAnalogInputV1`, 16 channels of
`channelVoltages()`, present on every shot. Small data mirrors it as
`ai/ch00 … ai/ch15`.

| channel | small data | meaning | threshold |
|---|---|---|---|
| `ch02` | `ai/ch02` | **CC** shutter monitor | `> 2` = open |
| `ch03` | `ai/ch03` | **VCC** shutter monitor | `> 2` = open |

Confirmed against the lab's production code, `xpp_sharing/utils.py:176-177`:

```python
cc  = np.array(f['ai/ch02'])   # CC monitor; large value typically indicates open
vcc = np.array(f['ai/ch03'])   # VCC monitor
```

The lines are cleanly binary — ~0.045 V low, ~5.05 V high, with nothing in
between (run 389: 3 596 counts in the bottom histogram bin, 36 407 in the top, 0
elsewhere). The threshold value is not delicate; anything in 0.5–4.5 V gives
identical results. The lab uses 2; a notebook comment says "vcc > 4 means open
shutter". Both agree on every event in both local runs.

The other 14 analog channels are unlabeled and sit at analog noise near 0 V
(|mean| < 0.05 V, no bimodality) in both runs.

### Observed branch states

| | run 389 (40 003 ev) | run 475 (3 201 ev) |
|---|---|---|
| `ch02` (CC) | 5.049 V constant, **100 % open** | 5.049 V constant, **100 % open** |
| `ch03` (VCC) | bimodal, **91.0 % open**, 20 transitions | 0.045 V constant, **0 % open** |

VCC's 20 transitions in run 389 are 10 contiguous closed blocks of ~360 events
each, recurring every ~3 600 events — a periodic chopping pattern, not per-shot
alternation.

**Caveat for both local runs: CC is stuck open.** All the discriminating
information rides on VCC. The lab's own selection reflects this — it reads `cc`
but never uses it, and defines both datasets from `vcc` alone
(`xpp_sharing/utils.py:250,307`):

```python
mask_excluded |= (vcc < 2)   # "VCCC" dataset
mask_excluded |= (vcc > 2)   # "CC" dataset  (complementary)
```

So as currently used these are complementary states of one line, and "both on"
is not distinguishable from "CC on" in run 389 or 475. Runs where `ch02` also
toggles would be needed to exercise the full 2-bit state.

### The split is physically visible in the monitors

Ratio of each monitor's mean between VCC-open and VCC-closed shots, run 389:

| monitor | ratio | interpretation |
|---|---|---|
| `gas_detector/f_11_ENRC` | 0.990 | upstream of split — unaffected |
| `ipm_hx2/sum` (HX2-SB1) | 0.989 | upstream — unaffected |
| `lomdiode/sum` (XppMon_Pim1) | 0.983 | unaffected |
| `ipm2/sum` (XPP-SB2) | 0.978 | **unaffected** |
| `diode2/sum` (XppSb3_Pim) | 0.978 | unaffected |
| `ipm3/sum` (XPP-SB3) | 0.971 | unaffected |
| `lombpm/sum` (XppMon_Pim0) | **2.412** | follows the branch |
| `diodeU/sum` (XppEnds_Ipm0) | **1.668** | follows the branch |
| Jungfrau `azav` sum | **2.039** | follows the branch |

**Consequence for the masking project: ipm2 sits on the wrong side of the
split.** It does not see the flux change that the Jungfrau sees, so it is a poor
per-shot normalizer here. This independently explains the earlier empirical
finding that ipm2 normalization hurt the pipeline. `diodeU` and `lombpm` track
the detector; ipm2 does not.

---

## 2. Per-shot intensity quantities

Many more than ipm2. All are per-shot and present in both the XTC and small data
unless noted.

### Beam-monitor BMMONs — `Bld.BldDataBeamMonitorV1`

`TotalIntensity()`, `X_Position()`, `Y_Position()`, `peakA[16]`, `peakT[16]`.

| small data | source | position |
|---|---|---|
| `ipm_hx2/` | `HX2-SB1-BMMON` | upstream, before the split |
| `ipm2/` | `XPP-SB2-BMMON` | before the split |
| `ipm3/` | `XPP-SB3-BMMON` | before the split |

Small data stores `sum`, `xpos`, `ypos`, `peaks[16]`.

### Diodes / IPMs — `Lusi.IpmFexV1` + raw `Ipimb.DataV2`

Each carries `channel[4]`, `sum`, `xpos`, `ypos`; the `Ipimb.DataV2` companion
holds the raw `channelN` counts, `channelNVolts`, `channelNps`, `channelNpsVolts`,
plus `triggerCounter` and `config0..2`.

| small data | source | notes |
|---|---|---|
| `diodeU/` | `XppEnds_Ipm0` | endstation — **after the split** |
| `diode2/` | `XppSb3_Pim` | before the split |
| `lomdiode/` | `XppMon_Pim1` | at the LODCM |
| `lombpm/` | `XppMon_Pim0` | **after the split** |

### The channels the lab actually uses

From `get_summary` (`xpp_sharing/utils.py:180-185`):

| name | field | run 389 median | run 475 median |
|---|---|---|---|
| `d1` | `diode2/channels[:, 0]` | 0.0539 | 0.0056 |
| `d5` | `diodeU/channels[:, 3]` | 0.0526 | 0.0000 |
| `d6` | `lombpm/channels[:, 2]` | 0.2065 | 0.0077 |
| `sample_diode` | `diodeU/channels[:, 0]` | 0.0178 | 0.0019 |
| `ipm2` | `ipm2/sum` | — | — |

`sample_diode` is the **normalizer** — the production pipeline divides each
assembled image by it, then clips negatives (`utils.py:271-275`). `d1` is used as
a *quality gate*, not a normalizer: events are kept only when `0.4 <= d1 <= 0.45`.

**That d1 band is run-specific and does not transfer.** It keeps 0.45 % of run
389 (179 events) and **0 events of run 475**, whose d1 never exceeds 0.157. Any
run-agnostic selection must re-derive this band per run rather than hardcode it.

### Upstream absolute energy — `Bld.BldDataFEEGasDetEnergy`

`f_11_ENRC`, `f_12_ENRC`, `f_21_ENRC`, `f_22_ENRC`, `f_63_ENRC`, `f_64_ENRC` —
FEE gas detector pulse energy in mJ, upstream of everything.

### Detector pixel sum

The sum of all Jungfrau pixels is the most direct measure of what the detector
received, but it is **not in small data per shot**. Options:

- decode the XTC frames (`iter_calibrated` already does this — accumulating a sum
  there is essentially free),
- or use `jungfrau1M_alcove/azav_azav` (3201, 1, 123), the per-shot azimuthal
  average, summed over q as a proxy. Correlates 0.879 with ipm2 on run 475.
- `Sums/jungfrau1M_alcove_calib` is run-level only.

### Not usable

`FEE-SPEC0` / `feeBld/hproj` (2048-bin FEE spectrometer) is present on only
77.6 % of events and its per-shot sums are large and negative
(median ≈ −3.6e7) — unnormalized or miscalibrated. Leave it out unless the
calibration is known.

---

## 3. Complete per-shot inventory, run 475

Everything in `evt.keys()` on every event:

| source | type | content |
|---|---|---|
| `XppEndstation.0:Jungfrau.0` (`jungfrau1M_alcove`) | `Jungfrau.ElementV2` | `frame()` (2,512,1024) uint16, `frameNumber()`, `fiducials()`, `ticks()` |
| `NoDetector.0:Evr.0` (`evr0`) | `EvrData.DataV4` | event-code list |
| `EBeam` | `Bld.BldDataEBeamV7` | `ebeamCharge`, `ebeamDumpCharge`, `ebeamEnergyBC1/BC2`, `ebeamL3Energy`, `ebeamLTU250/450`, `ebeamLTUAngX/Y`, `ebeamLTUPosX/Y`, `ebeamPhotonEnergy`, `ebeamPkCurrBC1/BC2`, `ebeamUndAngX/Y`, `ebeamUndPosX/Y`, `ebeamXTCAVAmpl/Phase`, `damageMask` |
| `PhaseCavity` | `BldDataPhaseCavityV1` | `charge1`, `charge2`, `fitTime1`, `fitTime2` |
| `FEEGasDetEnergy` | `BldDataFEEGasDetEnergyV1` | 6 × `f_*_ENRC` |
| `HX2-SB1-BMMON`, `XPP-SB2-BMMON`, `XPP-SB3-BMMON` | `BldDataBeamMonitorV1` | see §2 |
| `XppMon_Pim0/Pim1`, `XppSb3_Pim`, `XppEnds_Ipm0` | `Lusi.IpmFexV1` + `Ipimb.DataV2` | see §2 |
| `XPP-AIN-01` | `BldDataAnalogInputV1` | 16 voltages — **CC/VCC**, see §1 |
| — | `EventId` | timestamp, fiducials, run, vector, idxtime |

Configured but contributing **no per-event data**: `XppEndstation.0:Zyla.1`
(`zyla_sb5`), `NoDetector.0:Evr.1` (`evr1`, emits zero codes), `ControlData`
(`npvControls=0`, `npvMonitors=0`, `events=0` — run 475 is not a scan).

**There is no timetool, no Opal, no camera of any kind producing data in these
runs.** The alias table lists many (`xpp_opal1`, `alvium_tt`, `xtcav`, five
`epix_alc*`, several YAGs) but none are in the partition. The only per-shot
timing information available is `PhaseCavity.fitTime1/2`.

### EVR codes actually firing (evr0, 300 events of run 475)

| code | rate | config label |
|---|---|---|
| 40 / 140 | 100 % | — |
| 41 / 141 | 50 % | — |
| 42 / 142 | 25 % | — |
| 43 / 143 | 8.3 % | — |
| 44 / 144 | 4.2 % | — |
| 45 / 145 | 0.8 % | — |
| 46 / 146 | 0.4 % | — |
| 119 | 100 % | — |
| 137 | 99.3 % | `'Beam On'` |
| 90 | 66.7 % | `'Laser on'` (mislabeled — see §1) |
| 91 | 33.3 % | `'Laser off'` (mislabeled — see §1) |
| 162 | 0.7 % | — |

40–46 / 140–146 are a rate-division ladder. Run 389 shows the same pattern. Small
data reserves columns for codes 30–36, 92–98, 120, 138/139, 150, 164, 190–194,
215/216, all identically zero in both runs.

Small data's `lightStatus/xray` reproduces code 137 exactly, and
`lightStatus/laser` reproduces code 90 exactly (equivalently `not 91`). These are
faithful transcriptions of the EVR codes — and inherit the mislabeling.

### EPICS

383 PVs in `EpicsArch.0:NoDevice.0`, readable per event via
`ds.env().epicsStore()`. **No PV name or alias contains "CC" or "VCC"** — the
only regex matches are `ccm_*` (the MON-hutch channel-cut monochromator:
`ccm_E`, `ccm_Theta0`, `ccm_alio_*`, `ccm_x1/x2`) and `vac_Gcc_*` (cold-cathode
vacuum gauges). Neither is the branch state. **CC/VCC exist only as `ai/ch02` and
`ai/ch03`.**

The small-data producer saves 43 PVs per event (`att_T`, `slit_s*`, `ccm_E`,
`lom_E`, `lom_EC`, the `gon_*` and `robot_*` motors, …) and 383 once per run under
`epicsOnce/`. The delay used by the production pipeline is `epicsAll/delay`,
converted with `delay_ps = 0.9376321852982434 * (delay - 3.1)`
(`utils.py:191`). Both local runs are single-delay: run 389 sits at 2.999 ps,
run 475 at 4.998 ps.

---

## 4. Gotchas found while verifying this

- **`damage/<det> == 1` means the data is PRESENT**, not damaged. Absent
  detectors read 0 (`evr1`, `zyla_sb5`, `ControlData`, `scan`).
- **`psana.Detector('XPP-AIN-01').get(evt)` returns `None`.** The `DdlDetector`
  wrapper cannot handle this type. Use the raw accessor:
  ```python
  evt.get(psana.Bld.BldDataAnalogInputV1, psana.Source('BldInfo(XPP-AIN-01)')).channelVoltages()
  ```
  The same applies to `Lusi.IpmFexV1` (`TypeError: object of type 'IpmFexV1' has
  no len()`) — use `evt.get(psana.Lusi.IpmFexV1, psana.Source('BldInfo(...)'))`.
- **`ipm3/sum` goes negative** (run 389 mean ≈ −6 800, run 475 min ≈ −29 000).
  Do not use it as a normalizer without investigating the baseline.
- The EVR `desc()` strings live in `EvrData.ConfigV7` in the config store, not in
  the per-event `DataV4`. They are stock XPP labels; see §1.
