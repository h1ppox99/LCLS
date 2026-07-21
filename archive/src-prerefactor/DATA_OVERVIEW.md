# xppl1016922 — Data Overview & How to Use It

A practical guide to the local copy of LCLS experiment **`xppl1016922`** (XPP
instrument), aimed at someone who did **not** build the experiment and does
**not** have `psana` installed. It explains what the beamline produces, what
files are here, how they are structured, and how to pull the numbers into plain
numpy arrays (assembled detector images, geometry, masks, per-event scalars) —
ready for whatever analysis you want to do next.

The companion code is in this same `src/` folder and is **data access only**:

| file | what it does |
|------|--------------|
| `lcls_xpp.py` | read the small‑data HDF5 and the psana calib files as plain numpy — no psana needed |
| `read_xtc.py` | pull calibrated per‑event Jungfrau frames + scalars out of the raw XTC (needs psana) |
| `setup_psdm_layout.py` | build a psana‑readable symlink tree (real‑colon calib names) |
| `requirements.txt` | `pip install -r requirements.txt` (numpy, h5py) |

---

## 1. What kind of experiment this is

**LCLS** is an X‑ray free‑electron laser (XFEL): it delivers ultrashort
(~10–50 fs) X‑ray pulses at 120 Hz. **XPP** ("X‑ray Pump–Probe") is one of its
instruments, specialised in **time‑resolved** measurements: an **optical laser
pulse** ("pump") excites the sample and the **X‑ray pulse** ("probe") measures
the structural response a controlled **time delay** later. By scanning the
delay you build a movie of the dynamics.

Consequences that shape every data file here:

- **Everything is per‑pulse (per‑event).** Each X‑ray shot is one "event".
  The FEL fluctuates shot‑to‑shot, so almost every quantity (intensity
  monitors, photon energy, detector frame…) is recorded *per event* and you
  normalise/filter afterwards.
- **Laser‑on vs laser‑off.** The pump laser is fired on a subset of shots
  (here 2 of every 3 — see `lightStatus/laser`). "Pumped − unpumped" is the
  core of the analysis.
- **Timing/monitor diagnostics matter.** i0 intensity monitors (IPMs), gas
  detectors, diodes, e‑beam parameters and a "timetool" are all recorded so
  you can normalise and time‑sort events.

This particular experiment ran **runs ~1–481**. A *run* is one contiguous
acquisition (a fixed configuration, or one delay scan). Two runs are present
locally as reduced data (**389**, **475**) and one as full raw data (**475**).

---

## 2. The detectors and their parameters

Referenced by their psana *source name*:

### Jungfrau 1M — `XppEndstation.0:Jungfrau.0` (small‑data alias `jungfrau1M_alcove`)
The main area detector for this run. A **Jungfrau 1M** = **2 modules**, each
**512 × 1024** pixels → native data shape **`(2, 512, 1024)`**.

| parameter | value | where it comes from |
|-----------|-------|---------------------|
| pixel size | **75 µm** | `UserDataCfg/.../pixelsize` |
| assembled image | **1030 × 1064** | from `ix`/`iy` pixel maps |
| gain stages | **3** (auto gain‑switching) | calib arrays are `(3, 2, 512, 1024)` |
| sample–detector distance | **190 mm** | `azav__azav_dis_to_sam` |
| X‑ray wavelength | **1.2915 Å** (≈ 9.6 keV) | `azav__azav_lam` / `lom_E` |
| direct‑beam center | **(35.51, −35.22) mm** in the detector plane ≈ pixel **(col 1005, row 45)** | `azav__azav_xcen/ycen` |
| accessible q | **~0.01 – 2.43 Å⁻¹** (123 bins, 0.02 Å⁻¹ wide) | `azav__azav_q` |

The beam center sits near a **corner** of the detector — a deliberate offset
so a single 1M panel covers a wide q‑range.

> ⚠️ **Geometry gotcha (important).** The file also stores per‑pixel lab
> coordinates `x, y, z`. The in‑plane `x, y` are reliable, but the `z` map is
> the **stale psana default (100 mm)**, *not* the real 190 mm distance. The
> value that reproduces the physics is **`dis_to_sam = 190 mm`**. Use 190 mm and
> the flat‑detector model (a 1M panel is essentially flat); do not build geometry
> from the `z` map.

### Epix100a — `XppGon.0:Epix100a.1 … .5`
Five **Epix100a** panels (each **704 × 768**, gain **~0.06 keV/ADU**) mounted on
the XPP goniometer. **Calibration constants for them are present**, but neither
local small‑data run contains Epix data — so for runs 389/475 the Epix panels
were not read out / not saved. You'd see them if you process an XTC run that
used them.

### Zyla `zyla_sb5` and monitors
A **Zyla** sCMOS camera (`zyla_sb5`) is configured (see `Sums/zyla_sb5_*` and
`UserDataCfg/zyla_sb5`) but produced no image data in these runs. Many **beam
diagnostics** are saved every event — see §4.

---

## 3. What files are on disk

```
xppl1016922/
├── xtc/            raw per-event detector data (psana XTC) — run 475 only, ~5.8 GB
├── hdf5/smalldata/ reduced per-event summaries (HDF5) — runs 389 & 475
├── calib/          psana detector calibration constants
├── stats/summary/  per-run beamline summary web output (most runs)
└── results/        per-user analysis dirs — EMPTY skeleton in this copy
```

### 3a. Small‑data HDF5  — `hdf5/smalldata/xppl1016922_Run<NNNN>.h5`  ★ start here
Produced from XTC by **`smalldata_tools`**. One row per event; small enough to
load fully in memory. **This is the file the companion code reads.** Present:
`Run0389` (40 003 events), `Run0475` (3 201 events). Structure detailed in §4.

### 3b. Raw XTC — `xtc/xppl1016922-r0475-s0<0-4>-c00.xtc`
Full event stream (every pixel of every frame). A run is split across parallel
DAQ **streams** `s00…s04`; **all streams of a run are read together**. Reading
XTC **requires `psana`** (see §6). Only run 475 is here (~1.3 GB/stream). You
only need this if the quantity you want was *not* saved into small‑data
(e.g. full per‑event Jungfrau frames, or the Epix panels).

### 3c. Calibration — `calib/`
psana calib store. Layout: `calib/<DetType::CalibV1>/<Source>/<constant>/<START>-end.data`.
Constants present:

- **Epix100a** (`XppGon.0:Epix100a.1..5`): `pedestals`, `pixel_gain`,
  `pixel_rms`, `pixel_status`.
- **Jungfrau** (`XppEndstation.0:Jungfrau.0`): `pedestals`, `pixel_gain`,
  `pixel_offset`, `pixel_rms`, `pixel_status`, `dark_min`, `dark_max`, `geometry`.

Each `<START>-end.data` applies from run `START` onward until a higher‑numbered
file supersedes it. Pedestals/darks were regenerated at runs **110, 150, 175,
187, 227, 362, 477** (dark calibration runs). A `HISTORY` file logs provenance.
The files are **ASCII arrays** (numpy‑loadable) with a `#` header; shapes:
Epix `(704, 768)`, Jungfrau `(3, 2, 512, 1024)` for per‑gain constants.

> **You usually don't need to touch `calib/` directly.** The small‑data file
> already embeds the applied `ped`, `rms`, `gain`, `offset`, `pixel_status` and
> `mask` under `UserDataCfg/jungfrau1M_alcove/…`, and the `Sums/*_calib`
> images are already pedestal/gain‑corrected. `lcls_xpp.load_calib()` reads the
> raw files if you ever want them.

> **Filename quirk:** psana names contain `:` (e.g. `Epix100a::CalibV1`). This
> macOS‑made copy replaced every `:` with the invisible char **U+F022**, so
> typing the path literally fails. Use `lcls_xpp.resolve()` (it rewrites
> `:` → U+F022) or `os.listdir`, never a hand‑typed colon path.

### 3d. Beamline summaries — `stats/summary/BeamlineSummary/BeamlineSummary_Run<NNN>/`
Per‑run auto‑generated summary/plots (the web "run tables"). Present for most
runs; useful for a quick eyeball of a run without opening data.

### 3e. `results/` — **empty** directory skeleton in this copy. No analysis
scripts or notebooks were copied. Don't assume code lives here.

---

## 4. Inside the small‑data HDF5 (the important one)

Three logical parts:

**(A) Per‑event arrays** — first axis = number of events (3201 for run 475).

| group / dataset | meaning |
|---|---|
| `event_time`, `fiducials` | timestamp & 120 Hz pulse ID per event (for matching/sorting) |
| `lightStatus/xray`, `lightStatus/laser` | 1/0: was the X‑ray / pump‑laser on this shot |
| `ipm2/sum`, `ipm3/sum`, `ipm_hx2/sum` | i0 intensity monitors (BMMON). `ipm2` = XPP‑SB2, the usual normaliser. Also `peaks`,`xpos`,`ypos` |
| `gas_detector/f_*_ENRC` | FEE gas‑detector pulse energy (mJ), upstream i0 |
| `diode2`, `diodeU`, `lombpm`, `lomdiode` | photodiode/PIM channels (`channels`, `sum`) |
| `ai/ch00..15` | 16 analog‑input channels (user diodes/signals) |
| `ebeam/*` | electron‑beam params: `photon_energy` (per‑shot FEL energy), `L3_energy`, `charge`, position/angle… |
| `phase_cav/*` | phase‑cavity timing |
| `evr/code_*` | which EVR **event codes** fired this shot (triggers/flags). `xray on` ↔ code 137, `laser drop` ↔ code 91 |
| `scan/varStep` | index of the scanned step (all 0 in run 475 → static run) |
| `epics/<pv>` | selected EPICS PVs sampled per event (motors, attenuator `att_T`, slits, `lxt`/`lxt_vitara` laser timing, `ccm_E`, `lom_E`…) |
| `jungfrau1M_alcove/azav_azav` | **`(nevents, nphi=1, nq=123)`** — the azimuthal average already computed per event by smalldata_tools |
| `damage/<det>` | per‑event data‑validity flag per detector (0 = damaged/absent) |

> **There are no full per‑event Jungfrau frames in these small‑data files** —
> only the per‑event azimuthal average (`azav_azav`) plus run‑level sums. For
> full frames you must process the XTC with psana (§6).

**(B) `Sums/`** — detector image summed over the whole run.
`Sums/jungfrau1M_alcove_calib` is `(2, 512, 1024)`, the sum of calibrated
frames — a **real 2‑D detector image**, ready to assemble and analyse.
Also `*_dropped` (laser‑off sum) and `*_square` (for variance).

**(C) `UserDataCfg/`** — written once; the configuration + geometry.
`UserDataCfg/jungfrau1M_alcove` contains the full detector geometry:
`pixelsize`, `imgShape`, `ix`/`iy` (panel→image tiling), `x`/`y`/`z`
(per‑pixel positions), `mask`/`cmask`/`statusMask`, the calib arrays
(`ped`,`rms`,`gain`,`offset`,`pixel_status`), and the azimuthal‑integration
parameters smalldata_tools used (`azav__azav_center`, `..._dis_to_sam`,
`..._lam`, `..._q`, `..._qbins`, …). The `..._header` field is a human‑readable
summary of that geometry. `SmallData.jungfrau_geometry()` returns all of it as a
dict.

Browse any file yourself:

```python
from lcls_xpp import SmallData
sd = SmallData(475)
sd.tree()                     # full group/dataset tree with shapes
print(sd.keys("UserDataCfg/jungfrau1M_alcove"))
```

---

## 5. What the loader hands you (numpy arrays)

`lcls_xpp.py` stops at plain numpy — an assembled image, the geometry, and the
masks. What you do next (integration, fitting, plotting) is your own analysis.

```python
from lcls_xpp import SmallData

sd  = SmallData(475)
img = sd.sum_image()                   # (1030,1064) calibrated run-sum image
geo = sd.jungfrau_geometry()           # dict: everything below
q_ref, azav = sd.azav()                # (123,) and (nevents,1,123) — the average
                                       # smalldata_tools already computed per event
```

`img` is a **real assembled 2‑D detector image** (the native `(2,512,1024)`
panel stack tiled through the stored `ix`/`iy` maps by `SmallData.assemble()`).
`geo` carries the geometry you need to interpret it:

| key | value |
|-----|-------|
| `dist_mm` | 190 mm sample–detector distance (use this, **not** the `z` map) |
| `wavelength_A` | 1.2915 Å (≈ 9.6 keV) |
| `pixel_size_m` | 75 µm |
| `beam_center_mm` | `(35.51, −35.22)` mm ≈ assembled pixel `(col≈1005, row≈45)` |
| `mask` | `1 = good pixel`, `0 = bad` (invert if your tool wants `1 = ignore`) |
| `q`, `q_bin_edges` | the q grid smalldata used, `1/Å` |

To work with **individual events'** frames (not just the run sum) you first need
psana to read the XTC — §6 — then run each `(2,512,1024)` frame through the same
`sd.assemble(frame)`.

---

## 6. If you need the raw XTC / psana (only when small‑data isn't enough)

You are not on an LCLS machine, so you have two options:

- **Best effort locally — conda:** psana ships on the LCLS conda channel:
  ```bash
  conda create -n psana -c lcls-ii -c conda-forge python=3.9 psana
  # then, pointing SIT_PSDM_DATA at the parent of ./xtc and ./calib:
  #   ds = psana.DataSource('exp=xppl1016922:run=475:smd')
  ```
  This is heavy and version‑sensitive; the official source is
  <https://github.com/lcls-psana/psana>. It is only needed to (a) read full
  per‑event detector **frames** from XTC, or (b) regenerate small‑data.
- **Recommended:** stay in the small‑data HDF5 + `calib` ASCII files, which the
  companion code reads with just `numpy`/`h5py`. Everything in §2–§5 works
  without psana. Only reach for XTC when you specifically need raw per‑event
  frames or a detector that wasn't saved to small‑data.

---

## 7. Quick reference — run 475 numbers

```
events ................ 3201        laser on ... 66.7%    xray on ... 99.3%
detector .............. Jungfrau 1M (2, 512, 1024), 75 µm pixels
mono photon energy .... 9.60 keV  (λ = 1.2915 Å)
sample–det distance ... 190 mm
beam center ........... (35.51, −35.22) mm  ≈  (col 1005, row 45) px
q range ............... 0.01 – 2.43 Å⁻¹  (123 bins, Δq = 0.02 Å⁻¹)
assembled image ....... 1030 × 1064
```
Run 475 is a **static** run (no delay scan). Run 389 has the same layout with
40 003 events.
