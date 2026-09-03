# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A local copy of LCLS experiment **`xppl1016922`** (XPP instrument at SLAC), **plus an active
research project built on top of it**. The data layout mirrors the SLAC path
`/sdf/data/lcls/ds/xpp/xppl1016922/`, so relative paths from cluster scripts carry over.

The experiment spans runs ~1–481. A "run" is one contiguous acquisition, identified by a
zero-padded number (e.g. run 475). Only a couple of runs are present locally (see below).

**The project goal: automate detector masking.** We are building an automatic, run-agnostic
masker for the Jungfrau1M detector to replace the lab's manual hand-drawn mask + beam-center
recipe. The lab's current production method is the baseline we measure against and aim to beat.

## Repository map

Read `docs/DATA_xppl1016922.md` for the full data guide; this section is the orientation.

The repo is a single installable Python package, **`automask`** (top-level, `pip install -e .`),
sitting alongside the data mirror. There is no `src/` wrapper. Import project code as
`from automask.… import …` and run entry points with `python -m automask.<module>` — the old
`sys.path.insert` hacks are gone.

| path | role |
|------|------|
| **`automask/`** | **Main working area.** The auto-masking project, an installable package. Core: `masking.py` (STATS/REGULARIZERS/COMBINERS registries + `Channel`/`Pipeline`), `stats/` `regularization/` `combine/` (one file per method), `sample.py` (`Sample`, the per-run arrays a pipeline reads), `image_store.py` (selected-shot reductions + psana calibration constants, content-hashed cache), `run_profile.py` + `utils.py` (one XTC pass -> per-shot field table), `shot_selection.py` (field-native `Condition`/`ShotSelection`), `evaluation/` (all development and runtime evaluation), and `dataset.py` (loaders/score). Sweeps use Hydra through `conf/` + `studies/sweep_hyperparameters.py` + `scripts/*.sh`. `io/` are psana readers, `producers/` prewarm the cache, `studies/` contains only temporary research awaiting migration, `data/masks/` holds human references, and `outputs/` is generated and **gitignored**. Start here. |
| `automask/evaluation/` | **All evaluation code.** `labelled.py` scores against reference masks, `consistency.py` measures reproducibility on round-robin real-shot folds, and `azimuthal.py` contains the separate physical-consistency diagnostic. Fold reductions use `ImageStore`; see `docs/EVALUATION.md` and `docs/CONSISTENCY.md`. |
| `xpp_sharing/` | **The lab's current production method** (CO2 delay-scan notebooks + `utils.py`). Reference/baseline to improve on — manual mask, diode normalization, delay binning. Read-only. |
| `automask/io/` | psana readers (import as `automask.io.<name>`): `psana1.py` (`Psana1RunSource` — the local↔SLAC seam: `from_files` opens explicit streams, `from_experiment` uses the standard resolver), `read_xtc.py` (calibrated frames, calibration constants, panel index maps), `lcls1_adapters.py` (official `smalldata_tools` detector adapters for profiling). |
| `docs/`, `psana_env.sh` | `DATA_xppl1016922.md` (data layout + what a single shot records) + `PSANA_XTC.md` (how to open XTC) + `EVALUATION.md` and `CONSISTENCY.md`; and the env-activation script (repo root). |
| `xtc/` | Raw per-event detector data (psana XTC format). |
| `hdf5/smalldata/` | Reduced per-event HDF5 summaries. **Start data analysis here** — no psana needed. |
| `calib/` | psana detector calibration constants. |
| `stats/summary/` | Per-run beamline summary output (most runs). |
| `results/` | **Empty skeleton — ignore it.** No code or outputs were copied. |

## Environment (psana)

psana **is installed locally** and works. Activate it with:

```bash
source psana_env.sh      # sets SIT_* vars, activates conda env ana-4.0.66-py311
```

The Bash tool does not persist shell state between calls, so source it in the *same* command
that runs python: `source psana_env.sh && python -m automask.masking`.

- One-time setup: `pip install -e . --no-deps` makes `import automask` work from anywhere
  (`--no-deps` so pip doesn't touch numpy/scipy in the ana env and break psana).
- Reading small-data HDF5 and `calib/` needs `numpy`/`h5py`; dependencies are declared in `pyproject.toml`.
- Reading raw XTC frames needs psana (the env above). See `docs/PSANA_XTC.md`.

## Coding requirements

- **Comments** : limit the quantity of comments used in Python files to a small level

## Data on disk

**`hdf5/smalldata/xppl1016922_Run<NNNN>.h5`** (4-digit run) — one row per event, loads fully
in memory. Useful information to validate claims about data content, conventions etc... It should 
not be integrated in the masking pipeline, which directly uses `xtc`files.

**`xtc/xppl1016922-r<RUN>-s<STREAM>-c00.xtc`** — full event stream, needs psana. A run is
split across parallel DAQ streams `s00…s04` that psana normally merges by timestamp. Present:
- **Run 475** — all 5 streams (s00–s04) complete (~1.36 GB each). 3 201 events.
- **Run 389** — **4 streams** present (s00, s01, s03, s04; s02 missing), all of them
  **partially-downloaded and truncated**. psana merges what it can and yields **6 471
  events**, versus 40 003 in the merged small-data file. See the gotcha below.

**`calib/`** — psana calib store: `calib/<DetType::CalibV1>/<Source>/<constant>/<START>-end.data`.
Each `<START>-end.data` applies from run `START` until superseded. Files are numpy-loadable
ASCII arrays. Pedestals/darks were regenerated at runs 110, 150, 175, 187, 227, 362, 477. The
small-data files already embed the applied calibration, so you rarely need `calib/` directly.

## Detectors & geometry (Jungfrau1M, the main detector)

- **Jungfrau1M** — psana source `XppEndstation.0:Jungfrau.0`, small-data alias
  `jungfrau1M_alcove`. 2 modules of 512×1024 → native shape **`(2, 512, 1024)`**, uint16,
  3 gain stages. Assembles to a **1030×1064** image via stored `ix`/`iy` maps.
- **Epix100a** — `XppGon.0:Epix100a.1…5`, five 704×768 panels. Calib is present, but **neither
  local run (389/475) contains Epix data** — those panels were not read out/saved.
- Geometry: sample–detector **190 mm**, λ **1.2915 Å** (≈9.6 keV), 75 µm pixels, beam center
  near a detector corner at `(35.51, −35.22)` mm ≈ assembled pixel (col 1005, row 45),
  q ≈ 0.01–2.43 Å⁻¹.
- **Geometry gotcha:** use `dis_to_sam = 190 mm`, **not** the per-pixel `z` map (stale psana
  default of 100 mm). In-plane `x`/`y` are fine.

## Masking model

One run in, one boolean mask out (`True == masked`). Four stages, in order:

1. **Profile** — `utils.profile_run_values(run)` makes one XTC pass and returns a `RunProfile`:
   every per-shot field psana and `smalldata_tools` expose, discovered, not declared.
2. **Select** — `ShotSelection(where=(Condition(field, op, value), ...), trim=..., n_shots=...)`
   over those field names. The library has no beam/branch concepts; experimental meaning lives
   in the caller.
3. **Mask** — `Pipeline(channels=[Channel(stat, params, field_reg=..., mask_reg=...), ...],
   combiner=...)`. ONE list: whether a channel is part of the intensity-free floor is read from
   its stat's registered `kind`, never declared twice. `Pipeline.needs()` names every array the
   channels read; `Sample.from_store(run, selection, needs)` materializes exactly those — a name
   in `image_store.REDUCTIONS` is reduced over the selected shots, anything else is passed
   straight to `psana.Detector` as a calibration accessor (`pedestals`, `rms`, `status_as_mask`).
4. **Score** — `evaluation.evaluate` uses references during development;
   `evaluation.evaluate_consistency` measures shot/fold reproducibility without labels.

Conventions that bite if ignored:

- **Reductions are assembled, calibration constants are native panel.** Assembling scatters onto
  a zero-filled canvas, so a constant whose zero means something (`status_as_mask`) must be
  interpreted BEFORE `geometry.panel_to_asm`. `ImageStore.calibration` serves panel form only.
- **psana masks are `1 == good`; project masks are `True == masked`.** The consuming statistic
  converts; `io/read_xtc.detector_calibration` returns psana's values untouched.
- **Detector facts are asked of psana, never hardcoded** — panel shape from `Detector.shape(run)`,
  the assembled canvas from `(ix.max()+1, iy.max()+1)`, the gain-stage count from the constant's
  own leading axis. Do not reintroduce shape constants.
- **A `Sample` carries no ground truth.** The human mask comes from `evaluation.reference_mask(run)`.

## Gotchas

- **Colons in filenames.** psana names contain `:` (e.g. `Epix100a::CalibV1`,
  `XppGon.0:Epix100a.1`). This copy was made on macOS, which can't store `:`, so each `:` is
  replaced by the private-use char **U+F022**. Typing a literal colon path fails ("No such
  file or directory"). Use glob/tab-completion, `os.listdir`/`os.walk`,
  or a `calib/Epix100a*` wildcard — never a hand-typed colon path. Note psana itself reads
  the psdm tree (`$SIT_PSDM_DATA/xpp/xppl1016922/calib`, real colons), not the repo `calib/`.
- **Truncated run 389.** All four present streams are truncated. psana reads them fine —
  6 471 events decode cleanly (full Jungfrau raw frames included), then each stream stops at
  its truncation with an `EOF while reading datagram payload` warning, no crash. But:
  (a) **open it by explicit file path**, not `exp=xppl1016922:run=389` — the run-resolver
  rejects the incomplete layout ("XTC file(s) is empty"); use
  `automask.io.read_xtc.local_run_source(389)`, which globs the streams into a
  `Psana1RunSource`. `slac_run_source(389)` is the same thing via the standard resolver. (b) Those 6 471 events
  are a **subset** of the 40 003 in small data, so **XTC event indices do not line up with
  small-data row indices for run 389** — never join the two by index (run 475 is complete and
  does align). (c) The streams carry Jungfrau + beamline monitors (EBeam, gas detector,
  BMMONs, IPMs, the XPP-AIN-01 analog input) but **not** the Epix panels. (d) `.calib()`
  needs the calib dir wired into psana's search path (`Psana1RunSource` does this via
  `psana.setOption`; `automask/dev/setup_psdm_layout.py` builds the psdm tree); `.raw()`
  works regardless.
- **There is no laser, and `lightStatus/laser` is a lie.** One x-ray beam is split into two
  branches, **CC** and **VCC**, selected per shot by the analog voltages `ai/ch02` (CC) and
  `ai/ch03` (VCC), thresholded at 2 V. EVR codes 90/91 are labelled `'Laser on'`/`'Laser off'`
  in the stock XPP timing config and mean nothing here — do not use them as a shot filter.
  `lightStatus/xray` (EVR 137, `'Beam On'`) is genuine but says nothing about the branch.
  Likewise **ipm2 is upstream of the split** and is blind to it: normalize or rank shots on a
  downstream monitor (`sample_diode` = `diodeU/channels[:,0]`, `diodeU`, `lombpm`). Full
  evidence in **`docs/DATA_xppl1016922.md`**; the code side is `automask/selection/shot_selection.py`.
