# Run profiling

Stage 1. `inspect_run` makes one psana pass, caches the profile, and returns a
handle plus a `report.md` listing every per-shot field, its coverage, changing
values, geometry, and calibration evidence. Reuse a cached run with
`load_profile` instead of re-reading psana. The profile is the source of truth
for field names — copy them exactly, never infer a field from its alias.

Read the profile alongside `experiment_logs/` and answer the two questions below
before selecting shots.

## Confirm calibration

- **Is the data calibrated, and by which constants?** Small-data files embed the
  applied calibration; the psana calib store holds pedestals/darks regenerated
  at runs 110, 150, 175, 187, 227, 362, 477 — identify which set applies to this
  run.
- **Geometry** — sample–detector distance, PONI/beam center, pixel size,
  wavelength. For this experiment: `dis_to_sam = 190 mm`, λ ≈ 1.2915 Å
  (~9.6 keV), 75 µm pixels, beam center near a detector corner
  (~pixel col 1005, row 45), q ≈ 0.01–2.43 Å⁻¹.
- **Geometry gotcha** — use `dis_to_sam = 190 mm`, **not** the per-pixel `z` map
  (stale psana default of 100 mm). In-plane `x`/`y` are fine. This matters only
  for azimuthal/center-dependent channels (`sigma_clipping`).

## Identify the run's relevant variables

From `report.md` + experiment logs, name the fields the selection will use:

- **Beam-on** — `eventCode[137]` (genuine `Beam On`).
- **Upstream incident monitor** for the flux floor — e.g. `gas_detector/f_11_ENRC`
  or `ipm2/sum` (ipm2 is upstream of the beam split, blind to the branch).
- **Downstream monitors** if a branch-aware ranking is ever needed —
  `sample_diode` (`diodeU/channels[:,0]`), `diodeU`, `lombpm`.
- **Branch voltages** — `ai/ch02` (CC), `ai/ch03` (VCC), thresholded at 2 V;
  irrelevant to masking but present.

Do not use `lightStatus/laser` or EVR codes 90/91 as filters — there is no laser
and those labels mean nothing here. See `docs/DATA_xppl1016922.md` for the full evidence.

## Bounded-profile caveats

- A profile made with `max_events` is **development evidence, not a full-run
  result** — say so when reporting from it.
- **Truncated runs** (e.g. run 389) must be opened by explicit file path, not the
  run resolver; their XTC event indices do **not** align with small-data rows, so
  never join the two by index, and expect missing detectors/streams.
