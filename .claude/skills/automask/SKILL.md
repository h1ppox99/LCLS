---
name: automask
description: >-
  Operate the deterministic automask interface for LCLS run inspection, shot
  selection, Jungfrau image previews, detector-mask construction, and run-local
  mask validation. Use for automasking tasks, for interpreting automask handles
  or artifacts, and when choosing the next safe automask tool from partial
  existing results.
---

# Automask

Use the in-process `automask` tools as the execution boundary. Each tool keeps
its result alive in the host as an object you name by a short handle
(`prof-1`, `sel-2`, `pipe-1`); pass those handles between tools and pass
parameters inline as objects. Perform only the operations the request needs,
reuse handles you already hold, and compare candidates without mutating them.

## Establish the interface

Call `automask_catalog` before authoring or changing a pipeline. It defines the
registered masking capabilities, selection operators, reductions, default
parameters, and the field-native parameter shapes that `define_selection` and
`define_pipeline` accept. It does **not** define the fields present in a run.
Do not invent registry entries or pass ad hoc Python around the tools.

The tools write only inside the current run's working directory and never touch
`xtc/`, `calib/`, `hdf5/`, or `xpp_sharing/`; raw experiment and calibration
data are inputs only. Profiles, previews, masks, and reports are persisted for
you and their paths are returned in each tool's summary — reference those paths,
do not re-derive them.

## Choose the next operation

- Inspect unknown run contents with `inspect_run`; reuse a prior run with
  `load_profile` on its cached `profile/` directory instead of re-reading psana.
- Test a proposed selection with `describe_selection` before reading frames.
- Render evidence with `preview_selection` when selection quality or visible
  detector structure matters.
- Build a candidate with `build_mask` once its selection and pipeline handles
  are defined.
- Evaluate sensitivity with `validate_mask` before recommending a candidate; it
  returns a handle for the recommended pipeline.
- Reuse a handle you already hold when it already satisfies the current
  dependency, rather than recomputing it.

## Inspect and select shots

Read [run-and-selection.md](references/run-and-selection.md) before interpreting
fields, choosing conditions, normalization, trimming, or shot count. Use exact
field names from the inspection `report.md`. Check the stage counts from
`describe_selection` and stop if the selection is empty or contradicts the
intended physical state. Treat a profile made with `max_events` as development
evidence, not a full-run result. When the user requests no modifications, keep
selections and pipelines explicitly hypothetical, and do not call a selection
checked unless `describe_selection` actually ran.

## Design and validate masks

Read [mask-design.md](references/mask-design.md) before defining a pipeline or
judging a mask. Start from the catalog's canonical pipeline example (or the
default `define_pipeline` recipe), preserve the mandatory floor, and change one
understandable concern at a time. Inspect the per-channel layer counts in the
`build_mask` summary as well as the combined mask.

Do not claim scientific correctness from visual plausibility or stability
alone. State what evidence was checked, whether the run/profile was bounded, the
chosen selection and pipeline, the returned handles and artifact paths, and any
unresolved ambiguity.

## Handle failures

- On a missing field, return to the inspection report; do not guess aliases.
- On a rejected selection or pipeline, correct the inline params against the
  live `automask_catalog`; do not patch around the validation during a run.
- On psana or truncated-data warnings, preserve the warning and assess the run
  limitations before proceeding.
- A tool error is returned as a JSON `error` payload, not a crash — read it,
  fix the inputs, and retry with corrected handles or params.
