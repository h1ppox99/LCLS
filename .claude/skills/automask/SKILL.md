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

Produce one boolean mask per run for the Jungfrau1M detector (`True == masked`),
run-agnostic, to beat the lab's manual mask. Production is unsupervised — no
ground truth at masking time.

## The baseline contract

The pipeline carries a strong, **low-variance baseline** (default selection and
default channels; see [baseline.md](references/baseline.md)). Do not rebuild a
recipe from scratch — that reintroduces the variance and uncertainty the baseline
exists to remove.

The agent's job is to **reduce the baseline's bias for this run**: read the
evidence, notice where the baseline under-masks, over-masks, or misses an
artifact, and make one attributable change at a time. Anchor on the baseline,
deviate only on evidence, and ship the baseline when the evidence supports it.

## The tool loop

Use the in-process `automask` tools as the execution boundary. Each tool keeps
its result alive as an object you name by a short handle (`prof-1`, `sel-2`,
`pipe-1`); pass handles between tools and pass params inline as objects. Reuse
handles you already hold; compare candidates as distinct handles without mutating
them.

Call `automask_catalog` first — it defines the registered statistics,
regularizers, selection operators, reductions, and exact parameter shapes. It
does **not** list a run's fields. Do not invent registry entries.

1. **Profile** — `inspect_run` (or `load_profile` to reuse a cache) → profile
   handle + `report.md`. See [profiling.md](references/profiling.md).
2. **Select** — `define_selection` → `describe_selection` (check every stage
   count) → `preview_selection` (render the reduction the pipeline needs). See
   [shot-selection.md](references/shot-selection.md).
3. **Mask** — `define_pipeline` → `build_mask` (mask.npy + overview.png +
   per-channel explain panels). See [mask-design.md](references/mask-design.md).
4. **Validate** — first cross the mask against the selection image: list the
   anomalous structure visible in it (per-row/column intensity dips or spikes,
   large anomalous components — dark or bright) and confirm each is masked or a
   deliberate keep — judge the mask by what anomalous structure remains *unmasked*,
   never only by whether the layers already present look plausible. Then
   `validate_mask` (perturbations + fold consistency; returns the recommended
   pipeline handle). See
   [evidence-and-decisions.md](references/evidence-and-decisions.md).

The tools write only inside the run's working directory and never touch `xtc/`,
`calib/`, `hdf5/`, or `xpp_sharing/` — those are read-only inputs. Artifact paths
are returned in each summary; reference them, do not re-derive them.

## When to deviate

Deviate only when the evidence in
[evidence-and-decisions.md](references/evidence-and-decisions.md) shows the
baseline under-masks, over-masks, or misses an artifact class — then change one
understandable concern and re-check the same evidence. If the user asks for no
modifications, keep selections and pipelines explicitly hypothetical, and do not
call a selection checked unless `describe_selection` actually ran. Treat a
`max_events` profile as development evidence, not a full-run result.

## Reference map

- [baseline.md](references/baseline.md) — the default selection + pipeline. Start here.
- [profiling.md](references/profiling.md) — confirm calibration; name the run's variables.
- [shot-selection.md](references/shot-selection.md) — the selection levers.
- [mask-design.md](references/mask-design.md) — channels, the artifact→operator map, deliberate change.
- [evidence-and-decisions.md](references/evidence-and-decisions.md) — read explain.png / distributions / metrics and choose the change.

## Failure handling

- **Missing field** — return to the inspection `report.md`; do not guess aliases.
- **Rejected selection/pipeline** — correct the inline params against the live
  `automask_catalog`; do not patch around the validation.
- **psana or truncated-data warnings** — preserve the warning and assess the run
  limitations before proceeding.
- **Tool error** — returned as a JSON `error` payload, not a crash; read it, fix
  the inputs, retry with corrected handles or params.

Do not claim scientific correctness from visual plausibility or stability alone.
State what evidence was checked, whether the run/profile was bounded, the chosen
selection and pipeline, the returned handles and artifact paths, and any
unresolved ambiguity.
