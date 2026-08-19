# Run inspection and shot selection

## Evidence hierarchy

1. Use the current inspection's `report.md` for available fields, coverage,
   changing values, geometry, and calibration evidence.
2. Use `docs/DATA.md` for experiment-specific interpretation and
   `docs/DATA_OVERVIEW.md` for local data completeness.
3. Use experiment logs when the meaning or polarity of an analog condition is
   uncertain. Do not infer semantics from an alias alone.

## Experiment constraints

- Run 475 has a complete local five-stream XTC copy. Run 389 has four truncated
  streams and yields only a subset of the events in small data. Never join run
  389 XTC events to HDF5 rows by index.
- This experiment has no laser. EVR 90/91 labels are inherited timing labels and
  are not valid laser-state filters. EVR 137 represents x-ray beam presence but
  does not identify the CC/VCC branch.
- The beam is split into CC and VCC branches. `ai/ch02` observes CC switching and
  `ai/ch03` observes VCC switching around 2 V. Confirm the desired branch and
  inequality from observed distributions or experiment evidence before fixing
  a recipe.
- `ipm2` is upstream of the split. Prefer a downstream monitor such as
  `diodeU/channels[0]`, `diodeU/sum`, or `lombpm/sum` for normalization or
  ranking when its coverage supports the choice.

## Selection procedure

Use exact names copied from the inspection profile:

1. Express physical-state constraints as `where` conditions. All conditions are
   combined with AND.
2. Set `normalization` only to a covered downstream intensity field. This also
   rejects non-finite and zero values.
3. Apply a small symmetric percentile trim only when the selected intensity
   distribution shows outliers. Three percent at each tail is a starting
   candidate, not a fixed scientific rule.
4. Treat roughly 800 evenly spaced shots as a practical initial image sample.
   If fewer than 100 survive, flag the weak averaging evidence rather than
   silently relaxing the physical conditions.
5. Run `describe_selection`, inspect every stage count, then `preview_selection`
   for at least the reduction needed by the chosen pipeline. Use `mean` for
   average structure and `std`, `median`, or `mad` only when their different
   statistics are relevant.

`n_eligible` measures the normalization field's finite, nonzero effect after
the `where` conditions. Compare separately described selections when choosing
among normalization fields; a report's whole-run coverage does not make that
choice.

Keep competing selections as distinct handles and explain their physical
meaning. A visually sharp image is useful evidence, but does not by itself prove
that the selection represents the intended experimental state.
