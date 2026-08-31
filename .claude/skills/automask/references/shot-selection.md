# Shot selection

Stage 2. The technical levers of a `ShotSelection` and their behaviour. The
baseline selection is in [baseline.md](baseline.md); when to move off it, and how
to judge a resulting image, are in [evidence-and-decisions.md](evidence-and-decisions.md).
Use exact field names copied from the profile `report.md`. Check every stage
count with `describe_selection` before decoding frames; stop if a stage empties
the selection.

## Ground selection (before any run-specific field)

Apply in this order, on every run:

1. **Beam-on gate** — `eventCode[137] == 1` (drops ~1% no-beam shots).
2. **Low-flux floor** — a low-side cut on an upstream incident monitor at
   ~0.5·median. The darkest shots carry no coherent structure and only add noise.
3. **No per-shot normalization** — it erases the persistent structure masking
   depends on.

Keep bright shots; cut the high side only for confirmed saturation.

## Conditions (`where`)

- Express physical-state constraints as `Condition(field, op, value)`; all
  conditions are **ANDed**.
- Operators: `==`, `!=`, `<`, `<=`, `>`, `>=`, `between`, `in`, `not in`,
  `finite`, `nonzero`.
- The experiment logs name which fields carry physical meaning for the run;
  the library has no beam/branch concepts, so that meaning lives in the caller.
- A good working point is a selection where the experiment is on and most signal
  passes — beam-on plus the flux floor above.

## Normalization

- Set `normalization` only to a **covered downstream** intensity field. It also
  rejects non-finite and zero values.
- `n_eligible` reports the normalization field's finite, nonzero count after the
  `where` conditions — compare it across candidate normalization fields; a
  report's whole-run coverage does not settle the choice.
- The baseline uses `normalization=None`.

## Trim

Minor lever. A small **low-side** trim on incident flux helps slightly (~3% is a
starting candidate); high or symmetric trims **hurt** — they discard the
brightest, most informative shots. **Never trim the high tail.**

## Number of shots (`n_shots`)

More shots lower the background noise (error ∝ 1/√N) at the cost of time.

- **~800** default, **400** minimum.
- **~1600** for sharp features, **~400** for diffuse scenes.
- Below ~200 the averaging evidence is weak — flag it rather than silently
  relaxing the physical conditions.

`n_shots` samples evenly across the eligible shots; `None` selects all.

## Reductions as channels

Two complementary reductions, not one ranking:

- **`mean` + `std`** — signal plus extended artifacts. `std` exceeds `mean`
  wherever structure is coherent, and is complementary to `mean`.
- **`mad`** — the intensity-free instability floor (feeds `mad_variance`). Unlike
  `std` it does **not** light up on signal, so it isolates persistently unstable
  pixels. Not in the baseline; add it when instability is suspected.

## When to deviate

When the selection image is weak or a distribution looks wrong, the tiered fixes
(raise `n_shots`, check the fields, adjust the flux floor, add `mad`, and when to
escalate) live in
[evidence-and-decisions.md](evidence-and-decisions.md#judging-the-selection-image).
Keep competing selections as distinct handles and compare them — never mutate one
in place.
