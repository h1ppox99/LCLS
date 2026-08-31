# Evidence and decisions

The agent's core value: turning evidence for *this* run into one attributable
change to the baseline. The baseline is low-variance; deviate only on evidence
read here. Each section below says what to look at, then gives the fixes to try —
**first the cheap/safe ones, then the more invasive ones, and what to ask the
user about.** Apply one change at a time and re-read the same evidence.

## Judging the selection image

Good: meaningful persistent structure standing above the background, with real
signal (Bragg peaks, rings) intact. Bad: blurry, structureless, or a monitor
distribution that looks wrong.

If the image is weak or a distribution looks off:

- **First try:** raise `n_shots` if enough shots are eligible; confirm the
  `where` fields (especially the incident monitor) are the right ones against
  `report.md`; add the `mad` reduction to expose instability the mean hides.
- **Then try:** adjust the flux floor (relax to keep more shots, tighten to drop
  a dark tail); add a small low-side trim if the incident tail is bad; try a
  different downstream monitor field.
- **Last resort:** per-shot normalization on a covered downstream field — it
  usually erases structure, so only when nothing else helps.
- **Ask the user** when the correct monitor field is unclear, a distribution
  stays anomalous, or the image is still not maskable after the above.

## Reading explain.png

`build_mask` persists `overview.png` (mask over the image) and a per-channel
`explain.png` — for a field channel its robust-z field with the `(k, mode)` cut
drawn, for a pick (`hough_lines`) its hidden stages. Floor channels have no
panel.

Fixes by what the panel shows:

- **Cut bites into the bulk** (over-masking) → raise `k`.
- **Cut misses a visible defect** (under-masking) → lower `k`; if the defect is
  diffuse and never separates, add `blob_scale` to aggregate it.
- **Layer sits on real signal** (Bragg/ring) → wrong operator for that
  morphology; switch per the map in [mask-design.md](mask-design.md), or gate the
  area. If unsure whether it is signal, ask the user.
- **Layer scattered / empty while a defect is visible** → wrong operator; add the
  channel matched to the artifact.
- **`hough_lines` fires on a rim/edge** → normally the inpainted floor prevents
  this; if it persists, suspect a real line or a selection problem, not a knob.

## Reading pixel distributions

Look at the field histogram against its cut.

- **Defect separable** (distinct tail or mode) → place the threshold in the gap.
- **Not separable** → do not force a threshold. First add `blob_scale`
  aggregation; then switch to the operator matched to the morphology; if it still
  will not separate, the artifact may not be maskable from this evidence —
  escalate rather than inventing a cut.

## Reading validation and fold-consistency metrics

Run `validate_mask` before recommending any candidate. Stability means
robustness to the tested variations, **not** agreement with ground truth.

- **A channel that is fold-inconsistent** → likely a fold artifact; drop it or
  make it more robust before keeping it.
- **Consistency surfaces persistently unstable pixels** → add a `mad_variance`
  channel on the `mad` reduction.
- **Labelled evaluation** against a reference mask is development evidence only;
  keep it separate from production inference.

## Evidence to decision

The loop, one change per turn:

1. **Detect** — compare the baseline `overview.png` and explain panels to the
   selection image: under-masking, over-masking, or a missed artifact class.
2. **Diagnose** — which channel, and whether the cause is the threshold
   (`k`/`mode`), the field (wrong stat or needs aggregation), or the selection.
3. **Change one thing** — retune, add/adjust a regularizer, add a channel, or fix
   the selection. Define a new handle; never mutate a candidate in place.
4. **Re-check** — rebuild, re-read the same evidence, validate. Keep the change
   only if the evidence supports it; otherwise revert to the baseline.

## Pitfalls

- **A larger mask is not a better mask** — judge by whether each layer sits on a
  real artifact, never by masked-pixel count.
- **Never certify by eye or a single score** — verify each layer's evidence, then
  confirm with `validate_mask` and (in development) labelled evaluation.
- **Stability is not correctness**, and a sharp image does not prove the intended
  physical state.
- **Escalate ambiguous scientific judgments** (signal vs artifact) instead of
  hardening a weak heuristic into a confident mask.
