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
- **`hough_lines` masks a curved ring/arc** → it is running at
  `polarity="bright"`/`"both"`; revert to `dark`. Bright ridges here are
  diffraction rings (real signal), not line defects.

## Reading pixel distributions

Never accept a field `k` from the image or a documented default alone — read the
channel's robust-z distribution *quantitatively*. `explain.png` plots it (kept
grey vs masked crimson, log-scale, cut drawn); confirm the numbers by loading
`explain_<stat>.npy` and measuring the masked fraction and where the cut sits
relative to the bulk.

Decide from the distribution, not the image alone:

- **A clear separable outlier** (a gap or distinct second mode) → put the cut in
  the gap. A good split leaves the kept side roughly symmetric with no outlier and
  the masked side holding the whole outlier.
- **No clear outlier in the raw field, but a defect is still visible on the panel**
  → aggregate before thresholding (currently the `blob_scale` matched filter) and
  re-read the *aggregated* distribution. Aggregation is the test of whether the visually-suspected region is
  really an outlier: if it is, the aggregated distribution splits into two
  sub-distributions — the kept side roughly symmetric with no outlier, the masked
  side covering the entire outlier plus a small base reaching into the normal bulk.
- **No separation even after aggregation** (the masked side just blends
  continuously into the bulk; its area grows smoothly as `k` relaxes) → the cut
  would select noise, not a defect. Raise `k` until the masked side is only the
  outlier; if nothing separates at any `k`, there is nothing to mask here — escalate
  rather than inventing a cut.

Quantitative and visual reads must agree — a log-scale histogram can make a large
masked bulk-fraction look like a thin tail — and both must agree with the
fold-consistency check.

## Detecting a compact defect: the funnel

A compact defect is found through stages: stat field → `blob_scale` aggregate →
threshold → connected components → `area_gate`. `detection_funnel` (and the
per-channel `explain.png`) show each stage. The rule: **a defect visible in the
stat field but absent after a later stage means that stage is mis-parametrized —
never that there is nothing to mask.**

Adjust one knob at a time, coarse to fine, re-reading the funnel after each:

- **`k`** — from the aggregated distribution's bulk/tail knee.
- **`blob_scale.radii`** — if a defect visible in the stat field does not separate
  in the aggregate, add a radius near its apparent size.
- **`blob_scale.aspects` / `angles`** — only if an *elongated* defect (ellipse,
  rectangle) still under-separates; the default is a disk. The whole bank is
  applied at once (pixelwise `max` over every radius×aspect×angle, not one shape
  per pixel), so it never hides a defect — but each added kernel raises the noise
  floor and adds false positives. Use the smallest bank that works (e.g.
  `aspects=(1,2)`, `angles=(0,90)`) and **re-read `k`** afterward: the bulk→tail
  knee moves right as the bank widens.
- **`area_gate.min_area`** — from the component-size gap; keep it **size-only**.

Never gate this by shape: a circularity/eccentricity cut deletes irregular
defects (measured worse here). Elongation is handled in the aggregation
(`aspects`), not by rejecting components.

## Null and near-null channels

A channel masking 0 (or very few) px is **not** automatically clean — it is
either correct (no such artifact on this run) or a threshold that overshot a real
one. Distinguish them; never read a null as reassurance.

- A **pick** (`hough_lines`) legitimately returns 0 when there is simply no line.
  If this is what is observed, then it needs no action.
- A **field** channel (`variance`, `pedestal_z`, …) returning ~0 is a red flag
  whenever the selection image shows structure it should catch — e.g. `variance`
  (low) empty while a dark band/shadow is visible. The likely cause is `k`
  landing just past the artifact, not an absent artifact.
- **Test a suspicious field null:** temporarily loosen the threshold (lower `k`)
  together with an `area_gate`, rebuild, and look at what appears. If a coherent
  extended region emerges, the baseline `k` overshot a genuine artifact — keep the
  looser cut. If only scattered specks appear, the null was correct.

## Reading validation and fold-consistency metrics

Run `validate_mask` before recommending any candidate. Stability means
robustness to the tested variations, **not** agreement with ground truth.

Read `report.md` — it gives mean **and** worst-case per metric (pairwise IoU
mean/min, changed-area mean/max) and names the most-deviant fold per strategy.
Open `metrics.json` only when you need the raw per-fold breakdown.

- Fold consistency **cannot identify false negatives**: an artifact no channel caught cannot
become unstable, and an empty or near-empty mask is perfectly stable
(fold IoU ≈ 1.0) yet may be badly incomplete — never read stability as coverage.

- Fold consistency is the primary **false-positive** detector, and this is decisive for
threshold choice. Read the **fold IoU**: `1 − foldIoU` is (≈) the fraction of the masked
set that is *not* reproducible across folds — the fraction that is noise — and it is 
size-invariant. "Changed area %" grows with mask size and is only a secondary indicator.
A genuine coherent defect jitters only on its boundary, so it should give fold IoU very 
close to 1, the more so the larger it is.

- **Low fold IoU (below ~0.9), especially on a large layer** → the cut is masking
  noise (false positives). Raise `k` — or drop the channel — until the masked set
  is fold-stable. This must corroborate the distribution read above: a cut biting
  the bulk is exactly what shows up here as low fold IoU. If the two disagree,
  re-check both rather than trusting either alone.

## Evidence to decision

The loop, one change per turn:

1. **Detect** — cross the selection image against the mask *first*: enumerate the
   artifacts visible in the image (per-row/column intensity dips or spikes, large
   anomalous components — dark or bright) and confirm each is masked or a
   deliberate keep. Only then compare the explain panels for under-masking,
   over-masking, or a missed artifact class. Judge by what artifacts remain
   unmasked — not by whether present layers look plausible.
2. **Diagnose** — which channel, and whether the cause is the threshold
   (`k`/`mode`), the field (wrong stat or needs aggregation), or the selection.
3. **Change one thing** — retune, add/adjust a regularizer, add a channel, or fix
   the selection. Define a new handle; never mutate a candidate in place.
4. **Re-check** — rebuild, re-read the same evidence, validate. Keep the change
   only if the evidence supports it; otherwise revert to the baseline.

## Pitfalls

- **A larger mask is not a better mask** — judge by whether each layer sits on a
  real artifact, never by masked-pixel count.
- **Layer-by-layer review is blind to misses** — checking that each masked region
  sits on a real defect can only confirm what a channel already caught, never
  reveal what none did. Always cross the image against the mask (its complement),
  not just the mask against itself.
- **Never certify by eye or a single score** — verify each layer's evidence, then
  confirm with `validate_mask` and (in development) labelled evaluation.
- **Stability is not correctness**, and a sharp image does not prove the intended
  physical state.
- **Escalate ambiguous scientific judgments** (signal vs artifact) instead of
  hardening a weak heuristic into a confident mask.
