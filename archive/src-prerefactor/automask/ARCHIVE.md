# Archived masking ideas (xppl1016922 Jungfrau1M)

Explored in the 5-method study (run 475), then discarded. Kept as a record so we
don't re-try them blindly — but note the study scored each method against the
*whole* `human_Mask` using the `_dropped` RMS, so its scores understate anything
that only fires on the residual `T = human & ~G`. Ideas rejected here have already
come back in a better form (D -> `method_blackhat`); read the scores as "this exact
formulation failed", not "this signal is useless".
Scores are IoU / precision / recall vs `human_Mask`.

- **A — variance (RMS threshold):** flag pixels whose log-RMS is a robust-MAD outlier (dead ≈ 0 variance, hot ≈ huge). *(0.017 / 0.91 / 0.02 — precise but tiny recall; redundant with F.)*
- **B — horizontal/vertical filters:** flag pixels deviating strongly from their row- and column-median neighbours. *(0.007 / 0.80 / 0.01 — negligible coverage.)*
- **D — low-signal morphology:** threshold low-mean pixels, grow by morphological closing, keep large regions. *(0.041 / 0.16 / 0.05 — too imprecise, masks normal signal.)* **Superseded — morphology was revived successfully as `method_blackhat`.** What was wrong here was the *order*: thresholding first and closing after grows whatever the threshold got wrong. The black-hat inverts that (close first, threshold the closing *residual*), which measures each pixel against its local background instead of a global cut, and keeps *small* dark features rather than large regions. T-prec 0.905 — the most precise of the three residual detectors.
- **E — local z-score anomalies:** pixel vs 2D local-median / local-scale, keep clusters, drop single specks. *(0.001 / 1.00 / 0.001 — almost nothing survives.)*
- **F — persistently-negative pixels (the "clip negatives" idea):** flag `mean < −3·RMS`. *(0.012 / 1.00 / 0.012 — 100% precise but dead pixels only, fully redundant with A; `mean<=0` alone flags ~47% of the chip and is useless.)*

**Where this went:** the live pipeline is `methods.py` — geometry floor G, then
variance (A) and black-hat (C) on the residual (IoU 0.747). The per-pixel statistics
tooling (`pixel_stats.py`, `data/features/`) is a **source** for that pipeline,
not just a diagnostic.

## `method_hv` — dropped from the pipeline (run 475)

Kept as a function, removed from the combo: it makes the result **worse**
(`G|A|B|C` 0.739 vs `G|A|C` **0.747**). Of the 2072 px it adds that `G|A|C` lacks,
only **10.3%** are real targets — ~9 false positives per hit.

**It is not redundant with the black-hat, and don't record it as such.** Only 56%
of its hits overlap C, and **44% of them are on BRIGHT pixels** — which `black_tophat`
cannot reach by construction, being a dark-feature operator. Nothing else in the
pipeline sees bright defects at all. The bright hits are mostly *wrong here* because
run 475 is illuminated, so sharp bright structure is diffraction signal, not hot
pixels. On a **dark run** that inverts: bright outliers are genuine hot pixels (it is
how `pixel_status` is built) and `method_hv` would be the only tool here that sees
them. Re-tune `k` (12 was set against an illuminated run) before trusting it.

## Post-op alternatives (run 475)

`pad_mask` (5x5 square dilation) was applied to every method as a "general rule".
That rule is **gone**: A now uses `close_open` (IoU 0.742 -> **0.747**, and 3.6x
fewer good pixels discarded: 9211 -> 2539 FP). G and C still use `pad_mask`.

**Method: the first sweep here got the wrong answer, and the failure is instructive.**
It tested closing/opening at 5x5 -- the element tuned for *dilation* -- concluded
"dilation wins every comparison", and nearly closed the question. Closing fills
interiors, so it wants a *larger* element; at its own optimum (sq9) it ties dilation
(0.741 vs 0.742) at far higher precision (0.966 vs 0.928). **Never carry one
operator's hyperparameter over to another and call the comparison fair.**

- **the density of a mask decides what it can survive** -- measure it first
  (`binary_erosion` survivor count) instead of guessing:
  | mask | raw px | survives erode 3x3 | verdict |
  |---|---|---|---|
  | A variance | 8428 | **0** (salt-and-pepper) | dense *as a region*: `close_open` |
  | C blackhat | 667 | 0 (true specks) | `close_open` annihilates it -> `pad_mask` |
  | G geometry | 47344 | 34952 (1-px lines) | erosion eats the lines -> `pad_mask` |
- **`close_open` on A — the winner (sq9: IoU 0.7469, prec 0.978).** Close fuses the
  speckle into solid regions, open then drops the stragglers = "keep contiguous
  regions, discard lone specks". A real plateau, not an argmax fluke: sq7 0.7414 /
  sq8 0.7465 / **sq9 0.7469** / sq10 0.7437 / sq11 0.7416.
- **ORDER IS EVERYTHING: `close>open` works, `open>close` destroys.** Opening first
  erodes, and nothing here has a solid 3x3 core to survive (A -> 0 px, and 0.518 for
  every variant). Closing first *creates* the structure opening can then safely erode.
  This is the one result to remember from this file.
- **opening alone — never.** Anti-extensive; wipes A (8428 -> 0) and C (667 -> 0).
- **`close>dilate` / `dilate>close` — redundant.** At pad=2 the dilation already
  subsumes closing's hole-filling (50715 vs 50808 px, IoU 0.741 vs 0.742).
- **`close_open` does NOT transfer.** It kills C (667 -> 0) and shrinks G (47344 ->
  36792 vs dilation's 71776). It is an A-specific cleanup, not a new global rule.
- **`pad_mask`'s 5x5 is right for what still uses it** — square3 0.618 / **square5
  0.735** / square7 0.677, disk2 0.712, cross5 0.691 (measured pre-`close_open`).
  Independently re-derives the notebook's hand-chosen 5x5.

Figure: `images/padding_ops_A_run0475.png` — A raw / dilate sq5 / close sq5 / close
sq9 / open sq5 / close>open sq9, full chip + beam-stop zoom + TP/FP/FN.

## Intensity-banded features — the variance detector needs the mixed intensities

Premise: an all-shot sum blends near-dark and bright frames, so masking per
intensity band might expose defects that only misbehave at one illumination level.
Tested by rebuilding `mean`/`ustd` from only the shots in one `ipm2/sum` percentile
band (`../extract_band_features.py`) and running the UNCHANGED `G|A|C` pipeline on
each (`intensity_bands.py`, run 475, ~200 frames/band):

| features | IoU | A adds | A T-prec | C T-prec |
|---|---|---|---|---|
| all, published (dropped mean) | **0.747** | 3.60% | 0.955 | 0.905 |
| **all_raw, raw mean — the control** | 0.704 | 3.60% | 0.955 | 0.177 |
| 3-10% band (ipm2 15-190) | 0.471 | **0.00%** | 1.000 | 0.210 |
| 90-97% band (ipm2 14799-23553) | 0.476 | **0.07%** | 1.000 | 0.138 |

**Read this table against `all_raw`, not against the published 0.747.** The published
baseline feeds `method_blackhat` the `_dropped` mean from `pixel_stats.py`, which is
built from the run's 24 **beam-off** shots (see the `_dropped` section below) — a dark
map, no diffraction rings; the band means are plain calibrated means of LIT shots
(median ~0.24, rings visible). That is a different *estimator* — in fact a different
beam condition — not just different shots, so comparing a band to 0.747 confounds the
two. `all_raw` holds the estimator fixed (800-frame raw `umean`/`ustd`, all
intensities).

**Result 1 — A (variance) genuinely collapses when you bin by intensity.** Not an
artifact: `ustd` is raw in every row, so A is identical (3.60%) in both all-shot rows
and dies only in the bands. `method_variance` finds the beam stop as a LOW-variance
island, and the contrast it keys on is largely *shot-to-shot intensity variation* — an
illuminated pixel's variance is inflated by the run's 3-orders-of-magnitude intensity
swing, a shadowed pixel's is not. Bin by intensity and that swing is removed **by
construction**, so the bulk stops being high-variance and the island stops standing
out. Separation of the target from the bulk (median z of `log10(ustd)`, cut at
`z < -7`): **-3.27 all / -2.65 bright / -1.29 dim**; pixels reaching the cut:
**8428 / 631 / 0**. Frame count is not the cause: 201 vs 800 frames alone would widen
the log10-MAD from 0.084 to only **0.086**, but observed is **0.113** (bright) and
**0.277** (dim).

The dim band is the worse of the two and is near-dark (mean 0.038 ADU/px): no photons
means no shot noise anywhere, so nothing separates shadow from signal. The bright band
keeps partial contrast (shot noise ~ sqrt(I) still lifts lit pixels) but not enough to
reach the cut.

**Result 2 — C (black-hat) is flat in intensity; its precision comes from the fact
that `_dropped` means BEAM-OFF.** C barely moves across the bands (0.177 all-raw ->
0.210 dim -> 0.138 bright); the whole 0.905 -> ~0.18 gap is the *estimator*, not the
shots. See the section below for what `_dropped` actually is — this is a live finding
about the **published** pipeline, not about binning.

**Verdict: don't bin by intensity for these detectors** — the mixed-intensity run is a
feature, not a bug. If a future detector should be intensity-specific it must key on
something other than variance contrast (the `p*` maps in `data/features/` are kept for
that). Note both band masks stay ~99% precise: they find *less*, not *wrong*, so the
bands are safe but useless as a standalone mask.

Figures: `images/intensity_bands/pipeline_by_band_run0475.png` (A / C / combo /
agreement per column), `images/intensity_bands/features_by_band_run0475.png`.

## What `_dropped` means: BEAM-OFF shots — and `method_blackhat` is a dark-frame detector

Established while investigating Result 2 above, because an earlier draft of this file
guessed "photon dropping" and was **wrong**. `Sums/jungfrau1M_alcove_calib_dropped`
is the sum over the run's **dropped shots** — the X-ray-OFF events — not a photon
threshold, and not a droplet/photon-counting sum.

Evidence (run 475, 3201 events, 3177 x-ray on, **24 x-ray off**):

- **Frame count is exactly 24, proved.** By Cauchy-Schwarz a sum over `n` frames obeys
  `D^2 <= n*Q` (`D`=`_dropped`, `Q`=`_dropped_square`), so `max(D^2/Q)` is a hard lower
  bound on `n`, attained exactly when a pixel is constant across the frames. Measured:
  `max(D^2/Q) = 24.00` (99.999th pct also 24.0). The run has exactly 24 x-ray-off shots.
- **The magnitude is dark, not lit.** Per frame, mean|.| on live pixels: `_dropped`/24 =
  **0.056**, a real beam-off frame = 0.069, a LIT frame (`calib`/3177) = **0.118**.
- **The resulting map looks dark**: `mean_run0475_asm.npy` (= `_dropped`/N) shows ASIC
  structure and dead specks and **no diffraction rings**, unlike any lit mean.
- It also explains `methods.py`'s own note that "the baseline `_dropped` RMS is flat" —
  a beam-off RMS *is* flat, having no illumination structure to carry.

**Caveat — not reproduced bit-exactly.** Summing the 22 reachable x-ray-off frames with
`det.calib()` gives only corr **0.42** with `_dropped` on live pixels (0.957 expected for
22/24 shared frames), and median ratio 0.667 vs the expected 1.091. The frame *count*,
*darkness* and *character* all match, but the exact calibration path does not — likely a
common-mode / pedestal-version difference in the producer. Do not assume you can
regenerate `_dropped` from `det.calib()` and get the same array.

**Why this matters for the pipeline.** `method_blackhat` is fed `mean` from
`pixel_stats.py` = `_dropped`/N, so **C is not looking at the beam at all** — it is a
*dark-frame bad-pixel finder*. That is the whole source of its T-prec 0.905: on a
beam-off map there is no diffraction structure to trip on, so "darker than its local
surroundings" means a genuinely dead/bad pixel. Swap in a lit mean (`umean`) and the
rings and beam-stop edges become local dark features, and precision falls to 0.177.

Consequences:
- **Intensity-banding C is meaningless by construction** — every band mean is built from
  lit shots, so no band can ever recover the 0.905. That, not any intensity physics, is
  why C is flat across the bands.
- Anything that re-derives the black-hat's mean **must keep it beam-off**, or C's
  precision goes with it.
- **Latent normalization inconsistency (cosmetic, not a bug):** `pixel_stats.py` divides
  the 24-frame `_dropped` sums by `N` = 3201 (all events), so its `mean`/`rms` are the
  dark maps scaled by ~24/3201 (~133x too small). Harmless where it is used — every
  consumer thresholds on a robust MAD, and a global scale cancels — but the absolute
  ADU values in `mean_run0475*.npy` are meaningless, and `rms` there is ~proportional to
  the dark RMS rather than equal to it. Don't read those maps as physical ADU.
