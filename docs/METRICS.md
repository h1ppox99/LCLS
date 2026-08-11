# Label-free mask metrics

How to judge a Jungfrau mask when nobody has drawn the answer.

In development we have a human reference for runs 389 and 475, so a mask is
scored by IoU against it. In production that reference does not exist: a run
arrives, the pipeline emits a mask, and the only thing available is the data the
mask was built from. This document is the study of what can be measured under
that constraint, what each measurement is actually worth, and how much of it
survived a check against the human mask.

Code: `automask/unsupervised/`. Validation driver:
`automask/studies/metric_validation.py`. Production score card:
`python -m automask.unsupervised.report`.

## The two failure modes any label-free metric must survive

**1. Degenerate optima.** Almost every "is this mask self-consistent?" statistic
is maximised by masking nothing (perfectly reproducible) or masking everything
(perfectly isotropic, perfectly stationary). A metric that only rewards
consistency is a metric that rewards deleting the detector. Every metric below
is therefore either bounded to a fixed null (reproducibility → 1.0) or reported
**against a size-matched control**: a random mask that removes exactly as many
non-floor pixels as the candidate. The question is never "did the residual image
improve?" but "did it improve *more than dropping that many pixels at random*?".

**2. Mask metrics vs procedure metrics.** A frozen array is trivially stable
under resampling — it does not depend on the data. Reproducibility is a property
of the *procedure*, so `Candidate` carries a `make(sample)` callable, not an
array, and `Candidate.static=True` (the human mask, the lab mask) makes every
tier-1 metric return NaN instead of a meaningless 1.0. `MetricSpec.needs_maker`
enforces this.

## The resampling substrate

Everything in tiers 1 and 3 needs the same thing: per-pixel statistics on
*subsets* of shots. Recomputing those from XTC per subset is unaffordable, and
widening `ShotSelection` with a fold index would rekey the content hash and
invalidate the whole warm feature cache.

`unsupervised/folds.py` instead makes one XTC pass and accumulates the additive
per-pixel moments `(n, Σx, Σx²)` on **two** K = 10 fold axes (800 lit shots per
run, 80 per fold). Any subset of either axis then sums to exact mean/std for
that subset, in numpy. Two axes, because the splits below need opposite things
from a fold:

- **time blocks** — contiguous runs of shots in stream order;
- **dealt folds** — shots handed out round-robin, one to each fold in turn.

From the one cache:

- **split-half by shot parity** (dealt folds 0,2,4,… vs 1,3,5,…) — pure sampling
  noise, condition-balanced;
- **split-half by contiguous halves** (time blocks 0-4 vs 5-9) — sampling noise
  *plus* within-run drift and any change in condition mix;
- **cross-fold χ²** over all 10 **time blocks** — the tier-3 stationarity test,
  which needs chronology: dealt folds each span the whole run, so the same χ²
  over them would test overdispersion rather than stationarity.

`stab_alt − stab_time` is therefore a direct read of how much of the
irreproducibility is non-stationarity rather than counting statistics.

> ⚠ **The dealt axis is new and the tier-1 numbers below predate it.**
> Previously this module built contiguous blocks only, and `stab_alt`
> interleaved whole 80-shot *blocks*. That balanced nothing: the CC/VCC branch
> clusters in runs of a thousand-odd shots
> (`studies/loss_identification.py::exp_a2_interchangeable`, runs 378/389/396),
> so over ten folds the VCC-open fraction spanned 0.425 / 0.597 / 0.175, and on
> two of the three runs the block-alternating split came out *worse* balanced
> than the contiguous-halves split it existed to improve on. Dealt shot by shot,
> the spread falls to 0.025 / 0.016 / 0.050. **Every `stab_alt` figure in this
> document was measured under the old layout and needs re-running at SLAC**; the
> cache filename carries a layout tag so stale files rebuild rather than load.

One subtlety: resampled samples replace only `mean`/`std`, never the
geometry floor or the calibration constants, and `mean` keeps its exact zero
support. Otherwise the assembled canvas itself would flicker between draws and
tier-1 would measure the canvas instead of the detectors.

## Tier 0 — arithmetic sanity

Assumes only that a mask is a mask. No physics, no resampling.

| metric | statement | honest verdict |
|---|---|---|
| `plausible_frac` | masked fraction is inside 2–25% | a prior, not evidence; rules out collapse and runaway |
| `floor_containment` | the mask contains the geometry floor | a bug detector |
| `compactness` | fraction of added pixels in ≥4-px connected components | the only tier-0 metric with real discriminating power |

`compactness` is not circular even though masks come from intensity statistics:
it encodes a prior on *defect morphology* (dead ASICs, hot columns, gaps are
spatially coherent; single flagged pixels scattered uniformly are threshold
noise), which is independent of the intensity evidence the detectors threshold on.

## Tier 1 — is the estimator reproducible?

Assumes the mask should not depend on *which* shots you happened to record, or on
the third decimal of a hyperparameter.

- `stab_noise` — perturb `(mean, std)` by their analytic sampling laws
  (`mean ~ N(μ, σ²/n)`, `std ~ σ(1 + N(0,1)/√(2(n−1)))`), rebuild, IoU. Cheap,
  but assumes per-pixel independence, which common mode violates — hence:
- `stab_alt` — rebuild on even vs odd SHOTS (dealt folds), IoU between the two
  masks. Both sides carry the same mix of experimental conditions.
- `stab_time` — rebuild on the first vs second half of the run.
- `stab_hyper` — jitter every float knob by lognormal(0, 0.10), rebuild, IoU.
  Integers stay integers ≥ 1; strings and hardware constants
  (`FIXED_KNOBS = {asic, defectiveness_scale}`) are never touched.

All four are mean IoU in [0, 1] with null = 1.0. **Necessary, not sufficient**:
they cannot tell a stable-and-correct mask from a stable-and-wrong one.

## Tier 2 — azimuthal isotropy

First tier that uses physics. The sample is isotropic powder-like scattering, so
the true signal is a function of q alone: after masking, intensity within a ring
should vary across azimuthal sectors only by counting noise. Excess azimuthal
scatter is unmasked detector structure.

The statistic is an **effect size**, not an ANOVA F:

```
excess = sqrt(max(Var_between_sectors − σ̂²·mean(1/n_s), 0)) / |μ_ring|
```

F-tests grow without bound as pixels per cell grows, which would make "mask more
pixels" and "improve isotropy" indistinguishable. The excess is scale-free —
verified in `tests/test_unsupervised.py` to move by <1.5× across 50/200/800
pixels per cell.

Its **measured null floor** is ~0.19% of the ring mean at 12 sectors × 200
px/cell — not zero, because `Var_between` is itself estimated from 12 sector
means and lands above its expectation half the time. A 5% one-sector anomaly
reads 1.47%, so the working dynamic range is ≈8×. Effects below the floor are
not detectable, which is why the raw value is always reported next to its control:

- `azim_excess` — the raw effect size (lower is better).
- `azim_gain` — control excess minus candidate excess.
- `azim_winrate` — paired per-ring sign test against the size-matched control,
  null = 0.5.

Ring edges, sector edges and winsorisation thresholds are all fixed from the
**floor-only** pixel set, so no candidate is scored on a partition it chose.

### The frozen noise reference

`σ̂²` and `μ_ring` are also fixed from the floor-only set (`ring_reference`), not
read off the candidate's own survivors. Estimating them per candidate leaves a
milder form of the defect that sinks Welch's F: a mask that removes noisy pixels
shrinks `σ̂²`, shrinking the term subtracted from its own numerator, so the
statistic **rises when the mask works**. On a synthetic ring stack with one
sector at 8× the per-pixel variance and a real 4% anisotropy elsewhere, masking
the noisy sector moves the candidate-local excess *up* by +1.28 points, in 5 of 6
draws. With the reference frozen it moves down, in 6 of 6.

Freezing the pixel *set* is not sufficient on its own, and this is the part worth
knowing. The floor removes geometry and calib defects but **not** the intensity
defects the mask exists to find, so those pixels are still in the reference set.
A dof-weighted pool over it returns σ̂² = 644 against a true 100 on the field
above, and the inflated floor then swallows the real 4% anisotropy whole — every
candidate reads exactly `0.000%` and 67% of rings clip to zero. `ring_reference`
therefore takes the **median of the per-cell variances across the ring's
sectors**, which returns 100.2 on the same field. On a ring with no
variance-inflating defect the two reductions agree to three decimals, so the
change is a no-op on clean rings. All four properties are asserted in
`tests/test_unsupervised.py`.

`Var_between` stays centred on the candidate's own ring mean; the frozen `μ` is
the unit, not the centre. Centring on a foreign mean would fold a radial offset
`(μ_cand − μ_ref)²` into an azimuthal statistic.

## Tier 3 — event-axis stationarity

Strongest assumption: a healthy pixel's response is stationary over the run.
Per pixel, an inverse-variance-weighted χ² across the 10 fold means; reject at
`P_ANOM = 1e-3`.

Global beam drift would reject essentially every pixel, so each fold is divided
by a single global gain `g_k = median(mean_k / mean_all)` over live pixels. A
test injects 30% drift and confirms the rejection rate stays within 10× nominal;
without the gain it goes to ~1 and the metric measures the machine, not the
detector.

- `event_leak` — fraction of *surviving testable* pixels with p < 1e-3.
- `event_gain` — control leak minus candidate leak.

Two limitations stated up front: the test is **blind to constant and dead
pixels** by construction (a pixel that always reads 0 is perfectly stationary),
and it consumes the same second moments as the `variance` detector, so agreement
between them is weaker evidence than tier-2 agreement.

## Validation against the human mask

A surrogate is worth exactly its rank correlation with the truth. 21 candidate
masks — the production recipe, single-detector ablations, a k-sweep of the
variance threshold, TV extremes, morphological over/under-masking, two random
controls, the human mask and the lab mask — were scored on runs 389 and 475 with
every label-free metric, and each metric's Spearman ρ taken against the human
IoU and against **residual IoU** (agreement on the pixels the geometry floor does
not already explain, which is the part a masker actually has to get right).

Spearman, not Pearson: the metrics live on incomparable scales, and all we ask of
a surrogate is that it order masks correctly. Pooling is done on within-run
ranks, since absolute levels differ between runs.

### Rank correlations (21 candidates × 2 runs)

> ⚠ **The tier-2 rows below predate the frozen noise reference** and have not been
> re-measured against it. Re-running `studies/metric_validation.py` needs the
> archived frozen benchmark inputs, which are not rebuilt by the production
> package, so the change is verified only on synthetic fields
> where the truth is known by construction. The synthetic evidence says the clean
> case is unchanged to three decimals and only rings with a variance-inflating
> defect move — but `azim_excess`, `azim_gain` and `azim_winrate` should be
> treated as **unvalidated on real data** until the panel is re-run at SLAC.

| metric | tier | ρ(IoU) | p | ρ(residual IoU) | ρ within top half | per-run |
|---|---|---|---|---|---|---|
| `plausible_frac` | 0 | **+0.58** | 0.0001 | +0.31 | — | +0.56 / +0.61 |
| `floor_containment` | 0 | +0.41 | 0.007 | +0.15 | — | +0.41 / +0.41 |
| `compactness` | 0 | +0.03 | 0.84 | −0.22 | −0.33 | +0.27 / −0.09 ⚠ |
| `stab_alt` | 1 | +0.15 | 0.37 | −0.08 | +0.03 | +0.18 / +0.12 |
| `stab_time` | 1 | +0.11 | 0.51 | −0.14 | −0.04 | +0.15 / +0.08 |
| `stab_noise` | 1 | +0.06 | 0.74 | −0.16 | −0.54 | +0.05 / +0.05 |
| `stab_hyper` | 1 | +0.01 | 0.95 | −0.16 | +0.35 | +0.12 / −0.09 ⚠ |
| `azim_excess` | 2 | +0.41 | 0.008 | **+0.62** | −0.00 | +0.18 / +0.63 |
| `azim_gain` | 2 | +0.38 | 0.013 | **+0.64** | +0.22 | +0.28 / +0.48 |
| `azim_winrate` | 2 | +0.39 | 0.010 | **+0.70** | +0.44 | +0.28 / +0.52 |
| `event_gain` | 3 | +0.29 | 0.060 | +0.50 | −0.42 | +0.15 / +0.44 |
| `event_leak` | 3 | +0.30 | 0.058 | +0.49 | −0.43 | +0.09 / +0.49 |

Composite (tier-mean of within-run z-scores): ρ = **+0.72** (p 0.0003) on run 475,
**+0.40** (p 0.07) on run 389.

### What actually works

**`azim_winrate` is the load-bearing metric, and it is a gate, not a score.** On
real data it separates masks that found genuine detector structure from masks
that did not, sharply and with a correctly calibrated null:

| candidate class | run 389 | run 475 |
|---|---|---|
| geometry floor only, or a detector that added nothing | 0.00 | 0.00 |
| size-matched random pixels / random blobs | 0.45–0.50 | 0.51 |
| the lab's `cmask` (1.9% masked) | 0.14 | 0.27 |
| every mask containing real structure | 0.89–0.90 | 0.61–0.72 |

`random_px` scoring 0.500 on run 389 and 0.514 on run 475 is the strongest single
piece of evidence in this study: the null of a paired per-ring sign test against a
size-matched control comes out at exactly 0.5 on real detector data, so a win rate
above it is not an artefact of masking more pixels.

`azim_excess` tells the same story on an absolute scale — 0.177 for a mask that
adds nothing, 0.062–0.074 once real structure is removed on run 389 — a 2.5×
separation.

**But tier 2 saturates immediately after that gate.** Among the run-389 masks
that pass it, `azim_excess` spans 0.062–0.074 while true IoU spans 0.55–0.91, and
the *lowest* excess (0.062) belongs to the *worst* of them (`weighted_sum`, IoU
0.547). That is why `ρ within top half` collapses to ≈0 for every tier-2 metric
and why the composite ranks `weighted_sum` first on run 389. Isotropy answers
"was the gross structure removed?", not "was exactly the right set removed?" —
past a point, more masking always looks marginally more isotropic, and the
size-matched control only partly cancels it.

Pairing the gate with `plausible_frac` recovers most of what a single metric
cannot do. `azim_winrate > 0.55` **and** masked fraction in 2–25% admits, on both
runs, only masks with true IoU ≥ 0.61 — worst case rises from 0.10 (`lab_cmask`)
to 0.61, and both over-maskers (`dilate_r5` at 33%, `weighted_sum` at 27%) are
rejected on the fraction, not on isotropy. Within the surviving 0.61–0.91 band no
label-free metric here orders candidates reliably.

**Tier 1 does not rank, but it screens.** All four stability metrics have
ρ ≈ 0 against IoU, exactly as the degeneracy argument predicts — and the data
show the degeneracy directly: `floor_only` and `det_houg` score *exactly* 1.000 on
`stab_alt`, `stab_time`, `stab_noise` and `stab_hyper`, because a mask that adds
nothing cannot be perturbed. Read as a one-sided screen it still earns its place:
`stab_hyper` is 0.97–0.99 for every sound pipeline and 0.86 / 0.91 for
`weighted_sum`, the one candidate whose combiner is genuinely fragile. Low
stability is real evidence of a bad pipeline; high stability is no evidence of a
good one.

**Tier 3 is nearly inert here.** `event_leak` is constant at 0.004 across all 21
candidates on run 389 (zero discriminating power) and takes just two values on
run 475 — 0.007 for masks that added nothing, 0.006 for masks that did. The
ordering is correct but the dynamic range is one part in seven, and the top-half
ρ is negative. The measured leak also sits 4–6× above the nominal 1e-3 χ²
rate, i.e. real pixels are neither Poisson nor independent, so the absolute level
is not interpretable and only the control-relative `event_gain` is reported with a
verdict. Tier 3 stays in the package as a diagnostic (it is the only metric that
would catch a pixel that drifts without changing its run-averaged variance), not
as a selection criterion.

**Two tier-0 metrics are better than they look, for the wrong reason.**
`plausible_frac` (ρ +0.58) and `floor_containment` (ρ +0.41) are the highest
ρ(IoU) in the table, but they contain no information about *which* pixels were
masked — they score highly because the candidate panel deliberately includes
gross over- and under-maskers, and IoU against a 13–15% reference is largely
determined by matching that fraction. On residual IoU, the target that measures
actual masking skill, they fall to +0.31 and +0.15 while tier 2 rises to +0.62 to
+0.70. Prior-based sanity checks should be read as constraints, never as quality.

`compactness` is the one metric that failed outright: ρ +0.03, and its sign flips
between runs. The morphology prior is sound but the production masks are dominated
by large coherent regions in every candidate, so the statistic has almost no
variance to work with.

### Consequences for the project

1. **Ranking a k-sweep without labels is not currently possible.** On run 475 the
   variance threshold is genuinely selective (IoU 0.82 → 0.94 → 0.74 as k goes
   2.0 → 4.5 → 6.0) and no label-free metric resolves it; the composite puts
   `var_k2.5` (IoU 0.865) ahead of `production` (0.956). Hyperparameters must
   still be set on a labelled run and transferred.
2. **Accepting or rejecting a mask in production *is* possible.** The gate above
   catches the failures that matter — a pipeline that silently degenerates to the
   geometry floor, one whose detector stopped firing, one that ran away to a third
   of the canvas — all with a calibrated null and no reference mask.
3. **Residual IoU is the right development target.** Every metric looks better
   against it, and full IoU on a canvas whose floor is already 10% rewards
   getting the floor right, which is not the problem.
4. Run 389 is systematically the weaker validation run (its per-run ρ is roughly
   half run 475's on every tier-2/3 metric). It is the truncated run, with 6 471
   decoded events and no dedicated human reference, so it is scored against the
   run-475 mask. Treat run-475 numbers as the informative ones.

## How to read a production mask

`python -m automask.unsupervised.report` prints one card: every metric next to
the value it takes when its own null is true, and a coarse `ok / WEAK / n-a`
flag. It deliberately does not collapse to a single number — the useful output is
*where* a mask is weak (unreproducible? isotropy not improved? still leaking
non-stationary pixels?), and a single score invites over-trust it has not earned.

The production recipe on run 475 (13.31% masked) reads:

```
compactness        1.0000  ok     stab_noise    0.9975  ok     azim_excess  0.1630  --
floor_containment  1.0000  ok     stab_time     0.9634  ok     azim_gain    0.0142  ok
plausible_frac     1.0000  ok     stab_hyper    0.9876  ok     azim_winrate 0.6943  ok
stab_alt           0.9928  ok                                  event_leak   0.0063  --
                                                               event_gain   0.0003  ok
```

Note `stab_time` 0.963 against `stab_alt` 0.993: the balanced split is tighter
than the chronological one, so roughly 3% of the mask is not reproducible across
the run rather than across shots. That difference is the intended read of having
both — pure sampling noise costs 0.7%, non-stationarity costs the other 3%.

⚠ Both figures are from the old block-interleaved layout. Under it `stab_alt`
was not condition-balanced, so part of what is attributed to sampling noise here
is branch composition; the 3% gap is an upper bound on drift, not a measurement
of it. Re-run before quoting.
