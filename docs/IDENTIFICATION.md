# Masking as identification

What a bad pixel is, if you are willing to write down a forward model — and
which parts of that story survive being measured.

Code: `automask/identification/`. Driver: `automask/studies/loss_identification.py`.
Tests: `automask/tests/test_identification.py`.

## Running it at SLAC

Three stages, in this order, because each one can make the next pointless. One
SLURM script drives all of them:

```bash
sbatch automask/scripts/identification.sbatch survey 475   # seconds, no frames
sbatch automask/scripts/identification.sbatch gate   475   # 1 XTC pass per axis
sbatch automask/scripts/identification.sbatch frames 475   # A1, A3, M5, P3
```

| stage | cost | what it answers | stop if |
|---|---|---|---|
| `survey` | seconds | which condition axes this run has, and how many groups each resolves into | `delay` is absent — later stages then work off a weaker axis, and you want to know before paying for an XTC pass |
| `gate` | ~800 frames per axis | **M2R: does any axis move the sample signal at all?** | **REFUTED.** Then `tau` cannot be separated from `S` on this run and `frames` cannot tell you anything about it |
| `frames` | frozen inputs + fold cache | A1, A3, M5, P3 | — |

`gate` is its own job on purpose: it is the cheapest decisive measurement here,
and a negative result makes the rest of the study inapplicable to this
experiment no matter how well the estimators behave. Everything writes to
`automask/outputs/` and every XTC pass is cached, so only the first run of each
stage pays for it. Edit `ENVP`/`PSDM` in `psana_env.sh` for the cluster.

Locally, without psana (see *What is reachable from the local XTC*):

```bash
python -m automask.studies.loss_identification                       # bench claims
python -m automask.studies.loss_identification --claims A2 --runs 378 389 396
python -m automask.io.xtc_raw                                        # decode + validate
python -m automask.tests.test_identification
```

### Which files matter

Most of this branch is library and prose. To *run* experiments you touch three
things: `scripts/identification.sbatch` (the entry point),
`studies/loss_identification.py` (the claim register — every experiment lives
here, one function each), and `identification/` (the estimators they call).
`io/xtc_raw.py` is only used off-cluster, where psana is unavailable.

## The model

```
x_it = tau_i · F_t · A_i · S(q_i; theta_c(t))  +  F_t · J_i  +  eps_it
M    = {i : tau_i ≠ 1}  ∪  {i : J_i ≠ 0}
```

| symbol | meaning | status |
|---|---|---|
| `F_t` | incident flux, shot `t` | measured (downstream monitors) |
| `A_i = Ω_i P(χ_i, q_i)` | solid angle × polarization | known from calibration |
| `S(q; θ_c)` | isotropic sample scattering | unknown, **condition-dependent** |
| `tau_i ∈ [0,1]` | transmission — shadows | unknown, static, **multiplicative** |
| `J_i ≥ 0` | parasitic scattering — streaks | unknown, static, **additive** |
| `eps_it` | Poisson + read | variance estimable |

Detector defects are out of scope — the calib + geometry floor already removes
them. What is in scope is the structure the intensity statistics currently find
by thresholding, restated as: `tau` and `J` are parameters, and masking is
testing whether they depart from their null.

The three components compose **differently**, and that is the whole content of
the proposal. `tau` multiplies the condition-dependent signal; `J` is added to
it and does not depend on the condition. So differencing two conditions kills
`J` identically and leaves `tau` — which is why there are two stages and why
they have an order.

## Why this document is a list of things that could be false

The model above is a hypothesis with a lot of structure in it, and structure is
what makes an estimator efficient when it is right and confidently wrong when it
is not. Every load-bearing assertion is therefore registered as a `Claim` with a
written-down **falsifier**: the outcome that would kill it. A claim whose
falsifier cannot be stated does not go in the register.

The register splits in two, and the split turned out to be the most useful thing
about building it:

- **`bench` claims are about an ESTIMATOR.** True or false as mathematics, on a
  field where `tau`, `J`, `A` and `S` are known by construction
  (`identification/forward.py`). No beamtime makes a biased estimator unbiased,
  so these are settled now. **13 of them, all run below.**
- **`run` claims are about THIS EXPERIMENT.** Whether conditions move the
  signal, whether the sample is anisotropic, whether real artifacts are
  multiplicative or additive. No simulation answers these. **6 of them: one
  answered from the local XTC (A2, supported), five needing the cluster — and
  M2R among those is the gate the others depend on.**

## Results

Verdicts below are from the runs available in this checkout. **needs frames**
means the experiment is implemented and blocked on calibrated Jungfrau frames,
which the local XTC contains but which cannot be decoded here (see *What is
reachable from the local XTC* below).

| id | claim | verdict |
|---|---|---|
| A1 | conditions drift within a run, so folds must be alternating | needs frames |
| A2 | consecutive shots share conditions, so deal folds round-robin | **supported** (runs 378/389/396) |
| A3 | the sample is *not* truly isotropic | needs frames |
| A4 | a median ring reduction beats a pooled mean here | **supported** |
| M1 | only *within-ring* structure is identifiable | **supported** |
| M2 | identifying power is `Var_c(γ_{q,c})` and it caps the method | **supported** |
| M2R | some condition axis in THIS experiment supplies that power | **the gate — run first** |
| M7 | given the contrast, stage-1 error is counting noise | **refuted** |
| M3 | the contrast cancels `J` exactly | **supported** |
| M4 | the stage order is forced, not chosen | **supported** |
| M5 | real artifacts split into multiplicative and additive classes | needs frames |
| M6 | uncorrected polarization reads as an artifact | **supported** |
| P1 | a perimeter prior suppresses streaks | **supported** |
| P2 | Radon/Hough dominates a region prior on streaks | **supported** |
| P3 | the artifact classes are disjoint, so the union is exact | needs frames |
| H1 | polarization is *exactly* `m = ±2`, removable analytically | **mixed** |
| H2 | a sharp shadow is broadband to `m ~ 2π/Δχ` | **supported** |
| H3 | the harmonic cut is computable on this geometry | **mixed** |
| O1 | threshold-then-regularize is a lossy approximation to the joint MAP | **supported** |

### The one that failed, and what it changes

**M7 is the result worth reading first**, because it was not on the original
list — it is what the M2 experiment turned up on the way.

The notes say identifying power is `Var_c(γ_{q,c})` and that it caps the
method. M2 confirms the scaling: recovery error falls with contrast at a log-log
slope of **−0.44** against `Var_c(γ)`, where the model predicts −0.50, and the
measured power correctly reads zero when the conditions are made identical. So
the quantity is real, it is measurable, and it controls the error.

It is not, however, the binding constraint. Reducing each ring to a single
number — the natural reading of "divide by the ring mean" — leaves the ring's own
**radial gradient** inside the estimate. `q` varies across a ring bin, so
`S_c − S_c'` varies across it too, and a pixel at the inner edge is charged for
being at the inner edge. Measured:

```
per-ring constant denominator: empirical error is 9.7x the propagated counting noise
robust linear-in-q denominator:                    1.1x
```

The estimator is off by an order of magnitude from its own error bars, and the
excess vanishes only at the one ring where `S_c − S_c'` happens to be at an
extremum, which is exactly the signature of a slope. Fitting `a + b·q` inside
each ring (robustly, so the artifacts do not set their own baseline) brings it to
1.1× — noise-limited.

This is the same systematic `unsupervised/azimuthal.py::gradient_leakage`
documents for the second moment, appearing here in the first. **Consequence:
`twoway.stage1_transmission` and `stage2_parasitic` default to
`detrend="linear"`, and any "divide by the ring" step elsewhere should be read
as suspect until its gradient leakage is measured.** Nothing downstream of
stage 1 is interpretable without it: M6 and M4 both read as null effects under
the constant denominator purely because the leakage swamped them.

### A2 on real data: confirmed, and it says how to deal the folds

The note assumes consecutive shots share experimental conditions and concludes
"→ alternating folds". Measured on runs 378, 389 and 396, **both halves hold**,
and the measurement turns the qualitative rule into a quantitative one.

The CC/VCC branch does not vary shot to shot — it comes in long blocks, which is
precisely what makes neighbouring shots equivalent:

| run | adjacent shots agree | two random shots agree | mean run length |
|---|---|---|---|
| 378 | **99.5%** | 80.2% | 200 shots |
| 389 | **99.7%** | 81.1% | 336 shots |
| 396 | **98.6%** | 82.8% | 73 shots |

A shot and its neighbour are on the same branch essentially always; two shots
picked at random from the run only ~80% of the time. So dealing consecutive
shots into different folds hands each fold the same condition mix. The per-shot
flux monitor agrees (|acf| below the 2/√n band at every lag on 378 and 396; run
389 exceeds it at lag 10 only).

Dealing round-robin — one image to fold A, one to B, one to C, cycle — is the
scheme this supports, and it is decisively better than a contiguous split.
VCC-open fraction across 10 folds:

| run | dealt round-robin | contiguous blocks |
|---|---|---|
| 378 | 0.025 | **0.425** |
| 389 | 0.016 | **0.597** |
| 396 | 0.050 | **0.175** |

Contiguous folds differ in branch composition by up to 60 percentage points;
dealt folds by 2–5.

**The one way dealing could fail, and why it does not here.** Round-robin into
`K` folds is biased only if the condition's period is commensurate with `K` —
if the branch flipped every 10 shots, fold 0 would get one branch and fold 5 the
other. That is the opposite regime from this data: block lengths are 7×, 20× and
34× the fold count, so every fold samples every block. The rule is
`block length ≫ K`, and it is worth re-checking on any run whose branch pattern
looks different.

**This changed existing code.** `unsupervised/folds.py` used to build contiguous
blocks and then interleave whole *blocks* (`alternating()` = even folds vs odd
folds), which against 73–336-shot branch runs is close to arbitrary:

| run | `folds.alternating()` gap | `folds.halves()` gap |
|---|---|---|
| 378 | 0.098 | **0.003** |
| 389 | **0.027** | 0.211 |
| 396 | 0.150 | **0.015** |

On two of three runs the alternating split was *worse balanced* than the
contiguous-halves split it existed to improve on.

`folds.py` now accumulates **two** fold axes from the same single XTC pass:
contiguous time blocks, which `halves()` and the tier-3 stationarity χ² need
because those questions are about chronology; and dealt folds, which
`alternating()` now reads so that `stab_alt` compares two interchangeable
halves. The cache filename carries a layout tag, so stale caches rebuild rather
than load with the wrong meaning, and the cost is 2× cache size plus one extra
add per frame — the XTC pass, which is what actually costs, is still one.

**Consequence for `docs/METRICS.md`:** it reads the `stab_alt` − `stab_time` gap
(0.993 vs 0.963 on run 475) as "sampling noise vs within-run drift". Under the
old layout `stab_alt` was not condition-balanced, so part of that gap is branch
composition and the 3% is an upper bound on drift rather than a measurement of
it. Every tier-1 figure there was measured under the old layout and needs
re-running; the document now says so.

Two caveats on the numbers. Only one DAQ stream per run is present locally, so
"adjacent" means adjacent *within a stream*. Measured from the datagram
fiducials, that stride is exactly 5 beam shots — median fiducial step 15, beam
at 120 Hz against a 360 Hz fiducial clock, stream reading out at 24.0 Hz — so a
stream holds every 5th shot and true acquisition-order branch runs are **5×
longer** than the table says (1 000–1 700 shots, not 73–336). That only
strengthens the conclusion. And run 389's stream is truncated to 697 events
against the 40 003 of the full run, so its fold-imbalance figures are for that
subset.

### What holds

**M1 — identifiability.** On a field carrying both a compact shadow and a
ring-wide one, the estimator recovers the compact shadow at **100%** of its true
depth and the ring-wide shadow at **0%** (residual 0.08% of its depth, which is
the detrend's own curvature). The algebra says the two-way design is rank
deficient by exactly one per ring; an SVD of an explicit small design agrees
(deficiency 4 for 4 rings). This is not a limitation to work around — it is a
statement about what the data contain, and it should be checked against a real
reference mask before any of this is built (that check is the run-side of M1,
`identifiable_fraction`).

**M3 — the contrast cancels `J`.** Raising the streak amplitude from 0 to 0.6
moves the recovered transmission off-streak by at most `8×10⁻⁶`, against an
estimator noise of `5.5×10⁻³` — a factor of 660. On a noiseless field the test
asserts exact equality to `10⁻⁹`. The cancellation is algebraic, as claimed.

**M4 — the order is forced.** Run forward, the shadow reads 67σ in stage 1 and
the streak 14σ in stage 2. Run backwards, the "J-first" residual sees the
**shadow at 242σ against the streak at 69σ** — it is 3.5× more sensitive to the
artifact of the wrong class, which it would then hand to the curve prior. The
ordering is not a preference.

**M6 — polarization.** Dropping `P` from `A` multiplies the clean-pixel spread
by 7.6× and induces a spurious transmission swing of 0.27 — **60% of a real
shadow's depth, and 50× the estimator's own noise.** Uncorrected polarization
does not degrade this estimator, it invents artifacts in it.

**A4 — robust ring reduction.** Under contamination by a 45%-deep shadow:

| contamination | median RMSE | mean RMSE | trimmed RMSE |
|---|---|---|---|
| 0% | 0.500 | **0.392** | 0.427 |
| 2% | 0.550 | 0.962 | **0.476** |
| 5% | **0.725** | 2.259 | 0.723 |
| 10% | **1.206** | 4.521 | 1.284 |
| 20% | **2.599** | 8.983 | 2.979 |

The median costs 1.28× the mean's RMSE on a clean ring and beats it by 3.5× at
5% contamination. A 25%-trimmed mean tracks it closely — better at 2%, slightly
worse at 20%. The choice is justified, but "median" is not special: anything
robust does, and the gain over a plain mean only becomes large once a ring is
more than a couple of percent contaminated.

**P1/P2 — one prior is wrong for two of the three classes.** On an evidence
field carrying a compact blob and a streak of *equal total evidence*, the
perimeter MAP's two recalls move in **opposite directions** with λ:

```
lambda   blob recall   streak recall   noise kept
  0.05         0.609           0.581       0.990
  0.79         0.672           0.437       0.938
  2.00         0.806           0.027       0.240      <- best for the blob
  3.17         0.043           0.000       0.021
```

At the λ a user would pick — the one that best recovers the blob — the streak is
at 3%. The curve prior on the *same evidence field* gets **0.69 recall keeping
0.00 of the noise**, and at matched false-positive rate beats the perimeter MAP
0.86 to 0.45. This is the formal reason `hough_lines` has to be a separate
detector, made numerical.

**O1 — the greedy pipeline does lose something, but not much.** Same evidence
field, two ways of using it: the union of per-class MAPs on the continuous LLR
scores **IoU 0.771**; threshold-then-morphology scores **0.723**. The joint
model wins, by 0.048 IoU. That is a real margin and a small one — worth knowing
before rebuilding a pipeline around it. (The perimeter MAP *alone* scores 0.469,
consistent with P1: the gain comes from having the right prior per class, not
from being a MAP.)

### The harmonic cut: mixed, in an instructive way

**H1 — "exactly `m = ±2`" is true in the wrong domain.** `P = 1 − p sin²2θ cos²χ`
and `cos²χ = (1 + cos 2χ)/2`, so in the LINEAR domain the polarization factor is
exactly `m = 0` plus `m = 2` — measured leakage above `m = 2` is `< 10⁻³⁰`,
i.e. numerically zero. But the cut is proposed on `log S`, and `log(a − b cos 2χ)`
is not band-limited:

| 2θ | power above m=2, linear | power above m=2, log | m4/m2 amplitude |
|---|---|---|---|
| 5° | 4e-30 | 9.1e-07 | 0.0010 |
| 15° | 3e-30 | 7.5e-05 | 0.0087 |
| 28.9° | 3e-31 | 1.1e-03 | 0.0332 |

At this detector's maximum 2θ the log-domain residual is a 3.3% `m = 4`
component. Whether that matters depends on the shadow amplitude it competes
with, which is a run-side question — but "removable analytically" is only true
if the removal is done in the linear domain, before the log.

**H2 — the bandwidth rule holds, up to a constant.** A top-hat of angular width
`Δχ` loses half its power at `m₀ ≈ 0.44 · (2π/Δχ)` for narrow shadows (5°–20°)
and `≈ 0.25 ·(2π/Δχ)` for wide ones (45°–90°). So `2π/Δχ` is the right scaling
and `m₀` is the interpretable knob it is claimed to be — but the half-power point
sits a factor of 2–4 below `2π/Δχ`, so a cut set at the nominal value is more
aggressive than it looks.

**H3 — but the geometry does not allow the cut as written.** The beam sits near a
detector corner, so a ring is an arc, not a circle. Median arc coverage on a
representative canvas is **96°**, and full-circle harmonics are not orthogonal on
an arc: the design's condition number blows past any usable tolerance at
**m₀ = 0**. Rescaling each ring's covered arc to a full period restores
conditioning to **m₀ = 32**, which addresses angular features down to ~3°.

**Consequence: the harmonic cut must be defined on the arc, and `m₀` then means
"features larger than `W/m₀`", not `2π/m₀`.** That is still one interpretable
parameter, but it is a different one, and it varies ring to ring with `W`.
`harmonics.arc_phase` implements the rescaling; `angular_coverage` measures `W`
correctly (the obvious `max − min` overstates a wrapped arc as 359°, and even
`2π − largest gap` credits a two-piece ring with the space between the pieces).

## What is reachable from the local XTC

The checkout carries three single-stream XTC files — runs **378** (1 332 events),
**389** (697, truncated) and **396** (4 267) — and the full `calib/` tree. It does
not carry `automask/data/`, `hdf5/smalldata/`, or psana, which ships for
linux-64/osx-64 only and cannot be installed on this osx-arm64 machine.

That does not block everything, because XTC is a container of fixed
little-endian C structs and the per-shot scalars can be decoded directly.
`automask/io/xtc_raw.py` does exactly that — datagram walk, transition filtering,
and decoders for EVR codes, the analog-input shutter voltages, the IPM/diode
`fex` values and the gas detector — skipping the 2 MB detector payloads by
seeking rather than reading. `automask.io.scan_shots` prefers psana and falls
back to it, so `ShotSelection` and everything above it is unchanged.

Every decode is checked against something independent of it, because a struct
misread yields plausible numbers rather than an error:

| decode | check | result |
|---|---|---|
| EVR codes | code 137 is `'Beam On'` (DATA.md) | present on 99.2–99.4% of events; the code set is exactly the stock XPP one |
| analog input | ch02/ch03 are the CC/VCC shutters, at ~0 or ~5.05 V (DATA.md) | 16 channels, **100%** of shots on a rail |
| `IpmFexV1` | `sum` must equal its own four channels | max residual 4×10⁻⁷ (float32) |
| BLD source ids | branch ratio 2.41 (`lombpm`) / 1.67 (`diodeU`) vs ~0.98 upstream (DATA.md) | 0x22 and 0x28 identified consistently on all three runs |
| — confirmed by | `diodeU/channels[0]` median 0.0178, `lombpm/channels[2]` 0.2065 (DATA.md run 389) | 0.0189 and 0.2234 on the 697-event subset |

The two *upstream* monitors are deliberately left unnamed: their branch ratios
are 0.98 and 0.99, indistinguishable within any run's noise, and an earlier
version of the matcher assigned them oppositely on runs 389 and 396. Neither is
used by this project.

`python -m automask.io.xtc_raw` runs the decode and validation; three runs take
about 5 seconds. This unlocks **A2** and the condition-axis survey on real data.

**What it does not unlock.** A1, A3, M5 and P3 all need calibrated Jungfrau
frames. The raw frames are in the file (`typeid` 108, 2 097 208 bytes per event
= 2×512×1024 uint16 plus header) and `calib/` holds the pedestals, gain, offset,
rms, status and geometry. But turning raw into calibrated means reproducing
psana's pedestal/gain-stage/common-mode chain and parsing its geometry format,
and a subtly wrong calibration would produce artifacts indistinguishable from
the ones these experiments are trying to detect. That is a deliberate stopping
point, not an oversight — those four claims are better answered on a machine
with psana than on a reimplementation of it.

Also still psana-only: the `delay` condition axis, which lives in the EPICS
store. The survey reports it as unavailable rather than absent.

## What still needs a run, and what each would settle

These are implemented in `studies/loss_identification.py` and run with
`--runs 475`. Each needs the frozen inputs (`automask/data/`) and, for M5, one
extra XTC pass (`identification/conditions.py`).

**A1 — does the run drift?** Per-fold ring levels from the existing 10-fold
cache, gain-normalised, regressed on fold index. Reports the fraction of rings
drifting at |t| > 3 and the global gain swing. *Kills the alternating-fold
requirement if the drift is sampling noise.* Now more pointed than when it was
written: A2 shows the `stab_alt` − `stab_time` gap that `docs/METRICS.md`
attributes to drift has a branch-composition component, so this experiment is
what separates the two.

**A3 — is the sample really anisotropic?** Per-ring harmonic power on floor-only
pixels, fitted on the arc (H3), against the counting-noise null. Reports the
median azimuthal amplitude and how much of it sits at `m ≤ 4`. *Settles both the
motivation for robust reduction and the value of `m₀`* — the harmonic cut is
only useful if real anisotropy is genuinely low-order, and that number is
currently assumed, not known.

**M5 — are real artifacts multiplicative or additive?** Per pixel, a weighted
regression of `z̄^(c)` on the predicted ring signal across conditions: a slope
departing from 1 is a shadow, an intercept departing from 0 is a streak. Run on
reference-masked pixels against an unmasked control. *This is the experiment that
justifies or destroys the two-stage decomposition.* If masked pixels do not
separate into two populations, one stage would do.

**P3 — are the classes disjoint?** Connected components of the reference mask
beyond the floor, classified by elongation and fill ratio. *Kills the exactness
of `M = ∪ M_k` if a large fraction of mask pixels sits in components that are
simultaneously line-like and region-like.*

**Which condition axis?** M5 needs condition groups, and *which variable plays
the role of `c` is itself an open question.* `identification/conditions.py`
offers four, in decreasing order of expected identifying power:

| axis | what it is | cost |
|---|---|---|
| `delay` | the CO2 delay-scan motor (`epicsAll/delay`) — the axis the experiment was actually scanning | one EPICS pass; returns None if the run does not publish the PV |
| `branch` | CC vs VCC; different optics, possibly different `S` | free, already in `ShotMeta` |
| `time` | contiguous blocks — a proxy for `delay` in a scan run, a drift measure otherwise | free |
| `flux` | monitor quantiles | free, and it is the **negative control**: the model divides `F_t` out, so this should score ~zero identifying power unless the sample response is nonlinear |

`python -m automask.identification.conditions 475` labels a run under every
scheme without touching a frame, which is the cheap first thing to run: it says
how many shots each axis resolves into how many groups, and therefore whether
the expensive pass is worth making. Run M2's `identifying_power` on the result
before building anything — if no axis clears an SNR of 1, stage 1 cannot run on
this experiment and the rest is moot.

## Reading the numbers

`IdentifyingPower.snr` is a ratio of **variances** (`Var_c(γ)` excess over its
sampling floor), so it runs large — 3×10⁵ at a 2% signal contrast on the bench.
Take the square root for an amplitude ratio. What matters is the threshold: below
1 there is no usable contrast, and the bench correctly reads 0 when the
conditions are made identical.

The bench (`identification/forward.py`) is faithful where the estimators care —
corner geometry, so rings are arcs; the real 2θ range, so the polarization depth
is right; artifacts that compose multiplicatively and additively as the model
says; heteroscedastic noise; and error bars that carry their own χ² scatter. It
is crude where they do not — a ~200×200 canvas, a smooth powder profile with no
Bragg texture, and group means drawn from their sampling distribution rather than
accumulated shot by shot. No claim tested against it depends on canvas size, and
the two that would depend on real sample texture (A3, M5) are in the run list,
not simulated.

## Appendix — the proposal, as written

The source notes this document tests. Kept verbatim so each claim id above can
be traced back to the sentence it came from; where the two disagree, the
measurement above is the newer statement.


### Personal notes

A bad pixel is defined relative to a model of good behavior. Bad is also defined relative to the downstream usage : here $I(q)$.
This choice belongs to the agentic system or the experimentator.

**Assumptions** :
1. Evolution in time : conditions of experiment change as the experiment goes (#TODO evidence from data?). But we want a static mask for simplicity → use alternating folds to build folds with identical distribution of conditions.
2. Consecutive shots are interchangeable? *Yes* ( #TODO evidence from data?) → alternating folds
3. Real isotropy? *No* ( #TODO evidence from data?) → noise estimate using the median over sectors ( #TODO prove this is statistically more robust in our setting)

### Setting

One run : evolution of different experiment conditions

1. Isotropic signal $I(q)$ **dependent** of experiment conditions
2. Detector defects (bad pixels) **fixed** → calib + geometry mask
3. Experimental noise : geometrically coherent (regions, lines, streaks) **(in)dependent** of experiment conditions (assumed static for now)

**Goal** : extract experimental noise from multiple shot data

### Current formalization

### Forward model

The three components of the setting compose differently — this is what fixes the estimator.

$$x_{it} = \tau_i \cdot F_t A_i\, S(q_i;\theta_{c_t}) \;+\; F_t J_i \;+\; \varepsilon_{it}$$

| symbol | meaning | status |
| ---------------------------- | -------------------------- | ----------------------------------------- |
| $F_t$ | incident flux, shot $t$ | **measured** (monitors) |
| $A_i = \Omega_i P(\chi_i,q_i)$ | solid angle × polarization | **known** from calibration |
| $S(q;\theta_c)$ | isotropic scattering | unknown, **condition-dependent** |
| $\tau_i \in [0,1]$ | transmission — shadows | unknown, static, **multiplicative** |
| $J_i \ge 0$ | parasitic scattering — streaks | unknown, static, **additive** |
| $\varepsilon_{it}$ | Poisson + read | variance estimable |

$$M = \{i : \tau_i \ne 1\} \;\cup\; \{i : J_i \ne 0\}$$

Detector defects are **out of scope**: already removed by the calib + geometry floor.

### Log form

With $J = 0$ and $y_{it} = \log(x_{it}/F_t)$:

$$y_{it} = \underbrace{\log\tau_i + \log A_i}_{\alpha_i\ (\text{pixel})} \;+\; \underbrace{\log S(q_i;\theta_{c_t})}_{\gamma_{q_i,c_t}\ (\text{ring}\times\text{condition})} \;+\; \varepsilon_{it}$$

Two-way fixed effects. **Masking = testing $\alpha_i \ne \log A_i$ after removing $\gamma_{q,c}$.**

1. **Ring-wide artifacts are unidentifiable** — $\alpha_i$ and $\gamma_{q,c}$ are confounded for any component constant within a ring. Only *within-ring* variation of $\alpha$ is detectable.
2. **Identifying power $= \operatorname{Var}_c(\gamma_{q,c})$** — if conditions do not move the signal, nothing separates. Measurable, and it caps the method.
3. **Polarization belongs to $A_i$** — uncorrected, it sits in $\alpha_i$ and reads as artifact.

### Identification

Flux-normalise $z_{it} = x_{it}/F_t$, average within condition group $c$:

$$\bar z_i^{(c)} = \tau_i A_i\, S(q_i;\theta_c) + J_i$$

**Stage 1 — shadows.** Contrast two conditions; $J_i$ cancels exactly:

$$\Delta_i = \bar z_i^{(c)} - \bar z_i^{(c')} = \tau_i A_i\big[S(q_i;\theta_c) - S(q_i;\theta_{c'})\big] \qquad\Longrightarrow\qquad \boxed{\ \frac{\Delta_i}{\overline{\Delta}_{q_i}} = \frac{\tau_i}{\bar\tau_{q_i}}\ }$$

**Stage 2 — streaks.** Residual against the recovered transmission:

$$\hat J_i = \bar z_i^{(c)} - \hat\tau_i A_i \hat S(q_i;\theta_c)$$

The order is forced by the composition, not chosen.

### Geometric priors

One prior per class. A single prior is not merely suboptimal — it is wrong for two of the three.

| class | geometry | prior |
| ------ | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| shadow | compact region, sharp edge | $\Phi_{\text{reg}}(M) = \lambda\lvert\partial M\rvert = \lambda\sum_{(i,j)\in\mathcal{N}} w_{ij}\,\mathbb{1}[M_i \ne M_j]$ — submodular, exact MAP by graph cut |
| streak | thickened curve $\{i : d(i,\Gamma) < w\}$ | $\Phi_{\text{curve}} = \lambda_w w + \lambda_n$ — penalise width and count, **not length** |
| blob | disk, unknown radius | matched-filter bank $\max_R \ell_R(i)$; $\sqrt{N_R}$ is the whitening |

> [!warning] Why one prior fails
> $\Phi_{\text{reg}}$ has perimeter $\propto$ length for a thin line, so the region prior **suppresses streaks**. This is the formal reason `hough_lines` must be a separate detector: Hough/Radon *is* the MAP detector under $\Phi_{\text{curve}}$.

Artifact fields are class-disjoint $\Rightarrow M = \bigcup_k M_k$ is **exact**, not a heuristic.

### Shadow vs anisotropy — harmonic cut

Per ring, $\log S(q,\chi) = \sum_m s_m(q)e^{im\chi}$:

- **polarization** — exactly $m = \pm 2$, removable analytically
- **sample anisotropy** — low order in $m$
- **sharp shadow** of angular width $\Delta\chi$ — broadband to $m \sim 2\pi/\Delta\chi$

Project out $\lvert m \rvert \le m_0$ per ring: removes polarization exactly and most anisotropy, at the cost of blindness to shadows wider than $\sim 2\pi/m_0$. One interpretable parameter in place of several opaque ones.

### Objective

$$\hat M = \arg\max_{M = \cup_k M_k}\ \sum_i \ell_i(M_i) \;-\; \sum_k \Phi_k(M_k)$$

$\ell_i$ = LLR for $\alpha_i \ne \log A_i$. The current pipeline (statistic → threshold → regularise → combine) is a greedy approximation: it thresholds *before* the prior acts, discarding the evidence the prior needs.
