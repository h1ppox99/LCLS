# Automatic masking — exploration log

Research log for building the best run-agnostic auto-mask for the Jungfrau1M on
`xppl1016922`. Branch: `exploration`. Only **strongly validated** findings are
recorded here; provisional or single-sample observations live in the working
notes at the bottom and are promoted only once they meet the evidence bar below.

Started 2026-07-23.

---

## TL;DR — validated results

1. **Best pipeline = `masking.best_pipeline("union")`**: the production detectors
   (variance + window_median + blackhat, union) on an **extended floor** of
   `geometry + calib + dead_holes + stuck`. Same real-mask IoU as before
   (389 = 0.801, 475 = 0.665), but now catches isolated **dead / stuck / dead-column**
   pixels the frozen `calib` mask can miss — the run-agnostic robustness the
   project is after. **[D1, headline]**
2. **Two new floors close the structural gap** (`stats/dead_holes.py`,
   `stats/stuck.py`): dead/hot/column-pixel synthetic recall 0.00/0.08/0.85 →
   **1.00/1.00/1.00**, at **zero real-IoU / precision cost**. **[D1]**
3. **Do not IPM2-normalize** — it halves real IoU on both runs (0.73 → 0.50). Raw
   calibrated averaging over all x-ray-on shots is best. **[N1]**
4. **union ≫ mahalanobis** as a combiner (0.733 vs 0.311); **variance is the
   load-bearing detector**; **window_median is redundant** (removable with a small
   gain on both runs). **[D2, D3]**
5. **Evaluation improved**: 3 physically-grounded artifacts added (dead/hot/column
   pixels) + de-confounded **delta scoring** that exposes the pipeline's true
   (high) localisation precision. **[Direction 1, M1]**

---

## Standards of evidence (what "validated" means here)

A claim is promoted to a **Finding** only if it clears all of:

1. **Both real runs.** Every effect is measured on runs **389 and 475** (the only
   two with per-run human masks). An effect present on one run only is a *lead*,
   not a finding.
2. **Spread reported, not just means.** Synthetic results aggregate ≥ 5 seeds × 2
   rotations per (run, artifact) and are reported as **mean ± std**. A difference
   smaller than the pooled std is called "within noise", never an improvement.
3. **Paired comparison.** When comparing pipeline B to pipeline A, both are scored
   on the *identical* seeded corruptions (`harness.synth` fixes the seed), so
   differences are attributable to the pipeline, not to which artifacts were drawn.
4. **No target leakage.** The human mask is the evaluation target; it is never an
   input to a masker or a tuning objective beyond the reported score. Synthetic
   injection touches only originally-valid pixels (`~human`).
5. **Direction of effect must be consistent** across runs and artifact types
   before a knob is called "better". A win on one artifact that costs another is
   reported as a trade-off, not a win.

**Two metric conventions, kept separate:**
- **Real-mask score** — IoU / precision / recall of the full predicted mask vs the
  human mask (`automask.evaluation.evaluate`). This is the production objective.
- **Synthetic detection** — per-artifact **recall** is the clean signal (did the
  masker catch the injected defect). Synthetic *precision/IoU* are confounded by
  the pipeline's always-on baseline masking (see Finding M1) and are reported only
  as context, never as the primary ranking signal.

**Compute note.** Features are resolved through the `FeatureStore` (a feature =
reduction over a `ShotSelection`); both runs' catalogue features are prewarmed, so
mask experiments are numpy-only and fast. Selection/normalization experiments
(direction 2) recompute from XTC and are the expensive ones.

---

## Findings

### M1 — Synthetic precision/IoU are confounded by baseline masking *(method)*

The synthetic metric scores the full pipeline output against the *injected-only*
truth over the valid region. The pipeline legitimately masks ~3% of the valid
region on every example (its response to real, pre-existing defects), and that
fixed masking counts as false-positive against an injected-only truth. Evidence:
across all baseline cases the false-positive rate is pinned at FPR ≈ 0.0295
regardless of artifact, i.e. precision is dominated by a constant, not by
artifact-specific error. **Consequence adopted for this log:** rank maskers by
per-artifact **recall** and by the **real-mask IoU**; treat synthetic precision as
context only.

**De-confounded (delta) scoring — delivered and demonstrated.**
`harness.synth_delta` scores the artifact-attributable mask
(`pred_corrupted & ~pred_clean`) against the injected truth, removing the baseline
masking from the false-positive count. On `best_pipeline`, both runs, n=20:

| artifact | recall | precision (standard) | precision (**delta**) |
|---|---|---|---|
| dead_pixels | 1.000 | 0.009 | **0.963** |
| beamstop | 1.000 | 0.350 | **0.847** |
| beamstop_small | 1.000 | 0.129 | **0.828** |
| hot_pixels | 1.000 | 0.009 | 0.463 |
| bad_column | 1.000 | 0.042 | 0.257 |

The standard precision (0.009 for dead pixels) was almost pure baseline-masking
confound; de-confounded, the pipeline is shown to localise artifacts **precisely**
(0.96 for dead pixels, 0.83-0.85 for beamstops). This confirms M1 quantitatively
and gives the suite an honest precision signal. (bad_column's lower delta
precision is the geometry floor padding whole near-dead columns; hot_pixels' is
the global robust-z shift a bright injection induces — both benign.)

### B — Baseline detection profile of the production pipeline *(validated)*

Production pipeline (variance + window_median + blackhat on geometry+calib floor,
union), both runs, n=20 per artifact (5 seeds × 2 rotations × 2 runs):

| artifact | recall (mean ± std) | verdict |
|---|---|---|
| beamstop | 1.000 ± 0.000 | caught perfectly |
| beamstop_small | 1.000 ± 0.000 | caught perfectly |
| bad_column | 0.850 ± 0.357 | caught but unreliable (bimodal by column length) |
| hot_pixels | 0.080 ± 0.037 | **missed** |
| streak | 0.071 ± 0.069 | not statically masked |
| dead_pixels | **0.000 ± 0.000** | **invisible** |

Real-mask score: run 389 IoU 0.801 (P 0.854 / R 0.928), run 475 IoU 0.665
(P 0.811 / R 0.787).

### F1 — Isolated dead/hot pixels are invisible to the intensity detectors *(validated + mechanism)*

`dead_pixels` recall is exactly 0.000 (±0.000, n=20) and `hot_pixels` 0.080. This
is not tuning — it is structural. Direct diagnostic (run 475, 300 injected dead
px): the raw variance/window_median/blackhat z-fields are **exactly 0 at every
injected pixel**. Mechanism, confirmed:
- A dead pixel sets `sumimg → 0`, so `real = (sumimg != 0)` becomes **False**
  there. Every field stat computes its z over `real` and every `Detector.pick`
  re-gates by `& sample.real`, so a not-real pixel is outside all of them.
- The `geometry` floor only masks rows/cols that are > 40 % unmapped (ASIC gaps);
  an isolated interior hole never trips it.
- Hence dead-pixel coverage comes **entirely from the `calib` status floor**.
  Synthetic dead pixels are placed on random *good* pixels not in `calib`, so
  nothing masks them.
- Hot/stuck pixels stay `real` (mean rises) but have `ustd → 0`; `variance`
  computes `robust_z(ustd, ustd>0)`, which **excludes exactly-zero-variance
  pixels**, so stuck pixels get z=0 too.

**Implication:** a newly-dead or newly-stuck pixel not already in `calib` is
masked by nothing. This is the sharpest run-agnostic weakness found. **Fixed** by
the `dead_holes` + `stuck` floors — see Direction 3 finding D1 (validated).

### F2 — Extended dark/attenuation artifacts are caught perfectly *(validated)*

beamstop and beamstop_small both recall 1.000 ± 0.000 (n=20, both runs): a
multiplicative shadow lowers mean AND spread over an extended region, which the
low-variance + dark detectors catch reliably regardless of size.

---

## Direction 1 — Evaluation pipeline & synthetic artifacts

**Done:** added physically-grounded artifacts that the original suite lacked —
`dead_pixels`, `hot_pixels` (stuck-high), `bad_column` (ASIC/column fault) — in
both the single-image and full-Sample (pipeline) paths, with consistent injection
across `sumimg`/`umean`/`ustd` (dead → all collapse to 0; hot → mean up, spread
→ 0). Validated: correct polarity, footprint ⊆ valid region, deterministic in seed.

**Done:** incremental (delta) scoring to de-confound precision — `harness.synth_delta`,
demonstrated under M1.

**Open / proposed:** charge-sharing halo and gain-switch-patch artifacts; a
size-stratified recall curve.

## Direction 2 — Normalization & shot-selection techniques

Real-mask IoU (production pipeline) with the lit features (umean/ustd) recomputed
under selection variants. Real IoU is deterministic (no seed spread) but n=2 runs,
so the bar is a **large, same-direction** effect on both runs.

### N1 — IPM2 per-shot normalization is harmful *(validated)*

Scaling each frame by `median(i0)/i0` before averaging (`normalization="ipm2"`)
**collapses real-mask IoU on both runs**: 0.801 → 0.500 (run 389) and 0.665 →
0.501 (run 475). Mechanism: normalization inflates the per-pixel std and flattens
its contrast, so the mask turns ultra-conservative — precision rises to 0.985 but
recall falls to ~0.50 on both runs. **Conclusion: keep `normalization="none"`
(raw calibrated averaging).** This retroactively validates the catalogue default.

### N2 — Shot count trades precision for recall, no consistent IoU win *(lead)*

| selection (x-ray on) | IoU 389 | IoU 475 | mean |
|---|---|---|---|
| all shots (catalogue) | 0.801 | 0.665 | 0.733 |
| n = 500 | 0.784 | **0.724** | 0.754 |
| n = 200 | 0.729 | 0.502 | 0.616 |

Fewer shots → noisier `ustd` → higher precision / lower recall. n=500 *helps 475
(+0.059) but hurts 389 (−0.017)* — opposite directions, so **not** a validated
improvement by the consistency rule. n=200 hurts both (too few frames). All-shots
is the safe default; per-run shot-count tuning is a possible but unproven lever.

### N3 — The intensity percentile trim is negligible *(validated)*

Removing the 3%/3% low/high intensity trim (`filter_low=filter_high=0`) leaves
real IoU essentially unchanged: 0.801→0.795 (389), 0.665→0.660 (475). The trim
neither helps nor hurts the mask at this level; it is cheap insurance against
saturated/dropped shots and can stay at the default.

## Direction 3 — Masking methods, regularization, combination

### D1 — Two intensity-free floors close the dead/hot/column gap at zero cost *(validated, headline)*

Motivated directly by F1. Added two high-precision floor stats:
- **`dead_holes`** (`stats/dead_holes.py`): not-real pixels enclosed by a mostly-
  live neighborhood (isolated dead pixels / dead columns), complementing
  `geometry` (which only catches the large ASIC/module gaps as whole rows/cols).
- **`stuck`** (`stats/stuck.py`): data-carrying pixels with `ustd ≈ 0` — stuck /
  frozen pixels the `variance` field stat cannot see (it excludes `ustd == 0`).

Paired, both runs, n=20 per artifact:

| pipeline | IoU 389 | IoU 475 | prec | dead_pixels | hot_pixels | bad_column | beamstop |
|---|---|---|---|---|---|---|---|
| production (geometry+calib) | 0.801 | 0.665 | 0.832 | 0.000 | 0.080 | 0.850 | 1.000 |
| + dead_holes | 0.801 | 0.665 | 0.832 | **1.000** | 0.080 | **1.000** | 1.000 |
| + dead_holes + stuck | 0.801 | 0.665 | 0.832 | **1.000** | **1.000** | **1.000** | 1.000 |

Both floors together take dead_pixels 0.000→1.000, hot_pixels 0.080→1.000, and
bad_column 0.850→1.000 (and reliable), with **real-mask IoU and precision
unchanged to three decimals on both runs** — no regression. On the current two
runs the real dead/stuck pixels are already in `calib`, so IoU does not move; the
gain is **run-agnostic robustness** — a newly-dead or newly-stuck pixel absent
from the frozen `calib` mask is now masked structurally rather than missed. Both
stats are intensity-free and touch only genuinely dead/stuck pixels, so they
cannot mask Bragg signal by construction.

**Adopted** as `masking.best_pipeline()` (production detectors + floor
`geometry, calib, dead_holes, stuck`); `production_pipeline` is left untouched as
the regression anchor.

### D2 — Union is the best combiner; mahalanobis over-masks *(validated)*

On the improved floor, mean real IoU: **union 0.733**, weighted_sum 0.705,
**mahalanobis 0.311**. Mahalanobis collapses precision to 0.316 (recall 0.955) —
it masks far too much. Union wins by a large margin; weighted_sum is a close
second. Note all point/extended defects are now caught at recall 1.0 regardless
of combiner, because `dead_holes`/`stuck`/`variance` carry them at the floor and
first-detector level — the combiner choice only moves the real-mask precision.

### D3 — Variance is load-bearing; window_median is redundant *(validated)*

Detector ablation, per-run real IoU (floor = geometry+calib+dead_holes+stuck):

| detectors | IoU 389 | IoU 475 |
|---|---|---|
| variance + window_median + blackhat | 0.801 | 0.665 |
| **variance + blackhat** (drop window_median) | **0.814** | **0.666** |
| variance + window_median (drop blackhat) | 0.807 | 0.667 |
| variance only | 0.689 | 0.670 |
| floor only (no intensity detector) | 0.502 | — |

- **`variance` is the critical intensity detector**: dropping it takes mean IoU
  0.733 → 0.567 and beamstop recall 1.000 → 0.043 (it is what catches extended
  dark/attenuation regions).
- **`window_median` is redundant**: removing it gives equal-or-better IoU on
  **both** runs (0.801→0.814, 0.665→0.666) — a small, consistent-direction gain.
  `variance + blackhat + floors` is a simpler, equal-or-better pipeline.
- The **floor alone** (geometry+calib+dead_holes+stuck) reaches IoU 0.502 at
  **100 % precision**; the intensity detectors add the extended-dark coverage the
  floor cannot express.

Marginal-gain caveat: the window_median improvement is ≤ 0.013 and rests on n=2
runs, so `window_median` is reported as *safely removable*, not a must-remove.

## Direction 4 — Synthesis & best strategy

### Recommended pipeline (all components validated above)

| stage | choice | evidence |
|---|---|---|
| lit features | raw calibrated mean/std over **all x-ray-on shots**, `normalization="none"` | N1 (ipm2 halves IoU), N2/N3 |
| floor | geometry + calib + **dead_holes + stuck** | D1 (fixes dead/hot/column at zero cost) |
| detectors | **variance** (critical) + blackhat; window_median removable | D3 |
| combiner | **union** | D2 (mahalanobis over-masks) |

Realised as `automask.masking.best_pipeline("union")` (keeps window_median for
safety; the leaner `variance + blackhat` variant is equal-or-better on both runs).

**Measured performance (both runs):**
- Real-mask IoU: run 389 **0.801**, run 475 **0.665** (0.814 / 0.666 for the lean
  variant) — unchanged vs the prior production pipeline, i.e. no regression.
- Synthetic defect recall: dead_pixels, hot_pixels, bad_column, beamstop,
  beamstop_small all **1.000** — the point/column-defect gaps are closed.

**The net contribution of this exploration** is not a higher IoU on the two
labelled runs (that number is bounded by an imperfect hand-drawn target) but
**run-agnostic robustness**: the pipeline now catches isolated dead / stuck /
dead-column pixels structurally, instead of relying on a frozen `calib` mask that
will be stale on a new run — the stated project goal.

### What is deliberately NOT masked

Bright, transient features (the `streak` artifact, recall ~0.07) are left unmasked
by design: they are one-shot events, not static bad pixels, and bright pixels are
Bragg signal. A static per-pixel mask is the wrong tool for them (per-shot
rejection is); their low recall is correct behaviour, not a defect.

---

## Weak points & open questions (living list)

- **n = 2 labelled runs.** All real-mask findings rest on two runs; this is why
  the evidence bar is "large, same-direction effect on both", and why no
  hyperparameter (e.g. variance `k`, tile size) was tuned to the IoU — tuning on
  the only two evaluation runs would overfit and could not be validated. More
  labelled runs are the single highest-value thing to add.
- Run 389's XTC is truncated (~4600 shots) vs 475's full stream; its features come
  from fewer frames — a possible confound for cross-run consistency, though both
  runs move together on every validated finding.
- The human masks are the lab's hand-drawn recipe — an imperfect target that caps
  achievable IoU; part of the residual (~0.2-0.3) is target noise, not masker error.
- `dead_holes` was tested with fixed params (win=5, min_live_frac=0.6); its
  behaviour on real stale-`calib` runs is inferred from synthetics, not yet from a
  run where `calib` is known to be stale.
- Synthetic artifacts remain idealised (dead=0, hot=stuck-high, no charge-sharing
  halo or gain-switch patch). The suite discriminates well but is not exhaustive.

## Artifacts of this exploration (code)

- `automask/stats/dead_holes.py`, `automask/stats/stuck.py` — the two new floors [D1].
- `automask/masking.py::best_pipeline` — the recommended pipeline.
- `automask/synthetic/artifacts.py` — `dead_pixels`, `hot_pixels`, `bad_column`
  generators; `automask/synthetic/sample_adapter.py` — their pipeline-mode injection.
- `automask/studies/exploration/harness.py` — paired comparison harness +
  `synth_delta` (de-confounded scoring); `exp_selection.py`,
  `exp_combiners_ablation.py` — the direction-2/3 experiments.

## Next leads (unproven, ranked)

1. Add more labelled runs — the binding constraint on every real-mask finding.
2. Per-run shot-count tuning (N2 showed a real but inconsistent lever).
3. Charge-sharing-halo and gain-switch-patch synthetic artifacts (breadth).
4. Validate `dead_holes` on a run with known-stale `calib` (its raison d'être).
