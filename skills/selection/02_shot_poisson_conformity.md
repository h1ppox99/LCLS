---
name: xray-selection-shot-poisson-conformity
description: Per-shot Poisson-conformity screen — an AGENT-DECIDED cut APPENDED to the ipm cuts (01a/01b stay as they are). Judges every shot against the conditional null k_i ~ Poisson(lambda_i·f_t) with three separated statistics — flux conformity vs the monitor (u), self-normalized spatial-shape deviance (D), single-pixel tail spikes — fenced by population MAD after flux+time detrending, so shots are dropped for nonconformity, never for brightness. Run0475: 7/2779 flagged (~35% photon excess AND shifted shape — pointing-excursion class); 0 of the 28 above-p99 bright shots flagged.
category: selection
role: agent-decided shot cut (whole shots) — statistical conformity screen, appended after 01a/01b
gate: conditional — needs per-shot calibrated frames and a healthy monitor; runs after 01a (and after 01b when that cut was chosen)
status: not-wired
---

# Selection · 02 — Per-shot Poisson conformity (agent decides)

The transpose of [qa/08 pixel photon statistics](../qa/methods/08_pixel_photon_statistics.md):
same conditional null `k_i(t) ~ Poisson(λ̂_i · f_t)`, opposite sample axis — there one
*pixel* across shots (judges the detector), here one *shot* across pixels (judges the
shot). An **append** to the ipm cuts, not a replacement: 01a/01b keep their roles
(noise floor; baseline-compatible p99), this method adds an evidence-driven screen that
catches what a monitor-only threshold cannot see — and clears what it cannot clear.

## The decision the agent owns

> **Run this screen or not? At which `fence_k`? And per flagged class — drop, or route
> the evidence downstream instead of dropping?**
>
> Skipping is legitimate (cost: one calibration pass over all frames). Flagged shots are
> not automatically bad for every product: a pure monitor-mismatch shot poisons a
> per-shot-normalized sum but is harmless in a plain sum. The routing table below is the
> decision surface; the agent must record class counts and what it did with each.

## Principle

If pixels are healthy (qa/08's verdict) and the calibration is right, then *conditional
on the shot's true flux*, every healthy pixel's count is an independent Poisson draw.
A shot can violate this in exactly three separable ways, so the method computes three
statistics per shot over healthy pixels (`~status_bad`, `λ̂ > 0`):

| Statistic | Null it tests | Catches |
|---|---|---|
| `u` — standardized relative residual of `S_t/M_t` (total photons vs monitor prediction `M_t = f_t·Λ`) | monitor and detector agree on the shot's flux | monitor glitches, pointing/spectral excursions, saturation droop |
| `D/N` — mean per-pixel Poisson deviance vs the **self-normalized** expectation `μ_i = λ̂_i·(S_t/Λ)` | the shot's *spatial pattern* matches the run's, at the shot's **own** brightness | structure anomalies: shifted rings, panel events, localized bursts |
| `n_spike` — pixels with `k` above the per-pixel Poisson tail bound (`tail_p`) at the shot's own brightness | no single pixel is impossibly bright | zingers / cosmic-ray-like single-pixel events |

Three design rules, each learned the hard way on Run0475 (see Evidence):

1. **Never fence the Poisson-scaled `z = (S−M)/√M` on a full flux range.** The dominant
   residual of `S/M` is *multiplicative* (per-shot pointing jitter ~15 % MAD here), so z
   grows like √M and re-flags ordinary scatter on bright shots — brightness bias through
   the back door, the exact failure this method exists to remove. Work with the relative
   residual `r = S/M − 1`, standardized by the combined error model
   `σ_t = √(σ_mult² + 1/M_t)` (σ_mult from the bright half, where Poisson is negligible).
2. **Detrend along flux, then along time, before fencing.** The ipm2 response is
   nonlinear in flux level (`S/M` median walks 0.73 → 1.13 across the range — the
   plateau evidence of 01a, seen end-to-end), and slow pointing/spectral drift lives in
   time (normalization's residual, not selection's business). A rolling median along
   each axis absorbs both; what is fenced is "outlier against peers at the same flux and
   epoch", never the systematics.
3. **Self-normalize the deviance.** Computed against `λ̂_i·f_t` the deviance re-detects
   any flux mismatch (`u` already measures it) and the two flags collapse into one
   confounded verdict. Against `λ̂_i·(S_t/Λ)` the deviance is blind to flux and tests
   *shape only* — the two statistics separate cleanly, and the class labels mean what
   they say.

## Parameters

| CLI flag | Default | Meaning |
|---|---|---|
| `--photon-kev` | — (required) | single-photon energy for photonization (9.6 for Run0475) |
| `--low-ipm-threshold` | 200 | 01a's cut; this method screens the post-01a set |
| `--fence-k` | 6 | fence in population-MAD units for both `u` and `D` (same convention as qa/08) |
| `--detrend-window` | 101 | rolling-median window (shots) for the flux- and time-detrends |
| `--tail-p` | 1e-12 | per-pixel tail probability for spikes (expected false spikes ≈ `tail_p·N_pix·T` ≈ 0.002 run-wide) |
| `--spike-min` | 1 | spikes needed to flag the shot |

## Decision rules

### Evidence to gather before deciding

1. Are per-shot calibrated frames available at acceptable cost? (One pass ≈ 30 s for
   2 779 shots on a laptop.) If not, this method is not available — say so.
2. Did qa/08 pass? If the *detector* is out of conformity, per-shot verdicts are
   uninterpretable — fix pixels first.
3. Is the monitor itself sane (01a's plateau check)? `u` compares against ipm2; a dead
   monitor makes `u` meaningless (`D` and `n_spike` still work — they are monitor-free).

### Run / skip

**RUN** when per-shot normalization or weighting is planned (a single monitor-mismatch
shot enters the sum with the wrong weight), when the bright tail matters and a blanket
p99 cut is too crude, or when the product is a calibration standard (LaB6 sum) where a
few pointing-excursion shots smear the rings. **SKIP** (recorded, with reason) when only
a quick-look unweighted sum is needed, or frames are unavailable.

### Routing per flagged class

| Class | Signature | Default action | Alternative |
|---|---|---|---|
| `zinger_spike` | `n_spike ≥ spike_min` | drop shot | per-shot pixel veto, if the pipeline ever grows one |
| `structural` | `D` over fence (shape wrong at own brightness) | drop shot | none — shape cannot be re-weighted away |
| `excess/deficit_vs_monitor` | `u` over fence, `D` quiet | drop **iff** per-shot division by ipm2 is planned; else keep | down-weight; or route to [normalization/01](../normalization/01_per_shot_flux_ipm2.md) as evidence against monitor trust |

### Relation to 01b (append, not replacement)

01b's p99 cut stays, with its own rationale (baseline compatibility with the human
reference sum). When both run: 01a → 01b → 02 on the survivors. What 02 adds on top of
p99 is *specificity* — on Run0475 it flags mid-range excursion shots p99 never looks at,
and clears all 28 above-p99 shots that p99 would discard wholesale. When 01b is skipped
(evidence-driven branch), 02 covers the protective role: saturation droop or bright-end
monitor rollover surfaces as `deficit/excess_vs_monitor` at the bright end.

## Implementation

`scripts/shot_poisson_conformity.py`. Single calibration pass (pedestal, gain,
photonize — mirrors `pipeline/xtclib.py`), sparse per-shot nonzero-pixel cache, then
all statistics in memory. λ̂ comes from the same screened set (contamination by the
rare anomalous shots is O(flags/T) and diluted by robust fences).

```
python skills/selection/scripts/shot_poisson_conformity.py \
  --frames npy/frames_raw.npy --shot-table npy/shot_table.npz \
  --ped calib/ped.npy --gain calib/gain.npy --status-bad calib/status_bad.npy \
  --photon-kev 9.6 --out-dir outputs/<run>/selection/shot_conformity
```

## Evidence (Run0475)

2 779 shots screened (x-ray on, ipm2c ≥ 200; monitor CV 1.05 — full flux range, no
band). Poisson relative floor on totals 0.85 %; measured multiplicative residual
σ_mult ≈ 15 % — the run is *pointing-jitter dominated at the shot level*, which is
what forces design rules 1–2.

- **Flagged: 7/2779 (0.25 %),** all `structural`: photon excess of +30–36 % vs monitor
  **and** shape deviance above the fence *after* self-normalization — the beam genuinely
  moved on those shots (rings shifted), they are not merely mis-monitored. `nswitch`
  stays at its fixed-set baseline (566–572) on all of them — not a gain artifact.
- **Bright tail cleared: 0 of 28 shots above p99 flagged** (10 of them carry a +30–40 %
  monitor mismatch but their shape at their own brightness is normal — kept by default,
  droppable via the `u` route if per-shot division is planned). The p99 percentile and
  the conformity verdict select *different shots* — evidence that brightness and badness
  are independent axes here.
- **No spikes anywhere** at `tail_p = 1e-12` (clean LaB6 run).
- **Monitor nonlinearity measured end-to-end**: flux-binned `S/M` median 0.757 (ipm2c
  ≈ 300) → 0.998 (≈ 11 k) → 1.13 (bright end) — the detector-side view of 01a's
  "linear plateau above ≈ 3000".
- Design iteration recorded for honesty: a Poisson-z fence first flagged 19 shots, 16 of
  them bright — the √M artifact of rule 1; switching to the conditional relative
  residual dissolved those to zero and the confounded deviance then isolated the 7 real
  excursions once self-normalized (rule 3). 17 → 7 was systematics leaving the verdict.

## When to use

Any run with per-shot frames where shots enter a weighted/normalized product, a
calibration sum, or where the bright tail is scientifically valuable. Any detector +
any flux monitor — nothing is Jungfrau- or ipm2-specific beyond the calibration step.

## Trade-offs

- Needs a full frame pass (~30 s here) — orders costlier than the smalldata-only 01a/01b.
- The `u` fence inherits σ_mult: on a pointing-noisy run like this one it only catches
  gross monitor disagreement (|u| ≈ 2–3 for a 35 % excess). That is honest — on this run
  a 35 % mismatch *is* ordinary scatter — but it means `u` is a blunt instrument
  precisely when jitter is worst. A narrow monitor band (qa/08 style) sharpens it.
- The deviance fence sits at the p99.9 tail of a smooth distribution — the 7 flags are
  the extreme tail, not a separated cluster; expect the flag set to move by ±1–2 shots
  under reasonable `fence_k` choices. Report counts, don't over-read identity.
- λ̂ is estimated once from the screened set (no re-estimation loop); fine at 0.25 %
  contamination, revisit if a run flags ≫ 1 %.

## Do not

- **Don't fence `(S−M)/√M` on a full flux range** — brightness bias by construction
  (Evidence, design iteration). Use the standardized relative residual.
- **Don't drop a shot for being bright.** If a cut correlates with brightness, it must
  be because conformity failed there, and the report must show which statistic.
- **Don't skip the flux-detrend** when the monitor has a nonlinear response — otherwise
  the method silently measures ipm2's response curve, not shot anomalies.
- **Don't fold flagged shots into the pixel mask** — selection removes shots; masking
  removes pixels ([golden rule](README.md)).
- **Don't run it against a sick detector** — qa/08 gates this method.

## Outputs

`shot_conformity_report.json`: screen parameters and fences, `sigma_mult`,
`ratio_vs_flux_curve` (the monitor-response evidence), `n_flagged`, `class_counts`,
`bright_tail_demo` (above-p99 kept/flagged counts), per-shot detail for every flag,
`rejected_shot_indices` (the list `decisions.json → selection` consumes),
`spike_examples`. `shot_conformity.npz`: per-shot arrays (S, M, ratio, u, deviance,
n_spike, flags, classes). `shot_conformity.png`: ratio-vs-flux with conditional median,
standardized residual, detrended deviance, spikes-vs-intensity — flagged shots marked
on all four. Log run/skip + class counts + routing chosen; silent skipping is the enemy.

## Machine block

```json
{
  "method": "shot_poisson_conformity",
  "defaults": {
    "low_ipm_threshold": 200.0,
    "fence_k": 6.0,
    "detrend_window": 101,
    "tail_p": 1e-12,
    "spike_min": 1
  },
  "run0475": {
    "n_screened": 2779,
    "sigma_mult": 0.1496,
    "n_flagged": 7,
    "flag_fraction": 0.0025,
    "class_counts": {"structural": 7},
    "above_p99_total": 28,
    "above_p99_flagged": 0
  }
}
```

## Links

Part of: [selection](README.md). Companions: [01a](01a_low_ipm_exclusion.md) (runs
first; supplies the low threshold) · [01b](01b_high_ipm_exclusion.md) (kept as-is; 02
appends specificity on top of, or covers the protective role in place of, the p99 cut).
Transpose of: [qa/08 pixel photon statistics](../qa/methods/08_pixel_photon_statistics.md)
(same null, other axis; gates this method). Feeds:
[normalization/01](../normalization/01_per_shot_flux_ipm2.md) (`u` evidence on monitor
trust; `ratio_vs_flux_curve` is the plateau check seen from the detector side).
