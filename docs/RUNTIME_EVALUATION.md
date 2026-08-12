# Runtime evaluation

Runtime evaluation asks whether a masking procedure is trustworthy on a run for
which no reference mask exists. It does not estimate IoU and does not emit a
single quality score or hand-written pass/fail verdict.

```python
from automask.evaluation import evaluate_runtime

result = evaluate_runtime(pipeline, run=475, seed=0)
```

The returned `RuntimeEvaluation` contains three independent kinds of evidence.

## Sampling stability

Selected shots are dealt round-robin into ten folds. Repeated random half-folds
are compared with their complements, and the mask IoU distribution is reported
with its standard deviation. Half-fold partitions are sampled without
replacement. Because reductions are reconstructed from
real disjoint shots, this includes correlated detector and beam noise that an
independent per-pixel noise model would miss.

## Temporal stability

The mask is independently reconstructed from the first and second chronological
halves of the run. Their IoU measures sensitivity to detector drift and changes
in the experimental-condition mixture. It is reported separately from sampling
stability because the two quantities answer different questions.

## Azimuthal physical consistency

After solid-angle and polarization correction, excess scatter between
azimuthal sectors is measured within radial rings. The candidate is compared
with repeated random masks that remove the same number of pixels. The result
reports candidate excess, gain over matched controls, and the fraction of rings
in which the candidate wins. The distribution across repeated controls quantifies
Monte Carlo variability.

This diagnostic assumes approximately isotropic scattering. It must not be used
unchanged for strongly textured or intrinsically anisotropic samples.

## Structural outputs

`masked_fraction` is descriptive, not a quality score. `floor_contained` checks
the pipeline contract that geometry and calibration exclusions survive fusion.
No universal acceptable masked-fraction range is assumed.

The former analytic independent-pixel perturbation, arbitrary hyperparameter
jitter, compactness prior, and event-axis χ² score are not part of runtime
evaluation: their assumptions or empirical calibration were not strong enough
to support production decisions.
