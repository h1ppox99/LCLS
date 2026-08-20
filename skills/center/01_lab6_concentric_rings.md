---
name: xray-center-lab6-concentric-rings
description: Optional assembled-image center calibration from LaB6 concentric rings; activate only when the current context does not specify a center.
category: center
role: optional-submodule
gate: image center is not specified in the current run context
status: wired
---

# Center · 01 — LaB6 concentric-ring fit

This is a compact, optional submodule between reduction and masking. It resolves
the center of the assembled images; it does not define a “LaB6 ring center.”

## The decision the agent owns

> Has the image center already been specified in the current context?

## Principle

When a center is supplied, reuse it exactly and do not activate estimation. When
it is absent, the user or Agent assigns control points to at least two physically
indexed LaB6 rings. pyFAI supplies the relationship between ring orders;
deterministic code fits free radii and a common center. A separate validator then
checks every predicted in-field ring against independent image pixels. Vision is
evidence acquisition, not metrology.

## Parameters

| Parameter | Value | Meaning |
|---|---:|---|
| fitting reflections | LaB6 (100), (110) | minimum two indexed control rings |
| ring-order model | pyFAI `LaB6_SRM660c` | predicts all order relationships |
| absolute radii | free parameters | never fixed during center fitting |
| excluded anchors | 367, 531, 734 px | unindexed diffuse features, context only |
| radial sidebands | ±10 px | local contrast reference |
| search prior | broad lower-left beamstop region | instrument context, no ring radius |
| coordinates | assembled `(row, col)` | shared contract for downstream code |

## Decision rules

1. Read `center_context.json` before inspecting or fitting the image.
2. `specified=true` → write a `reuse` decision; do not estimate.
3. `specified=false` → assign at least two indexed rings, fit each independently,
   then fit a common center.
4. Independently test every pyFAI-predicted ring that intersects the image.
5. Missing or mislocated predicted orders → `ring_semantics_failure`; revise the
   ring assignment instead of accepting a match to one circle.
6. `revise` → use structured feedback, at most twice.
7. `escalate` → preserve the candidate and stop; never promote it to `center`.
8. Insufficient arc span is retained as uncertainty evidence; it is not by itself
   an escalation condition. Independent-center disagreement is a hard check only
   when both rings have enough angular support for that comparison.

## Implementation

```bash
python3 pipeline/step2c_estimate_center.py --out-dir outputs/<run> --source agent_estimate
```

The executor obtains the LaB6 radius ratios from pyFAI, searches a free (100)
radius, extracts ridge points, fits each circle independently, and performs an
equal-ring-weight joint fit. The independent validator re-samples pixels at all
in-field predicted orders before the executor emits its verdict.

## Evidence (Run0475)

Run0475 confirms 847 px = (100) and 1245 px = (110): observed radius ratio
1.4699 versus theoretical 1.4712. The 367/531/734 px anchors do not index to
LaB6. The outer (110) arc covers only a small azimuthal span, so it cannot by
itself determine a stable center. Treating 367 px as (100) predicts several
additional in-field rings that are absent, so the all-ring validator rejects that
semantic assignment even though one predicted order happens to overlap a ridge.

## When to use

Use only for calibration data with visible rings when no image center is present
in the run context. Downstream masking and I(q) use the resulting artifact.

## Trade-offs

Partial arcs couple radius and center, so their independent single-circle centers
can drift. Record angular span as uncertainty, while using the joint fit and
independent all-order pixel validation for the decision.

## Do not

- Do not run merely because the data are calibration data if a center is supplied.
- Do not use 367/531/734 px as LaB6 or center constraints.
- Do not tune wavelength, lattice constant, or reflection identity to force agreement.
- Do not convert an escalated `candidate_center` into an accepted `center`.
- Do not silently substitute a repository-wide hard-coded center.

## Outputs

- `image_center.json`: hypothesis, free-radius fits, verification, and decision.
- `image_center_overlay.png`: detected ridge points and free-fit lattice circles.
- `ring_validation.json`: independent per-order location checks from the pyFAI model.
- `ring_validation_overlay.png`: pass/fail overlay for every predicted in-field ring.
- `center_loop.jsonl`: one structured record per fit/revision attempt.
- `center_human_review.md`: targeted escalation packet when underdetermined.

## Links

Feeds [masking](../masking/README.md) and deterministic I(q) verification.
