"""sample -- the per-run arrays a masking pipeline reads.

`Sample` bundles the selected-shot reductions and calibration constants a
pipeline consumes; it is built from the selected-shot image cache
(`image_store`) in assembled detector `geometry`.
"""

from automask.sample.sample import DERIVED, Sample

__all__ = ["Sample", "DERIVED"]
