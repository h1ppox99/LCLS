"""
combine/union.py -- boolean-union of the evidence onto the floor.

Unions the intensity-free floor with every per-detector thresholded pick. This
is how a Pipeline fuses its channels into the final mask.
"""

from __future__ import annotations


def combine_masks(floor, picks):
    """Union the floor with every per-method pick (each already thresholded).
    `picks` is a dict[str, np.ndarray] for readability; the keys are not used."""
    combined = floor
    for m in picks.values():
        combined = combined | m
    return combined
