"""
Tests for automask.synthetic. Plain assert-style functions so they run either
under pytest or directly (pytest is not installed in the ana env):

    python -m automask.synthetic.tests.test_synthetic

They use a tiny synthetic image + mask (no frozen dataset needed) so the checks
are fast and self-contained.
"""

from __future__ import annotations

import numpy as np

from automask.synthetic.artifacts import ARTIFACTS, beamstop_shadow
from automask.synthetic.evaluate import _rotate
from automask.synthetic.metrics import masking_metrics


def _toy():
    """Small image + ground-truth mask; a border ring is marked invalid."""
    rng = np.random.default_rng(0)
    image = rng.uniform(1.0, 100.0, size=(64, 80))
    gt = np.zeros((64, 80), dtype=bool)
    gt[:4, :] = gt[-4:, :] = gt[:, :4] = gt[:, -4:] = True  # invalid border
    return image, gt


def test_seed_is_reproducible():
    image, gt = _toy()
    valid = ~gt
    for gen in ARTIFACTS.values():
        c1, m1 = gen(image, valid, np.random.default_rng(7))
        c2, m2 = gen(image, valid, np.random.default_rng(7))
        assert np.array_equal(c1, c2), f"{gen.__name__}: corruption not reproducible"
        assert np.array_equal(m1, m2), f"{gen.__name__}: injected mask not reproducible"


def test_original_image_unchanged():
    image, gt = _toy()
    original = image.copy()
    valid = ~gt
    for gen in ARTIFACTS.values():
        gen(image, valid, np.random.default_rng(1))
        assert np.array_equal(image, original), (
            f"{gen.__name__} mutated the source image"
        )


def test_no_invalid_pixel_modified():
    image, gt = _toy()
    valid = ~gt
    for gen in ARTIFACTS.values():
        corrupted, injected = gen(image, valid, np.random.default_rng(2))
        assert np.array_equal(corrupted[gt], image[gt]), (
            f"{gen.__name__} changed an originally-invalid pixel"
        )
        assert not injected[gt].any(), f"{gen.__name__} injected into an invalid pixel"


def test_perfect_prediction_scores_one():
    image, gt = _toy()
    valid = ~gt
    _, injected = ARTIFACTS["beamstop"](image, valid, np.random.default_rng(3))
    assert injected.any()
    m = masking_metrics(injected.copy(), injected, valid)
    for key in ("precision", "recall", "f1", "iou"):
        assert m[key] == 1.0, f"perfect prediction: {key}={m[key]} != 1.0"


def test_rotate_keeps_image_and_mask_aligned():
    image, gt = _toy()
    for deg in (0, 90, 180, 270):
        rimg, rgt = _rotate(image, gt, deg)
        assert rimg.shape == rgt.shape
        # a full turn made of 4 x 90-deg steps returns to the original
        assert np.array_equal(np.rot90(rimg, 4 - deg // 90), image)
        assert np.array_equal(np.rot90(rgt, 4 - deg // 90), gt)
    # invalid-pixel count is preserved (rotation only relabels positions)
    assert int(_rotate(image, gt, 90)[1].sum()) == int(gt.sum())


def test_beamstop_shapes_differ():
    image, gt = _toy()
    valid = ~gt
    kw = dict(center=[0.5, 0.5], radius=12.0, axis_ratio=1.0, softness=0.0)
    _, ell = beamstop_shadow(
        image, valid, np.random.default_rng(0), shape="ellipse", **kw
    )
    _, rect = beamstop_shadow(
        image, valid, np.random.default_rng(0), shape="rect", **kw
    )
    # a square box strictly contains the inscribed circle of the same half-extent
    assert rect.sum() > ell.sum()
    assert (ell & ~rect).sum() == 0


def test_empty_prediction_does_not_crash():
    image, gt = _toy()
    valid = ~gt
    _, injected = ARTIFACTS["streak"](image, valid, np.random.default_rng(4))
    m = masking_metrics(np.zeros_like(valid), injected, valid)
    assert m["recall"] == 0.0 and m["iou"] == 0.0 and m["f1"] == 0.0
    assert m["fpr"] == 0.0 and m["tp"] == 0


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _run_all()
