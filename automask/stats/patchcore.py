"""
stats/patchcore.py -- PatchCore anomaly field as a masking statistic.

The industrial-defect-detection standard (Roth et al. 2022), wired into the
STATS registry so it thresholds, regularizes and fuses exactly like `variance`
or `asic_polish`.

How it works: an ImageNet backbone (`wide_resnet50_2`, mid-level layers 2+3 --
deep enough to be textural, shallow enough to stay local) turns the frame into a
grid of patch descriptors. A **memory bank** of descriptors taken from KNOWN-GOOD
regions defines what normal looks like; a patch's anomaly score is the distance
to its nearest bank neighbour. Nothing is trained -- fitting the bank is a forward
pass plus a subsample, so the "2 labelled runs" problem does not apply.

Where the bank comes from is the one real design choice, and both answers are
implemented via `PatchCoreParams.bank_run`:

* ``bank_run=<other run>`` -- fit on another run's clean pixels (its human mask
  says which). This mirrors the lab's actual situation: one run IS hand-masked,
  and the question is whether that one label generalizes. Test-time is
  label-free.
* ``bank_run=None`` -- self-fit on the run under test, excluding only the
  deterministic geometry+calib floor. Fully unsupervised, but the bank then
  contains the very defects being looked for; they are rare, so a RANDOM
  subsample keeps the bank normal-dominated. (PatchCore's greedy coreset is
  deliberately NOT used here: it maximizes diversity, which preferentially keeps
  the outliers -- exactly the wrong bias when the "clean" set is contaminated.)

The frame is painted as an RGB image the backbone can read (`umean`, `ustd` and
the valid-pixel flag in the three channels), asinh-compressed and
percentile-stretched. Output is a robust-z field, `mode="high"` (far from normal
== defective).

Needs torch/torchvision/timm, so this module is deliberately NOT imported by
`stats/__init__.py` -- the psana env must stay torch-free. Import it explicitly
to register it:

    from automask.stats import patchcore   # noqa: F401  (registers "patchcore")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from automask.stats.base import StatSpec, register_stat, robust_z

_MODEL_CACHE: dict = {}
_BANK_CACHE: dict = {}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PatchCoreParams:
    k: float = 3.5
    mode: str = "high"              # far from the normal manifold == defective
    backbone: str = "wide_resnet50_2"
    known: str = "production"       # what already-trusted mask defines "not normal"
    bank_run: Optional[int] = None  # None == fit the bank on the run under test
    fill: str = "none"              # how masked pixels are erased from the INPUT
    #   "none"    -- leave them (the backbone sees dead bands and black gaps)
    #   "median"  -- constant fill: flat plateaus with STEP EDGES the CNN flags
    #   "nearest" -- nearest valid neighbour: no step, but streaky and too smooth
    #   "texture" -- smooth local level + noise matched to the local good scatter,
    #                so a filled region is statistically ordinary background
    n_bank: int = 30000             # descriptors kept in the memory bank
    neighbours: int = 1             # k of the k-NN distance
    blur_sigma: float = 4.0         # smoothing of the upsampled score map
    seed: int = 0


# ==========================================================================
#  what we already know is bad, before PatchCore ever runs
# ==========================================================================
def known_mask(sample, source: str) -> np.ndarray:
    """The trusted mask fed back into PatchCore: `floor` = geometry+calib only,
    `production` = the full production pipeline (floor + variance + hough_lines +
    asic_polish), `human` = the hand mask. Only `human` needs a label."""
    if source == "human":
        return np.asarray(sample.human, bool)
    from automask.masking import production_pipeline
    pipe = production_pipeline("union")
    if source == "floor":
        return pipe.floor(sample)
    if source == "production":
        return pipe.run(sample)
    raise ValueError(f"unknown `known` source {source!r}")


# ==========================================================================
#  rendering: the shot-selection reductions as an image the backbone can read
# ==========================================================================
def _stretch(a: np.ndarray, domain: np.ndarray) -> np.ndarray:
    med = float(np.median(a[domain]))
    sd = 1.4826 * float(np.median(np.abs(a[domain] - med))) or 1.0
    v = np.arcsinh((a - med) / sd)
    lo, hi = np.percentile(v[domain], [1, 99])
    return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)


def _fill(arr: np.ndarray, bad: np.ndarray, mode: str, seed: int) -> np.ndarray:
    """Erase `bad` pixels from an image so nothing there looks like a feature.

    The point is subtle and it is the whole game: a masked region must be made
    *uninteresting*, not merely blanked. A constant fill blanks it -- and hands
    the backbone a perfectly straight, perfectly sharp intensity step along every
    ASIC seam, which is a far stronger edge than any real defect. "texture"
    instead reproduces the local background level AND its scatter, so a filled
    patch is statistically indistinguishable from ordinary detector noise and
    scores no anomaly at all.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    good = ~bad
    if mode == "median":
        return np.where(bad, np.median(arr[good]), arr)
    ind = distance_transform_edt(bad, return_distances=False, return_indices=True)
    near = arr[tuple(ind)]
    if mode == "nearest":
        return np.where(bad, near, arr)
    if mode != "texture":
        raise ValueError(f"unknown fill mode {mode!r}")
    base = gaussian_filter(near, 8.0)               # local level, streaks removed
    w = gaussian_filter(good.astype(float), 8.0)
    scale = (gaussian_filter(np.where(good, np.abs(arr - base), 0.0), 8.0)
             / np.maximum(w, 1e-6)) / 0.7979        # mean|dev| -> sigma
    noise = np.random.default_rng(seed).standard_normal(arr.shape) * scale
    return np.where(bad, base + noise, arr)


def render(sample, known: Optional[np.ndarray] = None, fill: str = "none",
           seed: int = 0) -> np.ndarray:
    """(3, H, W) float32 in [0, 1] -- the frame AFTER the trusted masks.

    `known` is excluded from the contrast stretch (a huge dead band no longer
    sets the dynamic range for everything else) and, when `fill` is not "none",
    erased from the input itself along with the unmapped pixels. What reaches the
    backbone is then the cleaned frame: only the residual is left to be anomalous.
    """
    real = sample.real
    good = real & ~known if known is not None else real
    bad = ~good
    chans = []
    for arr in (np.asarray(sample.umean, float), np.asarray(sample.ustd, float)):
        if fill != "none":
            arr = _fill(arr, bad, fill, seed)
            chans.append(_stretch(arr, good))
        else:
            chans.append(_stretch(arr, good) * real)
    flag = np.ones_like(real, float) if fill != "none" else real.astype(float)
    return np.stack(chans + [flag]).astype(np.float32)


# ==========================================================================
#  backbone -> patch descriptors
# ==========================================================================
def _backbone(name: str):
    import timm
    import torch
    if name not in _MODEL_CACHE:
        model = timm.create_model(name, pretrained=True, features_only=True,
                                  out_indices=(1, 2))
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL_CACHE[name] = (model.eval().to(dev), dev)
    return _MODEL_CACHE[name]


def descriptors(img: np.ndarray, backbone: str):
    """(h, w, C) patch descriptors and the stride of the grid.

    Layer-2 and layer-3 maps are locally 3x3-average-pooled (PatchCore's
    neighbourhood aggregation: a descriptor should describe a patch, not a
    pixel), the deeper map is upsampled to the shallower grid, and the two are
    concatenated.
    """
    import torch
    import torch.nn.functional as F
    model, dev = _backbone(backbone)
    x = torch.from_numpy(img)[None].to(dev)
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    with torch.no_grad():
        feats = model((x - mean) / std)
        pooled = [F.avg_pool2d(f, 3, stride=1, padding=1) for f in feats]
        size = pooled[0].shape[-2:]
        up = [pooled[0]] + [F.interpolate(f, size=size, mode="bilinear",
                                          align_corners=False) for f in pooled[1:]]
        emb = torch.cat(up, dim=1)[0]                  # (C, h, w)
    return emb.permute(1, 2, 0).contiguous(), img.shape[1] / emb.shape[1]


def _patch_is_clean(exclude: np.ndarray, grid_shape, dev):
    """Grid cells whose whole receptive patch is free of `exclude` pixels."""
    import torch
    import torch.nn.functional as F
    m = torch.from_numpy(exclude.astype(np.float32))[None, None].to(dev)
    small = F.adaptive_max_pool2d(m, grid_shape)[0, 0]
    return small < 0.5


def fit_bank(sample, p: PatchCoreParams, exclude: np.ndarray):
    """Memory bank of descriptors from the pixels `exclude` does NOT cover."""
    import torch
    emb, _ = descriptors(render(sample, exclude, p.fill, p.seed), p.backbone)
    clean = _patch_is_clean(exclude, emb.shape[:2], emb.device)
    bank = emb[clean]
    if bank.shape[0] > p.n_bank:
        g = torch.Generator(device="cpu").manual_seed(p.seed)
        idx = torch.randperm(bank.shape[0], generator=g)[:p.n_bank]
        bank = bank[idx.to(bank.device)]
    return bank


def anomaly_map(sample, bank, p: PatchCoreParams,
                known: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-pixel distance-to-normal, upsampled and smoothed to image size."""
    import torch
    import torch.nn.functional as F
    from scipy.ndimage import gaussian_filter
    emb, _ = descriptors(render(sample, known, p.fill, p.seed), p.backbone)
    h, w, c = emb.shape
    q = emb.reshape(-1, c)
    dists = []
    for chunk in torch.split(q, 4096):
        d = torch.cdist(chunk, bank)
        dists.append(d.topk(p.neighbours, largest=False).values.mean(1))
    score = torch.cat(dists).reshape(1, 1, h, w)
    full = F.interpolate(score, size=sample.human.shape, mode="bilinear",
                         align_corners=False)[0, 0].cpu().numpy()
    return gaussian_filter(full, p.blur_sigma) if p.blur_sigma > 0 else full


# ==========================================================================
#  the registered statistic
# ==========================================================================
def _bank_for(sample, p: PatchCoreParams):
    """Fit (or serve) the memory bank, excluding everything already known bad.

    This is where the trusted detectors earn their keep: a self-fitted bank whose
    only exclusion is the geometry+calib floor is contaminated by the very
    defects being hunted (they land in the bank, so their distance to it is
    zero). Excluding the *production* mask instead removes them without any
    hand label, which is what makes an unsupervised bank work at all.
    """
    key = (p.backbone, p.bank_run if p.bank_run is not None else sample.run,
           p.known, p.fill, p.n_bank, p.seed)
    if key in _BANK_CACHE:
        return _BANK_CACHE[key]
    if p.bank_run is None:
        src = sample
    else:
        from automask.evaluation import load_sample
        src = load_sample(p.bank_run, features=("umean", "ustd", "pedestal"))
    _BANK_CACHE[key] = fit_bank(src, p, known_mask(src, p.known))
    return _BANK_CACHE[key]


def patchcore_stat(sample, p: PatchCoreParams) -> np.ndarray:
    """Robust-z of log10 distance-to-normal. High z == far from the bank.

    The z-scale is set over the trusted-good region only: a known dead band is
    not allowed to inflate the MAD and so raise the bar for everything the
    detector is actually being asked to find.
    """
    known = known_mask(sample, p.known)
    score = anomaly_map(sample, _bank_for(sample, p), p, known)
    return robust_z(score, sample.real & ~known, transform=np.log10)


def compute(sample, params: PatchCoreParams | None = None):
    return patchcore_stat(sample, params or PatchCoreParams())


register_stat(StatSpec(
    name="patchcore",
    compute=compute,
    params=PatchCoreParams,
    kind="field",
    mode="high",
    needs=("umean", "ustd"),
    doc="PatchCore: distance from an ImageNet-feature memory bank of normal "
        "patches; high z == unlike anything normal",
))
