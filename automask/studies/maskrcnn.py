"""
studies/maskrcnn.py -- can an OFF-THE-SHELF Mask R-CNN mask a detector?

Same contract as `masking.Pipeline`: consume the reductions of one
`ShotSelection` (the production `_LIT` selection -> `umean`, `ustd`) and emit a
boolean mask. Here the masker is torchvision's `maskrcnn_resnet50_fpn_v2` with
its COCO weights, used **zero-shot** -- no fine-tuning, no labels, nothing
learned from this experiment. Two runs of ground truth are far too few to train
an instance segmenter, so the question is whether the generic
objectness/segmentation prior transfers: does a network trained on people, cars
and dogs propose the dead blocks, seams and shadows of a Jungfrau frame when it
is shown one as an image?

The detector is used **class-agnostically** -- COCO's 91 categories are
meaningless on a diffraction frame, so every instance above a score threshold
counts as "some object is here" and the union of their masks is the predicted
mask. Because the right threshold is not knowable a priori, the study sweeps it
and reports the whole curve, with the best point marked; a single number would
hide that the operating point is being chosen on the test run.

Three input renderings are tried, since the network's prior is about *images*
and how the features are painted into RGB is the only free choice left:

    umean   -- lit-beam mean, grayscale-replicated
    ustd    -- lit-beam std, grayscale-replicated (the variance detector's input)
    stack   -- (umean, ustd, valid-pixel flag) as R/G/B

Each is percentile-stretched to [0, 1]; torchvision applies the ImageNet
normalization itself. Inference is a 512-crop sliding window, so structures are
seen at roughly the object scale COCO was trained at.

Scored against the human masks of runs 389 / 475, next to the production
pipeline and the deterministic geometry+calib floor.

Run (needs torch/torchvision, NOT the psana env):

    python -m automask.studies.maskrcnn
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torchvision

from automask.dataset import score
from automask.evaluation import EVAL_RUNS, load_sample
from automask.masking import production_pipeline

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "outputs", "figures")

CROP = 512
STRIDE = 256
VARIANTS = ("umean", "ustd", "stack")
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


# ==========================================================================
#  1. the masker's input: reductions of one ShotSelection, painted as an image
# ==========================================================================
@dataclass
class RunInput:
    run: int
    umean: np.ndarray
    ustd: np.ndarray
    human: np.ndarray
    floor: np.ndarray
    real: np.ndarray


def load_run(run: int, pipe) -> RunInput:
    sample = load_sample(run, features=pipe.features_needed() + ("umean",))
    return RunInput(run=run, umean=np.asarray(sample.umean, float),
                    ustd=np.asarray(sample.ustd, float), human=sample.human,
                    floor=pipe.floor(sample), real=sample.real)


def _stretch(a: np.ndarray, real: np.ndarray) -> np.ndarray:
    """Percentile-stretch one feature to [0, 1] for a network expecting photos.

    asinh first so Bragg peaks do not eat the whole dynamic range; invalid
    pixels are held at 0 (they read as black holes, which is what they are).
    """
    med = float(np.median(a[real]))
    sd = 1.4826 * float(np.median(np.abs(a[real] - med))) or 1.0
    v = np.arcsinh((a - med) / sd)
    lo, hi = np.percentile(v[real], [1, 99])
    return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1) * real


def render(data: RunInput, variant: str) -> np.ndarray:
    """(3, H, W) float32 in [0, 1] -- the frame as an RGB image."""
    u = _stretch(data.umean, data.real)
    s = _stretch(data.ustd, data.real)
    if variant == "umean":
        img = np.stack([u, u, u])
    elif variant == "ustd":
        img = np.stack([s, s, s])
    elif variant == "stack":
        img = np.stack([u, s, data.real.astype(float)])
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return img.astype(np.float32)


# ==========================================================================
#  2. zero-shot inference: sliding window, class-agnostic instance union
# ==========================================================================
def build_model(device, score_floor: float = 0.05):
    """COCO-pretrained Mask R-CNN, unmodified. `score_floor` only widens what the
    head returns so one pass can be re-thresholded afterwards."""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT", box_score_thresh=score_floor, box_detections_per_img=200)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, img: np.ndarray, device, thresholds=THRESHOLDS,
            mask_thr: float = 0.5) -> dict:
    """Masks for every score threshold in one sliding-window pass.

    Returns ``{threshold: bool mask}``. Detections are kept class-agnostically:
    a COCO label on a detector frame carries no meaning, only the objectness.
    """
    _, H, W = img.shape
    out = {float(t): np.zeros((H, W), dtype=bool) for t in thresholds}
    rows = sorted({*range(0, max(H - CROP, 0) + 1, STRIDE), H - CROP})
    cols = sorted({*range(0, max(W - CROP, 0) + 1, STRIDE), W - CROP})
    n_det = 0
    for r0 in rows:
        for c0 in cols:
            tile = torch.from_numpy(img[:, r0:r0 + CROP, c0:c0 + CROP]).to(device)
            pred = model([tile])[0]
            scores = pred["scores"].cpu().numpy()
            n_det += int((scores >= 0.5).sum())
            if scores.size == 0:
                continue
            masks = (pred["masks"][:, 0] >= mask_thr).cpu().numpy()
            for t in out:
                keep = scores >= t
                if keep.any():
                    out[t][r0:r0 + CROP, c0:c0 + CROP] |= masks[keep].any(0)
    return out, n_det


# ==========================================================================
#  3. study
# ==========================================================================
def agreement_rgb(pred, truth):
    rgb = np.ones(pred.shape + (3,))
    rgb[pred & truth] = (0.15, 0.55, 0.20)          # hit
    rgb[pred & ~truth] = (0.85, 0.20, 0.15)         # false positive
    rgb[~pred & truth] = (0.20, 0.35, 0.85)         # miss
    return rgb


def _row(name, pred, data):
    s = score(pred, data.human)
    res = score(pred & ~data.floor, data.human & ~data.floor)
    print(f"  {name:<34} masked {100*pred.mean():5.2f}%  IoU {s['iou']:.3f}  "
          f"prec {s['precision']:.3f}  rec {s['recall']:.3f}   |  "
          f"residual IoU {res['iou']:.3f}")
    return s


def figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(results)
    fig, axes = plt.subplots(n, 5, figsize=(22, 4.7 * n))
    axes = np.atleast_2d(axes)
    for row, res in enumerate(results):
        d, best = res["data"], res["best"]
        panels = [
            (res["render"].transpose(1, 2, 0), "rgb",
             f"run {d.run}: ShotSelection(beam=on, 800 shots) rendered "
             f"[{res['best_variant']}]"),
            (d.human, "bin", "human reference mask"),
            (res["prod"], "bin",
             f"production pipeline   IoU {res['s_prod']['iou']:.3f}"),
            (best, "bin", f"Mask R-CNN zero-shot (COCO), best of the sweep\n"
                          f"{res['best_variant']} @ score {res['best_thr']:.2f}   "
                          f"IoU {res['s_best']['iou']:.3f}"),
            (agreement_rgb(best, d.human), "rgb", "zero-shot vs human"),
        ]
        for col, (img, kind, title) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img if kind == "rgb" else img, cmap=None if kind == "rgb" else "magma",
                      vmin=None if kind == "rgb" else 0, vmax=None if kind == "rgb" else 1,
                      interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    handles = [Patch(color=(0.15, 0.55, 0.20), label="true positive"),
               Patch(color=(0.85, 0.20, 0.15), label="false positive"),
               Patch(color=(0.20, 0.35, 0.85), label="missed")]
    axes[0, 4].legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle("Off-the-shelf (COCO-pretrained, zero-shot) Mask R-CNN as a detector "
                 "masker, vs the production pipeline", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    print(f"\nfigure -> {path}")


def curve_figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.2),
                             squeeze=False)
    for ax, res in zip(axes[0], results):
        for variant, curve in res["curves"].items():
            ax.plot(THRESHOLDS, [c["iou"] for c in curve], marker="o", ms=3,
                    label=f"zero-shot [{variant}]")
        ax.axhline(res["s_prod"]["iou"], color="k", ls="--",
                   label=f"production ({res['s_prod']['iou']:.3f})")
        ax.axhline(score(res["data"].floor, res["data"].human)["iou"], color="0.5",
                   ls=":", label="geometry+calib floor")
        ax.set_xlabel("detection score threshold"); ax.set_ylabel("IoU vs human mask")
        ax.set_title(f"run {res['data'].run}"); ax.set_ylim(0, 1)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Zero-shot Mask R-CNN: IoU across the whole score-threshold sweep")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"figure -> {path}")


def main(runs: Sequence[int] = EVAL_RUNS, out: Optional[str] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  (COCO-pretrained Mask R-CNN, zero-shot -- no training)")
    pipe = production_pipeline("union")
    model = build_model(device)
    results = []
    for run in runs:
        data = load_run(run, pipe)
        prod = pipe.run(load_sample(run, features=pipe.features_needed()))
        print(f"\n=== run {run} ===")
        s_prod = _row("production pipeline", prod, data)
        _row("geometry+calib floor", data.floor, data)
        curves, best = {}, None
        for variant in VARIANTS:
            img = render(data, variant)
            masks, n_det = predict(model, img, device)
            curves[variant] = [score(masks[float(t)], data.human) for t in THRESHOLDS]
            i = int(np.argmax([c["iou"] for c in curves[variant]]))
            thr = float(THRESHOLDS[i])
            print(f"  -- rendering [{variant}]: {n_det} instances at score>=0.5 "
                  f"over the sliding window")
            _row(f"zero-shot @ 0.50 [{variant}]", masks[0.5], data)
            s = _row(f"zero-shot @ {thr:.2f} best [{variant}]", masks[thr], data)
            _row(f"zero-shot best | floor [{variant}]", masks[thr] | data.floor, data)
            if best is None or s["iou"] > best[0]["iou"]:
                best = (s, masks[thr], thr, variant, img)
        s_best, m_best, thr_best, var_best, img_best = best
        results.append({"data": data, "prod": prod, "s_prod": s_prod,
                        "curves": curves, "best": m_best, "s_best": s_best,
                        "best_thr": thr_best, "best_variant": var_best,
                        "render": img_best})
    base = out or os.path.join(OUT, "maskrcnn_zeroshot.png")
    figure(results, base)
    curve_figure(results, base.replace(".png", "_sweep.png"))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(out=a.out)
