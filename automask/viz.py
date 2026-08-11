"""
viz.py -- the project's figures, in four layers built on one rendering primitive.

Plotting lives here, *above* the FeatureStore, rather than as methods on the
frozen spec classes: rendering a selector's effect means resolving it to an
image, which reaches down into ``FeatureStore`` -> XTC. Keeping that here leaves
``ShotSelection`` / ``FeatureSpec`` pure numpy and psana-free (their whole point).

  1. primitive          ``show``               -- render one (1064,1030) array well.
  2. selection views     ``show_feature``       -- one feature/selection as an image.
                         ``compare_selections`` -- a grid to eyeball selector effects.
  3. mask evaluation     ``agree_rgb`` / ``save_agreement`` -- TP/FP/FN overlays.
  4. sweeps              ``plot_results_heatmap`` -- 2-D IoU heatmap over two knobs.

Every function takes an optional ``ax`` and returns its artist/figure, so the
same code works inline (Jupyter/IDE) and headless -- the Agg guard below picks a
non-interactive backend when there is no display.
"""
from __future__ import annotations
import os

import numpy as np
import matplotlib
if not os.environ.get("MPLBACKEND") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── 1. primitive ──────────────────────────────────────────────────────────────
def show(img, ax=None, *, mask=None, robust=True, vmin=None, vmax=None,
         cmap="magma", cbar=True, title=""):
    """Render one assembled detector array ``(1064, 1030)`` with sane defaults.

    Dead/zero and non-finite pixels are shown neutral (never counted in the
    colour scale). ``robust`` clips to the 1st–99th percentile of the valid
    pixels unless explicit ``vmin``/``vmax`` are given. Returns the ``AxesImage``;
    creates its own figure when ``ax`` is None. This is the single place display
    logic lives -- everything else composes it.
    """
    img = np.asarray(img, dtype=float)
    valid = np.isfinite(img) & (img != 0)
    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)
    if vmin is None and vmax is None and robust and valid.any():
        vmin, vmax = np.percentile(img[valid], [1, 99])

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6.2))
    disp = np.ma.array(img, mask=~valid)
    cmap_obj = matplotlib.colormaps[cmap].copy()
    cmap_obj.set_bad(color="0.15")
    im = ax.imshow(disp, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    if cbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    return im


# ── 2. selection / feature views ──────────────────────────────────────────────
def _resolve_spec(spec, reduction):
    """Coerce a catalogue name, a FeatureSpec, or a bare ShotSelection into a
    FeatureSpec the FeatureStore can key on (name is irrelevant to the cache)."""
    from automask.features import get_spec
    from automask.features.base import FeatureSpec
    from automask.shot_selection import ShotSelection
    if isinstance(spec, str):
        return get_spec(spec)
    if isinstance(spec, FeatureSpec):
        return spec
    if isinstance(spec, ShotSelection):
        return FeatureSpec("_", reduction, spec)
    raise TypeError(f"expected str | FeatureSpec | ShotSelection, got {type(spec).__name__}")


def _sel_label(sel, n_used=None):
    """Compact one-line summary of a ShotSelection's knobs, for panel titles.

    ``n_used`` (the actual shots the cached feature was built from) wins over the
    requested ``sel.n_shots`` when known -- so ``n=`` reflects availability, not
    just the config (e.g. a run with fewer accessible shots than requested)."""
    bits = [condition.label() for condition in sel.where]
    if not bits:
        bits.append("all shots")
    if n_used is not None:
        bits.append(f"n={n_used}")
    elif sel.n_shots is not None:
        bits.append(f"n={sel.n_shots}")
    if sel.trim is not None:
        bits.append(f"trim={sel.trim.field}[{sel.trim.low:g},{sel.trim.high:g}]")
    if sel.normalization is not None:
        bits.append(f"norm={sel.normalization}")
    return " ".join(bits)


def _n_used(store, run, spec):
    """Actual shots the cached feature used, from the sidecar (``None`` if absent)."""
    c = store.counts(run, spec)
    return c.get("n_used") if c else None


def show_feature(run, spec, ax=None, store=None, reduction="mean", out=None, **kw):
    """Render one feature image for ``run``.

    ``spec`` may be a catalogue name (``"umean"``), a ``FeatureSpec``, or a bare
    ``ShotSelection`` (paired with ``reduction``, default ``"mean"``). Resolved
    through the FeatureStore -- served from the warm cache if present, else
    computed from XTC (needs the psana env). Saves a PNG when ``out`` is given.
    """
    from automask.features import FeatureStore
    spec = _resolve_spec(spec, reduction)
    store = store or FeatureStore()
    img = store.get(run, spec)
    label = _sel_label(spec.selection, _n_used(store, run, spec))
    title = kw.pop("title", f"run {run:04d} — {spec.reduction} · {label}")
    im = show(img, ax=ax, title=title, **kw)
    if out:
        im.axes.figure.savefig(out, dpi=110, bbox_inches="tight")
    return im


def compare_selections(run, selections, reduction="mean", store=None,
                       shared_scale=True, out=None, **kw):
    """Grid of one feature panel per selection -- the 'effect of different shot
    selectors' figure.

    ``selections`` is a list of ShotSelection (or FeatureSpec / catalogue names).
    With ``shared_scale`` all panels share one 1–99th-percentile colour scale
    (so brightness differences between selectors are real, not per-panel
    autoscaled) plus a single shared colourbar. Returns the Figure.
    """
    from automask.features import FeatureStore
    store = store or FeatureStore()
    specs = [_resolve_spec(s, reduction) for s in selections]
    imgs = [store.get(run, s) for s in specs]

    vmin = vmax = None
    if shared_scale:
        pool = [i[np.isfinite(i) & (i != 0)] for i in imgs]
        pool = np.concatenate(pool) if any(p.size for p in pool) else np.array([])
        if pool.size:
            vmin, vmax = np.percentile(pool, [1, 99])

    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.6), squeeze=False)
    axes = axes[0]
    im = None
    for ax, spec, img in zip(axes, specs, imgs):
        im = show(img, ax=ax, robust=not shared_scale, vmin=vmin, vmax=vmax,
                  cbar=False,
                  title=_sel_label(spec.selection, _n_used(store, run, spec)), **kw)
    if shared_scale and im is not None:
        fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02)
    fig.suptitle(f"run {run:04d} — {reduction}", fontsize=13)
    if out:
        fig.savefig(out, dpi=110, bbox_inches="tight")
    return fig


def show_mask(mask, ax=None, color=(0.85, 0.1, 0.1), title=""):
    """Render a boolean mask on its own: masked pixels in ``color``, rest white."""
    mask = np.asarray(mask, dtype=bool)
    rgb = np.full((*mask.shape, 3), 0.96)
    rgb[mask] = color
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6.2))
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    # Off-white rather than white, with a frame: a sparse mask on a white page is
    # indistinguishable from an empty panel (hough_lines beyond the floor is 0.00%
    # on run 475 -- that has to read as "found nothing", not as a broken figure).
    for spine in ax.spines.values():
        spine.set_color("0.7")
    if title:
        ax.set_title(title, fontsize=11)
    return ax


def _detector_input(detector, sample):
    """The image a detector reads, as (feature name, assembled array), or None."""
    from automask.stats.base import STATS
    from automask.geometry import panel_to_asm
    for name in STATS[detector.stat].needs:
        if name in ("real", "center", "calib"):
            continue
        arr = getattr(sample, name, None)
        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim == 3:
            arr = panel_to_asm(arr, sample.run)
        if arr.shape == sample.sumimg.shape:
            return name, arr
    return None


def detector_panels(pipeline, sample, out=None, floor_row=True, store=None):
    """One row per masking channel: its input image left, the mask it gives right."""
    from automask.features import FEATURES, FeatureStore

    base = pipeline.floor(sample)
    rows, skipped = [], []
    if floor_row:
        rows.append((None, "sumimg", sample.sumimg))
    for d in pipeline.detectors:
        got = _detector_input(d, sample)
        if got is None:
            skipped.append(d.stat)
            continue
        rows.append((d, *got))
    if not rows:
        raise ValueError("no detector in this pipeline reads an assembled image")

    store = store or FeatureStore()
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 5.4 * len(rows)),
                             squeeze=False)
    for (d, fname, img), (ax_l, ax_r) in zip(rows, axes):
        spec = FEATURES.get(fname)
        if spec is not None and spec.selection is not None:
            label = f"{fname} — {_sel_label(spec.selection, _n_used(store, sample.run, spec))}"
        elif spec is not None:
            label = f"{fname} — calib constant (assembled)"
        else:
            label = f"{fname} — run sum"
        if d is None:
            mask, name = base, "+".join(pipeline.floor_stats) + " floor"
        else:
            mask, name = d.pick(sample) & ~base, f"{d.stat} (beyond floor)"
        show(img, ax=ax_l, title=label, cbar=False)
        show_mask(mask, ax=ax_r,
                  title=f"{name} — {100 * mask.mean():.2f}% masked")

    note = f"   (skipped: {', '.join(skipped)} — non-assembled input)" if skipped else ""
    fig.suptitle(f"run {sample.run:04d} — pipeline inputs and masks{note}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if out:
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return fig


# ── 3. mask evaluation ────────────────────────────────────────────────────────
def agree_rgb(pred, truth):
    """RGB agreement map: green=TP, red=FP, blue=FN, white=TN."""
    a = np.ones((*truth.shape, 3))
    a[pred & truth] = (0.0, 0.7, 0.0)
    a[pred & ~truth] = (0.9, 0.0, 0.0)
    a[~pred & truth] = (0.0, 0.3, 1.0)
    return a


def save_agreement(pred, floor, human, run, out, title=""):
    """Two-panel residual figure: pick-vs-residual-target error map + full mask."""
    from automask.dataset import score
    target = human & ~floor
    resid = score(pred & ~floor, target)
    full = score(pred, human)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
    ax[0].imshow(agree_rgb(pred & ~floor, target)); ax[0].axis("off")
    ax[0].set_title(f"residual (green=TP red=FP blue=FN)\n"
                    f"T-IoU={resid['iou']:.3f} p={resid['precision']:.2f} "
                    f"r={resid['recall']:.2f}", fontsize=11)
    ax[1].imshow(agree_rgb(pred, human)); ax[1].axis("off")
    ax[1].set_title(f"full mask vs human\nIoU={full['iou']:.3f} "
                    f"p={full['precision']:.2f} r={full['recall']:.2f}", fontsize=11)
    fig.suptitle(title or f"run {run} — masking result", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig)
    return out


# ── 4. sweeps ─────────────────────────────────────────────────────────────────
def plot_results_heatmap(csv_path, xcol, ycol, out, metric="mean_iou"):
    """2-D `metric` heatmap over two swept columns from a sweep results.csv."""
    import csv
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        raise ValueError(f"empty results csv: {csv_path}")
    xs = sorted({float(r[xcol]) for r in rows})
    ys = sorted({float(r[ycol]) for r in rows})
    H = np.full((len(ys), len(xs)), np.nan)
    for r in rows:
        i = ys.index(float(r[ycol])); j = xs.index(float(r[xcol]))
        H[i, j] = float(r[metric])
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(xs), 1.6 + 1.0 * len(ys)))
    im = ax.imshow(H, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, len(xs), 0, len(ys)])
    bi, bj = np.unravel_index(np.nanargmax(H), H.shape)
    ax.scatter([bj + 0.5], [bi + 0.5], marker="*", s=220, color="red", edgecolor="w",
               label=f"best {metric}={H[bi,bj]:.3f}")
    ax.set_xticks(np.arange(len(xs)) + 0.5); ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_yticks(np.arange(len(ys)) + 0.5); ax.set_yticklabels([f"{y:g}" for y in ys])
    ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"{metric} over ({xcol}, {ycol})")
    fig.colorbar(im, ax=ax, fraction=0.046, label=metric)
    fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out
