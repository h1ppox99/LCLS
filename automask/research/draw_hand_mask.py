#!/usr/bin/env python3
"""Interactive hand-mask editor for the Jungfrau1M assembled image.

Numpy + matplotlib ONLY -- no psana, no h5py, no `automask` import at runtime,
so this single file plus a couple of ``.npy`` arrays runs on a laptop with a
normal matplotlib GUI window.  Masks follow the project convention
``True == masked``.

Workflow (see ``bundle_kit`` for the cluster side):
  * You draw *on top of* a frozen "floor base" = pad(calib, 2) | geometry(pad 2)
    -- the pipeline's 100%-precision floor (padded pixel-status mask + ASIC/gap
    lines).  The base is always included and shown in blue; you add regions
    (beam stop, streaks, corner, beam center) in red and erase in green.
  * Final mask = ``(base | added) & ~erased`` == floor | hand.

Run standalone::

    python draw_hand_mask.py --sum mean_run0475_asm.npy \\
                             --base base_run0475_asm.npy --out human_run0475.npy

or from a notebook / the repo::

    from automask.research.draw_hand_mask import editor
    editor(run=475)

Controls (also printed on launch and shown in the title bar):
  r rect   l line   o circle   p polygon   b brush   -- pick a tool
  e        toggle ADD / ERASE
  [ / ]    brush / line / circle radius down / up
  drag     rect/circle/line: press-drag-release;  brush: press-drag to paint
  click    polygon: click vertices, ENTER closes, ESC cancels
  u undo   c clear-hand   h toggle base   g toggle background
  s save   q quit
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _has_display():
    """True if a GUI backend can plausibly open (avoid crashing headless Qt)."""
    import sys

    if sys.platform == "darwin" or sys.platform.startswith("win"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# --------------------------------------------------------------------------- #
# rasterization -- pure numpy, shape list -> boolean add/erase layers
# --------------------------------------------------------------------------- #
def _rasterize(shapes, shape):
    """Return (add, erase) boolean layers from a list of (kind, geom, op)."""
    add = np.zeros(shape, dtype=bool)
    era = np.zeros(shape, dtype=bool)
    for kind, geom, op in shapes:
        m = add if op == "add" else era
        if kind == "rect":
            r0, r1, c0, c1 = geom
            m[r0 : r1 + 1, c0 : c1 + 1] = True
        elif kind == "circle":
            r, c, rad = geom
            r0 = max(0, int(r - rad))
            r1 = min(shape[0], int(r + rad) + 1)
            c0 = max(0, int(c - rad))
            c1 = min(shape[1], int(c + rad) + 1)
            if r1 <= r0 or c1 <= c0:
                continue
            yy, xx = np.ogrid[r0:r1, c0:c1]
            m[r0:r1, c0:c1] |= (yy - r) ** 2 + (xx - c) ** 2 <= rad * rad
        elif kind == "line":
            r0, c0, r1, c1, rad = geom  # thick segment (capsule)
            # Effective half-width: rad==0 gives a continuous 1px line (a pure
            # rad*rad test would leave gaps on diagonals). Width ~= 2*rad+1.
            eff = max(float(rad), 0.5)
            pad = int(np.ceil(eff))
            lo_r = max(0, int(min(r0, r1)) - pad)
            hi_r = min(shape[0], int(max(r0, r1)) + pad + 1)
            lo_c = max(0, int(min(c0, c1)) - pad)
            hi_c = min(shape[1], int(max(c0, c1)) + pad + 1)
            if hi_r <= lo_r or hi_c <= lo_c:
                continue
            yy, xx = np.mgrid[lo_r:hi_r, lo_c:hi_c].astype(np.float64)
            dr, dc = r1 - r0, c1 - c0
            seg2 = dr * dr + dc * dc
            t = (
                0.0
                if seg2 == 0
                else np.clip(((yy - r0) * dr + (xx - c0) * dc) / seg2, 0, 1)
            )
            dist2 = (yy - (r0 + t * dr)) ** 2 + (xx - (c0 + t * dc)) ** 2
            m[lo_r:hi_r, lo_c:hi_c] |= dist2 <= eff * eff
        elif kind == "poly":
            if len(geom) < 3:
                continue
            from matplotlib.path import Path

            rr = [p[0] for p in geom]
            cc = [p[1] for p in geom]
            r0 = max(0, int(min(rr)))
            r1 = min(shape[0], int(max(rr)) + 1)
            c0 = max(0, int(min(cc)))
            c1 = min(shape[1], int(max(cc)) + 1)
            if r1 <= r0 or c1 <= c0:
                continue
            yy, xx = np.mgrid[r0:r1, c0:c1]
            pts = np.column_stack([xx.ravel(), yy.ravel()])  # (x=col, y=row)
            verts = [(p[1], p[0]) for p in geom]
            inside = Path(verts).contains_points(pts).reshape(yy.shape)
            m[r0:r1, c0:c1] |= inside
    return add, era


def compose(base, shapes):
    """Final mask = (base | added) & ~erased."""
    add, era = _rasterize(shapes, base.shape)
    return (base | add) & ~era


# --------------------------------------------------------------------------- #
# shape <-> json (resumable)
# --------------------------------------------------------------------------- #
def _shapes_to_json(shapes):
    return [
        {"kind": k, "geom": [list(v) for v in g] if k == "poly" else list(g), "op": op}
        for k, g, op in shapes
    ]


def _shapes_from_json(raw):
    out = []
    for s in raw:
        g = [tuple(v) for v in s["geom"]] if s["kind"] == "poly" else tuple(s["geom"])
        out.append((s["kind"], g, s.get("op", "add")))
    return out


# --------------------------------------------------------------------------- #
# background scaling for display
# --------------------------------------------------------------------------- #
def _disp(image):
    a = np.asarray(image, dtype=np.float64)
    a = np.maximum(a, 0)
    lo, hi = np.percentile(a[a > 0], (1, 99)) if (a > 0).any() else (0, 1)
    a = np.log1p(np.clip(a, 0, hi))
    return a


def _overlay(base, add, era, show_base=True):
    """RGBA overlay: base=blue, added=red, erased(from base)=green."""
    h, w = base.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    if show_base:
        b = base & ~era
        rgba[b] = (0.20, 0.45, 1.0, 0.45)
    rgba[add] = (1.0, 0.15, 0.15, 0.50)
    erased_from_base = era & base
    rgba[erased_from_base] = (0.10, 0.90, 0.30, 0.55)
    return rgba


# --------------------------------------------------------------------------- #
# the editor
# --------------------------------------------------------------------------- #
def editor(
    run=None,
    *,
    sum_img=None,
    base_mask=None,
    shapes_json=None,
    out=None,
    out_shapes=None,
    brush=8,
):
    """Launch the interactive editor.

    Provide either ``run`` (resolves arrays from the repo via automask.evaluation.dataset)
    or explicit ``sum_img`` / ``base_mask`` paths-or-arrays (laptop / kit mode).
    """
    image, base, dflt_out, dflt_json = _resolve_inputs(
        run, sum_img, base_mask, out, out_shapes
    )
    out = out or dflt_out
    # Derive the shapes sidecar from the mask name so it is unique per run.
    # (Kit mode has no --run, which otherwise made every session save to the
    # same "hand_hand.json".)
    out_shapes = out_shapes or (os.path.splitext(out)[0] + ".json")
    shapes_json = shapes_json or out_shapes  # auto-resume from the same file

    shapes = []
    if shapes_json and os.path.exists(shapes_json):
        with open(shapes_json) as f:
            shapes = _shapes_from_json(json.load(f))
        print(f"[load] resumed {len(shapes)} shapes from {shapes_json}")

    import matplotlib
    import matplotlib.pyplot as plt

    backend = matplotlib.get_backend().lower()
    # If we're stuck on a non-interactive backend (e.g. Agg) *and* a display is
    # actually available, try to grab a GUI backend.  We gate on a display first
    # because loading Qt/Tk with no display can hard-abort the interpreter (an
    # uncatchable crash, not a Python exception).  On a laptop matplotlib already
    # defaults to an interactive backend, so this block usually does nothing.
    if (
        backend.endswith("agg")
        and "inline" not in backend
        and "nbagg" not in backend
        and _has_display()
    ):
        for cand in ("QtAgg", "TkAgg"):
            try:
                plt.switch_backend(cand)
                break
            except Exception:
                continue
    backend = matplotlib.get_backend().lower()
    interactive = not (backend.endswith("agg") and "nbagg" not in backend)
    if not interactive:
        print(
            "[warn] no interactive backend (backend=%s): a window can't open. "
            "Use a laptop GUI, `ssh -X`, or `%%matplotlib widget` in a notebook. "
            "Returning the composed mask without drawing." % matplotlib.get_backend()
        )
        return compose(base, shapes)

    state = {
        "mode": "rect",
        "op": "add",
        "rad": brush,
        "press": None,
        "poly": [],
        "temp": [],
        "artist": None,
        "show_base": True,
        "show_bg": True,
    }

    fig, ax = plt.subplots(figsize=(11, 11))
    bg = ax.imshow(_disp(image), cmap="gray", origin="upper", interpolation="nearest")
    add, era = _rasterize(shapes, base.shape)
    ov = ax.imshow(_overlay(base, add, era), origin="upper", interpolation="nearest")
    ttl = ax.set_title("")
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    # Lock the view to the full image and disable autoscale: otherwise the
    # rubber-band `ax.plot` previews (line/poly) rescale the axes to the tiny
    # preview and the whole image "collapses".
    ax.set_xlim(-0.5, base.shape[1] - 0.5)
    ax.set_ylim(base.shape[0] - 0.5, -0.5)
    ax.set_autoscale_on(False)

    def data_xy(event):
        """Pointer (row, col) clamped to the image, valid even off the axes.

        `event.xdata/ydata` go None outside the axes, which froze drags at the
        border; the display->data transform still works there, so we use it and
        clamp so a shape dragged past the edge snaps flush to it.
        """
        if event.x is None or event.y is None:
            return None
        c, r = ax.transData.inverted().transform((event.x, event.y))
        r = min(max(float(r), 0.0), base.shape[0] - 1.0)
        c = min(max(float(c), 0.0), base.shape[1] - 1.0)
        return r, c

    def clear_temp():
        for a in state["temp"]:
            try:
                a.remove()
            except Exception:
                pass
        state["temp"] = []

    def refresh():
        add, era = _rasterize(shapes, base.shape)
        ov.set_data(_overlay(base, add, era, state["show_base"]))
        bg.set_visible(state["show_bg"])
        final = (base | add) & ~era
        ttl.set_text(
            f"run={run}  tool={state['mode']}  mode={state['op'].upper()}  "
            f"brush={state['rad']}px   masked={100 * final.mean():.2f}%  "
            f"shapes={len(shapes)}\n"
            "r rect  l line  o circle  p poly  b brush | e add/erase | [ ] size | "
            "u undo  c clear  h base  g bg | s save  q quit"
        )
        fig.canvas.draw_idle()

    # ---- committing shapes -------------------------------------------------
    def commit(shape):
        shapes.append(shape)
        clear_temp()
        state["press"] = None
        refresh()

    def on_press(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        tb = getattr(fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return
        r, c = event.ydata, event.xdata
        if state["mode"] == "poly":
            state["poly"].append((float(r), float(c)))
            xs = [p[1] for p in state["poly"]]
            ys = [p[0] for p in state["poly"]]
            clear_temp()
            state["temp"].append(
                ax.plot(xs, ys, "y.-", ms=8, lw=1, scalex=False, scaley=False)[0]
            )
            refresh()
            return
        state["press"] = (r, c)
        if state["mode"] == "brush":  # paint immediately
            commit(
                ("circle", (int(round(r)), int(round(c)), state["rad"]), state["op"])
            )
            state["press"] = (r, c)  # keep painting on drag

    def on_motion(event):
        if state["press"] is None:
            return
        xy = data_xy(event)
        if xy is None:
            return
        r0, c0 = state["press"]
        r, c = xy
        if state["mode"] == "brush":
            if (r - r0) ** 2 + (c - c0) ** 2 >= (state["rad"] / 2.0) ** 2:
                shapes.append(
                    (
                        "circle",
                        (int(round(r)), int(round(c)), state["rad"]),
                        state["op"],
                    )
                )
                state["press"] = (r, c)
                refresh()
            return
        # rubber-band preview for rect / circle
        if state["artist"] is not None:
            try:
                state["artist"].remove()
            except Exception:
                pass
        import matplotlib.patches as mp

        if state["mode"] == "rect":
            state["artist"] = mp.Rectangle(
                (min(c0, c), min(r0, r)),
                abs(c - c0),
                abs(r - r0),
                fill=False,
                ec="yellow",
                lw=1.5,
            )
        elif state["mode"] == "line":
            state["artist"] = ax.plot(
                [c0, c],
                [r0, r],
                "-",
                color="yellow",
                lw=max(1.5, state["rad"]),
                scalex=False,
                scaley=False,
            )[0]
        else:  # circle
            rad = ((r - r0) ** 2 + (c - c0) ** 2) ** 0.5
            state["artist"] = mp.Circle((c0, r0), rad, fill=False, ec="yellow", lw=1.5)
        if state["mode"] != "line":
            ax.add_patch(state["artist"])
        fig.canvas.draw_idle()

    def on_release(event):
        if state["press"] is None or state["mode"] in ("poly", "brush"):
            state["press"] = None
            return
        if state["artist"] is not None:
            try:
                state["artist"].remove()
            except Exception:
                pass
            state["artist"] = None
        r0, c0 = state["press"]
        xy = data_xy(event)
        r, c = xy if xy is not None else (r0, c0)
        if state["mode"] == "rect":
            rr0, rr1 = sorted((int(round(r0)), int(round(r))))
            cc0, cc1 = sorted((int(round(c0)), int(round(c))))
            if rr1 > rr0 and cc1 > cc0:
                commit(("rect", (rr0, rr1, cc0, cc1), state["op"]))
        elif state["mode"] == "line":
            commit(
                (
                    "line",
                    (
                        int(round(r0)),
                        int(round(c0)),
                        int(round(r)),
                        int(round(c)),
                        state["rad"],
                    ),
                    state["op"],
                )
            )
        else:
            rad = int(round(((r - r0) ** 2 + (c - c0) ** 2) ** 0.5))
            if rad > 0:
                commit(("circle", (int(round(r0)), int(round(c0)), rad), state["op"]))
        state["press"] = None

    def save():
        final = compose(base, shapes)
        np.save(out, final.astype(bool))
        with open(out_shapes, "w") as f:
            json.dump(_shapes_to_json(shapes), f, indent=1)
        png = os.path.splitext(out)[0] + ".png"
        fig.savefig(png, dpi=130)
        print(f"[save] mask -> {out}  ({100 * final.mean():.2f}% masked)")
        print(f"[save] shapes -> {out_shapes}   preview -> {png}")

    def on_key(event):
        k = (event.key or "").lower()
        if k in ("r", "l", "o", "p", "b"):
            state["mode"] = {
                "r": "rect",
                "l": "line",
                "o": "circle",
                "p": "poly",
                "b": "brush",
            }[k]
            state["poly"] = []
            clear_temp()
            refresh()
        elif k == "e":
            state["op"] = "erase" if state["op"] == "add" else "add"
            refresh()
        elif k in ("[", "]"):
            # rad 0 => 1px line (and 1px brush); width ~= 2*rad+1
            state["rad"] = max(0, state["rad"] + (1 if k == "]" else -1))
            refresh()
        elif k == "enter":
            if len(state["poly"]) >= 3:
                commit(("poly", list(state["poly"]), state["op"]))
            state["poly"] = []
            clear_temp()
            refresh()
        elif k == "escape":
            state["poly"] = []
            state["press"] = None
            clear_temp()
            refresh()
        elif k == "u":
            if shapes:
                shapes.pop()
                refresh()
        elif k == "c":
            shapes.clear()
            state["poly"] = []
            clear_temp()
            refresh()
        elif k == "h":
            state["show_base"] = not state["show_base"]
            refresh()
        elif k == "g":
            state["show_bg"] = not state["show_bg"]
            refresh()
        elif k == "s":
            save()
        elif k == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    print(__doc__.split("Controls")[1] if "Controls" in __doc__ else "")
    plt.show()
    return compose(base, shapes)


# --------------------------------------------------------------------------- #
# input resolution: repo mode (run=) vs kit mode (explicit paths/arrays)
# --------------------------------------------------------------------------- #
def _as_array(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    return np.load(x)


def run_image(run):
    """The assembled lit mean for `run`, from ImageStore (psana on a cache miss).

    Repo/cluster only -- kit mode never calls this, which is what keeps the
    laptop side numpy-only.
    """
    from automask.sample.image_store import ImageStore
    from automask.selection.presets import BEAM_ON_SELECTION

    return ImageStore().reduce(run, BEAM_ON_SELECTION, "mean").astype(np.float64)


def _resolve_inputs(run, sum_img, base_mask, out, out_shapes):
    image = _as_array(sum_img)
    base = _as_array(base_mask)
    if (image is None or base is None) and run is not None:
        # repo mode: compute from the run itself
        if image is None:
            image = run_image(run)
        if base is None:
            base = build_base(run)
    if image is None or base is None:
        raise ValueError(
            "need either run= (repo mode) or both sum_img= and base_mask= (kit mode)"
        )
    base = np.asarray(base).astype(bool)
    tag = f"run{run:04d}" if run is not None else "hand"
    dflt_out = f"human_Mask_{tag}_asm.npy"
    dflt_json = f"hand_{tag}.json"
    return image, base, dflt_out, dflt_json


def build_base(run):
    """The pipeline's own 100%-precision floor for `run`.  Repo/cluster only.

    Not a reimplementation of it: this runs `Pipeline.floor` over the registered
    floor channels (geometry + psana pixel status), so the reference the user
    draws == floor | hand is scored on exactly the same footing as the masker's
    floor, and stays that way if the floor changes.
    """
    from automask.mask import Pipeline, floor_channels
    from automask.sample import Sample
    from automask.selection.presets import BEAM_ON_SELECTION

    pipeline = Pipeline(floor_channels())
    sample = Sample.from_store(run, BEAM_ON_SELECTION, pipeline.needs())
    return pipeline.floor(sample).astype(bool)


# --------------------------------------------------------------------------- #
# cluster side: freeze base + bundle a portable kit to scp to a laptop
# --------------------------------------------------------------------------- #
def bundle_kit(run, dest):
    """Write {image, base, editor, README} for `run` into `dest` (a scp-able dir).

    Everything psana-dependent happens HERE, on the cluster; the kit itself is
    two .npy files the laptop opens with numpy alone.
    """
    os.makedirs(dest, exist_ok=True)
    image = run_image(run).astype(np.float32)
    base = build_base(run)
    np.save(os.path.join(dest, f"mean_run{run:04d}_asm.npy"), image)
    np.save(os.path.join(dest, f"base_run{run:04d}_asm.npy"), base)
    import shutil

    shutil.copy(os.path.abspath(__file__), os.path.join(dest, "draw_hand_mask.py"))
    readme = (
        f"# Hand-mask kit for run {run}\n\n"
        "Needs only numpy + matplotlib on your laptop:\n"
        "    pip install numpy matplotlib\n\n"
        "Draw:\n"
        f"    python draw_hand_mask.py --sum mean_run{run:04d}_asm.npy \\\n"
        f"        --base base_run{run:04d}_asm.npy --out human_Mask_run{run:04d}_asm.npy\n\n"
        f"Then scp human_Mask_run{run:04d}_asm.npy (+ hand_run{run:04d}.json) back.\n"
    )
    with open(os.path.join(dest, "README.md"), "w") as f:
        f.write(readme)
    print(f"[kit] run {run}: base={100 * base.mean():.2f}% masked -> {dest}")
    return dest


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", type=int, default=None, help="repo mode: run number")
    p.add_argument(
        "--sum",
        dest="sum_img",
        default=None,
        help="kit mode: sum image .npy (background)",
    )
    p.add_argument(
        "--base", dest="base_mask", default=None, help="kit mode: dead-pixel base .npy"
    )
    p.add_argument(
        "--shapes",
        dest="shapes_json",
        default=None,
        help="resume from this shapes .json",
    )
    p.add_argument("--out", default=None, help="output mask .npy")
    p.add_argument("--brush", type=int, default=8, help="initial brush radius (px)")
    p.add_argument(
        "--bundle",
        metavar="DEST",
        default=None,
        help="cluster: write a portable kit for --run into DEST and exit",
    )
    args = p.parse_args()
    if args.bundle:
        if args.run is None:
            p.error("--bundle requires --run")
        bundle_kit(args.run, args.bundle)
        return
    editor(
        run=args.run,
        sum_img=args.sum_img,
        base_mask=args.base_mask,
        shapes_json=args.shapes_json,
        out=args.out,
        brush=args.brush,
    )


if __name__ == "__main__":
    main()
