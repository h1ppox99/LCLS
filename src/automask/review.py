#!/usr/bin/env python3
"""Interactively accept or reject proposed masks for images in ``data/``.

Mask convention: boolean ``True`` means excluded/masked.  A masker is supplied
as ``module:function`` and must accept one 2-D NumPy array and return a boolean
array with the same shape.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib
import inspect
import io
import os
import random
import shutil
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DEFAULT_IMAGES = DATA / "images"
LABELS = DATA / "labels"
PROPOSALS = DATA / "proposals"
REFERENCES = DATA / "reference_masks"
VERDICTS = LABELS / "verdicts.csv"
Masker = Callable[[np.ndarray], np.ndarray]


def placeholder_masker(image: np.ndarray) -> np.ndarray:
    """Visible no-op used only to smoke-test the UI before an adapter exists."""
    return np.zeros(image.shape, dtype=bool)


def resolve_masker(spec: str | None) -> tuple[Masker, str]:
    if not spec:
        return placeholder_masker, "review:placeholder_masker"
    if ":" not in spec:
        raise ValueError("masker must be written as module:function")
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"{spec} is not callable")
    return function, spec


def load_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        image = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        key = "image" if "image" in archive else archive.files[0]
        image = archive[key]
    elif suffix in {".tif", ".tiff"}:
        import tifffile
        image = tifffile.imread(path)
    else:
        raise ValueError(f"unsupported image type: {path}")
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D array, got {image.shape}")
    return image


def validate_mask(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape != image.shape:
        raise ValueError(f"mask shape {mask.shape} does not match image shape {image.shape}")
    if mask.dtype != np.bool_:
        if not np.all((mask == 0) | (mask == 1)):
            raise TypeError("mask must be boolean or contain only 0/1")
        mask = mask.astype(bool)
    return mask


def display_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    transformed = np.sign(finite) * np.log1p(np.abs(finite))
    low, high = np.percentile(transformed, (1.0, 99.8))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def display_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    return np.sign(image) * np.log1p(np.abs(image))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def masker_fingerprint(masker: Masker, spec: str) -> str:
    try:
        source = inspect.getsource(masker).encode()
    except (OSError, TypeError):
        source = repr(masker).encode()
    return hashlib.sha256(spec.encode() + b"\0" + source).hexdigest()


FIELDS = [
    "timestamp_utc", "image_id", "image_path", "image_sha256", "verdict",
    "proposal_path", "proposal_sha256", "masker", "masker_sha256", "masked_fraction",
]


def read_latest_verdicts() -> dict[str, str]:
    if not VERDICTS.exists():
        return {}
    with VERDICTS.open(newline="", encoding="utf-8") as stream:
        return {row["image_id"]: row["verdict"] for row in csv.DictReader(stream)}


def append_verdict(row: dict[str, str]) -> None:
    LABELS.mkdir(parents=True, exist_ok=True)
    exists = VERDICTS.exists()
    with VERDICTS.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@dataclass
class ReviewItem:
    path: Path
    image: np.ndarray
    mask: np.ndarray
    proposal_path: Path


class ReviewSession:
    def __init__(self, paths: list[Path], masker: Masker, masker_spec: str, revisit: bool = False):
        self.paths = paths
        self.masker = masker
        self.masker_spec = masker_spec
        self.masker_hash = masker_fingerprint(masker, masker_spec)
        self.done = set() if revisit else set(read_latest_verdicts())
        self.position = 0
        self.history: list[str] = []
        self.item: ReviewItem | None = None

    def next_path(self) -> Path | None:
        while self.position < len(self.paths):
            path = self.paths[self.position]
            self.position += 1
            if path.stem not in self.done:
                return path
        return None

    def prepare(self, path: Path) -> ReviewItem:
        image = load_image(path)
        mask = validate_mask(self.masker(image), image)
        PROPOSALS.mkdir(parents=True, exist_ok=True)
        proposal_path = PROPOSALS / f"{path.stem}__{self.masker_hash[:12]}.npy"
        np.save(proposal_path, mask, allow_pickle=False)
        return ReviewItem(path, image, mask, proposal_path)

    def advance(self) -> ReviewItem | None:
        path = self.next_path()
        if path is None:
            self.item = None
            return None
        try:
            self.item = self.prepare(path)
        except Exception as exc:
            raise RuntimeError(f"failed while preparing {path}") from exc
        return self.item

    def record(self, verdict: str) -> None:
        if self.item is None:
            return
        item = self.item
        proposal_hash = file_sha256(item.proposal_path)
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "image_id": item.path.stem,
            "image_path": str(item.path.relative_to(ROOT)),
            "image_sha256": file_sha256(item.path),
            "verdict": verdict,
            "proposal_path": str(item.proposal_path.relative_to(ROOT)),
            "proposal_sha256": proposal_hash,
            "masker": self.masker_spec,
            "masker_sha256": self.masker_hash,
            "masked_fraction": f"{item.mask.mean():.9f}",
        }
        append_verdict(row)
        reference = REFERENCES / f"{item.path.stem}.npy"
        REFERENCES.mkdir(parents=True, exist_ok=True)
        if verdict == "yes":
            shutil.copyfile(item.proposal_path, reference)
        elif reference.exists():
            reference.unlink()
        self.done.add(item.path.stem)
        self.history.append(item.path.stem)
        self.item = None


class MaskReviewer:
    def __init__(self, paths: list[Path], masker: Masker, masker_spec: str, revisit: bool = False):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self.plt = plt
        self.session = ReviewSession(paths, masker, masker_spec, revisit=revisit)
        self.figure, self.axes = plt.subplots(1, 2, figsize=(14, 7))
        self.figure.subplots_adjust(bottom=0.16, wspace=0.04)
        yes_axis = self.figure.add_axes((0.35, 0.035, 0.12, 0.065))
        no_axis = self.figure.add_axes((0.53, 0.035, 0.12, 0.065))
        self.yes_button = Button(yes_axis, "Yes [Y]", color="#b8e6b8", hovercolor="#83d283")
        self.no_button = Button(no_axis, "No [N]", color="#f2b6b6", hovercolor="#e68181")
        self.yes_button.on_clicked(lambda _: self.record("yes"))
        self.no_button.on_clicked(lambda _: self.record("no"))
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.show_next()

    def show_next(self) -> None:
        try:
            item = self.session.advance()
        except Exception:
            self.plt.close(self.figure)
            raise
        if item is None:
            for axis in self.axes:
                axis.clear()
                axis.axis("off")
            self.axes[0].text(0.5, 0.5, "Review complete", ha="center", va="center", fontsize=22)
            self.figure.canvas.draw_idle()
            return
        image, mask = item.image, item.mask
        shown = display_image(image)
        low, high = display_limits(image)
        for axis in self.axes:
            axis.clear()
            axis.imshow(shown, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
            axis.axis("off")
        overlay = np.zeros((*mask.shape, 4), dtype=float)
        overlay[mask] = (1.0, 0.05, 0.05, 0.55)
        self.axes[1].imshow(overlay, interpolation="nearest")
        self.axes[0].set_title("Diffraction image")
        self.axes[1].set_title(f"Proposed mask (red, {100 * mask.mean():.2f}% excluded)")
        self.figure.suptitle(
            f"{item.path.stem}  |  reviewed {len(self.session.done)}/{len(self.session.paths)}"
        )
        self.figure.canvas.draw_idle()

    def record(self, verdict: str) -> None:
        self.session.record(verdict)
        self.show_next()

    def on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key in {"y", "enter"}:
            self.record("yes")
        elif key == "n":
            self.record("no")
        elif key in {"q", "escape"}:
            self.plt.close(self.figure)

    def run(self) -> None:
        self.plt.show()


class WebMaskReviewer:
    """Dependency-free browser UI for SSH/headless compute nodes."""

    def __init__(
        self, paths: list[Path], masker: Masker, masker_spec: str,
        revisit: bool = False, host: str = "127.0.0.1", port: int = 8765,
    ):
        self.session = ReviewSession(paths, masker, masker_spec, revisit=revisit)
        self.host, self.port = host, port
        self.session.advance()

    def render_png(self) -> bytes:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        item = self.session.item
        if item is None:
            return b""
        image, mask = item.image, item.mask
        shown = display_image(image)
        low, high = display_limits(image)
        figure = Figure(figsize=(14, 7), tight_layout=True)
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 2)
        for axis in axes:
            axis.imshow(shown, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
            axis.axis("off")
        overlay = np.zeros((*mask.shape, 4), dtype=float)
        overlay[mask] = (1.0, 0.05, 0.05, 0.55)
        axes[1].imshow(overlay, interpolation="nearest")
        axes[0].set_title("Diffraction image")
        axes[1].set_title(f"Proposed mask (red, {100 * mask.mean():.2f}% excluded)")
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=110)
        return buffer.getvalue()

    def page(self) -> bytes:
        item = self.session.item
        if item is None:
            content = "<h1>Review complete</h1><p>You can stop the server with Ctrl-C.</p>"
        else:
            image_id = html.escape(item.path.stem)
            content = f"""
              <h1>{image_id}</h1>
              <p>Reviewed {len(self.session.done)} of {len(self.session.paths)} images.</p>
              <img src="/image.png" alt="Original image and proposed mask">
              <div class="buttons">
                <form method="post" action="/verdict"><button class="yes" name="value" value="yes">Yes <kbd>Y</kbd></button></form>
                <form method="post" action="/verdict"><button class="no" name="value" value="no">No <kbd>N</kbd></button></form>
              </div>
            """
        document = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Diffraction mask review</title>
<style>
body {{ margin: 0 auto; max-width: 1500px; padding: 12px 24px 30px; font: 16px sans-serif; background: #17191d; color: #eee; text-align: center; }}
h1 {{ margin: 4px 0; font-size: 22px; }} p {{ color: #bbb; margin: 6px; }}
img {{ display: block; margin: 10px auto; max-width: 100%; max-height: calc(100vh - 180px); object-fit: contain; background: white; }}
.buttons {{ display: flex; justify-content: center; gap: 28px; }} form {{ margin: 0; }}
button {{ border: 0; border-radius: 7px; padding: 13px 55px; font-size: 20px; cursor: pointer; }}
.yes {{ background: #77cf87; }} .no {{ background: #e27c7c; }} kbd {{ font-size: 13px; }}
</style></head><body>{content}
<script>
document.addEventListener('keydown', event => {{
  const key = event.key.toLowerCase();
  if (key === 'y' || key === 'n') {{
    const value = key === 'y' ? 'yes' : 'no';
    fetch('/verdict', {{method:'POST', headers:{{'Content-Type':'application/x-www-form-urlencoded'}}, body:'value='+value}})
      .then(() => window.location.reload());
  }}
}});
</script></body></html>"""
        return document.encode("utf-8")

    def record(self, verdict: str) -> None:
        self.session.record(verdict)
        self.session.advance()

    def run(self) -> None:
        reviewer = self

        class Handler(BaseHTTPRequestHandler):
            def send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/image.png" and reviewer.session.item is not None:
                    self.send_bytes(reviewer.render_png(), "image/png")
                elif path == "/":
                    self.send_bytes(reviewer.page(), "text/html; charset=utf-8")
                else:
                    self.send_bytes(b"Not found\n", "text/plain", 404)

            def do_POST(self) -> None:
                if urlparse(self.path).path != "/verdict":
                    self.send_bytes(b"Not found\n", "text/plain", 404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                values = parse_qs(self.rfile.read(length).decode("utf-8"))
                verdict = values.get("value", [""])[0]
                if verdict not in {"yes", "no"}:
                    self.send_bytes(b"Expected yes or no\n", "text/plain", 400)
                    return
                reviewer.record(verdict)
                self.send_response(303)
                self.send_header("Location", "/")
                self.end_headers()

            def log_message(self, format: str, *args) -> None:
                return

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        print(f"Headless mask reviewer: http://{self.host}:{self.port}", flush=True)
        if self.host in {"127.0.0.1", "localhost"}:
            print(
                f"If this is a remote host, forward it from your laptop: "
                f"ssh -L {self.port}:127.0.0.1:{self.port} <user>@<host>",
                flush=True,
            )
        print("Press Ctrl-C to stop without losing completed verdicts.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


def find_images(directory: Path, limit: int | None) -> list[Path]:
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".npy", ".npz", ".tif", ".tiff"})
    return paths[:limit] if limit else paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--masker", help="image-to-mask callable as module:function")
    parser.add_argument("--tag", default="", help="configuration/hyperparameter tag saved with this proposal")
    parser.add_argument("--limit", type=int, help="review at most this many images")
    parser.add_argument("--revisit", action="store_true", help="include images with an existing verdict")
    parser.add_argument("--ordered", action="store_true", help="use filename order instead of a deterministic shuffle")
    interface = parser.add_mutually_exclusive_group()
    interface.add_argument("--web", action="store_true", help="serve a browser UI (recommended over SSH)")
    interface.add_argument("--gui", action="store_true", help="force the Matplotlib desktop GUI")
    parser.add_argument("--host", default="127.0.0.1", help="web UI bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8765, help="web UI port")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    masker, spec = resolve_masker(args.masker)
    paths = find_images(args.images.resolve(), None)
    if not paths:
        raise SystemExit(f"no supported images found in {args.images}")
    if not args.ordered:
        random.Random(20260718).shuffle(paths)
    if args.limit:
        paths = paths[:args.limit]
    provenance_spec = spec + (f" | {args.tag}" if args.tag else "")
    linux_headless = sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    if args.web or (linux_headless and not args.gui):
        WebMaskReviewer(
            paths, masker, provenance_spec, revisit=args.revisit,
            host=args.host, port=args.port,
        ).run()
    else:
        MaskReviewer(paths, masker, provenance_spec, revisit=args.revisit).run()


if __name__ == "__main__":
    main()
