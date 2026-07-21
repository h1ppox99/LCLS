#!/usr/bin/env python3
"""Hand-curate the image dataset: show one random image, keep it or remove it.

A stripped-down sibling of ``review.py``.  There is no mask and no masker: each
step shows a single diffraction image sampled at random from ``data/images/``,
and you decide whether it stays in the dataset (**Keep**) or is dropped
(**Remove**).  Decisions are appended to ``data/labels/curation.csv`` and
removed files are *moved* (not deleted) to ``data/images_removed/`` so the action
is reversible.  Each already-decided image is skipped, so re-running resumes
where you left off and never shows the same image twice.

Headless nodes get a browser UI automatically; pass ``--gui`` for the desktop
Matplotlib window when a display is available.
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Reuse the image loading / display-scaling helpers from the mask reviewer.
from review import display_image, display_limits, load_image


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DEFAULT_IMAGES = DATA / "images"
LABELS = DATA / "labels"
REMOVED = DATA / "images_removed"
CURATION = LABELS / "curation.csv"

FIELDS = ["timestamp_utc", "image_id", "image_path", "verdict"]


def read_decided() -> set[str]:
    if not CURATION.exists():
        return set()
    with CURATION.open(newline="", encoding="utf-8") as stream:
        return {row["image_id"] for row in csv.DictReader(stream)}


def append_verdict(row: dict[str, str]) -> None:
    LABELS.mkdir(parents=True, exist_ok=True)
    exists = CURATION.exists()
    with CURATION.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@dataclass
class CurateItem:
    path: Path
    image: object  # np.ndarray


class CurateSession:
    """Draws un-decided images in a random order and records keep/remove."""

    def __init__(self, paths: list[Path], seed: int | None = None):
        decided = read_decided()
        self.pending = [p for p in paths if p.stem not in decided]
        random.Random(seed).shuffle(self.pending)
        self.total = len(paths)
        self.decided_count = len(decided)
        self.item: CurateItem | None = None

    def advance(self) -> CurateItem | None:
        while self.pending:
            path = self.pending.pop()
            try:
                self.item = CurateItem(path, load_image(path))
            except Exception as exc:
                print(f"skipping {path.name}: {exc}", file=sys.stderr, flush=True)
                continue
            return self.item
        self.item = None
        return None

    def record(self, verdict: str) -> None:
        if self.item is None:
            return
        item = self.item
        if verdict == "remove":
            REMOVED.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item.path), str(REMOVED / item.path.name))
        append_verdict({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "image_id": item.path.stem,
            "image_path": str(item.path.relative_to(ROOT)),
            "verdict": verdict,
        })
        self.decided_count += 1
        self.item = None


# --------------------------------------------------------------------------- #
# Matplotlib desktop UI (used when a display is available and --gui is set).
# --------------------------------------------------------------------------- #
class DesktopCurator:
    def __init__(self, session: CurateSession):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self.plt = plt
        self.session = session
        self.figure, self.axis = plt.subplots(figsize=(8, 8))
        self.figure.subplots_adjust(bottom=0.14)
        keep_axis = self.figure.add_axes((0.30, 0.03, 0.16, 0.07))
        drop_axis = self.figure.add_axes((0.54, 0.03, 0.16, 0.07))
        self.keep_button = Button(keep_axis, "Keep [K]", color="#b8e6b8", hovercolor="#83d283")
        self.drop_button = Button(drop_axis, "Remove [R]", color="#f2b6b6", hovercolor="#e68181")
        self.keep_button.on_clicked(lambda _: self.record("keep"))
        self.drop_button.on_clicked(lambda _: self.record("remove"))
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.show_next()

    def show_next(self) -> None:
        item = self.session.advance()
        self.axis.clear()
        self.axis.axis("off")
        if item is None:
            self.axis.text(0.5, 0.5, "No images left", ha="center", va="center", fontsize=22)
        else:
            shown = display_image(item.image)
            low, high = display_limits(item.image)
            self.axis.imshow(shown, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
            self.axis.set_title(item.path.stem)
            self.figure.suptitle(f"decided {self.session.decided_count}/{self.session.total}")
        self.figure.canvas.draw_idle()

    def record(self, verdict: str) -> None:
        self.session.record(verdict)
        self.show_next()

    def on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key == "k":
            self.record("keep")
        elif key == "r":
            self.record("remove")
        elif key in {"q", "escape"}:
            self.plt.close(self.figure)

    def run(self) -> None:
        self.plt.show()


# --------------------------------------------------------------------------- #
# Dependency-free browser UI (default on headless / SSH compute nodes).
# --------------------------------------------------------------------------- #
class WebCurator:
    def __init__(self, session: CurateSession, host: str = "127.0.0.1", port: int = 8766):
        self.session = session
        self.host, self.port = host, port
        self.session.advance()

    def render_png(self) -> bytes:
        import io
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        item = self.session.item
        if item is None:
            return b""
        figure = Figure(figsize=(8, 8), tight_layout=True)
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        shown = display_image(item.image)
        low, high = display_limits(item.image)
        axis.imshow(shown, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
        axis.axis("off")
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=110)
        return buffer.getvalue()

    def page(self) -> bytes:
        item = self.session.item
        if item is None:
            content = "<h1>No images left</h1><p>You can stop the server with Ctrl-C.</p>"
        else:
            image_id = html.escape(item.path.stem)
            content = f"""
              <h1>{image_id}</h1>
              <p>Decided {self.session.decided_count} of {self.session.total} images.</p>
              <img src="/image.png" alt="Diffraction image">
              <div class="buttons">
                <form method="post" action="/verdict"><button class="keep" name="value" value="keep">Keep <kbd>K</kbd></button></form>
                <form method="post" action="/verdict"><button class="remove" name="value" value="remove">Remove <kbd>R</kbd></button></form>
              </div>
            """
        document = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Dataset curation</title>
<style>
body {{ margin: 0 auto; max-width: 1100px; padding: 12px 24px 30px; font: 16px sans-serif; background: #17191d; color: #eee; text-align: center; }}
h1 {{ margin: 4px 0; font-size: 22px; }} p {{ color: #bbb; margin: 6px; }}
img {{ display: block; margin: 10px auto; max-width: 100%; max-height: calc(100vh - 180px); object-fit: contain; background: white; }}
.buttons {{ display: flex; justify-content: center; gap: 28px; }} form {{ margin: 0; }}
button {{ border: 0; border-radius: 7px; padding: 13px 55px; font-size: 20px; cursor: pointer; }}
.keep {{ background: #77cf87; }} .remove {{ background: #e27c7c; }} kbd {{ font-size: 13px; }}
</style></head><body>{content}
<script>
document.addEventListener('keydown', event => {{
  const key = event.key.toLowerCase();
  if (key === 'k' || key === 'r') {{
    const value = key === 'k' ? 'keep' : 'remove';
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
        curator = self

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
                if path == "/image.png" and curator.session.item is not None:
                    self.send_bytes(curator.render_png(), "image/png")
                elif path == "/":
                    self.send_bytes(curator.page(), "text/html; charset=utf-8")
                else:
                    self.send_bytes(b"Not found\n", "text/plain", 404)

            def do_POST(self) -> None:
                if urlparse(self.path).path != "/verdict":
                    self.send_bytes(b"Not found\n", "text/plain", 404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                values = parse_qs(self.rfile.read(length).decode("utf-8"))
                verdict = values.get("value", [""])[0]
                if verdict not in {"keep", "remove"}:
                    self.send_bytes(b"Expected keep or remove\n", "text/plain", 400)
                    return
                curator.record(verdict)
                self.send_response(303)
                self.send_header("Location", "/")
                self.end_headers()

            def log_message(self, format: str, *args) -> None:
                return

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        print(f"Headless dataset curator: http://{self.host}:{self.port}", flush=True)
        if self.host in {"127.0.0.1", "localhost"}:
            print(
                f"If this is a remote host, forward it from your laptop: "
                f"ssh -L {self.port}:127.0.0.1:{self.port} <user>@<host>",
                flush=True,
            )
        print("Press Ctrl-C to stop without losing recorded decisions.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


def find_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in {".npy", ".npz", ".tif", ".tiff"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--seed", type=int, help="fix the random sampling order (default: fresh each run)")
    interface = parser.add_mutually_exclusive_group()
    interface.add_argument("--web", action="store_true", help="serve a browser UI (recommended over SSH)")
    interface.add_argument("--gui", action="store_true", help="force the Matplotlib desktop GUI")
    parser.add_argument("--host", default="127.0.0.1", help="web UI bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8766, help="web UI port")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    paths = find_images(args.images.resolve())
    if not paths:
        raise SystemExit(f"no supported images found in {args.images}")
    session = CurateSession(paths, seed=args.seed)

    linux_headless = sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    if args.web or (linux_headless and not args.gui):
        WebCurator(session, host=args.host, port=args.port).run()
    else:
        DesktopCurator(session).run()


if __name__ == "__main__":
    main()
