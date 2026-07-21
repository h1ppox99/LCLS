#!/usr/bin/env python3
\"\"\"Score an image-to-mask callable against user-accepted reference masks.\"\"\"
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from dataset import score
from review import DATA, DEFAULT_IMAGES, REFERENCES, ROOT, load_image, resolve_masker, validate_mask


def evaluate(masker_spec: str, images: Path = DEFAULT_IMAGES) -> dict:
    masker, _ = resolve_masker(masker_spec)
    per_image = []
    for reference_path in sorted(REFERENCES.glob(\"*.npy\")):
        image_path = images / f\"{reference_path.stem}.npy\"
        if not image_path.exists():
            continue
        image = load_image(image_path)
        prediction = validate_mask(masker(image), image)
        truth = np.load(reference_path, allow_pickle=False).astype(bool)
        metrics = score(prediction, truth)
        per_image.append({\"image_id\": reference_path.stem, **metrics})
    if not per_image:
        raise RuntimeError(\"no accepted reference masks match the image directory\")
    summary = {
        \"masker\": masker_spec,
        \"images\": len(per_image),
        \"mean_iou\": float(np.mean([row[\"iou\"] for row in per_image])),
        \"mean_precision\": float(np.mean([row[\"precision\"] for row in per_image])),
        \"mean_recall\": float(np.mean([row[\"recall\"] for row in per_image])),
        \"per_image\": per_image,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(\"--masker\", required=True, help=\"image-to-mask callable as module:function\")
    parser.add_argument(\"--images\", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument(\"--output\", type=Path, help=\"optional JSON output path\")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    result = evaluate(args.masker, args.images.resolve())
    rendered = json.dumps(result, indent=2) + \"\
    if args.output:
        args.output.write_text(rendered, encoding=\"utf-8\")
    print(rendered, end=\"\")


if __name__ == \"__main__\":
    main()