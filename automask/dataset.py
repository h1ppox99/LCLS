"""
dataset.py -- numpy-only access to the frozen masking dataset.

No psana, no h5py, no LCLS filesystem: this only loads frozen benchmark arrays.
All masks follow one convention: bool, True == masked.

    from automask.dataset import load_image, load_mask, list_images, list_masks

    img  = load_image("sum_calib_run0475")        # (1064, 1030) float32
    gt   = load_mask("human_Mask")                # (1064, 1030) bool, True==masked
"""
from __future__ import annotations
import os, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_IMG = os.path.join(_DATA, "images")
_MSK = os.path.join(_DATA, "masks")


def _load(dirpath: str, name: str, form: str) -> np.ndarray:
    if name.endswith(".npy"):
        name = name[:-4]
    if not name.endswith(("_asm", "_panel")):
        name = f"{name}_{form}"
    path = os.path.join(dirpath, name + ".npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}\navailable: {sorted(os.listdir(dirpath))}")
    return np.load(path)


def load_image(name: str, form: str = "asm") -> np.ndarray:
    """Load a sum image. form='asm' (1064,1030) or 'panel' (2,512,1024)."""
    return _load(_IMG, name, form)


def load_mask(name: str, form: str = "asm") -> np.ndarray:
    """Load a reference mask (bool, True==masked). form='asm' or 'panel'."""
    return _load(_MSK, name, form).astype(bool)


def list_images() -> list[str]:
    return sorted(n[:-4] for n in os.listdir(_IMG) if n.endswith(".npy"))


def list_masks() -> list[str]:
    return sorted(n[:-4] for n in os.listdir(_MSK) if n.endswith(".npy"))


def manifest() -> dict:
    with open(os.path.join(_DATA, "manifest.json")) as f:
        return json.load(f)


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    """IoU / precision / recall of a predicted mask vs a reference (True==masked)."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return {"iou": iou, "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "fn": fn}


if __name__ == "__main__":
    print("images:", list_images())
    print("masks :", list_masks())
