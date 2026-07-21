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