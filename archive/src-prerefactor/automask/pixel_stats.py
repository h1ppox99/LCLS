from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../automask
sys.path.insert(0, AUTOMASK)
from dataset import load_image, load_mask, manifest

FEAT = os.path.join(AUTOMASK, "data", "features")
IMG_OUT = os.path.join(AUTOMASK, "outputs", "figures")
os.makedirs(FEAT, exist_ok=True)
os.makedirs(IMG_OUT, exist_ok=True)