from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../automask
sys.path.insert(0, AUTOMASK)
from dataset import load_image, load_mask, score
from methods import geometry_mask, method_variance, method_blackhat, _agree

FEATURES = os.path.join(AUTOMASK, "data", "features")
IMAGES = os.path.join(AUTOMASK, "outputs", "figures", "intensity_bands")
RESULTS = os.path.join(AUTOMASK, "outputs", "masks")
os.makedirs(IMAGES, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)