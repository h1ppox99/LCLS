#!/usr/bin/env python3
"""
variance_sweep.py -- threshold sweep for the low-variance masking method.

Geometry is applied first (free floor).  We then treat the low-variance score
s = -z  (z = robust-MAD z-score of log uSTD) as a continuous detector for the
residual bad pixels T = human_Mask & ~G, evaluated over the candidate set
(real & ~G).  Produces:
  * precision-recall curve + average-precision (PR-AUC)
  * ROC curve + ROC-AUC
  * precision / recall / F1 vs threshold k
  * combined G|A(k) IoU vs human_Mask vs threshold k  (the metric that matters)
and prints the best operating points.

Run:  python variance_sweep.py     (from src/automask/)
"""
from __future__ import annotations
import os, sys
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../automask
sys.path.insert(0, AUTOMASK)
from dataset import load_mask, score
from methods import geometry_mask, load_all, pad_mask

FIG_DIR = os.path.join(AUTOMASK, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
RUN = 475


def curves(score_flat, y):
    """PR and ROC from a continuous score (descending) vs boolean truth y."""
    order = np.argsort(-score_flat)
    y = y[order].astype(np.int64)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    P, N = y.sum(), (1 - y).sum()
    recall = tp / P
    precision = tp / np.maximum(tp + fp, 1)
    fpr = fp / N
    ap = np.sum(np.diff(np.r_[0, recall]) * precision)          # PR-AUC
    roc_auc = np.trapz(recall, fpr)                             # ROC-AUC
    return precision, recall, fpr, ap, roc_auc


def main():
    sumimg, mean, var, human = load_all(RUN)
    real = sumimg != 0
    G = geometry_mask(real)
    T = human & ~G
    cand = real & ~G                                            # candidate pixels

    m = var > 0
    lr = np.log10(var[m]); c = np.median(lr)
    s = np.median(np.abs(lr - c)) * 1.4826 + 1e-12
    z = np.zeros(var.shape); z[m] = (lr - c) / s
    lowscore = -z                                              # higher = more low-variance

    precision, recall, fpr, ap, roc_auc = curves(lowscore[cand], T[cand])
    print(f"low-variance feature on residual T:  PR-AUC(AP)={ap:.3f}  ROC-AUC={roc_auc:.3f}")

    # threshold sweep with the ACTUAL method (dilated mask), on residual + combined
    ks = np.arange(1.5, 9.01, 0.25)
    prec_k, rec_k, f1_k, iou_k = [], [], [], []
    for k in ks:
        A = pad_mask((z < -k) & real)          # same 5x5 padding as the method
        Ao = A & ~G
        st = score(Ao, T); sc = score(G | A, human)
        prec_k.append(st["precision"]); rec_k.append(st["recall"])
        f1_k.append(0 if st["precision"] + st["recall"] == 0 else
                    2 * st["precision"] * st["recall"] / (st["precision"] + st["recall"]))
        iou_k.append(sc["iou"])
    prec_k, rec_k, f1_k, iou_k = map(np.array, (prec_k, rec_k, f1_k, iou_k))

    kbest_iou = ks[np.argmax(iou_k)]; kbest_f1 = ks[np.argmax(f1_k)]
    print(f"best combined IoU={iou_k.max():.3f} at k={kbest_iou:.2f}  "
          f"(current default k=5.0 -> IoU={iou_k[np.argmin(abs(ks-7))]:.3f})")
    print(f"best residual F1 ={f1_k.max():.3f} at k={kbest_f1:.2f}")

    # ---- figure ----------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    ax[0, 0].plot(recall, precision, lw=1.5)
    ax[0, 0].set_title(f"Precision-Recall (residual T)   AP={ap:.3f}")
    ax[0, 0].set_xlabel("recall"); ax[0, 0].set_ylabel("precision")
    ax[0, 0].set_xlim(0, 1); ax[0, 0].set_ylim(0, 1.02); ax[0, 0].grid(alpha=.3)

    ax[0, 1].plot(fpr, recall, lw=1.5); ax[0, 1].plot([0, 1], [0, 1], "k--", lw=.7)
    ax[0, 1].set_title(f"ROC   AUC={roc_auc:.3f}")
    ax[0, 1].set_xlabel("false-positive rate"); ax[0, 1].set_ylabel("true-positive rate")
    ax[0, 1].set_xlim(0, 1); ax[0, 1].set_ylim(0, 1.02); ax[0, 1].grid(alpha=.3)

    ax[1, 0].plot(ks, prec_k, label="precision (T)")
    ax[1, 0].plot(ks, rec_k, label="recall (T)")
    ax[1, 0].plot(ks, f1_k, label="F1 (T)", lw=2)
    ax[1, 0].axvline(7, color="gray", ls=":", label="current k=7")
    ax[1, 0].set_title("residual precision / recall / F1 vs threshold k")
    ax[1, 0].set_xlabel("z threshold k"); ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)

    ax[1, 1].plot(ks, iou_k, color="C3", lw=2)
    ax[1, 1].axvline(7, color="gray", ls=":")
    ax[1, 1].axvline(kbest_iou, color="C2", ls="--", label=f"best k={kbest_iou:.2f}")
    ax[1, 1].axhline(0.473, color="k", ls=":", lw=.7, label="geometry alone")
    ax[1, 1].set_title(f"combined G∪A(k) IoU vs human   (max {iou_k.max():.3f})")
    ax[1, 1].set_xlabel("z threshold k"); ax[1, 1].set_ylabel("IoU"); ax[1, 1].legend()
    ax[1, 1].grid(alpha=.3)

    fig.suptitle(f"xppl1016922 run {RUN} — low-variance threshold sweep", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG_DIR, f"variance_sweep_run{RUN:04d}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
