"""Figures for Note 02 (ML pipeline deep dive).

Run from this folder:  python figures_02.py
Figures are written to ../images/02_*.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close()
    print("saved", name)


# 1. Sigmoid and binary cross-entropy
z = np.linspace(-8, 8, 400)
s = 1 / (1 + np.exp(-z))
p = np.linspace(0.001, 0.999, 400)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(z, s, lw=2)
ax[0].axhline(0.5, ls="--", c="gray"); ax[0].axvline(0, ls="--", c="gray")
ax[0].set(title="Sigmoid  σ(z) = 1/(1+e^-z)", xlabel="z = wᵀx + b", ylabel="P(y=1|x)")
ax[1].plot(z, s * (1 - s), lw=2, c="tab:orange")
ax[1].set(title="Derivative σ'(z) = σ(z)(1-σ(z)), max 0.25", xlabel="z")
ax[2].plot(p, -np.log(p), lw=2, label="y = 1: -ln p")
ax[2].plot(p, -np.log(1 - p), lw=2, label="y = 0: -ln(1-p)")
ax[2].set(title="Binary cross-entropy per sample", xlabel="predicted p", ylabel="loss", ylim=(0, 5))
ax[2].legend()
save("02_sigmoid_bce.png")

# 2. Threshold trade-off on synthetic scores (10% positives)
rng = np.random.default_rng(0)
neg = rng.normal(-1.0, 1.0, 900)
pos = rng.normal(1.0, 1.0, 100)
scores = 1 / (1 + np.exp(-np.r_[neg, pos]))
y = np.r_[np.zeros(900), np.ones(100)]
prec, rec, thr = precision_recall_curve(y, scores)
f1 = 2 * prec * rec / (prec + rec + 1e-12)
fpr, tpr, _ = roc_curve(y, scores)
auc = roc_auc_score(y, scores)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(thr, prec[:-1], label="precision"); ax[0].plot(thr, rec[:-1], label="recall")
ax[0].plot(thr, f1[:-1], label="F1")
ax[0].set(title="Moving the threshold", xlabel="threshold t", ylabel="value"); ax[0].legend()
ax[1].plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}"); ax[1].plot([0, 1], [0, 1], "--", c="gray", label="random")
ax[1].set(title="ROC curve", xlabel="FPR = 1 - specificity", ylabel="TPR = recall"); ax[1].legend()
ax[2].plot(rec, prec, lw=2); ax[2].axhline(0.1, ls="--", c="gray", label="random = prevalence 0.1")
ax[2].set(title="Precision-recall curve", xlabel="recall", ylabel="precision"); ax[2].legend()
save("02_threshold_roc_pr.png")
print(f"synthetic AUC = {auc:.4f}")
