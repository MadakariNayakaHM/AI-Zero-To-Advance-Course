"""Figures for Note 04 (Supervised learning: classification). Writes ../images/04_clf_*.png.

Run from this folder:  python figures_04.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"
sig = lambda z: 1 / (1 + np.exp(-z))


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close()
    print("saved", name)


# 1. Linear vs logistic on the slide data, and what one extreme point does
X = np.array([27, 2, 100, 40, 14], dtype=float)
y = np.array([0, 0, 1, 1, 0])
X2, y2 = np.append(X, 400), np.append(y, 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, (xs, ys, title) in zip(axes, [(X, y, "Slide data (5 reviews)"), (X2, y2, "Add one review with 400 words")]):
    grid = np.linspace(0, xs.max() * 1.05, 400)
    lin = LinearRegression().fit(xs[:, None], ys)
    log = LogisticRegression(C=1e4, max_iter=10000).fit(xs[:, None], ys)
    ax.scatter(xs, ys, c=[ORANGE if v else BLUE for v in ys], s=50, zorder=3, ec="white")
    ax.plot(grid, lin.predict(grid[:, None]), color=GREY, label="linear regression")
    ax.plot(grid, log.predict_proba(grid[:, None])[:, 1], color=ORANGE, label="logistic regression")
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    xl = (0.5 - lin.intercept_) / lin.coef_[0]
    xg = -log.intercept_[0] / log.coef_[0, 0]
    ax.axvline(xl, color=GREY, ls="--", lw=1)
    ax.axvline(xg, color=ORANGE, ls="--", lw=1)
    ax.set_title(f"{title}\nlinear threshold x={xl:.1f}, logistic x={xg:.1f}", fontsize=10)
    ax.set_xlabel("word count")
    ax.set_ylim(-0.3, 1.4)
axes[0].set_ylabel("prediction / P(y=1)")
axes[0].legend(loc="upper left", fontsize=8)
save("04_clf_linear_vs_logistic.png")

# 2. Cost as a function of w: squared error is non-convex, log-loss is convex
xw = np.array([2, 3, 5, 26, 27, 28, 29, 30, 1, 4, 6, 31, 32], dtype=float)
yw = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1])
ws = np.linspace(-3, 3, 601)
mse, ce = [], []
for w in ws:
    z = w * (xw - 16)
    mse.append(np.mean((sig(z) - yw) ** 2) / 2)
    ce.append(np.mean(yw * np.logaddexp(0, -z) + (1 - yw) * np.logaddexp(0, z)))
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
axes[0].plot(ws, mse, color=GREY)
axes[0].set_title("Squared error of g(z): flat plateaus,\nnon-convex", fontsize=10)
axes[1].plot(ws, ce, color=ORANGE)
axes[1].set_title("Log-loss: convex, one minimum", fontsize=10)
for ax in axes[:2]:
    ax.set_xlabel("w   (model g(w·(x−16)))")
    ax.set_ylabel("cost J(w)")
f = np.linspace(0.001, 0.999, 400)
axes[2].plot(f, -np.log(f), color=ORANGE, label="y = 1:  −log f")
axes[2].plot(f, -np.log(1 - f), color=BLUE, label="y = 0:  −log(1−f)")
axes[2].plot(f, (1 - f) ** 2, color=GREY, ls="--", label="(1−f)² for y = 1")
axes[2].set_ylim(0, 5)
axes[2].set_xlabel("predicted probability f")
axes[2].set_title("Per-sample loss", fontsize=10)
axes[2].legend(fontsize=8)
save("04_clf_cost_convexity.png")

# 3. Decision boundaries: the two slide examples
fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"wspace": 0.35})
g = np.linspace(-0.5, 4, 300)
A, B = np.meshgrid(g, g)
axes[0].contourf(A, B, sig(A + B - 3), levels=20, cmap="coolwarm", alpha=0.6)
axes[0].contour(A, B, A + B - 3, levels=[0], colors="k")
axes[0].set_title("w1=w2=1, b=−3  →  x1 + x2 = 3", fontsize=10)
g = np.linspace(-2, 2, 300)
A, B = np.meshgrid(g, g)
cs = axes[1].contourf(A, B, sig(A ** 2 + B ** 2 - 1), levels=20, cmap="coolwarm", alpha=0.6)
axes[1].contour(A, B, A ** 2 + B ** 2 - 1, levels=[0], colors="k")
axes[1].set_title("features x1², x2²; w1=w2=1, b=−1  →  circle", fontsize=10)
for ax in axes:
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_aspect("equal")
fig.colorbar(cs, ax=axes, shrink=0.8, label="g(z) = P(y=1)")
plt.savefig(os.path.join(OUT, "04_clf_boundaries.png"), bbox_inches="tight")
plt.close()
print("saved 04_clf_boundaries.png")

# 4. Threshold sweep and ROC on the 10-point example
scores = np.array([0.95, 0.85, 0.80, 0.70, 0.55, 0.45, 0.40, 0.30, 0.20, 0.10])
truth = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 0])
fpr, tpr, thr = roc_curve(truth, scores)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].scatter(scores, np.zeros_like(scores), c=[ORANGE if t else BLUE for t in truth], s=80, ec="white", zorder=3)
for s, t in zip(scores, truth):
    axes[0].text(s, 0.08, str(t), ha="center")
axes[0].axvline(0.5, color="k", ls="--")
axes[0].text(0.51, -0.18, "threshold 0.5:\nTP=4 FP=1 FN=1 TN=4", fontsize=8)
axes[0].set_yticks([]); axes[0].set_ylim(-0.3, 0.3); axes[0].set_xlabel("model score  (label shown above each point)")
axes[0].set_title("Moving the threshold trades FP for FN", fontsize=10)
axes[1].step(fpr, tpr, where="post", color=ORANGE, lw=2)
axes[1].fill_between(fpr, tpr, step="post", alpha=0.15, color=ORANGE)
axes[1].plot([0, 1], [0, 1], color=GREY, ls=":")
axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate (recall)")
axes[1].set_title(f"ROC curve, AUC = {roc_auc_score(truth, scores):.2f}", fontsize=10)
save("04_clf_roc.png")

# 5. Six classifiers on the same 2-D data
Xm, ym = make_moons(n_samples=300, noise=0.25, random_state=0)
models = {"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression()),
          "k-NN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
          "Gaussian Naive Bayes": GaussianNB(),
          "Decision Tree": DecisionTreeClassifier(random_state=0),
          "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0),
          "SVM (RBF)": make_pipeline(StandardScaler(), SVC())}
pad = 0.3
xx, yy = np.meshgrid(np.linspace(Xm[:, 0].min() - pad, Xm[:, 0].max() + pad, 300),
                     np.linspace(Xm[:, 1].min() - pad, Xm[:, 1].max() + pad, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, (name, m) in zip(axes.ravel(), models.items()):
    m.fit(Xm, ym)
    Z = m.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    ax.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap="coolwarm", s=12, ec="white", lw=0.3)
    ax.set_title(f"{name}  (train acc {m.score(Xm, ym):.2f})", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
save("04_clf_model_boundaries.png")
