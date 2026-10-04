"""Extra figures for Note 04 (Span, Independence, Basis & Dimension).

Run from this folder:  python figures_04.py
Figures are written to ../images/ with the prefix 04x_ (no clash with 04_shear.png).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import idctn
from sklearn.decomposition import PCA

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close()
    print("saved", name)


def grid(ax, b1, b2, color, rng=range(-6, 7)):
    for k in rng:
        p = np.array([k * b1 - 30 * b2, k * b1 + 30 * b2])
        q = np.array([k * b2 - 30 * b1, k * b2 + 30 * b1])
        ax.plot(p[:, 0], p[:, 1], color=color, lw=0.6, alpha=0.5)
        ax.plot(q[:, 0], q[:, 1], color=color, lw=0.6, alpha=0.5)


def arrow(ax, v, color, label, start=(0, 0), offset=(0.15, 0.0), ls="-"):
    ax.annotate("", xy=(start[0] + v[0], start[1] + v[1]), xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2, ls=ls))
    if label:
        ax.text(start[0] + v[0] + offset[0], start[1] + v[1] + offset[1], label, color=color, fontsize=10)


# 1. Same point, two coordinate systems ("spectacles")
x = np.array([4, 5])
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
e1, e2 = np.array([1, 0]), np.array([0, 1])
u, v = np.array([2, 5]), np.array([1, 5])
for ax, (b1, b2, n1, n2, c, title) in zip(axs, [
        (e1, e2, "e1", "e2", GREY, "Standard basis: x = 4·e1 + 5·e2, coordinates (4, 5)"),
        (u, v, "u=(2,5)", "v=(1,5)", BLUE, "Basis B = {u, v}: x = 3u − 2v, coordinates (3, −2)")]):
    grid(ax, b1, b2, c, rng=range(-20, 21))
    arrow(ax, b1, ORANGE, n1, offset=(0.3, -1.6) if n1 != "e1" else (0.2, -0.6))
    arrow(ax, b2, AQUA, n2, offset=(-1.6, 0.3))
    ax.scatter(*x, s=70, color="k", zorder=5)
    ax.text(x[0] + 0.4, x[1] + 0.3, "x = (4, 5)", fontsize=10)
    ax.axhline(0, c="k", lw=0.8); ax.axvline(0, c="k", lw=0.8)
    ax.set_xlim(-3, 9); ax.set_ylim(-3, 17); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
arrow(axs[1], 3 * u, ORANGE, "3u", offset=(0.2, 0), ls="--")
arrow(axs[1], -2 * v, AQUA, "−2v", start=tuple(3 * u), offset=(0.6, 4), ls="--")
save("04x_change_of_basis.png")

# 2. The 64 DCT-II basis images (the JPEG basis for 8x8 blocks)
fig, axs = plt.subplots(8, 8, figsize=(6.5, 6.5))
for i in range(8):
    for j in range(8):
        c = np.zeros((8, 8)); c[i, j] = 1
        axs[i, j].imshow(idctn(c, type=2, norm="ortho"), cmap="gray", vmin=-0.25, vmax=0.25)
        axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
fig.suptitle("The 64 orthonormal 2-D DCT basis images (frequency increases right and down)", fontsize=9)
save("04x_dct_basis.png")

# 3. PCA picks a better basis for a correlated 2-D cloud
rng = np.random.default_rng(0)
X = rng.multivariate_normal([0, 0], [[3, 2], [2, 2]], size=500)
pca = PCA(n_components=2).fit(X)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(X[:, 0], X[:, 1], s=6, color=GREY, alpha=0.4, label="500 samples")
for k, (comp, var, col) in enumerate(zip(pca.components_, pca.explained_variance_, [BLUE, ORANGE])):
    arrow(ax, 2 * np.sqrt(var) * comp, col, f"q{k + 1}")
    ax.plot([], [], color=col, lw=2, label=f"q{k + 1}: {pca.explained_variance_ratio_[k]:.1%} of variance")
arrow(ax, (2.5, 0), AQUA, "e1", offset=(0.1, -0.5)); arrow(ax, (0, 2.5), AQUA, "e2", offset=(-0.6, 0.1))
ax.set_aspect("equal"); ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
ax.axhline(0, c="k", lw=0.5); ax.axvline(0, c="k", lw=0.5)
ax.set_title("Standard basis (e1, e2) vs PCA basis (q1, q2)", fontsize=10)
ax.legend(loc="lower right", fontsize=8)
save("04x_pca_basis.png")
