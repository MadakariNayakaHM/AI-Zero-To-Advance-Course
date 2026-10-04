"""Extra figures for Note 01 (Linear Systems, Determinant & Inverse).

Run from this folder:  python figures_01.py
Figures are written to ../images/ with the prefix 01x_ (no clash with 01_three_cases.png).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close()
    print("saved", name)


def parallelogram(ax, A, color, label):
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    img = A @ sq
    ax.fill(sq[0], sq[1], color=GREY, alpha=0.15)
    ax.plot(sq[0], sq[1], color=GREY, lw=1, ls="--", label="unit square (area 1)")
    ax.fill(img[0], img[1], color=color, alpha=0.25)
    ax.plot(img[0], img[1], color=color, lw=2, label=label)
    ax.axhline(0, c="k", lw=0.8)
    ax.axvline(0, c="k", lw=0.8)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)


# 1. Determinant = area scaling factor
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
A = np.array([[2, 1], [1, 2]])
parallelogram(axs[0], A, BLUE, "image under A, area |det| = 3")
axs[0].set_title("A = [[2,1],[1,2]], det = 3")
axs[0].set_xlim(-0.5, 3.5); axs[0].set_ylim(-0.5, 3.5)
F = np.array([[0, 1], [1, 0]])
parallelogram(axs[1], F, ORANGE, "image under F, area 1, flipped")
axs[1].set_title("F = [[0,1],[1,0]], det = -1 (orientation flips)")
axs[1].set_xlim(-0.5, 1.8); axs[1].set_ylim(-0.5, 1.8)
S = np.array([[2, 1], [4, 2]])
parallelogram(axs[2], S, AQUA, "image under S, area 0 (a segment)")
axs[2].set_title("S = [[2,1],[4,2]], det = 0 (collapse)")
axs[2].set_xlim(-0.5, 3.5); axs[2].set_ylim(-0.5, 6.5)
save("01x_det_area.png")

# 2. Least-squares line through 4 points, with residuals
x = np.array([0, 1, 2, 3.]); y = np.array([1, 3, 4, 4.])
X = np.c_[np.ones_like(x), x]
c, m = np.linalg.solve(X.T @ X, X.T @ y)
fig, ax = plt.subplots(figsize=(5.5, 4))
xs = np.linspace(-0.3, 3.3, 50)
ax.plot(xs, c + m * xs, color=BLUE, lw=2, label=f"least-squares line y = {c:.1f} + {m:.1f}x")
for xi, yi in zip(x, y):
    ax.plot([xi, xi], [yi, c + m * xi], color=ORANGE, lw=1.5)
ax.plot([], [], color=ORANGE, lw=1.5, label="residuals (sum of squares = 1)")
ax.scatter(x, y, s=50, color=GREY, zorder=3, label="data (0,1), (1,3), (2,4), (3,4)")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8, loc="lower right")
ax.set_title("Normal equations: XᵀX w = Xᵀy")
save("01x_least_squares_line.png")

# 3. Ill-conditioning: nearly parallel lines
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
xs = np.linspace(-1, 3, 200)
for ax, b2, sol, title in [(axs[0], 2.0001, (1, 1), "b = (2, 2.0001)  →  x = (1, 1)"),
                           (axs[1], 2.0002, (0, 2), "b = (2, 2.0002)  →  x = (0, 2)")]:
    ax.plot(xs, 2 - xs, color=BLUE, lw=2, label="x + y = 2")
    ax.plot(xs, (b2 - xs) / 1.0001, color=ORANGE, lw=2, ls="--", label=f"x + 1.0001y = {b2}")
    ax.scatter(*sol, s=60, color=GREY, zorder=3, label=f"solution {sol}")
    ax.set_title(title); ax.set_xlim(-1, 3); ax.set_ylim(-1, 3); ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
fig.suptitle("Ill-conditioned system (κ ≈ 40 000): a change of 0.0001 in b moves the solution by √2")
save("01x_ill_conditioned.png")
