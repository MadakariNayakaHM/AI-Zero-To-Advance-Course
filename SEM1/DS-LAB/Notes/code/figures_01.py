"""Figures for Note 01 (NumPy foundations). Writes ../images/01_*.png.

Run:  python3 figures_01.py
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
LIGHT = "#e8f0fb"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})


def grid(ax, M, x0, y0, face=LIGHT, edge=BLUE, fmt="{:g}", ghost=None):
    """Draw matrix M with its top-left cell at (x0, y0); cells are 1x1, y grows downward."""
    M = np.atleast_2d(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            g = ghost is not None and ghost[i, j]
            ax.add_patch(Rectangle((x0 + j, y0 + i), 1, 1, facecolor="white" if g else face,
                                   edgecolor=edge, lw=1.4, ls="--" if g else "-"))
            ax.text(x0 + j + 0.5, y0 + i + 0.5, fmt.format(M[i, j]), ha="center", va="center",
                    color=MUTED if g else INK, fontsize=12)


def canvas(ax, w, h):
    ax.set_xlim(-0.5, w)
    ax.set_ylim(h, -1.2)
    ax.set_aspect("equal")
    ax.axis("off")


# 1. Axis sums on the (2, 5) array from the class notebook
a = np.arange(10).reshape(2, 5)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
ax = axes[0]
canvas(ax, 6.5, 4.2)
grid(ax, a, 0, 0)
for j in range(5):
    ax.annotate("", xy=(j + 0.5, 2.25), xytext=(j + 0.5, 0.1),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))
grid(ax, a.sum(axis=0)[None, :], 0, 2.6, face="#fde8de", edge=ORANGE)
ax.set_title("a.sum(axis=0): collapse the rows\n→ one total per column, shape (5,)", fontsize=11)
ax = axes[1]
canvas(ax, 7.8, 4.2)
grid(ax, a, 0, 0)
for i in range(2):
    ax.annotate("", xy=(5.85, i + 0.5), xytext=(0.1, i + 0.5),
                arrowprops=dict(arrowstyle="->", color=AQUA, lw=1.6))
grid(ax, a.sum(axis=1)[:, None], 6.2, 0, face="#dcf3ea", edge=AQUA)
ax.set_title("a.sum(axis=1): collapse the columns\n→ one total per row, shape (2,)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "01_axis_sums.png"), dpi=150)
plt.close(fig)

# 2. Broadcasting (3,1) + (4,) -> (3,4)
col = np.array([[10], [20], [30]])
row = np.array([1, 2, 3, 4])
fig, ax = plt.subplots(figsize=(11, 3.4))
canvas(ax, 18.5, 3.4)
ghost_c = np.ones((3, 4), bool); ghost_c[:, 0] = False
grid(ax, np.repeat(col, 4, axis=1), 0, 0, ghost=ghost_c)
ax.text(2, -0.5, "col, shape (3,1)\nstretched along axis 1", ha="center", fontsize=10)
ax.text(4.7, 1.5, "+", fontsize=20, ha="center", va="center")
ghost_r = np.ones((3, 4), bool); ghost_r[0, :] = False
grid(ax, np.repeat(row[None, :], 3, axis=0), 5.4, 0, face="#fde8de", edge=ORANGE, ghost=ghost_r)
ax.text(7.4, -0.5, "row, shape (4,) → (1,4)\nstretched along axis 0", ha="center", fontsize=10)
ax.text(10.1, 1.5, "=", fontsize=20, ha="center", va="center")
grid(ax, col + row, 10.8, 0, face="#dcf3ea", edge=AQUA)
ax.text(12.8, -0.5, "result, shape (3,4)", ha="center", fontsize=10)
ax.text(15.3, 1.5, "dashed cells are never\nstored: NumPy reuses\nthe same values\n(no copy is made)",
        fontsize=10, color=MUTED, va="center")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "01_broadcasting.png"), dpi=150)
plt.close(fig)

# 3. Python loop vs vectorized sum of squares
sizes = np.array([10, 100, 1_000, 10_000, 100_000, 1_000_000])
loop_t, vec_t = [], []
for n in sizes:
    lst, arr = list(range(n)), np.arange(n)
    reps = max(1, 200_000 // n)
    t0 = time.perf_counter()
    for _ in range(reps):
        sum(v * v for v in lst)
    loop_t.append((time.perf_counter() - t0) / reps)
    t0 = time.perf_counter()
    for _ in range(reps * 10):
        (arr * arr).sum()
    vec_t.append((time.perf_counter() - t0) / (reps * 10))
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.loglog(sizes, np.array(loop_t) * 1e3, "o-", color=ORANGE, lw=2, label="Python loop: sum(v*v for v in lst)")
ax.loglog(sizes, np.array(vec_t) * 1e3, "o-", color=BLUE, lw=2, label="NumPy: (arr*arr).sum()")
ax.set_xlabel("number of elements n (log scale)")
ax.set_ylabel("time per run, ms (log scale)")
ax.set_title(f"Same answer, ~{loop_t[-1] / vec_t[-1]:.0f}× faster at n = 1,000,000 (this machine)")
ax.legend(frameon=False)
ax.grid(True, which="major", color="#dddddd", lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "01_vectorization_speed.png"), dpi=150)
plt.close(fig)
print("wrote", sorted(f for f in os.listdir(OUT) if f.startswith("01_")))
