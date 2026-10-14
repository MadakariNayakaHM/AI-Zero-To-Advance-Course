"""Figures for Note 03 (minimum edit distance). Writes ../images/03_*.png.

Run:  python3 figures_03.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})


def med_table(x, y, sub=1):
    n, m = len(x), len(y)
    D = np.zeros((n + 1, m + 1), dtype=int)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = 0 if x[i - 1] == y[j - 1] else sub
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + s)
    return D


def backtrace(D, x, y, sub):
    i, j, path = len(x), len(y), [(len(x), len(y))]
    while i > 0 or j > 0:
        s = 0 if i and j and x[i - 1] == y[j - 1] else sub
        if i and j and D[i, j] == D[i - 1, j - 1] + s:
            i, j = i - 1, j - 1
        elif i and D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
        path.append((i, j))
    return path


# 1. The cell recipe: where D(i, j) comes from
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.set_xlim(0, 6.4); ax.set_ylim(0, 4.2); ax.axis("off")
cells = {(1, 2.3): ("D(i-1, j-1)", AQUA), (3.4, 2.3): ("D(i-1, j)", ORANGE),
         (1, 0.3): ("D(i, j-1)", BLUE), (3.4, 0.3): ("D(i, j)", INK)}
for (cx, cy), (lab, col) in cells.items():
    ax.add_patch(Rectangle((cx, cy), 2.0, 1.3, fc="white", ec=col, lw=2.2))
    ax.text(cx + 1.0, cy + 0.65, lab, ha="center", va="center", fontsize=12, color=col, weight="bold")
arrow = dict(arrowstyle="-|>", mutation_scale=18, lw=2)
ax.add_patch(FancyArrowPatch((3.0, 2.3), (3.5, 1.6), color=AQUA, **arrow))
ax.add_patch(FancyArrowPatch((4.4, 2.3), (4.4, 1.6), color=ORANGE, **arrow))
ax.add_patch(FancyArrowPatch((3.0, 0.95), (3.4, 0.95), color=BLUE, **arrow))
ax.text(2.0, 3.85, "diagonal: copy (+0) if letters match,\nelse substitute (+1, or +2 in Levenshtein-2)",
        ha="center", fontsize=9.5, color=AQUA)
ax.text(5.5, 3.0, "from the previous\nrow: delete x(i)\n+1", ha="left", fontsize=9.5, color=ORANGE)
ax.text(0.05, 1.75, "from the left:\ninsert y(j)  +1", ha="left", fontsize=9.5, color=BLUE)
ax.set_title("Each cell = the cheapest of three neighbours + that move's cost", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "03_cell_recipe.png"), dpi=150)
plt.close(fig)

# 2. intention -> execution, sub = 2: full table with the backtrace path (slide orientation, # at the bottom)
x, y = "INTENTION", "EXECUTION"
D = med_table(x, y, 2)
path = set(backtrace(D, x, y, 2))
n, m = len(x), len(y)
fig, ax = plt.subplots(figsize=(7.6, 7.0))
ax.imshow(D, origin="lower", cmap="Blues", alpha=0.55)
for i in range(n + 1):
    for j in range(m + 1):
        on = (i, j) in path
        ax.text(j, i, D[i, j], ha="center", va="center", fontsize=11,
                color=ORANGE if on else INK, weight="bold" if on else "normal")
        if on:
            ax.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fill=False, ec=ORANGE, lw=2))
ax.set_xticks(range(m + 1)); ax.set_xticklabels(["#"] + list(y))
ax.set_yticks(range(n + 1)); ax.set_yticklabels(["#"] + list(x))
ax.set_xlabel("target: EXECUTION  (moving right = insert)")
ax.set_ylabel("source: INTENTION  (moving up = delete)")
ax.set_title("Edit-distance table, substitution cost 2: D(9, 9) = 8\norange = one optimal backtrace path", fontsize=12)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "03_intention_execution_table.png"), dpi=150)
plt.close(fig)

# 3. Brute force vs dynamic programming
def delannoy(n):
    T = [[1] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            T[i][j] = T[i - 1][j] + T[i][j - 1] + T[i - 1][j - 1]
    return T[n][n]

ns = np.arange(1, 31)
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.semilogy(ns, [delannoy(int(k)) for k in ns], color=ORANGE, lw=2.4, label="edit paths to check by brute force")
ax.semilogy(ns, (ns + 1) ** 2, color=BLUE, lw=2.4, label="cells the DP table fills, (n+1)²")
ax.set_xlabel("length of both strings, n")
ax.set_ylabel("count (log scale)")
ax.set_title("Why dynamic programming: paths explode, the table does not", fontsize=12)
ax.annotate("n = 9 (intention):\n1,462,563 paths vs 100 cells", xy=(9, delannoy(9)), xytext=(11, 1e3),
            arrowprops=dict(arrowstyle="->", color=MUTED), fontsize=9.5, color=MUTED)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "03_paths_vs_cells.png"), dpi=150)
plt.close(fig)
print("wrote 03_cell_recipe.png, 03_intention_execution_table.png, 03_paths_vs_cells.png")
